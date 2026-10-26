from __future__ import annotations

import pytest

from codex_task_tree.models import (
    CandidateProposal,
    CapabilityNeed,
    NodeStatus,
    PlanProposal,
    RoundKind,
    RoundPhase,
    RunStatus,
)
from codex_task_tree.runtime import hard_complete, install_plan
from codex_task_tree.scheduler import select_next

from .conftest import authorize, exploration_report, scope_review_report, spec
from .test_tree import children_report, report_for


def test_declined_prerequisite_defers_dependents_but_preserves_independent_work(setup):
    runtime, judge, _ = setup
    judge.denied.add("candidate-B")
    authorize(runtime, RoundKind.DECOMPOSE)

    def worker(context, *_):
        report = children_report(context, ["A", "B", "C", "D", "E"])
        report.capability_proposals = [
            CapabilityNeed(
                id=f"cap-{name}",
                owner_task_id=context["task_id"],
                description=f"Required {name} outcome",
                root_requirement_ids=["r1"],
            )
            for name in ["A", "B", "C", "D", "E"]
        ]
        for candidate, capability in zip(
            report.candidates, report.capability_proposals, strict=True
        ):
            candidate.capability_ids = [capability.id]
        report.candidates[2].depends_on = ["candidate-B"]
        report.candidates[3].depends_on = ["candidate-C"]
        report.plan_proposals[0].capability_ids = [c.id for c in report.capability_proposals]
        return report

    runtime.worker = worker
    runtime.step()
    root = runtime.state.nodes[runtime.state.root_task_id]
    assert [runtime.state.nodes[k].spec.title for k in runtime.state.children(root.id)] == [
        "A",
        "E",
    ]
    assert {gap.id for gap in root.open_gaps} == {
        "unplanned:cap-B",
        "unplanned:cap-C",
        "unplanned:cap-D",
    }
    assert len([kind for kind, _, _ in judge.calls if kind == "NECESSARY"]) == 5
    assert not hard_complete(runtime.state, root.id)


def test_partial_replacement_does_not_retire_existing_required_task(setup):
    runtime, judge, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["old"])
    runtime.step()
    root = runtime.state.root_task_id
    old = runtime.state.children(root)[0]
    previous = runtime.state.nodes[root].current_plan_revision_id
    judge.denied.add("candidate-newB")
    authorize(runtime, RoundKind.DECOMPOSE)

    def replacement(context, *_):
        report = children_report(context, ["newA", "newB"])
        proposal = report.plan_proposals[0]
        proposal.supersedes_task_ids = [old]
        proposal.replacement_map = {old: [c.id for c in report.candidates]}
        return report

    runtime.worker = replacement
    runtime.step()
    assert runtime.state.children(root) == [old]
    assert runtime.state.nodes[old].status != NodeStatus.SUPERSEDED
    assert runtime.state.nodes[root].current_plan_revision_id == previous


def test_replacement_preserves_history_transfers_dependencies_and_skips_old_work(setup):
    runtime, _, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["old", "consumer"])
    runtime.step()
    state = runtime.state
    root = state.root_task_id
    old, consumer = state.children(root)
    state.nodes[consumer].depends_on = [old]
    previous = state.nodes[root].current_plan_revision_id
    work = next(
        r for r in state.rounds.values() if r.task_id == root and r.kind == RoundKind.EXECUTE
    )
    cap_id = state.plans[previous].capability_ids[0]
    state.candidates["new"] = CandidateProposal(
        id="new",
        spec=spec("gzip", child=True),
        capability_ids=[cap_id],
        reason="replace failed route",
    )
    proposal = PlanProposal(
        id="replacement",
        capability_ids=[cap_id],
        candidate_ids=["new"],
        retained_child_ids=[consumer],
        supersedes_task_ids=[old],
        replacement_map={old: ["new"]},
        rationale="same obligations via gzip",
    )
    install_plan(state, work.id, proposal, [])
    assert state.nodes[old].status == NodeStatus.SUPERSEDED
    assert old in state.nodes[root].child_ids and previous in state.plans
    new = state.children(root)[-1]
    assert state.nodes[consumer].depends_on == [new]
    assert all(r.phase == RoundPhase.CANCELLED for r in state.rounds.values() if r.task_id == old)
    assert select_next(state).work_ref.id == next(
        r.id for r in state.rounds.values() if r.task_id == new
    )


def test_unresolved_execution_prevents_replacement(setup):
    runtime, _, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["old"])
    runtime.step()
    state = runtime.state
    root, old = state.root_task_id, state.children(state.root_task_id)[0]
    state.nodes[old].status = NodeStatus.RUNNING
    work = next(
        r for r in state.rounds.values() if r.task_id == root and r.kind == RoundKind.EXECUTE
    )
    state.candidates["new"] = CandidateProposal(
        id="new", spec=spec("new", child=True), capability_ids=["cap"], reason="replace"
    )
    proposal = PlanProposal(
        id="new-plan",
        capability_ids=["cap"],
        candidate_ids=["new"],
        supersedes_task_ids=[old],
        replacement_map={old: ["new"]},
        rationale="replacement",
    )
    with pytest.raises(ValueError, match="unresolved execution"):
        install_plan(state, work.id, proposal, [])
    assert state.nodes[old].status == NodeStatus.RUNNING


def test_verification_preserves_existing_wave_remainder(setup):
    runtime, _, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["A", "B"])
    runtime.step()
    ref = runtime.claim_next_work()
    wave = runtime.state.scheduler.wave.model_copy(deep=True)
    runtime.state.rounds[ref.id].phase = RoundPhase.COMMITTED
    runtime.state.scheduler.active_work = None
    runtime.queue_verification(runtime.state.root_task_id)
    choice = select_next(runtime.state)
    assert choice.work_ref.id == wave.round_ids[wave.next_index]
    assert choice.wave_after.next_index == wave.next_index + 1
    runtime.state.rounds[choice.work_ref.id].phase = RoundPhase.COMMITTED
    runtime.state.scheduler.wave = choice.wave_after
    assert select_next(runtime.state).work_ref.kind == "VERIFICATION"


def test_new_nodes_do_not_reset_global_no_progress(setup):
    runtime, _, _ = setup
    runtime.state.execution_profile.max_run_no_progress = 3
    runtime.state.execution_profile.max_node_no_progress = 20
    counter = 0

    def worker(context, *_):
        nonlocal counter
        if context["kind"] == "EXPLORE":
            return exploration_report(context)
        if context["kind"] == "SCOPE_REVIEW":
            return scope_review_report(context)
        counter += 1
        return children_report(context, [f"delegated-{counter}"])

    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = worker
    for _ in range(12):
        if runtime.state.status != RunStatus.ACTIVE:
            break
        runtime.step()
    assert runtime.state.status == RunStatus.BLOCKED
    assert runtime.state.progress.rounds_without_substantive_progress == 3
    assert len(runtime.state.nodes) >= 4


def test_reverification_failure_creates_a_stable_gap_and_repair(setup):
    runtime, _, workspace = setup
    runtime.worker = lambda context, *_: report_for(context)
    runtime.step()
    root = runtime.state.root_task_id
    (workspace / "Root.txt").write_text("change")
    runtime.command_exec = lambda *_: {"exit_code": 1, "stdout": "", "stderr": "real failure"}
    runtime.step()
    assert runtime.state.nodes[root].open_gaps
    assert any(
        r.phase == RoundPhase.QUEUED and r.purpose == "repair_or_replan"
        for r in runtime.state.rounds.values()
    )
