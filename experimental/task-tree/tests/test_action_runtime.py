from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from codex_task_tree.action_protocol import (
    ChildTask,
    DrillDownReport,
    ExecutionReport,
    ExploreBrief,
    VisitReport,
)
from codex_task_tree.action_runtime import ActionRuntime, completion_current, result_basis
from codex_task_tree.codex_adapter import WorkerReportError
from codex_task_tree.models import (
    ActionVerdict,
    CriterionAssessment,
    ExecutionProfile,
    MustSolveAssessment,
    MustSolveItem,
    NodeResult,
    NodeStatus,
    RootApproval,
    ResponsibilityAnswer,
    RoundKind,
    RoundPhase,
    RunStatus,
    WorkRef,
    digest,
)
from codex_task_tree.runtime import queue_round
from codex_task_tree.store import RunStore

from .conftest import spec


def item(problem):
    return MustSolveItem(
        problem=problem,
        verification_method="Inspect the delivered result and supporting evidence",
        expected_output="An evidenced answer to " + problem,
    )


class Worker:
    supports_continuation = True
    last_usage = None

    def __init__(self, workspace):
        self.workspace, self.visits, self.actions, self.phases = workspace, [], [], []
        self.drill_down = False
        self.force_finish = False
        self.fail_criterion = False
        self.block = False
        self.empty_split = False
        self.children = None
        self.omit_item_assessments = False
        self.fail_item = False

    def execution_owner(self):
        return {"pid": "fake", "start": "fake"}

    def ensure_stopped(self, owner):
        pass

    def recover(self, work, report_type):
        return None

    def run_action_visit(self, context, model, on_started, timeout, submit, validator=None, **_):
        thread = context.get("continuation_thread_id") or f"visit-{len(self.visits)}"
        on_started(thread, "explore")
        self.visits.append((thread, context))
        brief = ExploreBrief(
            summary="Checked current obligation and actual files",
            evidence_refs=["input.txt"],
            input_paths=["input.txt"],
            no_split_case="One bounded outcome; source checks are supporting steps",
        )
        answer = submit(brief)
        # A repeated tool delivery is idempotent, not another Luna judgment.
        assert submit(brief) == answer
        report = model(**self.identity(context), decision_id=answer["decision_id"])
        validator(report)
        return report

    @staticmethod
    def identity(context):
        return {
            key: context[key] for key in ("task_id", "round_id", "spec_revision", "input_digest")
        }

    def run_action_decision(self, context, on_started, timeout, on_owner, validator=None):
        self.actions.append(context)
        on_owner(self.execution_owner())
        on_started(f"luna-{len(self.actions)}", "decision")
        result = context["task"]["result"]
        action = "finish" if result else "execute"
        if result and result["child_result_versions"] != {
            n["id"]: n["result"]["revision"] for n in context["children"]
        }:
            action = "execute"
        if self.drill_down and not context["children"] and context["task"]["is_root"]:
            action = "drill_down"
        if self.force_finish:
            action = "finish"
        return ActionVerdict(
            action=action,
            reason="fixture evidence supports this action",
            acceptance=[
                CriterionAssessment(
                    criterion_id=context["task"]["spec"]["acceptance"][0]["id"],
                    status="FAIL" if self.fail_criterion else "PASS",
                    reason="inspected actual fixture",
                    evidence_refs=["input.txt"],
                )
            ],
            must_solve_assessments=[] if self.omit_item_assessments else [
                MustSolveAssessment(
                    must_solve=contract["problem"] if isinstance(contract, dict) else contract,
                    status="PASS" if result else "UNKNOWN",
                    reason="checked the delivery" if result else "no delivery yet",
                    evidence_refs=["input.txt"] if result else [],
                )
                for contract in context["task"]["must_solve"]
            ],
            blocked_on=["private device unavailable"] if self.block else [],
        )

    def run_with_feedback(self, context, model, on_started, timeout, validator=None, **_):
        on_started(context["continuation_thread_id"], context["kind"])
        self.phases.append(context)
        if context["kind"] == "DECOMPOSE":
            report = DrillDownReport(
                **self.identity(context),
                summary="two independent required outcomes",
                inline_work=["verify the input file"] if self.empty_split else [],
                children=[]
                if self.empty_split
                else self.children
                or [
                    ChildTask(
                        id=name,
                        spec=spec(name, child=True),
                        must_solve=[item(f"deliver {name} outcome")],
                        root_contribution="supports the root's usable storage",
                        reason="necessary outcome",
                    )
                    for name in ["A", "B"]
                ],
                no_deeper_question_reason="no related evidence path remains" if self.empty_split else "",
            )
        else:
            title = context["task"]["spec"]["title"]
            path = title + ".txt"
            (self.workspace / path).write_text("actual result " + title)
            report = ExecutionReport(
                **self.identity(context),
                summary="actual current delivery " + title,
                goal_answer="Answered " + context["task"]["spec"]["goal"],
                must_solve_answers=[
                    ResponsibilityAnswer(
                        must_solve=contract["problem"] if isinstance(contract, dict) else contract,
                        answer="Answered " + (contract["problem"] if isinstance(contract, dict) else contract),
                        verification_status="FAIL" if self.fail_item else "PASS",
                        verification_result="Inspected the delivered result against the expected output",
                        evidence_refs=[path],
                    )
                    for contract in context["task"]["must_solve"]
                ],
                inline_work_answers=[
                    ResponsibilityAnswer(
                        must_solve=question,
                        answer="Checked " + question,
                        verification_status="PASS",
                        verification_result="Inspected the delivered result",
                        evidence_refs=[path],
                    ) for question in context["inline_work"]
                ],
                deliverable_paths=[path],
                evidence_refs=[path],
                input_paths=["input.txt"],
                acceptance=[
                    CriterionAssessment(
                        criterion_id=context["task"]["spec"]["acceptance"][0]["id"],
                        status="PASS", reason="checked", evidence_refs=[path]
                    )
                ],
            )
        if validator:
            validator(report)
        return report


@pytest.fixture
def action_runtime(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.txt").write_text("stable input")
    store = RunStore(tmp_path / "controller", "actions")
    state = ActionRuntime.create(
        store, "Implement usable storage", workspace,
        ExecutionProfile(max_worker_rounds=60, max_parallel_siblings=1)
    )
    state.preparation_required = False
    for work in state.rounds.values():
        work.phase = RoundPhase.CANCELLED
    root = state.nodes[state.root_task_id]
    root.specs = [spec()]
    root.must_solve = [item("independently deliver the original usable storage obligation")]
    state.root_approval = RootApproval(
        approved_spec_revision=1,
        contract_hash=root.spec.contract_hash(),
        user_confirmation_ref="fixture approval",
    )
    queue_round(state, root.id, RoundKind.EXPLORE, "approved")
    store.save(state, state.revision)
    worker = Worker(workspace)
    return ActionRuntime(store, worker, None, None)


def run_to_stop(runtime, limit=30):
    for _ in range(limit):
        runtime.step()
        if runtime.state.status != RunStatus.ACTIVE:
            return
    pytest.fail("scheduler did not stop")


def test_leaf_executes_and_luna_finishes_without_another_explore(action_runtime):
    r = action_runtime
    r.step()
    assert r.state.status == RunStatus.SUCCEEDED
    assert [c["kind"] for c in r.worker.phases] == ["EXECUTE"]
    assert r.worker.phases[0]["continuation_thread_id"] == r.worker.visits[0][0]
    assert len(r.worker.actions) == 2
    assert len(r.worker.visits) == 1
    assert r.store.load().nodes[r.state.root_task_id].worker_thread_id == r.worker.visits[0][0]
    assert completion_current(r.state, r.state.root_task_id)
    assert completion_current(r.store.load(), r.state.root_task_id)
    assert not r.state.verification_jobs
    assert not r.state.decisions
    assert len(r.state.action_decisions) == 2
    first, second = [d.verdict for d in r.state.action_decisions.values()]
    assert first.action == "execute" and first.must_solve_assessments[0].status == "UNKNOWN"
    assert second.action == "finish" and second.must_solve_assessments[0].status == "PASS"
    assert all(context["root_goal"] == r.state.user_request for context in r.worker.actions)


def test_luna_receives_facts_without_worker_routing_opinions(action_runtime):
    r = action_runtime
    r.step()
    assert len(r.worker.actions) == 2
    for record in r.state.action_decisions.values():
        request = json.loads(Path(record.request_ref).read_text())
        assert "purpose" not in request
        assert "last_drill_down" not in request
        assert "granularity_assessment" not in request["exploration"]
        assert "planning_reason" not in request["task"]
        if request.get("last_visit"):
            assert "decision" not in request["last_visit"]
            assert "granularity_assessment" not in request["last_visit"]["brief"]
        for assessment in request["task"]["delivery_assessment"]["acceptance"]:
            assert "status" not in assessment
        for answer in (request["task"]["result"] or {}).get("must_solve_answers", []):
            assert "verification_status" not in answer


def test_luna_retry_resumes_node_thread_without_reexploring(action_runtime):
    r = action_runtime
    decide = r.worker.run_action_decision
    execute = r.worker.run_with_feedback

    def retry_once(context, *args, **kwargs):
        verdict = decide(context, *args, **kwargs)
        if (context["task"]["result"] or {}).get("revision") == 1:
            verdict.action = "execute"
            verdict.reason = "the original must_solve still needs a second delivery"
        return verdict

    def add_progress(context, *args, **kwargs):
        report = execute(context, *args, **kwargs)
        if context["kind"] == "EXECUTE" and context["task"]["result"]:
            (r.state.workspace / "Root.txt").write_text("second completed delivery")
            report.evidence_refs.append("https://example.invalid/second-source")
        return report

    r.worker.run_action_decision = retry_once
    r.worker.run_with_feedback = add_progress
    run_to_stop(r)
    assert len(r.worker.visits) == 1
    assert len(r.worker.actions) == 3
    assert [phase["continuation_thread_id"] for phase in r.worker.phases] == [
        r.worker.visits[0][0], r.worker.visits[0][0]
    ]
    assert r.worker.phases[1]["action_decision"]["verdict"]["action"] == "execute"
    assert r.state.nodes[r.state.root_task_id].result_revision == 2


def test_luna_crash_after_execution_reuses_saved_result(action_runtime):
    r = action_runtime
    decide = r.worker.run_action_decision
    crashed = False

    def interrupt_once(context, *args, **kwargs):
        nonlocal crashed
        if context["task"]["result"] and not crashed:
            crashed = True
            raise KeyboardInterrupt("simulated controller crash before Luna verdict")
        return decide(context, *args, **kwargs)

    r.worker.run_action_decision = interrupt_once
    with pytest.raises(KeyboardInterrupt):
        r.step()
    saved = r.store.load()
    assert saved.nodes[saved.root_task_id].result_revision == 1

    resumed = ActionRuntime(r.store, r.worker, None, None)
    run_to_stop(resumed)
    root = resumed.state.nodes[resumed.state.root_task_id]
    assert root.result_revision == 1
    assert root.worker_thread_id == r.worker.visits[0][0]
    assert len(r.worker.phases) == 1


def test_finish_with_remaining_work_is_corrected_in_luna_turn(action_runtime):
    r = action_runtime
    decide = r.worker.run_action_decision
    corrected = []

    def decide_with_correction(context, *args, validator=None, **kwargs):
        verdict = decide(context, *args, validator=None, **kwargs)
        if verdict.action == "finish":
            verdict.remaining_work = ["the parent will integrate this child later"]
            with pytest.raises(ValueError, match="finish must have no current-task"):
                validator(verdict)
            corrected.append(True)
            verdict.remaining_work = []
        validator(verdict)
        return verdict

    r.worker.run_action_decision = decide_with_correction
    run_to_stop(r)
    assert corrected
    assert r.state.status == RunStatus.SUCCEEDED


def test_children_admitted_directly_and_parent_must_execute_own_obligation(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    assert len(root.child_ids) == 2
    assert len(r.state.action_decisions) == 1  # no per-candidate judge
    original = root.spec.model_dump()
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED
    assert r.state.nodes[root.id].spec.model_dump() == original
    executions = [c for c in r.worker.phases if c["kind"] == "EXECUTE"]
    assert [c["task"]["spec"]["title"] for c in executions] == ["A", "B", "Root"]
    parent = executions[-1]
    root_thread = r.worker.visits[0][0]
    assert parent["continuation_thread_id"] == root_thread
    assert len({visit[0] for visit in r.worker.visits}) == 3
    assert all(c["result"] for c in parent["children"])
    assert all(c["result"]["goal_answer"] for c in parent["children"])
    assert all(
        [a["must_solve"] for a in c["result"]["must_solve_answers"]]
        == [item["problem"] for item in c["must_solve"]]
        for c in parent["children"]
    )
    assert parent["task"]["must_solve"] == [item.model_dump() for item in root.must_solve]
    assert all(c["root_goal"] == r.state.user_request for c in r.worker.phases)
    assert not any(
        work.kind in {RoundKind.REVIEW, RoundKind.RECONCILE, RoundKind.SCOPE_REVIEW}
        for work in r.state.rounds.values()
    )


def test_split_can_plan_available_work_even_with_external_blocker(action_runtime):
    r = action_runtime
    r.worker.drill_down = r.worker.block = True
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    assert len(root.child_ids) == 2
    assert root.status == NodeStatus.WAITING_CHILDREN
    assert r.state.action_decisions[next(iter(r.state.action_decisions))].verdict.blocked_on
    r.worker.block = False
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED


@pytest.mark.parametrize("failure", ["no_delivery", "failed_acceptance"])
def test_finish_cannot_skip_actual_delivery_or_failed_acceptance(action_runtime, failure):
    r = action_runtime
    if failure == "failed_acceptance":
        r.worker.fail_criterion = True
    r.worker.force_finish = True
    run_to_stop(r)
    assert r.state.status == RunStatus.BLOCKED
    assert not completion_current(r.state, r.state.root_task_id)


def test_narrow_children_cannot_complete_parent_without_new_integrated_result(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    r.step()  # A executes
    r.step()  # B executes
    r.worker.force_finish = True
    run_to_stop(r)
    root = r.state.nodes[r.state.root_task_id]
    assert root.status == NodeStatus.BLOCKED
    assert "no own executed delivery" in root.block_reason


def test_empty_split_is_bounded_and_never_reported_as_success(action_runtime):
    r = action_runtime
    r.worker.drill_down, r.worker.empty_split = True, True
    r.step()
    pending = [w for w in r.state.rounds.values() if w.phase == RoundPhase.QUEUED]
    assert len(pending) == 1
    assert pending[0].kind == RoundKind.EXECUTE
    assert pending[0].routing_basis == "inline_after_empty_drill"
    r.worker.drill_down = False
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED
    assert len(r.state.nodes) == 1
    assert len([p for p in r.worker.phases if p["kind"] == "EXECUTE"]) == 1
    assert next(p for p in r.worker.phases if p["kind"] == "EXECUTE")["inline_work"] == [
        "verify the input file"
    ]
    executed = next(w for w in r.state.rounds.values() if w.kind == RoundKind.EXECUTE)
    incomplete = ExecutionReport.model_validate_json(Path(executed.worker_report_ref).read_text())
    incomplete.inline_work_answers = []
    with pytest.raises(ValueError, match="inline_work_answers"):
        r._validate_report(executed, incomplete)


def test_inherited_children_share_parent_worker_session_but_not_luna(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.worker.children = [
        ChildTask(id=name, spec=spec(name, child=True), must_solve=[item(name)],
                  root_contribution="root", reason="distinct outcome", session_mode="inherit")
        for name in ("A", "B")
    ]
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    root_thread = root.worker_thread_id
    assert all(r.state.nodes[key].session_mode == "inherit" for key in root.child_ids)
    assert all(r.store.load().nodes[key].session_mode == "inherit" for key in root.child_ids)
    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: Worker(workspace)
    assert r._parallel_candidates() == []
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED
    assert all(thread == root_thread for thread, _ in r.worker.visits)
    assert all(p["continuation_thread_id"] == root_thread for p in r.worker.phases)
    assert len({decision.thread_id for decision in r.state.action_decisions.values()}) == len(r.state.action_decisions)


def test_mixed_sessions_keep_inherited_root_work_out_of_parallel_branches(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.worker.children = [
        ChildTask(id=name, spec=spec(name, child=True), must_solve=[item(name)],
                  root_contribution="root", reason="distinct outcome", session_mode=mode)
        for name, mode in (("A", "inherit"), ("B", "isolated"), ("C", "isolated"))
    ]
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    children = {r.state.nodes[key].spec.title: key for key in root.child_ids}
    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: Worker(workspace)
    r._ensure_branch_workspaces()
    assert children["A"] not in r.state.branch_workspaces
    assert set(r.state.branch_workspaces) == {children["B"], children["C"]}
    assert {work.task_id for work in r._parallel_candidates()} == {
        children["B"], children["C"]
    }
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED
    assert r.state.nodes[children["A"]].worker_thread_id is None
    assert r.state.nodes[root.id].worker_thread_id == r.worker.visits[0][0]


def test_down_probe_can_add_one_focused_child(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.worker.children = [
        ChildTask(
            id="deeper_question",
            spec=spec("deeper_question", child=True),
            must_solve=[item("answer one deeper question")],
            root_contribution="adds evidence for the root answer",
            reason="one related question has a distinct evidence path",
        )
    ]
    run_to_stop(r)
    root = r.state.nodes[r.state.root_task_id]
    assert r.state.status == RunStatus.SUCCEEDED
    assert len(root.child_ids) == 1
    assert len([p for p in r.worker.phases if p["kind"] == "EXECUTE"]) == 2


def test_pending_duplicate_reuses_dependency_without_second_execution(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.worker.children = [
        ChildTask(
            id=key,
            spec=spec("Same", child=True),
            must_solve=[item("same outcome")],
            root_contribution="needed",
            reason="same required behavior",
        )
        for key in ["one", "two"]
    ]
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    assert len(root.child_ids) == 1
    assert root.reuse_links["two"] == root.child_ids[0]
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED
    assert (
        len([c for c in r.worker.phases if c["kind"] == "EXECUTE" and not c["task"]["is_root"]])
        == 1
    )


def test_wait_cycle_is_rejected_atomically(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.worker.children = [
        ChildTask(
            id=key,
            spec=spec(key, child=True),
            must_solve=[item(key)],
            root_contribution="required",
            reason="required",
            depends_on=[dep],
        )
        for key, dep in [("A", "B"), ("B", "A")]
    ]
    r.step()
    assert len(r.state.nodes) == 1
    assert "cycle" in next(work.error for work in r.state.rounds.values() if work.error)


def test_unrelated_file_does_not_invalidate_but_consumed_input_does(action_runtime):
    r = action_runtime
    run_to_stop(r)
    (r.state.workspace / "unrelated.txt").write_text("sibling output")
    assert completion_current(r.state, r.state.root_task_id)
    (r.state.workspace / "input.txt").write_text("changed input")
    assert not completion_current(r.state, r.state.root_task_id)
    r.step()
    assert r.state.status != RunStatus.SUCCEEDED
    changes = r.worker.visits[-1][1]["changes_since_visit"]
    assert any(path.endswith("input.txt") for path in changes)
    assert r.worker.visits[-1][0] == r.worker.visits[0][0]


def test_external_blocker_parks_instead_of_false_finish(action_runtime):
    r = action_runtime
    r.worker.block = True
    run_to_stop(r)
    assert r.state.status == RunStatus.BLOCKED
    assert not r.worker.phases


def test_recovery_reviews_partial_effects_in_fresh_visit(action_runtime):
    r = action_runtime
    ref = r.claim_next_work()
    work = r.state.rounds[ref.id]
    work.phase, work.thread_id = RoundPhase.WORKER_RUNNING, "interrupted"
    work.execution_owner = r.worker.execution_owner()
    r.save()
    r.recover_active()
    ready = [work for work in r.state.rounds.values() if work.phase == RoundPhase.QUEUED]
    assert len(ready) == 1
    assert ready[0].kind == RoundKind.EXPLORE
    assert not ready[0].authorization_round_id


def test_persisted_action_request_contains_actual_result_and_original_question(action_runtime):
    r = action_runtime
    run_to_stop(r)
    record = list(r.state.action_decisions.values())[-1]
    request = json.loads(open(record.request_ref).read())
    assert request["root_goal"] == "Implement usable storage"
    assert request["task"]["result"]["summary"] == "actual current delivery Root"
    assert request["task"]["spec"]["acceptance"][0]["description"] == "Root behavior works"
    assert request["controller_facts"]["input_fingerprints"]


def test_parent_can_split_again_after_integrating_without_losing_responsibility(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    decide = r.worker.run_action_decision
    phase = r.worker.run_with_feedback

    def another_round(context, *args, **kwargs):
        verdict = decide(context, *args, **kwargs)
        if (
            context["task"]["is_root"]
            and context["task"]["result"]
            and len(context["children"]) == 2
        ):
            verdict.action = "drill_down"
            verdict.reason = "integration uncovered a remaining necessary outcome"
        return verdict

    def split_missing(context, *args, **kwargs):
        if context["kind"] == "DECOMPOSE" and context["task"]["result"]:
            r.worker.children = [
                ChildTask(
                    id="C",
                    spec=spec("C", child=True),
                    must_solve=[item("C")],
                    root_contribution="remaining root responsibility",
                    reason="needed",
                )
            ]
        return phase(context, *args, **kwargs)

    r.worker.run_action_decision = another_round
    r.worker.run_with_feedback = split_missing
    run_to_stop(r)
    root = r.state.nodes[r.state.root_task_id]
    assert r.state.status == RunStatus.SUCCEEDED
    assert len(root.child_ids) == 3
    assert len(root.result_history) == 2
    assert len(root.result_history[-1].child_result_versions) == 3
    assert root.must_solve == [item("independently deliver the original usable storage obligation")]


def test_changed_child_reopens_only_affected_branch_and_ancestors(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    run_to_stop(r)
    (r.state.workspace / "A.txt").write_text("changed product")
    r.audit_files()
    by_title = {n.spec.title: n for n in r.state.nodes.values()}
    assert by_title["A"].status == NodeStatus.READY
    assert by_title["B"].status == NodeStatus.SUCCEEDED
    assert by_title["Root"].status == NodeStatus.WAITING_CHILDREN
    assert r.pending(by_title["A"].id)
    assert not r.pending(by_title["B"].id)


def test_judge_output_schema_accepts_new_exploration_object():
    from codex_task_tree.codex_adapter import strict_schema

    schema = strict_schema(ActionVerdict, {"exploration": {"summary": "current evidence"}})
    assert schema["properties"]["action"]["enum"] == ["drill_down", "execute", "finish"]
    assert ActionVerdict.model_validate({"action": "split", "reason": "saved legacy decision"}).action == "drill_down"


def test_transport_failure_preserves_execution_for_recovery(action_runtime):
    r = action_runtime

    class TransportError(Exception):
        pass

    def disconnected(*args, **kwargs):
        raise TransportError("connection closed before execution was confirmed stopped")

    r.worker.run_action_visit = disconnected
    with pytest.raises(TransportError):
        r.step()
    saved = r.store.load()
    assert saved.status == RunStatus.BLOCKED
    assert saved.scheduler.active_work
    assert r.store.acquire_workspace(saved.workspace).unresolved_execution_refs


def test_failed_luna_call_retries_new_visit_without_same_visit_request_loop(action_runtime):
    r = action_runtime
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(args[0]["round_id"])
        raise RuntimeError("temporary provider failure")

    def visit(context, model, started, timeout, submit, validator, **kwargs):
        started(f"visit-{len(calls)}")
        brief = ExploreBrief(
            summary="actual exploration",
            evidence_refs=["input.txt"],
            no_split_case="one bounded outcome",
        )
        for _ in range(2):
            with pytest.raises(RuntimeError):
                submit(brief)
        report = VisitReport(**r.worker.identity(context), decision_id="unavailable")
        validator(report)

    r.worker.run_action_decision = unavailable
    r.worker.run_action_visit = visit
    run_to_stop(r)
    assert r.state.status == RunStatus.BLOCKED
    assert len(calls) == len(set(calls)) == r.state.execution_profile.max_node_no_progress
    assert not r.state.action_decisions


def test_input_is_root_task_and_starts_exploration_without_approval(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.txt").write_text("stable input")
    store = RunStore(tmp_path / "controller", "prepare-actions")
    ActionRuntime.create(store, "The original full question", workspace, ExecutionProfile())
    r = ActionRuntime(store, Worker(workspace), None, None)
    assert r.state.status == RunStatus.ACTIVE
    assert r.state.background["root_goal"] == r.state.user_request
    assert not r.state.preparation_required
    assert not r.state.preparations
    assert [work.kind for work in r.state.rounds.values()] == [RoundKind.EXPLORE]
    root = r.state.nodes[r.state.root_task_id]
    assert root.spec.goal == r.state.user_request
    assert root.spec.requirements[0].description == r.state.user_request
    assert [entry.problem for entry in root.must_solve] == [r.state.user_request]
    assert all(entry.verification_method and entry.expected_output for entry in root.must_solve)
    assert r.state.root_approval.contract_hash == root.spec.contract_hash()
    r.step()
    assert r.worker.visits[0][1]["kind"] == "EXPLORE"
    assert r.state.status == RunStatus.SUCCEEDED


def test_execution_report_must_answer_goal_and_each_must_solve(action_runtime):
    r = action_runtime
    work = next(iter(r.state.rounds.values()))
    root = r.state.nodes[r.state.root_task_id]
    report = ExecutionReport(
        task_id=root.id,
        round_id=work.id,
        spec_revision=root.spec.revision,
        input_digest=work.input_digest,
        summary="partial delivery",
        goal_answer="Answered the goal",
    )
    with pytest.raises(ValueError, match="must_solve_answers"):
        r._validate_report(work, report)

    report.must_solve_answers = [
        ResponsibilityAnswer(must_solve=root.must_solve[0].problem, answer="claimed answer")
    ]
    with pytest.raises(ValueError, match="verification status and result"):
        r._validate_report(work, report)


@pytest.mark.parametrize("action", ["drill_down", "execute", "finish"])
def test_luna_cannot_choose_any_action_without_each_must_solve_assessment(action_runtime, action):
    r = action_runtime
    r.worker.omit_item_assessments = True
    r.worker.drill_down = action == "drill_down"
    r.worker.force_finish = action == "finish"
    with pytest.raises(ValueError, match="must cover every current-task item in order"):
        r.step()
    assert r.state.status == RunStatus.BLOCKED
    assert "must cover every current-task item in order" in r.state.block_reason
    assert not r.state.action_decisions


def test_luna_item_pass_requires_evidence_even_before_execution(action_runtime):
    r = action_runtime
    decide = r.worker.run_action_decision

    def unsupported_pass(context, *args, **kwargs):
        verdict = decide(context, *args, **kwargs)
        verdict.must_solve_assessments[0].status = "PASS"
        verdict.must_solve_assessments[0].evidence_refs = []
        return verdict

    r.worker.run_action_decision = unsupported_pass
    with pytest.raises(ValueError, match="PASS needs actual evidence"):
        r.step()
    assert not r.state.action_decisions


def test_luna_must_assess_each_item_when_task_has_multiple_obligations(action_runtime):
    r = action_runtime
    r.state.nodes[r.state.root_task_id].must_solve.append(item("second obligation"))
    decide = r.worker.run_action_decision

    def omit_second(context, *args, **kwargs):
        verdict = decide(context, *args, **kwargs)
        verdict.must_solve_assessments.pop()
        return verdict

    r.worker.run_action_decision = omit_second
    with pytest.raises(ValueError, match="cover every current-task item"):
        r.step()
    assert not r.state.action_decisions


def test_legacy_string_item_is_also_assessed_before_routing(action_runtime):
    r = action_runtime
    r.state.nodes[r.state.root_task_id].must_solve = ["legacy obligation"]
    r.step()
    assert r.state.status == RunStatus.SUCCEEDED
    first = next(iter(r.state.action_decisions.values())).verdict
    assert [(entry.must_solve, entry.status) for entry in first.must_solve_assessments] == [
        ("legacy obligation", "UNKNOWN")
    ]


def test_worker_failed_item_check_cannot_finish_even_with_luna_pass(action_runtime):
    r = action_runtime
    r.worker.fail_item = True
    run_to_stop(r)
    assert r.state.status == RunStatus.BLOCKED
    assert "must_solve Worker verification did not pass" in r.state.nodes[r.state.root_task_id].block_reason


def test_new_child_must_solve_requires_three_fields(action_runtime):
    with pytest.raises(ValueError, match="verification_method"):
        ChildTask(
            id="missing_check", spec=spec("Missing", child=True),
            must_solve=[{"problem": "measure latency", "expected_output": "milliseconds"}],
            root_contribution="root answer", reason="needed",
        )


def test_old_string_contract_keeps_its_completion_basis(action_runtime):
    r = action_runtime
    root = r.state.nodes[r.state.root_task_id]
    root.must_solve = ["legacy obligation"]
    root.result_history = [NodeResult(
        revision=1, round_id="old", summary="done", goal_answer="done",
        must_solve_answers=[ResponsibilityAnswer(must_solve="legacy obligation", answer="done")],
    )]
    old_result = root.result_history[-1].model_dump(
        mode="json", exclude={"verification_ids", "decision_ids", "gaps"}
    )
    old_result["must_solve_answers"] = [{"must_solve": "legacy obligation", "answer": "done"}]
    assert result_basis(r.state, root.id) == digest({
        "spec": root.spec.contract_hash(), "must_solve": root.must_solve,
        "root_contribution": root.root_contribution, "root_goal": r.state.user_request,
        "result": old_result, "children": {}, "dependencies": {}, "files": {},
    })


def test_independent_sibling_explorations_overlap_and_commit_serially(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()  # Root creates A and B.
    barrier = threading.Barrier(2)
    visits = []

    class ParallelWorker(Worker):
        def run_action_visit(self, *args, **kwargs):
            visits.append(args[0]["task_id"])
            barrier.wait(timeout=5)
            return super().run_action_visit(*args, **kwargs)

    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: ParallelWorker(workspace)
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    assert set(visits) == set(root.child_ids)
    assert not r.state.scheduler.active_siblings
    assert all(
        any(work.task_id == child and work.kind == RoundKind.EXPLORE
            and work.phase == RoundPhase.COMMITTED for work in r.state.rounds.values())
        for child in root.child_ids
    )
    assert all(any(
        work.kind == RoundKind.EXECUTE and work.phase == RoundPhase.COMMITTED
        for work in r.state.rounds.values() if work.task_id == child
    ) for child in root.child_ids)
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED


def test_root_branches_execute_concurrently_in_isolated_workspaces(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    barrier = threading.Barrier(2)
    locations = {}

    class BranchWorker(Worker):
        def run_with_feedback(self, context, *args, **kwargs):
            if context["kind"] == "EXECUTE":
                title = context["task"]["spec"]["title"]
                locations[title] = self.workspace
                barrier.wait(timeout=5)
            return super().run_with_feedback(context, *args, **kwargs)

    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: BranchWorker(workspace)
    r.step()
    assert set(locations) == {"A", "B"}
    assert locations["A"] != locations["B"]
    assert all(path.is_file() for path in (r.state.workspace / "A.txt", r.state.workspace / "B.txt"))
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED


def test_finished_branch_refills_a_slot_without_waiting_for_slow_branch(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.worker.children = [
        ChildTask(id=name, spec=spec(name, child=True), must_solve=[item(name)],
                  root_contribution="root", reason="needed")
        for name in ("A", "B", "C")
    ]
    r.step()
    b_started, c_started = threading.Event(), threading.Event()

    class BranchWorker(Worker):
        def run_with_feedback(self, context, *args, **kwargs):
            if context["kind"] == "EXECUTE":
                title = context["task"]["spec"]["title"]
                if title == "B":
                    b_started.set()
                    assert c_started.wait(timeout=5)
                elif title == "C":
                    assert b_started.wait(timeout=5)
                    c_started.set()
            return super().run_with_feedback(context, *args, **kwargs)

    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: BranchWorker(workspace)
    r.step()
    assert b_started.is_set() and c_started.is_set()
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED


def test_parallel_branch_conflict_preserves_first_output(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    root = r.state.nodes[r.state.root_task_id]

    class CollidingWorker(Worker):
        def run_with_feedback(self, context, model, on_started, timeout, validator=None, **kwargs):
            report = super().run_with_feedback(
                context, model, on_started, timeout, validator=None, **kwargs
            )
            if context["kind"] == "EXECUTE" and context["task"]["spec"]["title"] == "B":
                deadline = time.monotonic() + 5
                while not (r.state.workspace / "A.txt").is_file() and time.monotonic() < deadline:
                    time.sleep(0.01)
                (self.workspace / "A.txt").write_text("B must not overwrite A")
                report.deliverable_paths = ["A.txt"]
                report.changed_paths = ["A.txt"]
                report.evidence_refs = ["A.txt"]
                for answer in report.must_solve_answers:
                    answer.evidence_refs = ["A.txt"]
                for assessment in report.acceptance:
                    assessment.evidence_refs = ["A.txt"]
            validator(report)
            return report

    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: CollidingWorker(workspace)
    r.step()
    assert (r.state.workspace / "A.txt").read_text() == "actual result A"
    assert r.state.nodes[root.child_ids[0]].status == NodeStatus.SUCCEEDED
    assert r.state.nodes[root.child_ids[1]].status != NodeStatus.SUCCEEDED


def test_existing_serial_branch_results_survive_parallel_migration(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    r.step()  # Complete one child using the original shared workspace.
    first = r.state.nodes[root.child_ids[0]]
    assert first.status == NodeStatus.SUCCEEDED
    saved_revision = first.result_revision

    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: Worker(workspace)
    r.step()
    assert first.result_revision == saved_revision
    assert (r.state.workspace / "A.txt").is_file()
    assert root.child_ids[1] in r.state.branch_workspaces
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED


def test_dependent_siblings_are_not_dispatched_together(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.worker.children = [
        ChildTask(id="A", spec=spec("A", child=True), must_solve=[item("A")],
                  root_contribution="root", reason="needed"),
        ChildTask(id="B", spec=spec("B", child=True), must_solve=[item("B")],
                  root_contribution="root", reason="needed", depends_on=["A"]),
    ]
    r.step()
    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: Worker(workspace)
    assert r._parallel_candidates() == []
    r.step()
    assert not r.state.scheduler.active_siblings


def test_parallel_sibling_reports_recover_after_commit_interruption(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: Worker(workspace)
    original = r._commit_report
    interrupted = False

    def fail_once(work, report):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("crash before sibling commit")
        return original(work, report)

    r._commit_report = fail_once
    with pytest.raises(KeyboardInterrupt):
        r.step()
    saved = r.store.load()
    assert len(saved.scheduler.active_siblings) == 2
    assert any(saved.rounds[ref.id].worker_report_ref for ref in saved.scheduler.active_siblings)

    resumed = ActionRuntime(r.store, r.worker, None, None,
                            worker_factory=lambda workspace: Worker(workspace))
    resumed.step()
    assert not resumed.state.scheduler.active_siblings
    assert all(resumed.state.rounds[ref.id].phase in {RoundPhase.COMMITTED, RoundPhase.INTERRUPTED}
               for ref in saved.scheduler.active_siblings)
    run_to_stop(resumed)
    assert resumed.state.status == RunStatus.SUCCEEDED


def test_parallel_split_keeps_other_live_round_identity(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    first, second = r.state.nodes[r.state.root_task_id].child_ids
    live = next(w for w in r.state.rounds.values() if w.task_id == first and w.phase == RoundPhase.QUEUED)
    split = next(w for w in r.state.rounds.values() if w.task_id == second and w.phase == RoundPhase.QUEUED)
    split.kind = RoundKind.DECOMPOSE
    r.state.scheduler.active_siblings = [
        WorkRef(kind="ROUND", id=live.id), WorkRef(kind="ROUND", id=split.id)
    ]
    report = DrillDownReport(
        task_id=second, round_id=split.id,
        spec_revision=r.state.nodes[second].spec.revision,
        input_digest="", summary="no useful child",
        no_deeper_question_reason="the current task is independently executable",
    )
    r._commit_split(split, report)
    assert r.state.rounds[live.id] is live
    live.worker_report_ref = "saved-worker-report"
    r.save()
    assert r.store.load().rounds[live.id].worker_report_ref == "saved-worker-report"


def test_parallel_recovery_reuses_frozen_files_when_refs_were_not_saved(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    r.state.execution_profile.max_parallel_siblings = 2
    r.worker_factory = lambda workspace: Worker(workspace)
    commit = r._commit_report

    def crash_after_freeze(work, report):
        if work.kind == RoundKind.EXECUTE:
            raise KeyboardInterrupt("crash before execution commit")
        return commit(work, report)

    r._commit_report = crash_after_freeze
    with pytest.raises(KeyboardInterrupt):
        r.step()
    saved = r.store.load()
    frozen = next(
        saved.rounds[ref.id] for ref in saved.scheduler.active_siblings
        if saved.rounds[ref.id].kind == RoundKind.EXECUTE
        and saved.rounds[ref.id].worker_report_ref
    )
    assert Path(frozen.worker_report_ref).is_file()
    assert Path(frozen.delivery_ref).is_file()
    frozen.worker_report_ref = None
    frozen.delivery_ref = None
    r.store.save(saved, saved.revision)

    resumed = ActionRuntime(r.store, r.worker, None, None,
                            worker_factory=lambda workspace: Worker(workspace))
    resumed.step()
    recovered = resumed.state.rounds[frozen.id]
    assert recovered.worker_report_ref and recovered.delivery_ref
    assert recovered.phase == RoundPhase.COMMITTED


def test_parallel_sibling_report_failure_does_not_discard_other_sibling(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    r.state.execution_profile.max_parallel_siblings = 2

    class FailingWorker(Worker):
        def run_action_visit(self, context, *args, **kwargs):
            if context["task_id"] == root.child_ids[0]:
                raise WorkerReportError("invalid structured response")
            return super().run_action_visit(context, *args, **kwargs)

    r.worker_factory = lambda workspace: FailingWorker(workspace)
    r.step()
    assert r.state.status == RunStatus.ACTIVE
    assert not r.state.scheduler.active_siblings
    assert any(
        work.task_id == root.child_ids[1] and work.kind == RoundKind.EXPLORE
        and work.phase == RoundPhase.COMMITTED
        for work in r.state.rounds.values()
    )
    assert r.state.nodes[root.child_ids[0]].status in {NodeStatus.READY, NodeStatus.BLOCKED}
