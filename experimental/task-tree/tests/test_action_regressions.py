"""Recovery, scheduling and context regressions from the protocol-three review."""

import json
from collections import Counter
from pathlib import Path

import pytest

from codex_task_tree.action_context import bounded_context
from codex_task_tree.action_protocol import (
    ChildReplacement,
    ChildTask,
    ExploreBrief,
    RemoteInputObservation,
    VisitReport,
)
from codex_task_tree.action_runtime import ActionRuntime, completion_current, judge_context
from codex_task_tree.models import NodeStatus, RoundPhase, RunStatus

from .conftest import spec
from .test_action_runtime import action_runtime as runtime_fixture
from .test_action_runtime import item, run_to_stop


@pytest.fixture
def action_runtime(tmp_path):
    return runtime_fixture.__wrapped__(tmp_path)


def queued(r, task_id=None):
    return next(
        w
        for w in r.state.rounds.values()
        if w.phase == RoundPhase.QUEUED and (task_id is None or w.task_id == task_id)
    )


def test_replay_never_restamps_old_delivery_with_changed_input(action_runtime, monkeypatch):
    r = action_runtime

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(r, "_commit_execution", interrupted)
    with pytest.raises(KeyboardInterrupt):
        r.step()
    old_receipt = next(w.delivery_ref for w in r.state.rounds.values() if w.delivery_ref)
    old_hashes = json.loads(Path(old_receipt).read_text())["input_fingerprints"]
    (r.state.workspace / "input.txt").write_text("new input after the old execution")
    resumed = ActionRuntime(r.store, r.worker, None, None)
    resumed.step()
    result = resumed.state.nodes[resumed.state.root_task_id].result_history[-1]
    assert result.input_fingerprints == old_hashes
    context = resumed.context_for(queued(resumed))
    assert any(p.endswith("input.txt") for p in context["changes_since_visit"])


def test_receipt_survives_crash_before_state_commit(action_runtime, monkeypatch):
    r = action_runtime
    original = r._freeze_delivery

    def freeze_then_crash(*args):
        original(*args)
        raise KeyboardInterrupt()

    monkeypatch.setattr(r, "_freeze_delivery", freeze_then_crash)
    with pytest.raises(KeyboardInterrupt):
        r.step()
    resumed = ActionRuntime(r.store, r.worker, None, None)
    resumed.step()
    assert resumed.state.nodes[resumed.state.root_task_id].result_revision == 1
    assert len(r.worker.phases) == 1


def test_same_goal_different_responsibility_is_not_duplicate(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    r.worker.children = [
        ChildTask(
            id=str(i),
            spec=spec("CSV", child=True),
            must_solve=[item(outcome)],
            root_contribution="CSV correctness",
            reason="necessary",
        )
        for i, outcome in enumerate(["quoted newlines", "missing values"])
    ]
    r.step()
    assert len(r.state.nodes[r.state.root_task_id].child_ids) == 2


def test_parent_exploration_reaches_child(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    decide = r._decide

    def enriched(work, context, brief):
        brief.findings = ["Distinct parent discovery about the data distribution"]
        return decide(work, context, brief)

    r._decide = enriched
    r.step()
    context = r.context_for(queued(r))
    assert "Distinct parent discovery" in json.dumps(context["ancestors"])
    assert context["task"]["planning_reason"] == "necessary outcome"


def test_failed_judge_preserves_exploration_for_next_worker(action_runtime):
    r = action_runtime

    def unavailable(*args, **kwargs):
        raise RuntimeError("provider temporarily unavailable")

    def visit(context, model, started, timeout, submit, validator, **kwargs):
        started("failed-judge")
        with pytest.raises(RuntimeError):
            submit(
                ExploreBrief(
                    summary="expensive saved discovery",
                    granularity_assessment="independent outcomes",
                )
            )
        validator(VisitReport(**r.worker.identity(context), decision_id="unavailable"))

    r.worker.run_action_visit = visit
    r.worker.run_action_decision = unavailable
    r.step()
    context = r.context_for(queued(r))
    assert context["last_visit"]["brief"]["summary"] == "expensive saved discovery"
    assert context["failed_rounds"][0]["exploration_ref"]


def test_remote_input_uses_probe_evidence_not_local_exists(action_runtime):
    r = action_runtime
    ref = "ssh hk:/mnt/data/results"
    r.state.input_refs.append(ref)
    r.save()
    decide = r._decide

    def remote(work, context, brief):
        brief.required_input_paths = [ref]
        brief.remote_inputs = [
            RemoteInputObservation(
                ref=ref, accessible=True, observed_version="snapshot-1", evidence_refs=["input.txt"]
            )
        ]
        return decide(work, context, brief)

    r._decide = remote
    r.step()
    assert r.worker.phases[0]["kind"] == "EXECUTE"
    request = r.worker.actions[0]
    assert request["controller_facts"]["missing_required_inputs"] == []
    assert ref not in request["controller_facts"]["input_fingerprints"]


def test_new_source_evidence_counts_but_rewording_does_not(action_runtime):
    r = action_runtime
    decide, execute = r.worker.run_action_decision, r.worker.run_with_feedback
    sources = [f"https://example.invalid/source-{i}" for i in range(4)]

    def keep_working(*args, **kwargs):
        verdict = decide(*args, **kwargs)
        verdict.action = "execute"
        return verdict

    def new_evidence(*args, **kwargs):
        result = execute(*args, **kwargs)
        result.summary = "wording " + str(len(r.worker.phases))
        result.evidence_refs.append(sources[min(len(r.worker.phases) - 1, 3)])
        return result

    r.worker.run_action_decision = keep_working
    r.worker.run_with_feedback = new_evidence
    for _ in range(4):
        r.step()
    assert r.state.nodes[r.state.root_task_id].no_progress_rounds == 0
    for _ in range(3):
        r.step()
    assert r.state.nodes[r.state.root_task_id].status == NodeStatus.BLOCKED


def test_queue_refresh_reads_zero_files_and_audit_hashes_each_file_once(
    action_runtime, monkeypatch
):
    import codex_task_tree.context as module
    from codex_task_tree.scheduler import build_work_schedule

    r = action_runtime
    r.worker.drill_down = True
    run_to_stop(r)
    counts = Counter()
    original = module.file_hash

    def counted(path):
        counts[str(path.resolve())] += 1
        return original(path)

    monkeypatch.setattr(module, "file_hash", counted)
    r.refresh_parents()
    build_work_schedule(r.state)
    assert not counts
    r.audit_files()
    assert len(counts) == 4
    assert set(counts.values()) == {1}


def test_undeclared_worker_write_reopens_affected_completed_branch(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    execute = r.worker.run_with_feedback

    def changes_child(context, *args, **kwargs):
        result = execute(context, *args, **kwargs)
        if context["kind"] == "EXECUTE" and context["task"]["is_root"]:
            (r.state.workspace / "A.txt").write_text("unreported parent modification")
        return result

    r.worker.run_with_feedback = changes_child
    for _ in range(6):
        r.step()
    nodes = {n.spec.title: n for n in r.state.nodes.values()}
    assert nodes["A"].status == NodeStatus.READY
    assert nodes["B"].status == NodeStatus.SUCCEEDED
    assert not completion_current(r.state, nodes["Root"].id)


def test_blocked_child_wakes_parent_and_can_be_replaced(action_runtime):
    r = action_runtime
    r.worker.drill_down = True
    decide, phase = r.worker.run_action_decision, r.worker.run_with_feedback
    blocked_id = None

    def blocked_once(context, *args, **kwargs):
        nonlocal blocked_id
        verdict = decide(context, *args, **kwargs)
        if context["task"]["spec"]["title"] == "A" and blocked_id is None:
            blocked_id = context["task_id"]
        if context["task_id"] == blocked_id:
            verdict.blocked_on = ["chosen approach cannot access source; alternative available"]
        if context["task"]["is_root"] and any(
            n["status"] == "BLOCKED" for n in context["children"]
        ):
            verdict.action = "drill_down"
            verdict.reason = "replace failed approach preserving obligation"
        return verdict

    def replacement(context, *args, **kwargs):
        result = phase(context, *args, **kwargs)
        if context["kind"] == "DECOMPOSE" and blocked_id:
            old = r.state.nodes[blocked_id]
            result.children = [
                ChildTask(
                    id="alternative",
                    spec=old.spec,
                    must_solve=old.must_solve,
                    root_contribution=old.root_contribution,
                    reason="use alternative source",
                )
            ]
            result.replacements = [
                ChildReplacement(
                    task_id=blocked_id,
                    replacement_id="alternative",
                    reason="chosen source unavailable",
                )
            ]
        return result

    r.worker.run_action_decision = blocked_once
    r.worker.run_with_feedback = replacement
    run_to_stop(r)
    assert r.state.status == RunStatus.SUCCEEDED
    old = r.state.nodes[blocked_id]
    assert old.status == NodeStatus.SUPERSEDED
    assert r.state.nodes[old.replaced_by_task_id].planning_reason == "use alternative source"
    assert r.state.nodes[r.state.root_task_id].replan_attempts == 1
    assert any(c["purpose"] == "replan_blocked_inputs" for _, c in r.worker.visits)


def test_context_budget_keeps_contracts_and_exposes_omitted_references(action_runtime):
    r = action_runtime
    context = r.context_for(queued(r))
    context["background"]["large_document"] = "x" * 100_000
    context["catalog"] = [{"goal": "y" * 200} for _ in range(1000)]
    packed = bounded_context(context, "/authoritative/context.json")
    assert len(json.dumps(packed, ensure_ascii=False).encode()) <= 64_000
    assert packed["task"]["spec"] == context["task"]["spec"]
    assert packed["root"]["same_as"] == "/task"
    assert packed["root_goal"] == context["root_goal"]
    assert packed["catalog"]["omitted_to"] == "/authoritative/context.json"
    context["task"]["must_solve"] = ["protected" * 100_000]
    with pytest.raises(ValueError, match="No criteria or adverse evidence"):
        bounded_context(context, "/authoritative/context.json")


def test_luna_keeps_compact_peer_catalog_when_full_catalog_exceeds_budget(action_runtime):
    context = action_runtime.context_for(queued(action_runtime))
    context["catalog"] = [
        {
            "id": "peer",
            "parent_id": context["task_id"],
            "status": "READY",
            "title": "Existing model evidence task",
            "goal": "Collect the model evidence",
            "must_solve": "large optional detail" * 5000,
        }
    ]
    packet = judge_context(context, ExploreBrief(summary="Current task evidence checked"))
    sent = bounded_context(packet, "/saved/request.json")
    assert sent["catalog"] == [{
        "id": "peer",
        "parent_id": context["task_id"],
        "status": "READY",
        "title": "Existing model evidence task",
        "goal": "Collect the model evidence",
    }]


def test_compacted_child_context_keeps_goal_and_must_solve_answers(action_runtime):
    context = action_runtime.context_for(queued(action_runtime))
    answer = {
        "revision": 1,
        "goal_answer": "The child goal is answered with measured evidence",
        "must_solve_answers": [
            {"must_solve": "measure latency", "answer": "Measured 3 ms under stated conditions"}
        ],
    }
    context["children"] = [
        {
            "id": f"child-{index}",
            "status": "SUCCEEDED",
            "block_reason": None,
            "must_solve": ["measure latency"],
            "root_contribution": "latency evidence",
            "result": answer,
            "result_report_ref": f"/saved/child-{index}.json",
            "large_optional_details": "x" * 1500,
        }
        for index in range(5)
    ]
    packed = bounded_context(context, "/saved/context.json", limit=9000)
    assert [child["result"] for child in packed["children"]] == [answer] * 5
    assert all("omitted_to" in child for child in packed["children"])


def test_luna_packet_keeps_current_evidence_when_prior_exploration_is_large():
    context = {
        "kind": "JUDGE",
        "root": {"id": "task-1"},
        "task": {
            "id": "task-1",
            "spec": {"acceptance": ["required"]},
            "result": {"summary": "r" * 25_000},
            "delivery_assessment": {"acceptance": [{"reason": "a" * 10_000}]},
            "latest_exploration": {
                "round_id": "prior-round",
                "ref": "/saved/prior.json",
                "brief": {
                    "summary": "prior finding",
                    "remaining_work": ["unresolved counterexample"],
                    "findings": ["old" * 5_000],
                },
            },
        },
        "exploration": {"summary": "current result", "findings": ["c" * 15_000]},
    }
    packet = bounded_context(context, "/saved/current.json")
    assert len(json.dumps(packet, ensure_ascii=False).encode()) <= 64_000
    assert packet["exploration"] == context["exploration"]
    assert packet["task"]["result"] == context["task"]["result"]
    assert packet["task"]["latest_exploration"]["remaining_work"] == [
        "unresolved counterexample"
    ]
    assert packet["task"]["latest_exploration"]["omitted_to"] == "/saved/current.json"


@pytest.mark.parametrize("weaken", [False, True])
def test_replanning_is_bounded_and_cannot_weaken_acceptance(action_runtime, weaken):
    r = action_runtime
    r.worker.drill_down = True
    decide, phase = r.worker.run_action_decision, r.worker.run_with_feedback

    def blocked(context, *args, **kwargs):
        verdict = decide(context, *args, **kwargs)
        if not context["task"]["is_root"] and context["task"]["spec"]["title"] == "A":
            verdict.blocked_on = ["source unavailable"]
        elif context["task"]["is_root"] and context["children"]:
            verdict.action = "drill_down"
        return verdict

    def replace(context, *args, **kwargs):
        result = phase(context, *args, **kwargs)
        if context["kind"] == "DECOMPOSE" and context["children"]:
            child = next(n for n in context["children"] if n["status"] == "BLOCKED")
            old = r.state.nodes[child["id"]]
            contract = old.spec.model_copy(deep=True)
            if weaken:
                contract.acceptance[0].required = False
            result.children = [
                ChildTask(
                    id="retry",
                    spec=contract,
                    must_solve=old.must_solve,
                    root_contribution=old.root_contribution,
                    reason="another approach",
                )
            ]
            result.replacements = [
                ChildReplacement(
                    task_id=old.id, replacement_id="retry", reason="try alternate source"
                )
            ]
        return result

    r.worker.run_action_decision, r.worker.run_with_feedback = blocked, replace
    run_to_stop(r, limit=40)
    root = r.state.nodes[r.state.root_task_id]
    assert root.status == NodeStatus.BLOCKED
    assert root.replan_attempts <= r.state.execution_profile.max_node_no_progress
    assert not completion_current(r.state, root.id)
    if weaken:
        assert len(r.state.nodes) == 3
        assert "preserve the full original responsibility" in root.block_reason


def test_remote_version_change_cannot_finish_old_delivery(action_runtime):
    r = action_runtime
    ref = "ssh hk:/mnt/data/results"
    decide = r._decide
    current_version = "v1"

    def remote(work, context, brief):
        brief.required_input_paths = [ref]
        brief.remote_inputs = [
            RemoteInputObservation(
                ref=ref,
                accessible=True,
                observed_version=current_version,
                evidence_refs=["input.txt"],
            )
        ]
        return decide(work, context, brief)

    r._decide = remote
    r.step()
    current_version = "v2"
    r.step()
    assert r.state.status == RunStatus.ACTIVE
    assert "remote input must be rechecked" in next(
        w.error for w in r.state.rounds.values() if w.error
    )


def test_optional_cited_url_does_not_force_repeated_probe(action_runtime):
    r = action_runtime
    decide = r._decide
    calls = 0

    def optional_source(work, context, brief):
        nonlocal calls
        calls += 1
        if calls == 1:
            brief.remote_inputs = [RemoteInputObservation(
                ref="https://docs.example.invalid/published-table",
                accessible=True, observed_version="published-table-v1",
                evidence_refs=["input.txt"],
            )]
        return decide(work, context, brief)

    r._decide = optional_source
    r.step()  # Source was observed while preparing the deliverable.
    r.step()  # The delivered result remains checkable without rereading every citation.
    assert r.state.status == RunStatus.SUCCEEDED


def test_explicit_reuse_cannot_erase_different_obligation(action_runtime):
    from codex_task_tree.action_protocol import DrillDownReport
    from codex_task_tree.models import RoundKind
    from codex_task_tree.runtime import queue_round

    r = action_runtime
    r.worker.drill_down = True
    r.step()
    root = r.state.nodes[r.state.root_task_id]
    source = r.state.nodes[root.child_ids[0]]
    r.worker.children = [
        ChildTask(
            id="different",
            spec=source.spec,
            must_solve=[item("a different obligation")],
            root_contribution=source.root_contribution,
            reason="looks similar",
            reuse_task_id=source.id,
        )
    ]
    work = queue_round(r.state, root.id, RoundKind.DECOMPOSE, "invalid-reuse")
    report = DrillDownReport(
        task_id=root.id,
        round_id=work.id,
        spec_revision=root.spec.revision,
        input_digest="",
        summary="reuse",
        children=r.worker.children,
    )
    with pytest.raises(ValueError, match="matching full responsibility"):
        r._commit_split(work, report)
    assert len(r.state.nodes) == 3
