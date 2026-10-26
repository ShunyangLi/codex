import json

import pytest

from codex_task_tree.codex_adapter import strict_schema
from codex_task_tree.decision_context import decision_packet
from codex_task_tree.jev import DecisionContextTooLarge
from codex_task_tree.models import (
    ContextReviewReport,
    DecisionBrief,
    DecisionClaim,
    NodeStatus,
    RoundKind,
    RoundPhase,
    RunStatus,
    TaskNode,
    WaveCursor,
)
from codex_task_tree.runtime import certificate_valid, queue_round
from codex_task_tree.scheduler import select_next

from .conftest import authorize, exploration_report, spec
from .test_tree import children_report, report_for


def test_finite_batch_does_not_let_new_shallow_retries_starve_deeper_work(setup):
    runtime, _, _ = setup
    state = runtime.state
    for work in state.rounds.values():
        work.phase = RoundPhase.COMMITTED
    state.scheduler.wave = None
    root = state.nodes[state.root_task_id]
    child = TaskNode(
        id="deeper",
        parent_id=root.id,
        depth=1,
        created_seq=state.sequence(),
        specs=[spec("deeper", child=True)],
    )
    state.nodes[child.id] = child
    older = queue_round(state, child.id, RoundKind.EXPLORE, "older-deeper-work")
    shallow = queue_round(state, root.id, RoundKind.EXPLORE, "shallow")
    first = select_next(state)
    assert first.work_ref.id == shallow.id
    state.scheduler.wave = first.wave_after
    shallow.phase = RoundPhase.COMMITTED
    queue_round(state, root.id, RoundKind.EXPLORE, "new-shallow-retry")
    assert select_next(state).work_ref.id == older.id


def test_old_wave_snapshot_round_trips_without_losing_unclaimed_work():
    wave = WaveCursor.model_validate({"depth": 1, "round_ids": ["a", "b"], "next_index": 1})
    restored = WaveCursor.model_validate_json(wave.model_dump_json())
    assert restored.round_ids == ["a", "b"]
    assert restored.work_refs[restored.next_index].id == "b"


@pytest.mark.parametrize("repair_limit", [0, 2])
def test_local_decision_size_error_releases_slot_for_independent_work(setup, repair_limit):
    runtime, judge, _ = setup
    runtime.state.execution_profile.max_context_repairs = repair_limit
    runtime.save()
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda ctx, *_: children_report(ctx, ["A", "B"])
    runtime.step()
    a, b = runtime.state.children(runtime.state.root_task_id)
    judge.denied.add("SHOULD_SPLIT")
    original = runtime.judge

    def decide(kind, target, payload):
        if kind == "SHOULD_SPLIT" and target == a:
            raise DecisionContextTooLarge("oversized fixture")
        return original(kind, target, payload)

    runtime.judge = decide
    runtime.worker = lambda ctx, *_: exploration_report(ctx)
    runtime.step()  # A exploration and direct routing error
    assert runtime.state.status == RunStatus.ACTIVE
    assert runtime.state.nodes[a].status == NodeStatus.NEEDS_CONTEXT
    assert runtime.state.scheduler.active_work is None
    job = next(iter(runtime.state.decision_jobs.values()))
    assert job.owner_task_id == a and job.status == "WAITING_CONTEXT"
    assert job.repair_attempts == bool(repair_limit)
    assert sum(r.kind == RoundKind.CONTEXT_REVIEW for r in runtime.state.rounds.values()) == bool(
        repair_limit
    )
    assert runtime.state.rounds[job.source_ref.id].worker_report_ref
    assert runtime.store.load().decision_jobs[job.id] == job
    choice = select_next(runtime.state)
    assert runtime.state.rounds[choice.work_ref.id].task_id == b
    runtime.step()
    assert any(
        r.task_id == b and r.kind == RoundKind.EXPLORE and r.phase == RoundPhase.COMMITTED
        for r in runtime.state.rounds.values()
    )


@pytest.mark.parametrize("mode", ["normal", "worker_budget_exhausted", "cited_evidence_changed"])
def test_context_review_resumes_saved_delivery_without_executing_product_twice(setup, mode):
    runtime, judge, workspace = setup
    original = runtime.judge
    executed = []

    def decide(kind, target, payload):
        if kind == "COMPLETE" and "decision_brief_override" not in payload:
            raise DecisionContextTooLarge("needs a context brief")
        return original(kind, target, payload)

    def worker(context, *_):
        if context["kind"] == "CONTEXT_REVIEW":
            return ContextReviewReport(
                task_id=context["task_id"],
                round_id=context["round_id"],
                spec_revision=context["spec_revision"],
                input_digest=context["input_digest"],
                decision_job_id=context["pending_decision"]["id"],
                decision_brief=DecisionBrief(
                    summary="Independently inspected current result",
                    claims=[
                        DecisionClaim(
                            id="c1",
                            statement="actual bytes exist",
                            evidence_refs=["Root.txt"],
                            criterion_ids=["ok"],
                        )
                    ],
                ),
                input_paths=["background/index.json"],
            )
        executed.append(context["round_id"])
        (workspace / "Root.txt").write_text("actual bytes")
        return report_for(context, deliverable_paths=["Root.txt"], completion_claim=True)

    runtime.worker, runtime.judge = worker, decide
    runtime.step()
    root = runtime.state.root_task_id
    job = next(iter(runtime.state.decision_jobs.values()))
    assert job.resume_phase == "DECIDING"
    assert runtime.state.nodes[root].status == NodeStatus.NEEDS_CONTEXT
    runtime.step()
    assert job.status == "QUEUED"
    assert "Root.txt" in job.brief_fingerprints
    if mode == "worker_budget_exhausted":
        runtime.state.execution_profile.max_worker_rounds = (
            runtime.state.progress.worker_attempts_started
        )
        runtime.save()
    if mode == "cited_evidence_changed":
        (workspace / "Root.txt").write_text("changed after independent inspection")
    runtime.step()
    assert len(executed) == 1
    if mode == "cited_evidence_changed":
        assert not certificate_valid(runtime.state, root)
        assert runtime.state.nodes[root].status == NodeStatus.NEEDS_CONTEXT
        assert runtime.state.decision_jobs[job.id].repair_attempts == 2
        return
    assert certificate_valid(runtime.state, root)
    assert runtime.state.decision_jobs[job.id].status == "RESOLVED"
    payload = next(payload for kind, _, payload in judge.calls if kind == "COMPLETE")
    assert (
        payload["decision_brief_override"]["decision_brief"]["claims"][0]["statement"]
        == "actual bytes exist"
    )


def test_context_error_does_not_release_an_unresolved_native_execution(setup):
    runtime, _, _ = setup
    work = runtime.claim_next_work()
    source = runtime.state.rounds[work.id]
    source.phase = RoundPhase.WORKER_RUNNING
    runtime.save()

    def judge(*_):
        raise DecisionContextTooLarge("unexpected decision during native execution")

    runtime.judge = judge
    with pytest.raises(DecisionContextTooLarge) as captured:
        runtime.decide("COMPLETE", source.task_id, {})
    with pytest.raises(DecisionContextTooLarge):
        runtime._park_decision(captured.value)
    assert runtime.state.status == RunStatus.BLOCKED
    assert runtime.state.scheduler.active_work == work
    assert not runtime.state.decision_jobs


@pytest.mark.parametrize(
    "evidence_reviews", [[], [{"criterion_id": "k1", "scope": "checked scope only"}]]
)
def test_completion_never_silently_discards_unreviewed_large_artifact(evidence_reviews):
    text = json.dumps({"rows": list(range(10000))})
    state = {
        "task": {"id": "root"},
        "worker_results": [{"task_id": "root", "delivery_report": {"summary": "actual findings"}}],
        "artifact_contents": [
            {
                "artifact_id": "a",
                "task_id": "root",
                "path": "/data.json",
                "current_content": {"text": text, "sha256": "hash"},
                "evidence_reviews": evidence_reviews,
            }
        ],
        "verification": [{"status": "FAIL", "actual_output": "a failure must remain visible"}],
        "coverage": [{"status": "GAP"}],
        "result_draft": {"gaps": ["unresolved"]},
    }
    packet, manifest = decision_packet("COMPLETE", state)
    assert packet["verification"] == state["verification"]
    assert packet["coverage"] == state["coverage"]
    assert packet["result_draft"] == state["result_draft"]
    assert state["artifact_contents"][0]["current_content"]["text"] == text
    content = packet["artifact_contents"][0]["current_content"]
    assert ("text" not in content) == bool(evidence_reviews)
    assert bool(manifest) == bool(evidence_reviews)
    if evidence_reviews:
        assert (
            "not supplied" in content["content"]
            and manifest[0]["evidence_reviews"] == evidence_reviews
        )


def test_parent_packet_retains_invalid_child_proof_and_actual_direct_child_result():
    state = {
        "task": {"id": "parent"},
        "artifact_contents": [],
        "worker_results": [
            {"task_id": "parent", "parent_id": None},
            {
                "task_id": "child",
                "parent_id": "parent",
                "delivery_report": {"summary": "actual child result"},
            },
            {"task_id": "grandchild", "parent_id": "child"},
        ],
        "child_completion_proofs": [{"task_id": "child", "currently_valid": False}],
    }
    packet, _ = decision_packet("COMPLETE", state)
    assert len(packet["worker_results"]) == 2
    assert packet["worker_results"][1]["delivery_report"]["summary"] == "actual child result"
    assert not packet["child_completion_proofs"][0]["currently_valid"]


def test_worker_identity_is_bound_in_output_schema():
    identity = {
        "task_id": "task",
        "round_id": "round",
        "spec_revision": 4,
        "input_digest": "digest",
        "pending_decision": {"id": "job"},
    }
    schema = strict_schema(ContextReviewReport, identity)
    for key in ("task_id", "round_id", "spec_revision", "input_digest"):
        assert schema["properties"][key]["enum"] == [identity[key]]
    assert schema["properties"]["decision_job_id"]["enum"] == ["job"]
