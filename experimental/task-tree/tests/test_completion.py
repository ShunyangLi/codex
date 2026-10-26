from pathlib import Path

import pytest

from codex_task_tree.jev import JevDecisionMaker
from codex_task_tree.models import CheckReviewReport, NodeStatus, RoundKind, RunStatus
from codex_task_tree.runtime import certificate_valid, queue_round

from .test_tree import report_for


@pytest.mark.parametrize(
    "content", ["actual result: 12 of 80, with stated limitations", b"\xff\x00binary"]
)
def test_complete_receives_actual_report_material_and_verification(setup, content):
    runtime, judge, workspace = setup

    def worker(context, *_):
        path = workspace / "result.txt"
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        report = report_for(
            context,
            deliverable_paths=["result.txt"],
            evidence=["Concrete observations with a stated limitation"],
            completion_claim=True,
        )
        report.summary = "Worker claims the task is complete"
        return report

    runtime.worker = worker
    runtime.step()
    payload = next(payload for kind, _, payload in judge.calls if kind == "COMPLETE")
    result = payload["worker_results"][0]
    assert result["delivery_report"]["evidence"] == [
        "Concrete observations with a stated limitation"
    ]
    assert result["delivery_report"]["completion_claim"] is True
    material = payload["artifact_contents"][0]["current_content"]
    if isinstance(content, bytes):
        assert "text" not in material and "binary" in material["content"]
    else:
        assert material["text"] == content
    assert payload["verification"][0]["status"] == "PASS"
    assert payload["verification"][0]["actual_output"] == "verified\n"
    assert certificate_valid(runtime.state, runtime.state.root_task_id)


def test_reverification_sends_current_bytes_alongside_historical_worker_delivery(setup):
    runtime, judge, workspace = setup

    def worker(context, *_):
        (workspace / "Root.txt").write_text("original delivered bytes")
        return report_for(context, deliverable_paths=["Root.txt"], completion_claim=True)

    runtime.worker = worker
    runtime.step()
    revision = runtime.state.nodes[runtime.state.root_task_id].result_revision
    (workspace / "Root.txt").write_text("changed but formally reverified behavior")
    runtime.worker = lambda *_: pytest.fail("reverification must not rerun the worker")
    runtime.step()
    payload = [p for kind, _, p in judge.calls if kind == "COMPLETE"][-1]
    artifact = payload["artifact_contents"][0]
    assert artifact["current_content"]["text"] == "changed but formally reverified behavior"
    assert artifact["current_content"]["sha256"] != artifact["delivered_sha256"]
    assert payload["worker_results"][0]["current_result"]["revision"] == revision
    assert certificate_valid(runtime.state, runtime.state.root_task_id)


def test_review_completion_retains_original_delivery_and_adds_current_review(setup):
    runtime, judge, workspace = setup

    def deliver(context, *_):
        (workspace / "Root.txt").write_text("actual worker product")
        return report_for(context, deliverable_paths=["Root.txt"], evidence=["original result"])

    runtime.worker = deliver
    runtime.step()
    root = runtime.state.nodes[runtime.state.root_task_id]
    runtime.state.status = RunStatus.ACTIVE
    root.status = NodeStatus.READY
    queue_round(runtime.state, root.id, RoundKind.REVIEW, "independent-review")
    runtime.save()

    def review(context, *_):
        return CheckReviewReport(
            task_id=context["task_id"],
            round_id=context["round_id"],
            spec_revision=context["spec_revision"],
            input_digest=context["input_digest"],
            summary="independent review only",
            evidence=["review evidence"],
        )

    runtime.worker = review
    runtime.step()
    payload = [p for kind, _, p in judge.calls if kind == "COMPLETE"][-1]
    result = payload["worker_results"][0]
    assert result["delivery_kind"] == "EXECUTE"
    assert result["delivery_report"]["evidence"] == ["original result"]
    assert result["latest_round_report"]["summary"] == "independent review only"
    assert payload["artifact_contents"][0]["current_content"]["text"] == "actual worker product"


def test_corrupt_delivery_snapshot_cannot_be_presented_as_actual_evidence(setup):
    runtime, judge, workspace = setup

    def worker(context, *_):
        (workspace / "Root.txt").write_text("actual delivered bytes")
        return report_for(context, deliverable_paths=["Root.txt"], completion_claim=True)

    runtime.worker = worker
    runtime.step()
    root = runtime.state.root_task_id
    artifact = next(iter(runtime.state.artifacts.values()))
    Path(artifact.snapshot_ref).write_text("corrupted archive")
    calls = len(judge.calls)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        runtime.publish_certificate(root, runtime.decision_context(root))
    assert len(judge.calls) == calls


def test_full_completion_material_over_limit_blocks_without_truncating(setup):
    runtime, _, workspace = setup
    runtime.state.execution_profile.max_jev_input_bytes = 1001
    runtime.judge = JevDecisionMaker(runtime.state.execution_profile, runtime.store, lambda: 10)

    def worker(context, *_):
        (workspace / "Root.txt").write_text("actual result\n" * 1000)
        return report_for(context, deliverable_paths=["Root.txt"], completion_claim=True)

    runtime.worker = worker
    runtime.step()
    assert runtime.state.status == RunStatus.ACTIVE
    assert runtime.state.nodes[runtime.state.root_task_id].status == NodeStatus.NEEDS_CONTEXT
    assert next(iter(runtime.state.decision_jobs.values())).status == "WAITING_CONTEXT"
    assert not runtime.state.nodes[runtime.state.root_task_id].completion_certificate_id
