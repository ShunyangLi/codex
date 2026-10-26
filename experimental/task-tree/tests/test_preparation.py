import json
from pathlib import Path

import pytest

from codex_task_tree.codex_adapter import CodexAdapter, instructions_for_kind
from codex_task_tree.context import build_context, preparation_view
from codex_task_tree.models import ExecutionProfile, RoundKind, RoundPhase, RunStatus, SpecProposal
from codex_task_tree.runtime import TaskRuntime
from codex_task_tree.scheduler import select_next
from codex_task_tree.store import RunStore

from .conftest import FakeJudge, local_command, preparation_report, spec


def new_runtime(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RunStore(tmp_path / "controller", "prepared")
    TaskRuntime.create(store, "understand then solve", workspace, ExecutionProfile())
    judge = FakeJudge()
    return TaskRuntime(store, None, judge, local_command), workspace, judge


def test_general_preparation_contract_is_saved_and_goal_decomposition_reuses_it(tmp_path):
    runtime, workspace, _ = new_runtime(tmp_path)
    adapter = CodexAdapter.__new__(CodexAdapter)
    adapter.profile = runtime.state.execution_profile

    def worker(context, *_):
        return preparation_report(context, workspace)

    worker.report_contract = adapter.report_contract
    runtime.worker = worker
    runtime.step()
    work = next(r for r in runtime.state.rounds.values() if r.kind == RoundKind.PREPARE)
    manifest = json.loads(Path(work.context_manifest_ref).read_text())
    instructions = manifest["execution_contract"]["developer_instructions"]
    assert instructions == instructions_for_kind("PREPARE")
    assert "general task-tree harness" in instructions
    assert "background/knowledge.md" in instructions
    assert "verified facts, assumptions, unknowns" in instructions
    assert "COMPLETE the approved parent goal" in instructions_for_kind("DECOMPOSE")


def test_preparation_precedes_specification_and_is_shared_without_creating_tasks(tmp_path):
    runtime, workspace, judge = new_runtime(tmp_path)
    calls = []

    def worker(context, *_):
        calls.append(context)
        if context["kind"] == "PREPARE":
            assert context["preparation"] is None
            return preparation_report(context, workspace)
        assert context["preparation"]["index"]["known"] == "CSV storage"
        return SpecProposal(
            task_id=context["task_id"], round_id=context["round_id"], proposed_spec=spec()
        )

    runtime.worker = worker
    runtime.step()
    root = runtime.state.nodes[runtime.state.root_task_id]
    assert len(runtime.state.nodes) == 1 and not root.result_history and not root.specs
    assert runtime.state.status == RunStatus.ACTIVE and not judge.calls
    snapshot = preparation_view(runtime.state)
    (workspace / "background/index.json").write_text(
        '{"sources":[],"known":"unverified replacement"}'
    )
    assert preparation_view(runtime.state) == snapshot
    runtime.step()
    assert runtime.state.status == RunStatus.AWAITING_APPROVAL and not runtime.state.root_approval
    assert [c["kind"] for c in calls] == ["PREPARE", "SPECIFY"]
    runtime.approve_root(root.spec.contract_hash(), "actual user confirmation")
    ref = select_next(runtime.state).work_ref
    assert build_context(runtime.state, ref.id)["preparation"] == snapshot
    assert runtime.decision_context(root.id)["preparation"]["index"]["known"] == "CSV storage"


def test_jev_receives_actual_verified_knowledge_text_without_inventing_json_contents(tmp_path):
    runtime, workspace, _ = new_runtime(tmp_path)

    def worker(context, *_):
        report = preparation_report(context, workspace)
        (workspace / "background/field_map.md").write_text(
            "seniority: recorded code; labels unknown"
        )
        (workspace / "background/profile.json").write_text('{"rows": 42}')
        report.artifact_paths += ["background/field_map.md", "background/profile.json"]
        return report

    runtime.worker = worker
    runtime.step()
    prepared = preparation_view(runtime.state)
    assert list(prepared["knowledge_documents"].values()) == [
        "seniority: recorded code; labels unknown"
    ]
    assert any(a["path"].endswith("profile.json") for a in prepared["artifacts"])
    (workspace / "background/field_map.md").write_text("unverified replacement")
    assert preparation_view(runtime.state) == prepared


@pytest.mark.parametrize("bad", ["missing", "outside", "absolute"])
def test_invalid_preparation_retries_before_specification(tmp_path, bad):
    runtime, workspace, _ = new_runtime(tmp_path)
    calls = []

    def worker(context, *_):
        calls.append(context["kind"])
        report = preparation_report(context, workspace)
        if len(calls) == 1:
            if bad == "absolute":
                report.artifact_paths = [str(workspace / "background/index.json")]
            else:
                report.artifact_paths = [
                    "background/index.json",
                    "background/missing.json" if bad == "missing" else "product.py",
                ]
        return report

    runtime.worker = worker
    runtime.step()
    assert not runtime.state.current_preparation_id
    assert any(r.phase == RoundPhase.FAILED for r in runtime.state.rounds.values())
    if bad == "absolute":
        assert any("workspace-relative" in (r.error or "") for r in runtime.state.rounds.values())
    runtime.step()
    assert calls == ["PREPARE", "PREPARE"] and runtime.state.current_preparation_id
    assert runtime.state.progress.worker_attempts_started == 2
    assert select_next(runtime.state).work_ref.id in runtime.state.rounds
    assert len(runtime.state.nodes) == 1


def test_staged_preparation_recovers_without_replaying_worker(tmp_path):
    runtime, workspace, _ = new_runtime(tmp_path)
    ref = runtime.claim_next_work()
    work = runtime.state.rounds[ref.id]
    work.input_digest = "durably-bound-context"
    context = build_context(runtime.state, work.id)
    context["input_digest"] = work.input_digest
    report = preparation_report(context, workspace)
    work.worker_report_ref = runtime.store.write_json(
        f"reports/{work.id}/worker.json", report.model_dump(mode="json")
    )
    work.phase = RoundPhase.VERIFYING
    runtime.save()
    restored = TaskRuntime(
        runtime.store, lambda *_: pytest.fail("must use staged background"), None, None
    )
    restored.step()
    assert restored.state.current_preparation_id == work.id
    assert len(restored.state.preparations) == 1
    assert restored.state.nodes[work.task_id].result_history == []


def test_repreparation_preserves_history_budget_and_requires_new_specification(tmp_path):
    runtime, workspace, _ = new_runtime(tmp_path)

    def worker(context, *_):
        if context["kind"] == "PREPARE":
            return preparation_report(context, workspace)
        proposed = spec(goal="goal informed by preparation " + str(context["next_spec_revision"]))
        proposed.revision = context["next_spec_revision"]
        return SpecProposal(
            task_id=context["task_id"], round_id=context["round_id"], proposed_spec=proposed
        )

    runtime.worker = worker
    runtime.step()
    runtime.step()
    first = runtime.state.current_preparation_id
    old_contract = runtime.state.nodes[runtime.state.root_task_id].spec.contract_hash()
    elapsed = runtime.state.progress.active_elapsed_seconds
    runtime.begin_preparation()
    assert not runtime.state.current_preparation_id and not runtime.state.root_approval
    assert runtime.state.progress.worker_attempts_started == 2
    runtime.step()
    runtime.step()
    assert len(runtime.state.preparations) == 2 and first in runtime.state.preparations
    assert runtime.state.progress.worker_attempts_started == 4
    assert runtime.state.progress.active_elapsed_seconds >= elapsed
    assert len(runtime.state.nodes[runtime.state.root_task_id].specs) == 2
    assert runtime.state.status == RunStatus.AWAITING_APPROVAL
    with pytest.raises(ValueError, match="current proposed contract"):
        runtime.approve_root(old_contract, "old confirmation")
    artifact = runtime.state.artifacts[runtime.state.preparations[first].index_artifact_id]
    assert Path(artifact.snapshot_ref).is_file()
    assert [r.kind for r in runtime.state.rounds.values()] == [
        RoundKind.PREPARE,
        RoundKind.SPECIFY,
        RoundKind.PREPARE,
        RoundKind.SPECIFY,
    ]


@pytest.mark.parametrize("invalid", ["missing", "empty", "duplicate"])
def test_preparation_requires_a_source_based_investigation_inventory(tmp_path, invalid):
    runtime, workspace, judge = new_runtime(tmp_path)

    def worker(context, *_):
        report = preparation_report(context, workspace)
        path = workspace / "background/index.json"
        index = json.loads(path.read_text())
        if invalid == "missing":
            del index["investigation_inventory"]
        elif invalid == "empty":
            index["investigation_inventory"] = []
        else:
            index["investigation_inventory"] *= 2
        path.write_text(json.dumps(index))
        return report

    runtime.worker = worker
    runtime.step()
    assert not runtime.state.current_preparation_id
    assert not judge.calls
    assert select_next(runtime.state).work_ref.id in runtime.state.rounds
    assert all(r.kind == RoundKind.PREPARE for r in runtime.state.rounds.values())
