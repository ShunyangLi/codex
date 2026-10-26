import sys

import pytest

from codex_task_tree.checks import latest_record
from codex_task_tree.context import fingerprints, read_roots, workspace_path
from codex_task_tree.models import RoundKind, RoundPhase
from codex_task_tree.runtime import certificate_valid

from .conftest import adopt, authorize, exploration_report, exploration_worker
from .test_routing import begin_exploration, next_kind, step_route
from .test_tree import report_for


def test_registered_round_context_can_be_cited_without_granting_parent_directory(setup):
    runtime, _, workspace = setup
    work = next(iter(runtime.state.rounds.values()))
    registered = runtime.store.write_json("contexts/exact.json", {"task": work.task_id})
    sibling = runtime.store.write_json("contexts/not-registered.json", {"not": "authorized"})
    work.context_manifest_ref = registered
    assert fingerprints(workspace, [registered], read_roots(runtime.state))
    with pytest.raises(ValueError, match="outside workspace"):
        fingerprints(workspace, [sibling], read_roots(runtime.state))


@pytest.mark.parametrize("unauthorized", ["unlisted", "symlink", "uri"])
def test_declared_inputs_allow_fingerprinting_without_granting_writes(tmp_path, unauthorized):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    source = data / "source.csv"
    source.write_text("old data")
    refs = [str(data)]
    before = fingerprints(workspace, [str(data)], refs)
    source.write_text("new data")
    assert before != fingerprints(workspace, [str(data)], refs)
    with pytest.raises(ValueError, match="outside workspace"):
        workspace_path(workspace, str(source))
    outside = tmp_path / "unlisted.csv"
    outside.write_text("not authorized")
    if unauthorized == "symlink":
        (data / "escape.csv").symlink_to(outside)
        target = data
    else:
        target = outside
    if unauthorized == "uri":
        refs = ["https://example.invalid/data"]
    with pytest.raises(ValueError, match="declared read-only inputs"):
        fingerprints(workspace, [str(target)], refs)


def test_external_dataset_change_invalidates_formal_checks_and_completion(setup):
    runtime, _, workspace = setup
    source = workspace.parent / "data.csv"
    source.write_text("input")
    runtime.state.input_refs = [str(source)]
    root = runtime.state.nodes[runtime.state.root_task_id]
    item = root.spec.acceptance[0]
    item.check_spec.command = [
        sys.executable,
        "-B",
        "-c",
        f"from pathlib import Path; assert Path({str(source)!r}).read_text() == 'input'",
    ]
    item.check_spec.artifact_paths = [str(source), "Root.txt"]
    item.adopted_check_revision_id = None
    adopt(runtime.state, runtime.store, root.id)
    runtime.state.root_approval.contract_hash = root.spec.contract_hash()
    authorize(runtime)

    def worker(context, *_):
        (workspace / "Root.txt").write_text("actual delivery")
        return report_for(context, deliverable_paths=["Root.txt"])

    runtime.worker = worker
    runtime.step()
    assert certificate_valid(runtime.state, root.id)
    current = latest_record(runtime.state, root.id, item.id)
    assert current and str(source) in current.input_paths
    source.write_text("changed upstream")
    assert not certificate_valid(runtime.state, root.id)
    assert latest_record(runtime.state, root.id, item.id) is None
    ids = runtime.verify(root.id, "changed-dataset")
    assert runtime.state.verifications[ids[0]].status == "FAIL"


def test_declared_dataset_changes_revoke_routes_even_if_worker_only_lists_workspace(setup):
    runtime, judge, workspace = setup
    source = workspace.parent / "data.csv"
    source.write_text("input")
    runtime.state.input_refs = [str(source)]
    begin_exploration(runtime)
    judge.denied.add("SHOULD_SPLIT")
    runtime.worker = exploration_worker
    step_route(runtime)
    assert next_kind(runtime) == RoundKind.EXECUTE
    source.write_text("changed upstream")
    runtime.worker = lambda *_: pytest.fail("stale dataset route must not execute")
    runtime.step()
    assert next_kind(runtime) == RoundKind.EXPLORE
    assert runtime.state.progress.worker_attempts_started == 1


def test_dataset_change_during_exploration_is_detected_before_jev(setup):
    runtime, judge, workspace = setup
    source = workspace.parent / "data.csv"
    source.write_text("input")
    runtime.state.input_refs = [str(source)]
    begin_exploration(runtime)

    def concurrent_update(context, *_):
        source.write_text("changed upstream")
        return exploration_report(context)

    runtime.worker = concurrent_update
    runtime.step()
    assert not judge.calls
    assert any(
        r.phase == RoundPhase.FAILED and "read-only round inputs changed" in r.error
        for r in runtime.state.rounds.values()
    )
