from __future__ import annotations

import json

import pytest

from codex_task_tree.cli import main, parser
from codex_task_tree.codex_adapter import CodexAdapter
from codex_task_tree.models import RoundPhase, RunStatus, WorkRef


def test_status_is_available_while_controller_holds_run_lock(setup, capsys):
    runtime, _, _ = setup
    with runtime.store.locked():
        assert main(["--home", str(runtime.store.root), "status", runtime.state.run_id]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == runtime.state.run_id


def test_cancel_before_execution_releases_workspace_without_erasing_history(setup):
    runtime, _, workspace = setup
    ref = runtime.claim_next_work()
    assert main(["--home", str(runtime.store.root), "cancel", runtime.state.run_id]) == 0
    restored = runtime.store.load()
    assert restored.status == RunStatus.CANCELLED
    assert restored.rounds[ref.id].phase == RoundPhase.INTERRUPTED
    assert restored.scheduler.active_work is None


def test_cancel_cannot_release_a_still_running_executor(setup, monkeypatch):
    runtime, _, _ = setup
    ref = runtime.claim_next_work()
    runtime.state.rounds[ref.id].execution_owner = {"pid": "123", "start": "test"}
    runtime.store.execution_reservation([ref.id])
    runtime.save()

    def alive(_):
        raise RuntimeError("executor is still alive")

    monkeypatch.setattr(CodexAdapter, "ensure_stopped", alive)
    assert main(["--home", str(runtime.store.root), "cancel", runtime.state.run_id]) == 1
    assert runtime.store.load().scheduler.active_work == ref


def test_cancel_stops_parallel_sibling_owners_before_releasing_workspace(setup, monkeypatch):
    runtime, _, _ = setup
    work = next(iter(runtime.state.rounds.values()))
    runtime.state.scheduler.active_siblings = [WorkRef(kind="ROUND", id=work.id)]
    work.phase = RoundPhase.WORKER_RUNNING
    work.execution_owner = {"pid": "123", "start": "fixture"}
    runtime.store.execution_reservation([work.id])
    runtime.save()
    stopped = []
    monkeypatch.setattr(CodexAdapter, "ensure_stopped", lambda owner: stopped.append(owner))
    assert main(["--home", str(runtime.store.root), "cancel", runtime.state.run_id]) == 0
    saved = runtime.store.load()
    assert stopped == [work.execution_owner]
    assert not saved.scheduler.active_siblings
    assert saved.status == RunStatus.CANCELLED


@pytest.mark.parametrize("required,expected", [(True, "PREPARE"), (False, "SPECIFY")])
def test_resume_before_approval_returns_to_preparation_or_specification(
    setup, monkeypatch, required, expected
):
    from codex_task_tree import cli
    from codex_task_tree.models import NodeStatus

    runtime, _, _ = setup
    state = runtime.state
    state.root_approval = None
    state.preparation_required = required
    state.status = RunStatus.BLOCKED
    state.nodes[state.root_task_id].status = NodeStatus.BLOCKED
    for work in state.rounds.values():
        work.phase = RoundPhase.FAILED
    runtime.save()
    monkeypatch.setattr(cli, "run", lambda store: None)
    assert (
        main(
            [
                "--home",
                str(runtime.store.root),
                "resume",
                state.run_id,
                "--reason",
                "report format fixed",
            ]
        )
        == 0
    )
    saved = runtime.store.load()
    queued = [r for r in saved.rounds.values() if r.phase == RoundPhase.QUEUED]
    assert len(queued) == 1 and queued[0].kind == expected
    assert not saved.root_approval


def test_start_defaults_to_three_action_protocol(tmp_path, monkeypatch):
    from codex_task_tree import cli

    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\necho fixture-codex\n")
    binary.chmod(0o755)
    (tmp_path / "codex-code-mode-host").write_text("fixture host")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    seen = []
    monkeypatch.setattr(cli, "run", lambda store: seen.append(store.load()))
    assert main(["--home", str(tmp_path / "controller"), "start", "--codex-bin", str(binary),
                 "--workspace", str(workspace), "Original task question"]) == 0
    state = seen[0]
    assert state.schema_version == 3
    assert state.background["root_goal"] == "Original task question"
    assert state.nodes[state.root_task_id].spec.goal == "Original task question"
    assert state.root_approval.contract_hash == state.nodes[state.root_task_id].spec.contract_hash()
    assert not state.preparation_required
    assert [r.kind.value for r in state.rounds.values()] == ["EXPLORE"]
    assert state.execution_profile.max_parallel_siblings == 3


def test_start_accepts_parallel_sibling_limit():
    args = parser().parse_args(["start", "--parallel-siblings", "2", "Task"])
    assert args.parallel_siblings == 2
    args = parser().parse_args(["start", "--parallel-branches", "3", "Task"])
    assert args.parallel_siblings == 3


def test_old_saved_run_without_parallel_setting_stays_serial(setup):
    runtime, _, _ = setup
    with runtime.store.connect() as db:
        raw = json.loads(db.execute(
            "SELECT state_json FROM run_state WHERE run_id=?", (runtime.state.run_id,)
        ).fetchone()[0])
        raw["execution_profile"].pop("max_parallel_siblings")
        db.execute("UPDATE run_state SET state_json=? WHERE run_id=?",
                   (json.dumps(raw), runtime.state.run_id))
    assert runtime.store.load().execution_profile.max_parallel_siblings == 1
