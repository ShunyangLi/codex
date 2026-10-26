from __future__ import annotations

import json

import pytest

from codex_task_tree.cli import main
from codex_task_tree.codex_adapter import CodexAdapter
from codex_task_tree.models import RoundPhase, RunStatus


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
