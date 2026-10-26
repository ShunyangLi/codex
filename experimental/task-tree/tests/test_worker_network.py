from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from codex_task_tree import codex_adapter
from codex_task_tree.cli import parser
from codex_task_tree.codex_adapter import CodexAdapter
from codex_task_tree.models import ExecutionProfile, WorkerReport


@pytest.mark.parametrize("network", [False, True])
@pytest.mark.parametrize(
    "kind",
    ["PREPARE", "EXECUTE", "RECONCILE", "EXPLORE", "SCOPE_REVIEW", "DECOMPOSE", "REVIEW", "SPECIFY", "DECISION", None],
)
def test_network_opt_in_preserves_role_filesystem_policy(tmp_path, network, kind):
    adapter = object.__new__(CodexAdapter)
    adapter.workspace = tmp_path
    adapter.profile = ExecutionProfile(
        permission_profile="workspace-write-network" if network else "workspace-write"
    )
    policy = adapter.sandbox_policy(kind).root
    assert policy.network_access is network
    writable = kind in {"PREPARE", "EXECUTE", "RECONCILE"}
    assert policy.type == ("workspaceWrite" if writable else "readOnly")
    if writable:
        assert [path.root for path in policy.writable_roots] == [str(tmp_path)]
        assert policy.exclude_tmpdir_env_var and policy.exclude_slash_tmp


@pytest.mark.parametrize("network", [False, True])
def test_frozen_check_network_matches_execution_profile(tmp_path, network):
    adapter = object.__new__(CodexAdapter)
    adapter.workspace = tmp_path
    adapter.profile = ExecutionProfile(
        permission_profile="workspace-write-network" if network else "workspace-write"
    )
    captured = []

    def execute(params):
        captured.append(params)
        return SimpleNamespace(model_dump=lambda: {"exit_code": 0})

    adapter.codex = SimpleNamespace(
        _client=SimpleNamespace(config=SimpleNamespace(rpc_timeout_seconds=1)),
        command_exec=execute,
    )
    assert adapter.command_exec(["ssh", "hk", "true"], tmp_path, 10) == {"exit_code": 0}
    assert captured[0].sandbox_policy.root.network_access is network
    assert captured[0].sandbox_policy.root.type == "workspaceWrite"


def test_network_turn_uses_native_policy_model_and_report_schema(monkeypatch, tmp_path):
    captured = {}
    adapter = object.__new__(CodexAdapter)
    adapter.workspace = tmp_path
    adapter.profile = ExecutionProfile(
        permission_profile="workspace-write-network", worker_model="gpt-6-sol", worker_effort="xhigh"
    )

    def start_turn(thread_id, wire_input, *, params, for_handle):
        captured.update(thread_id=thread_id, params=params, for_handle=for_handle)
        return SimpleNamespace(turn=SimpleNamespace(id="turn")), "subscription"

    client = SimpleNamespace(
        _start_turn=start_turn, config=SimpleNamespace(rpc_timeout_seconds=1)
    )
    adapter.codex = SimpleNamespace(_client=client)

    class Handle:
        def __init__(self, actual_client, thread_id, turn_id, *, _subscription):
            assert actual_client is client
            assert (thread_id, turn_id, _subscription) == ("thread", "turn", "subscription")
            self.id = turn_id

        def run(self):
            return SimpleNamespace(status="completed", final_response="{}", usage=None)

    monkeypatch.setattr(codex_adapter, "TurnHandle", Handle)
    thread = SimpleNamespace(
        id="thread", read=lambda **_: SimpleNamespace(thread=SimpleNamespace(turns=[]))
    )
    started = []
    adapter._run_turn(
        thread, {"kind": "SCOPE_REVIEW"}, WorkerReport, {"kind": "SCOPE_REVIEW"},
        lambda *args: started.append(args), time.monotonic() + 10,
    )
    params = captured["params"]
    assert params.model == "gpt-6-sol" and params.effort == "xhigh"
    assert params.sandbox_policy.root.type == "readOnly"
    assert params.sandbox_policy.root.network_access is True
    assert params.output_schema["type"] == "object"
    assert captured["for_handle"] is True
    assert started == [("thread", "turn")]


def test_cli_network_access_requires_explicit_opt_in():
    assert not parser().parse_args(["start", "task"]).allow_worker_network
    assert parser().parse_args(["start", "task", "--allow-worker-network"]).allow_worker_network
