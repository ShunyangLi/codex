from __future__ import annotations

import asyncio
import queue
from unittest.mock import AsyncMock, Mock

from openai_codex import AsyncCodex, Codex
from openai_codex.client import CodexClient, CodexConfig
from openai_codex.types import CommandExecParams, CommandExecResponse


def test_command_exec_forwards_exact_wire_fields():
    codex = object.__new__(Codex)
    response = CommandExecResponse(exit_code=0, stdout="ok", stderr="")
    codex._client = Mock()
    codex._client.request.return_value = response
    assert (
        codex.command_exec(
            CommandExecParams(command=["python", "check.py"], cwd="/project", timeout_ms=500)
        )
        is response
    )
    codex._client.request.assert_called_once_with(
        "command/exec",
        {"command": ["python", "check.py"], "cwd": "/project", "timeoutMs": 500},
        response_model=CommandExecResponse,
    )


def test_async_command_exec_initializes_and_propagates_errors():
    async def scenario():
        codex = AsyncCodex()
        codex._ensure_initialized = AsyncMock()
        codex._client.request = AsyncMock(side_effect=RuntimeError("denied"))
        try:
            await codex.command_exec(CommandExecParams(command=["false"]))
        except RuntimeError as error:
            assert str(error) == "denied"
        else:
            raise AssertionError("RPC errors must not become successful checks")
        codex._ensure_initialized.assert_awaited_once()

    asyncio.run(scenario())


def test_rpc_deadline_discards_waiter_without_claiming_execution_stopped():
    client = CodexClient(CodexConfig(rpc_timeout_seconds=0.01))
    client._write_message = Mock()
    client._router = Mock()
    client._router.create_response_waiter.return_value.get.side_effect = queue.Empty
    try:
        client._request_raw("command/exec", {"command": ["sleep", "1"]})
    except TimeoutError as error:
        assert "may still be active" in str(error)
    else:
        raise AssertionError("RPC timeout must propagate")
    client._router.discard_response_waiter.assert_called_once()
