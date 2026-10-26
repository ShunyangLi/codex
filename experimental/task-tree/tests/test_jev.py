from __future__ import annotations

import json
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread

import pytest

from codex_task_tree.codex_adapter import parse_report, strict_schema
from codex_task_tree.jev import (
    COMMON_INSTRUCTIONS,
    PREDICATES,
    PROMPT_VERSION,
    JevDecisionMaker,
    compact_json_materials,
    decision_endpoint,
    decision_instructions,
    decision_key,
    decision_policy_digest,
    encode_references,
    validate_response,
)
from codex_task_tree.models import ExecutionProfile, SpecProposal
from codex_task_tree.store import RunStore

from .conftest import spec


@pytest.mark.parametrize("kind", list(PREDICATES))
def test_each_decision_has_fixed_guidance_and_one_final_proposition(kind):
    instructions = decision_instructions(kind)
    assert instructions.startswith(COMMON_INSTRUCTIONS)
    assert instructions.count("【唯一判题】") == 1
    assert instructions.endswith(PREDICATES[kind])
    assert PREDICATES[kind].count("？") == 1
    assert (
        not {"INFORMATION_GAIN", "EXPLORATION_ADEQUATE", "EXECUTABLE", "PLAN_ACCEPTABLE"}
        & PREDICATES.keys()
    )


def test_common_guidance_and_criteria_changes_invalidate_policy(monkeypatch):
    from codex_task_tree import jev

    original = decision_policy_digest()
    monkeypatch.setattr(jev, "COMMON_INSTRUCTIONS", COMMON_INSTRUCTIONS + "Updated policy")
    assert decision_policy_digest() != original
    changed = decision_policy_digest()
    monkeypatch.setitem(jev.DECISION_CRITERIA, "SHOULD_SPLIT", "Updated criterion")
    assert decision_policy_digest() != changed


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, "0.8", None])
def test_invalid_decision_is_not_a_negative_or_positive_score(score):
    with pytest.raises(ValueError):
        validate_response(
            {"model": "model", "answers": {"decision": {"type": "noul", "noul": score}}}, "model"
        )


def test_durable_valid_negative_is_never_resampled(tmp_path, monkeypatch):
    from codex_task_tree import jev

    monkeypatch.setattr(socket, "getfqdn", lambda *_: "localhost")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"model": "test", "answers": {"decision": {"type": "noul", "noul": 0.01}}}
                ).encode()
            )

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        store = RunStore(tmp_path, "wire")
        judge = JevDecisionMaker(
            ExecutionProfile(jev_url=f"http://127.0.0.1:{server.server_port}", jev_model="test"),
            store,
            lambda: 10,
        )
        first = judge("COMPLETE", "task", {"evidence": "missing", "hash": "a" * 64})
        second = judge("COMPLETE", "task", {"evidence": "missing", "hash": "a" * 64})
        assert not first.accepted and first == second
        assert len(calls) == 1
        changed = judge("COMPLETE", "task", {"evidence": "missing", "hash": "b" * 64})
        assert changed.id != first.id and len(calls) == 2
        assert calls[0] == calls[1]  # Equal symbolic packets still bind distinct original evidence.
        assert first.prompt_version == PROMPT_VERSION
        assert calls[0]["questions"] == {
            "decision": {"type": "noul", "instructions": decision_instructions("COMPLETE")}
        }
        monkeypatch.setattr(jev, "COMMON_INSTRUCTIONS", COMMON_INSTRUCTIONS + "Updated policy")
        revised = judge("COMPLETE", "task", {"evidence": "missing", "hash": "a" * 64})
        assert revised.id != first.id and len(calls) == 3
        assert not revised.accepted
    finally:
        server.shutdown()
        server.server_close()


def test_oversized_input_is_blocked_without_truncation(tmp_path):
    judge = JevDecisionMaker(
        ExecutionProfile(max_jev_input_bytes=1001), RunStore(tmp_path, "large"), lambda: 10
    )
    with pytest.raises(RuntimeError, match="no context was truncated"):
        judge("COMPLETE", "task", {"data": "x" * 2000})


def test_reference_encoding_is_reversible_and_preserves_semantics(tmp_path):
    task = "task_" + "a" * 32
    source = {
        "tasks": {
            task: {"goal": "Preserve every CSV row and reopen after restart", "hash": "b" * 64}
        },
        "evidence": [task + ":roundtrip", str(tmp_path / "reports" / (task + ".json"))],
        "literal": "@jev_ref_1 must stay literal",
    }
    encoded, references = encode_references(source, str(tmp_path))
    assert encoded["tasks"][next(iter(encoded["tasks"]))]["goal"] == source["tasks"][task]["goal"]
    restored = re.sub(
        "|".join(re.escape(token) for token in sorted(references, key=len, reverse=True)),
        lambda match: references[match.group()],
        json.dumps(encoded),
    )
    assert json.loads(restored) == source
    assert len(json.dumps(encoded)) < len(json.dumps(source))


def test_json_material_compaction_preserves_tokens_and_only_changes_materials():
    source = (
        ' { "rows": [ {"text": "  keep \\t \\"quoted\\" \\\\ 中文  "}, 10000000000000000000000000000000000001, '
        '-0.00000000000000000001e+12 ], "duplicate": 1, "duplicate": 2 }\n'
    )
    state = {
        "review_material": {"result.json": source, "notes.md": source},
        "artifact_contents": [{"path": "/result.json", "current_content": {"text": source}}],
        "frozen_files": {"definition.json": {"text": source}},
        "prompt": source,
    }
    encoded, records = compact_json_materials(state)
    text = encoded["review_material"]["result.json"]
    assert json.loads(text, parse_float=str, parse_int=str) == json.loads(
        source, parse_float=str, parse_int=str
    )
    assert '"duplicate":1,"duplicate":2' in text
    assert "10000000000000000000000000000000000001" in text
    assert "-0.00000000000000000001e+12" in text
    assert '"  keep \\t \\"quoted\\" \\\\ 中文  "' in text
    assert encoded["prompt"] == encoded["review_material"]["notes.md"] == source
    assert encoded["artifact_contents"][0]["current_content"]["text"] == text
    assert encoded["frozen_files"]["definition.json"]["text"] == text
    assert state["review_material"]["result.json"] == source
    assert len(records) == 3
    for invalid in [' {"broken": } ', ' {"invalid": NaN} ']:
        assert compact_json_materials({"broken.json": invalid}) == ({"broken.json": invalid}, [])


def test_json_material_transport_keeps_original_context_and_cached_answer(tmp_path):
    store = RunStore(tmp_path, "json-materials")
    judge = JevDecisionMaker(
        ExecutionProfile(jev_model="test", max_jev_input_bytes=12000), store, lambda: 10
    )
    state = {"review_material": {"result.json": json.dumps({"rows": list(range(100))}, indent=100)}}
    assert len(json.dumps(state).encode()) > 12000
    calls = []

    def open_request(request, **_):
        calls.append(request.data)
        return BytesIO(
            json.dumps(
                {"model": "test", "answers": {"decision": {"type": "noul", "noul": 0.01}}}
            ).encode()
        )

    judge.opener.open = open_request
    first = judge("EVIDENCE_SUPPORTED", "task", state)
    assert judge("EVIDENCE_SUPPORTED", "task", state) == first
    assert len(calls) == 1 and len(calls[0]) < 12000
    request = json.loads(calls[0])
    assert request["state"]["json_material_encoding"]["paths"] == ["/review_material/result.json"]
    assert json.loads((Path(first.request_ref).parent / "full-context.json").read_text()) == state
    assert json.loads(request["state"]["context"]["review_material"]["result.json"]) == {
        "rows": list(range(100))
    }


def test_strict_map_wire_schema_round_trips_nested_models():
    proposal = SpecProposal(task_id="root", round_id="spec", proposed_spec=spec())
    proposal.proposed_spec.constraints = {"language": "Python"}
    value = proposal.model_dump(mode="json")
    value["proposed_spec"]["constraints"] = [{"key": "language", "value": "Python"}]
    assert parse_report(SpecProposal, json.dumps(value)) == proposal
    schema = strict_schema(SpecProposal)
    constraints = schema["$defs"]["TaskSpec"]["properties"]["constraints"]
    assert constraints["type"] == "array"


@pytest.mark.parametrize("suffix", ["", "/compatible-mode/v1", "/compatible-mode/v1/systemone"])
def test_system_one_endpoint_handles_local_root_and_compatible_base(suffix):
    endpoint = decision_endpoint("https://example.com" + suffix)
    assert endpoint.endswith("/v1/systemone") and "/v1/v1/" not in endpoint


def test_env_file_is_not_executed_and_process_key_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('OTHER=$(touch should-not-exist)\nexport QWEN_KEY="file-only-secret"\n')
    profile = ExecutionProfile(decision_key_env="QWEN_KEY", decision_env_file=str(env))
    assert decision_key(profile) == "file-only-secret"
    assert not (tmp_path / "should-not-exist").exists()
    monkeypatch.setenv("QWEN_KEY", "process-only-secret")
    assert decision_key(profile) == "process-only-secret"
    assert "secret" not in profile.model_dump_json()


@pytest.mark.parametrize("mode", ["ok", "unauthorized", "redirect"])
def test_cloud_protocol_uses_auth_without_persisting_key_or_following_redirect(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(socket, "getfqdn", lambda *_: "localhost")
    monkeypatch.delenv("QWEN_KEY", raising=False)
    secret = "file-only-bearer-value"
    env = tmp_path / ".env"
    env.write_text("QWEN_KEY=" + secret + "\n")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), body))
            self.send_response(200 if mode == "ok" else (401 if mode == "unauthorized" else 302))
            if mode == "redirect":
                self.send_header("Location", "/must-not-follow")
            self.end_headers()
            response = {
                "model": "decision-model-preview",
                "answers": {"decision": {"type": "noul", "noul": 0.01}},
                "usage": {"input_tokens": 80},
            }
            self.wfile.write(
                json.dumps(
                    response if mode == "ok" else {"error": "failed; echo " + secret}
                ).encode()
            )

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    store = RunStore(tmp_path / "controller", "cloud")
    try:
        profile = ExecutionProfile(
            jev_url=f"http://127.0.0.1:{server.server_port}/compatible-mode/v1",
            jev_model="decision-model-preview",
            decision_key_env="QWEN_KEY",
            decision_env_file=str(env),
        )
        judge = JevDecisionMaker(profile, store, lambda: 10)
        if mode == "ok":
            result = judge("COMPLETE", "failed-test", {"actual_check": "FAIL"})
            assert not result.accepted
            other = JevDecisionMaker(
                profile.model_copy(update={"jev_url": profile.jev_url + "/other/v1"}),
                store,
                lambda: 10,
            )
            assert other("COMPLETE", "failed-test", {"actual_check": "FAIL"}).id != result.id
        else:
            with pytest.raises(RuntimeError, match="HTTP") as error:
                judge("COMPLETE", "failed-test", {"actual_check": "FAIL"})
            assert secret not in str(error.value)
            assert len(calls) == 1
        assert calls[0][0] == "/compatible-mode/v1/systemone"
        assert calls[0][1] == "Bearer " + secret
        assert "questions" in calls[0][2] and "messages" not in calls[0][2]
        assert all(secret not in p.read_text() for p in store.directory.rglob("*.json"))
    finally:
        server.shutdown()
        server.server_close()
