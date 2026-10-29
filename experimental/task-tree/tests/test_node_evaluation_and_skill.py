import json
import re
import sys
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from codex_task_tree import decision_skill, jev
from codex_task_tree.checks import propose_check, review_file
from codex_task_tree.codex_adapter import CodexAdapter, instructions_for_kind
from codex_task_tree.decision_context import decision_packet
from codex_task_tree.jev import (
    DecisionContextTooLarge,
    JevDecisionMaker,
    encode_references,
    prepare_request,
)
from codex_task_tree.models import (
    CheckChangeProposal,
    CheckSpec,
    ExecutionProfile,
    NodeStatus,
    RoundKind,
)
from codex_task_tree.runtime import certificate_valid
from codex_task_tree.scheduler import select_next
from codex_task_tree.store import RunStore

from .conftest import authorize, exploration_report, local_command, scope_review_report
from .test_tree import report_for


def test_failed_acceptance_can_repair_a_leaf_without_creating_children(setup):
    runtime, judge, workspace = setup
    judge.denied.add("SHOULD_SPLIT")
    executions, commands = [], []

    def command(*args):
        commands.append(args)
        if len(commands) == 1:
            return {"exit_code": 1, "stdout": "actual defect", "stderr": ""}
        return local_command(*args)

    def worker(context, *_):
        if context["kind"] == "EXPLORE":
            return exploration_report(context)
        if context["kind"] == "SCOPE_REVIEW":
            return scope_review_report(context)
        assert context["kind"] == "EXECUTE"
        executions.append(context["round_id"])
        assert context["decision_support"]["model"] == "gpt-6-luna"
        assert context["decision_support"]["controller_run_directory"] == str(
            runtime.store.directory
        )
        (workspace / "Root.txt").write_text("fixed result" if len(executions) > 1 else "defect")
        return report_for(
            context,
            deliverable_paths=["Root.txt"],
            completion_claim=True,
            **({"coverage_updates": []} if len(executions) == 1 else {}),
        )

    runtime.command_exec, runtime.worker = command, worker
    runtime.step()
    assert not any(kind == "COMPLETE" for kind, _, _ in judge.calls)
    assert runtime.state.rounds[select_next(runtime.state).work_ref.id].kind == RoundKind.EXPLORE
    for _ in range(3):
        runtime.step()
    root = runtime.state.root_task_id
    assert len(executions) == 2 and not runtime.state.children(root)
    assert certificate_valid(runtime.state, root)
    assert any(record.status == "FAIL" for record in runtime.state.verifications.values())


def test_failed_semantic_completion_returns_to_probe_without_auto_splitting(setup):
    runtime, judge, workspace = setup
    judge.denied.add("COMPLETE")

    def worker(context, *_):
        (workspace / "Root.txt").write_text("checks pass but goal not fulfilled")
        return report_for(context, deliverable_paths=["Root.txt"], completion_claim=True)

    runtime.worker = worker
    runtime.step()
    assert runtime.state.nodes[runtime.state.root_task_id].status == NodeStatus.READY
    assert runtime.state.rounds[select_next(runtime.state).work_ref.id].kind == RoundKind.EXPLORE
    assert [kind for kind, _, _ in judge.calls] == ["COMPLETE"]
    assert not runtime.state.children(runtime.state.root_task_id)


def test_reviewed_history_can_shrink_without_removing_explicit_adverse_evidence():
    original = {
        "root": {"goal": "approved goal", "acceptance": ["restart works"]},
        "task": {"id": "root"},
        "scope_review": {
            "evidence": ["supporting details " * 10000],
            "omissions": ["restart behavior still missing"],
            "aspect_assessments": [{"status": "NEEDS_PROBE", "subject": "restart"}],
        },
        "verification": [{"status": "FAIL", "actual_output": "restart failed"}],
        "decision_brief_override": {
            "decision_brief": {"summary": "restart is unresolved", "unresolved": ["restart"]},
            "reviewed_sections": ["scope_review"],
            "review_ref": "review.json",
        },
    }
    packet, manifest = decision_packet("SHOULD_SPLIT", original)
    assert len(json.dumps(packet)) < len(json.dumps(original)) / 10
    assert packet["root"] == original["root"]
    assert packet["verification"] == original["verification"]
    retained = packet["retained_adverse_information"]
    assert any(item["value"] == ["restart behavior still missing"] for item in retained)
    assert any(
        item["value"].get("status") == "NEEDS_PROBE"
        for item in retained
        if isinstance(item["value"], dict)
    )
    assert original["scope_review"]["evidence"] and manifest


@pytest.mark.parametrize("section", ["root", "task", "verification", "coverage", "result_draft"])
def test_context_reviewer_cannot_remove_protected_material(section):
    with pytest.raises(ValueError, match="protected"):
        decision_packet("COMPLETE", {"decision_brief_override": {"reviewed_sections": [section]}})


def test_parent_uses_current_child_scope_without_repeating_descendant_logs():
    records = [{"task_id": "root", "status": "PASS", "actual_output": "integration verified"}]
    records += [{"task_id": "leaf", "status": "PASS", "actual_output": "checked " * 50000}]
    records += [{"task_id": "leaf", "status": "FAIL", "actual_output": "failure remains visible"}]
    original = {
        "task": {"id": "root"},
        "task_catalog": [
            {"id": "child", "parent_id": "root"},
            {"id": "leaf", "parent_id": "child"},
        ],
        "artifact_contents": [],
        "verification": records,
        "child_completion_proofs": [
            {
                "task_id": "child",
                "currently_valid": True,
                "certificate": {"verification_ids": ["v1"]},
            }
        ],
        "worker_results": [
            {"task_id": "root"},
            {
                "task_id": "child",
                "parent_id": "root",
                "delivery_report": {"summary": "actual result"},
            },
        ],
    }
    packet, _ = decision_packet("COMPLETE", original)
    assert len(json.dumps(packet)) < len(json.dumps(original)) / 10
    assert packet["verification"] == [records[0], records[-1]]
    assert packet["worker_results"][1]["delivery_report"]["summary"] == "actual result"
    assert original["child_completion_proofs"][0]["certificate"]["verification_ids"] == ["v1"]


def test_worker_skill_preview_matches_real_wire_and_does_not_call_api(
    tmp_path, monkeypatch, capsys
):
    context = {"task": {"id": "task"}, "evidence": "actual observation"}
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context))
    monkeypatch.setattr(
        jev, "decision_key", lambda *_: pytest.fail("preview cannot access API keys")
    )
    assert (
        decision_skill.main(
            ["--context", str(path), "--kind", "COMPLETE", "--run-directory", str(tmp_path)]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    real = prepare_request("COMPLETE", "decision-model-preview", context, str(tmp_path))
    assert result["wire_bytes"] == len(real.encoded)
    assert result["provider_token_count"] is None
    assert list(tmp_path.iterdir()) == [path]
    assert decision_skill.instructions() not in instructions_for_kind("CONTEXT_REVIEW")


def test_working_budget_is_enforced_even_below_provider_byte_ceiling(tmp_path):
    profile = ExecutionProfile(max_jev_input_bytes=262144, jev_context_target_bytes=10000)
    store = RunStore(tmp_path, "working")
    judge = JevDecisionMaker(profile, store, lambda: 10)
    with pytest.raises(DecisionContextTooLarge, match="working limit"):
        judge("SHOULD_SPLIT", "task", {"data": "x" * 20000})
    transport = json.loads(
        next((store.directory / "decisions").glob("*/transport.json")).read_text()
    )
    assert transport["wire_bytes"] < transport["hard_limit_bytes"]
    assert transport["working_budget_bytes"] == 10000


@pytest.mark.parametrize(
    "status,body",
    [(400, "context_length_exceeded"), (413, "too large"), (422, "too many input tokens")],
)
def test_provider_token_limit_becomes_local_context_repair_not_a_verdict(tmp_path, status, body):
    calls = []

    class Opener:
        def open(self, request, **kwargs):
            calls.append(request)
            raise urllib.error.HTTPError(request.full_url, status, body, {}, BytesIO(body.encode()))

    judge = JevDecisionMaker(ExecutionProfile(), RunStore(tmp_path, "provider_limit"), lambda: 10)
    judge.opener = Opener()
    with pytest.raises(DecisionContextTooLarge):
        judge("COMPLETE", "task", {"actual": "result"})
    assert len(calls) == 1


def test_child_receives_inherited_clauses_without_other_branches_future_acceptance():
    root = {
        "goal": "whole product",
        "scope": ["whole scope"],
        "constraints": {"permission": "read only"},
        "acceptance": [
            {"id": "A1", "description": "child outcome"},
            {"id": "A2", "description": "sibling outcome"},
        ],
        "requirements": [
            {"id": "R1", "acceptance_ids": ["A1"]},
            {"id": "R2", "acceptance_ids": ["A2"]},
        ],
    }
    context = {
        "root": root,
        "root_task_id": "root",
        "task": {
            "id": "child",
            "parent_id": "root",
            "spec": {"parent_criterion_ids": ["A1"], "root_requirement_ids": ["R1"]},
        },
        "artifact_contents": [],
    }
    packet, _ = decision_packet("COMPLETE", context)
    assert packet["root"]["acceptance"] == [root["acceptance"][0]]
    for key in ("goal", "scope", "constraints"):
        assert packet["root"][key] == root[key]
    context["task"]["spec"]["parent_criterion_ids"] = ["not-a-root-clause"]
    packet, _ = decision_packet("COMPLETE", context)
    assert packet["root"] == root  # Unknown/transitive mappings cannot silently drop clauses.
    context["task"]["parent_id"] = "another-parent"
    context["task"]["spec"]["parent_criterion_ids"] = ["A1"]
    packet, _ = decision_packet("COMPLETE", context)
    assert packet["root"] == root  # Equal local IDs do not establish an inherited root mapping.


def test_workspace_prefix_encoding_is_reversible_and_keeps_relative_source_names(tmp_path):
    workspace = str(tmp_path / "workspace")
    original = {
        "workspace_root": workspace,
        "evidence": [workspace + "/src/storage.py", "other exact content"],
    }
    encoded, refs = encode_references(original, str(tmp_path / "controller"), workspace)
    assert "/src/storage.py" in encoded["evidence"][0]
    restored = re.sub(
        "|".join(re.escape(key) for key in sorted(refs, key=len, reverse=True)),
        lambda match: refs[match[0]],
        json.dumps(encoded),
    )
    assert json.loads(restored) == original


@pytest.mark.parametrize("source_name", ["full-context.json", "source.json"])
def test_frozen_check_code_is_reused_only_for_exact_reviewed_current_bytes(setup, source_name):
    runtime, judge, workspace = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    (workspace / ".checks").mkdir()
    script = workspace / ".checks/check.py"
    script.write_text("print('verified')\n" + "# original implementation\n" * 100)
    root.spec.acceptance[0].check_spec = CheckSpec(
        command=[sys.executable, str(script)],
        artifact_paths=["Root.txt"],
    )
    runtime.state.root_approval.contract_hash = root.spec.contract_hash()
    definition = propose_check(
        runtime.state,
        runtime.store,
        root.id,
        "independent-fixture-proposer",
        CheckChangeProposal(
            id="frozen-check",
            criterion_id="ok",
            old_check_revision_id=root.spec.acceptance[0].adopted_check_revision_id,
            check_spec=root.spec.acceptance[0].check_spec,
            file_paths=[".checks/check.py"],
            reason="independently reviewed check implementation",
        ),
    )
    definition.status = "ADOPTED"
    root.spec.acceptance[0].adopted_check_revision_id = definition.id
    frozen = Path(definition.bundle_ref) / "files/.checks/check.py"
    payload = {"frozen_files": {"files/.checks/check.py": review_file(frozen)}}
    decision = judge("CHECK_CHANGE", definition.id, payload)
    decision.request_ref = runtime.store.write_json("frozen-review/request.json", {})
    runtime.store.write_json(f"frozen-review/{source_name}", payload)
    runtime.state.decisions[decision.id] = decision
    definition.review_refs.append(decision.id)
    authorize(runtime)
    runtime.save()

    def worker(context, *_):
        (workspace / "Root.txt").write_text("actual product")
        return report_for(
            context, deliverable_paths=["Root.txt", ".checks/check.py"], completion_claim=True
        )

    runtime.worker = worker
    runtime.step()
    context = runtime.completion_context(root.id, runtime.decision_context(root.id))
    packet, _ = decision_packet("COMPLETE", context)
    material = next(item for item in packet["artifact_contents"] if item["path"] == str(script))
    assert "text" not in material["current_content"]
    script.write_text(script.read_text() + "# changed after accepted review\n")
    context = runtime.completion_context(root.id, runtime.decision_context(root.id))
    packet, _ = decision_packet("COMPLETE", context)
    material = next(item for item in packet["artifact_contents"] if item["path"] == str(script))
    assert material["current_content"]["text"].endswith("# changed after accepted review\n")


@pytest.mark.parametrize("claim_criterion", [None, "different", "actual"])
def test_evidence_body_deferral_requires_independent_observation_for_that_criterion(
    claim_criterion,
):
    original = {
        "criterion": {"id": "actual", "description": "approved behavior"},
        "review_material": {"result.json": json.dumps({"observations": list(range(10000))})},
        "behavioral_check": {"exit_code": 1, "stdout": "actual failure must remain visible"},
    }
    if claim_criterion:
        original["decision_brief_override"] = {
            "decision_brief": {
                "claims": [
                    {
                        "criterion_ids": [claim_criterion],
                        "evidence_refs": ["result.json"],
                        "statement": "independent observation",
                    }
                ]
            }
        }
    packet, _ = decision_packet("EVIDENCE_SUPPORTED", original)
    assert packet["criterion"] == original["criterion"]
    assert packet["behavioral_check"] == original["behavioral_check"]
    assert isinstance(packet["review_material"]["result.json"], dict) == (
        claim_criterion == "actual"
    )
    assert isinstance(original["review_material"]["result.json"], str)


def test_native_prompt_profile_change_invalidates_previous_routing(setup):
    runtime, _, _ = setup
    work = next(r for r in runtime.state.rounds.values() if r.kind == RoundKind.EXECUTE)
    assert runtime.authorization_current(work)
    runtime.state.execution_profile.native_prompt_profile = "open-task-v1"
    assert not runtime.authorization_current(work)


def test_native_profile_cannot_silently_run_on_an_older_binary(tmp_path):
    binary = tmp_path / "old-codex"
    binary.write_bytes(b"old executable without native method")
    with pytest.raises(ValueError, match="not compiled"):
        CodexAdapter(
            ExecutionProfile(codex_bin=str(binary), native_prompt_profile="open-task-v1"), tmp_path
        )
