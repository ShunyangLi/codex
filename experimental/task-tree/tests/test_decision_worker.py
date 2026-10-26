import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_task_tree.codex_adapter import CodexAdapter
from codex_task_tree.decision_worker import CodexDecisionMaker, question_catalog, required_context
from codex_task_tree.jev import DecisionWorkerUnavailable
from codex_task_tree.models import NodeResult, RoundKind, TaskNode
from codex_task_tree.runtime import reuse_scope_covers

from .conftest import authorize, spec
from .test_tree import children_report


def submission(ref, text="Two independently verifiable outcomes", criterion="ok"):
    return {
        "decision_brief": {
            "summary": "Relevant observations",
            "claims": [
                {
                    "id": "fact",
                    "statement": text,
                    "evidence_refs": [ref],
                    "criterion_ids": [criterion],
                }
            ],
        }
    }


class Transport:
    def __init__(self, runtime, judge):
        self.profile, self.store, self.time_left = (
            runtime.state.execution_profile,
            runtime.store,
            lambda: 600,
        )
        self.judge = judge

    def __call__(self, *args):
        return self.judge(*args)


def test_native_decision_repairs_in_one_session_and_never_resamples_valid_no(setup):
    runtime, judge, workspace = setup
    judge.denied.add("SHOULD_SPLIT")
    runtime.state.execution_profile.jev_context_target_bytes = 10000
    evidence = workspace / "observations.txt"
    evidence.write_text("Bounded verified observation")
    received, sessions = [], []

    def run(context, tool, started, timeout):
        sessions.append(context)
        started("decision-thread", "turn")
        received.append(tool(submission(str(evidence), criterion="invalid")))
        received.append(tool(submission(str(evidence), text="x" * 20000)))
        received.append(tool(submission(str(evidence))))
        received.append(tool(submission(str(evidence))))

    worker = CodexDecisionMaker(
        SimpleNamespace(run_decision=run), Transport(runtime, judge), lambda: runtime.state
    )
    payload = {"root": runtime.state.nodes[runtime.state.root_task_id].spec.model_dump(mode="json")}
    first = worker("SHOULD_SPLIT", runtime.state.root_task_id, payload)
    second = worker("SHOULD_SPLIT", runtime.state.root_task_id, payload)
    assert not first.accepted and first == second
    assert len(sessions) == 1 and len(judge.calls) == 1
    assert [r.get("code") for r in received[:2]] == ["INVALID_INPUT", "INPUT_TOO_LARGE"]
    assert received[2]["ok"] and received[3]["cached"]
    assert first.worker_thread_id == "decision-thread"
    sent = judge.calls[0][2]
    assert "exploration" not in sent and "artifact_contents" not in sent
    evidence.write_text("changed observation")
    with pytest.raises(ValueError, match="evidence changed"):
        worker("SHOULD_SPLIT", runtime.state.root_task_id, payload)


def test_other_task_results_never_enter_question_catalog(setup):
    runtime, _, _ = setup
    node = runtime.state.nodes[runtime.state.root_task_id]
    node.result_history.append(NodeResult(revision=1, round_id="old", summary="PRIVATE RESULT"))
    catalog = question_catalog(runtime.state)
    assert "PRIVATE RESULT" not in json.dumps(catalog)
    assert catalog[0]["question"] == node.spec.goal and "scope" in catalog[0]


@pytest.mark.parametrize("opportunities", [[], [{"question": "historical nonempty evidence"}]])
def test_routing_source_explains_empty_legacy_candidates_without_losing_evidence(
    setup, opportunities
):
    runtime, judge, workspace = setup
    evidence = workspace / "facts.txt"
    evidence.write_text("Two independent outcomes share verified inputs")
    payload = {
        "root": runtime.state.nodes[runtime.state.root_task_id].spec.model_dump(mode="json"),
        "exploration_draft": {
            "split_opportunities": opportunities,
            "findings": ["independent outcome A", "independent outcome B"],
            "uncertainties": ["preserve adverse observations"],
        },
        "controller_facts": {"focused_execution_blockers": []},
    }

    def run(context, tool, started, timeout):
        source = json.loads(Path(context["source_context_ref"]).read_text())
        explored = source["exploration_draft"]
        if opportunities:
            assert explored["split_opportunities"] == opportunities
        else:
            assert "split_opportunities" not in explored
        assert explored["findings"] == payload["exploration_draft"]["findings"]
        assert explored["uncertainties"] == ["preserve adverse observations"]
        facts = context["protected_context"]["controller_facts"]
        assert facts["focused_execution_blockers"] == []
        assert facts["candidate_proposals_require_prior_split_authorization"]
        assert facts["empty_legacy_exploration_candidates_are_expected"]
        started("protocol-thread", "turn")
        assert tool(submission(str(evidence)))["ok"]

    worker = CodexDecisionMaker(
        SimpleNamespace(run_decision=run), Transport(runtime, judge), lambda: runtime.state
    )
    worker("SHOULD_SPLIT", runtime.state.root_task_id, payload)
    assert payload["exploration_draft"]["split_opportunities"] == opportunities
    assert payload["controller_facts"] == {"focused_execution_blockers": []}


def test_ancestor_acceptance_does_not_become_extra_leaf_work():
    root, leaf = spec("whole-store"), spec("compress-only", child=True)
    root.constraints = {"preserve": "must inherit"}
    root.scope = ["US-only observations"]
    packet = required_context(
        "SHOULD_SPLIT",
        {
            "root": root.model_dump(mode="json"),
            "task": {"id": "leaf", "spec": leaf.model_dump(mode="json")},
        },
    )
    assert "acceptance" not in packet["root"]
    assert packet["root"]["constraints"] == root.constraints
    assert packet["root"]["scope"] == root.scope
    assert packet["task"]["spec"]["acceptance"][0]["description"] == leaf.acceptance[0].description


def test_unproductive_split_returns_to_fresh_routing_without_inventing_children(setup):
    from codex_task_tree.models import DecompositionReport
    from codex_task_tree.scheduler import select_next

    runtime, judge, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)

    def worker(context, *_):
        return DecompositionReport(
            task_id=context["task_id"],
            round_id=context["round_id"],
            spec_revision=context["spec_revision"],
            input_digest=context["input_digest"],
            summary="One specified compression primitive with several acceptance inputs",
            unsplittable_reason="Every proposed case tests the same output; no independent necessary deliverable remains.",
        )

    runtime.worker = worker
    runtime.step()
    assert len(runtime.state.nodes) == 1
    assert runtime.state.rounds[select_next(runtime.state).work_ref.id].kind == RoundKind.EXPLORE
    assert not any(kind == "COMPLETE" for kind, _, _ in judge.calls)


def test_necessary_partition_does_not_exhaust_stagnation_before_leaf_execution(setup):
    from codex_task_tree.models import CapabilityNeed, RunStatus

    runtime, _, _ = setup
    runtime.state.execution_profile.max_run_no_progress = 1
    runtime.save()
    authorize(runtime, RoundKind.DECOMPOSE)

    def worker(context, *_):
        report = children_report(context, ["csv", "gzip"])
        report.capability_proposals = [
            CapabilityNeed(
                id=name,
                owner_task_id=context["task_id"],
                description=name,
                root_requirement_ids=["r1"],
            )
            for name in ("csv", "gzip")
        ]
        for candidate, capability in zip(
            report.candidates, report.capability_proposals, strict=True
        ):
            candidate.capability_ids = [capability.id]
        report.plan_proposals[0].capability_ids = ["csv", "gzip"]
        return report

    runtime.worker = worker
    runtime.step()
    assert runtime.state.status == RunStatus.ACTIVE
    assert len(runtime.state.children(runtime.state.root_task_id)) == 2
    assert runtime.state.progress.rounds_without_substantive_progress == 0
    assert not runtime.state.nodes[runtime.state.root_task_id].completion_certificate_id


def test_large_catalog_can_be_selected_without_truncating_current_obligations(setup):
    runtime, judge, workspace = setup
    runtime.state.execution_profile.jev_context_target_bytes = 10000
    root_id = runtime.state.root_task_id
    for i in range(30):
        item = spec(f"peer-{i}", child=True)
        item.scope = ["s" * 1000]
        runtime.state.nodes[str(i)] = TaskNode(
            id=str(i), parent_id=root_id, depth=1, created_seq=i + 10, specs=[item]
        )
    evidence = workspace / "facts.txt"
    evidence.write_text("actual observation")
    answers = []

    def run(context, tool, started, timeout):
        started("catalog-thread", "turn")
        args = submission(str(evidence))
        answers.append(tool(args))
        args["catalog_task_ids"] = ["invented-task"]
        answers.append(tool(args))
        args["catalog_task_ids"] = [root_id]
        answers.append(tool(args))

    worker = CodexDecisionMaker(
        SimpleNamespace(run_decision=run), Transport(runtime, judge), lambda: runtime.state
    )
    payload = {"root": runtime.state.nodes[root_id].spec.model_dump(mode="json")}
    worker("SHOULD_SPLIT", root_id, payload)
    assert [a.get("code") for a in answers[:2]] == ["INPUT_TOO_LARGE", "INVALID_INPUT"]
    packet = judge.calls[0][2]
    assert len(packet["task_catalog"]) == 1
    assert packet["root"]["goal"] == payload["root"]["goal"]
    assert (
        packet["root"]["acceptance"][0]["description"]
        == payload["root"]["acceptance"][0]["description"]
    )


def test_every_decision_gets_a_new_native_thread(setup):
    runtime, judge, workspace = setup
    evidence = workspace / "evidence.txt"
    evidence.write_text("facts")
    threads = []

    def run(context, tool, started, timeout):
        threads.append(len(threads))
        started(f"thread-{len(threads)}", "turn")
        tool(submission(str(evidence)))

    worker = CodexDecisionMaker(
        SimpleNamespace(run_decision=run), Transport(runtime, judge), lambda: runtime.state
    )
    payload = {"root": runtime.state.nodes[runtime.state.root_task_id].spec.model_dump(mode="json")}
    a = worker("SHOULD_SPLIT", "a", payload)
    b = worker("SHOULD_SPLIT", "b", payload)
    assert a.worker_thread_id != b.worker_thread_id and len(threads) == 2


def test_failed_decision_is_not_an_invented_no(setup):
    runtime, judge, _ = setup
    worker = CodexDecisionMaker(
        SimpleNamespace(run_decision=lambda *_: None),
        Transport(runtime, judge),
        lambda: runtime.state,
    )
    with pytest.raises(DecisionWorkerUnavailable, match="valid verdict"):
        worker("SHOULD_SPLIT", "root", {})
    assert not judge.calls


def test_unconfirmed_native_execution_never_becomes_releasable_decision_error(setup):
    from codex_task_tree.codex_adapter import UnresolvedWorkerExecution

    runtime, judge, _ = setup

    def run(*_):
        raise UnresolvedWorkerExecution("interruption unconfirmed")

    worker = CodexDecisionMaker(
        SimpleNamespace(run_decision=run), Transport(runtime, judge), lambda: runtime.state
    )
    with pytest.raises(UnresolvedWorkerExecution):
        worker("SHOULD_SPLIT", "root", {})
    assert not judge.calls


def test_decision_failure_parks_only_its_branch_and_preserves_ready_sibling(setup):
    from codex_task_tree.models import NodeStatus, RoundPhase, RunStatus
    from codex_task_tree.scheduler import select_next

    from .conftest import exploration_report, scope_review_report

    runtime, judge, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["A", "B"])
    runtime.step()
    a, b = runtime.state.children(runtime.state.root_task_id)

    def decide(kind, target, payload):
        if kind == "SHOULD_SPLIT" and target == a:
            raise DecisionWorkerUnavailable("bounded provider failure")
        return judge(kind, target, payload)

    runtime.judge = decide
    runtime.worker = lambda context, *_: (
        scope_review_report(context)
        if context["kind"] == "SCOPE_REVIEW"
        else exploration_report(context)
    )
    runtime.step()
    runtime.step()
    runtime.step()
    assert runtime.state.status == RunStatus.ACTIVE
    assert runtime.state.nodes[a].status == NodeStatus.BLOCKED
    job = next(iter(runtime.state.decision_jobs.values()))
    assert runtime.state.rounds[job.source_ref.id].phase == RoundPhase.FAILED
    assert select_next(runtime.state).work_ref.id in {
        r.id for r in runtime.state.rounds.values() if r.task_id == b
    }


def test_reuse_requires_all_target_root_mappings_before_calling_jev():
    source = spec("same-work", child=True)
    target = source.model_copy(deep=True)
    target.root_requirement_ids.append("missing")
    assert not reuse_scope_covers(target, source)
    source.root_requirement_ids.append("missing")
    assert not reuse_scope_covers(target, source)  # declaration alone is insufficient
    source.acceptance[0].root_requirement_ids.append("missing")
    assert reuse_scope_covers(target, source)


def test_pending_duplicate_waits_for_existing_work_and_keeps_own_acceptance(setup):
    runtime, judge, _ = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    existing = TaskNode(
        id="existing",
        parent_id=root.id,
        depth=1,
        created_seq=99,
        specs=[spec("already-planned", child=True)],
    )
    runtime.state.nodes[existing.id] = existing
    runtime.save()
    authorize(runtime, RoundKind.DECOMPOSE)

    def worker(context, *_):
        report = children_report(context, ["new-proposal"])
        report.candidates[0].possible_duplicate_task_ids = [existing.id]
        return report

    runtime.worker = worker
    runtime.step()
    child = runtime.state.nodes[runtime.state.children(root.id)[0]]
    assert child.duplicate_of_task_id == existing.id
    assert existing.id in child.depends_on
    assert child.completion_certificate_id is None
    assert child.spec.acceptance
    assert any(kind == "DUPLICATE" for kind, _, _ in judge.calls)


def test_report_feedback_reuses_thread_and_preserves_bad_report():
    adapter = object.__new__(CodexAdapter)
    seen, errors = [], []
    reports = ["not json", '{"worker_model": "gpt-6-sol"}']

    def run(thread, prompt, report_type, identity, on_started, deadline):
        seen.append((thread, prompt))
        return SimpleNamespace(final_response=reports.pop(0))

    from codex_task_tree.models import ExecutionProfile

    adapter._run_turn = run
    result = adapter._collect_report(
        "same-thread",
        {},
        ExecutionProfile,
        lambda *_: None,
        999999999999,
        on_feedback=lambda error, raw: errors.append((error, raw)),
    )
    assert result.worker_model == "gpt-6-sol"  # default preserved
    assert len(seen) == 2 and seen[0][0] == seen[1][0]
    assert seen[1][1]["validation_error"]["code"] == "INVALID_REPORT"
    assert errors[0][1] == "not json"


def test_dynamic_decision_tool_rejects_other_thread_and_restores_handler(tmp_path):
    from codex_task_tree.models import ExecutionProfile

    adapter = object.__new__(CodexAdapter)
    adapter.profile, adapter.workspace = ExecutionProfile(), tmp_path
    binary = tmp_path / "compiled-worker"
    binary.write_bytes(b"jev_decide")
    adapter.profile.codex_bin = str(binary)
    calls, starts = [], []

    def original(*_):
        return {"prior": True}

    def start(params):
        starts.append(params)
        return SimpleNamespace(thread=SimpleNamespace(id="bound-thread"))

    client = SimpleNamespace(_approval_handler=original, thread_start=start)
    adapter.codex = SimpleNamespace(_client=client)

    def collect(*_):
        wrong = client._approval_handler(
            "item/tool/call", {"threadId": "other", "tool": "jev_decide", "arguments": {}}
        )
        assert not wrong["success"] and not calls
        answer = client._approval_handler(
            "item/tool/call",
            {"threadId": "bound-thread", "tool": "jev_decide", "arguments": {"brief": "submitted"}},
        )
        assert answer["success"] and len(calls) == 1
        assert client._approval_handler("item/commandExecution/requestApproval", {}) == {
            "decision": "decline"
        }

    adapter._collect_report = collect
    adapter.run_decision({}, lambda args: calls.append(args) or {"ok": True}, lambda *_: None, 10)
    assert client._approval_handler is original
    assert starts[0]["sandbox"] == "read-only"
    assert starts[0]["dynamicTools"][0]["name"] == "jev_decide"


def test_decision_tool_fails_before_launching_a_thread_on_old_binary(tmp_path):
    from codex_task_tree.models import ExecutionProfile

    binary = tmp_path / "old-worker"
    binary.write_bytes(b"old fork without tool capability")
    adapter = object.__new__(CodexAdapter)
    adapter.profile = ExecutionProfile(codex_bin=str(binary))
    with pytest.raises(ValueError, match="rebuild codex-cli"):
        adapter.run_decision({}, lambda *_: None, lambda *_: None, 10)
