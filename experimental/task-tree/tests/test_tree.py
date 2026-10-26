from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from codex_task_tree.checks import propose_check
from codex_task_tree.context import build_context, decomposition_view
from codex_task_tree.models import (
    CandidateProposal,
    CapabilityNeed,
    CheckChangeProposal,
    CheckSpec,
    CoverageItem,
    DecompositionReport,
    ExecutionProfile,
    Gap,
    NodeStatus,
    PlanProposal,
    RoundKind,
    RoundPhase,
    RunStatus,
    SpecProposal,
    TaskNode,
    WorkerReport,
    digest,
)
from codex_task_tree.runtime import (
    TaskRuntime,
    certificate_valid,
    hard_complete,
    validate_wait_graph,
)
from codex_task_tree.scheduler import select_next
from codex_task_tree.store import RunStore

from .conftest import (
    FakeJudge,
    adopt,
    authorize,
    exploration_report,
    local_command,
    preparation_report,
    scope_review_report,
    spec,
)


def report_for(context, **kwargs):
    if not context["task"]["parent_id"] and "coverage_updates" not in kwargs:
        kwargs["coverage_updates"] = [
            CoverageItem(**{**item, "status": "SUPPORTED", "evidence_refs": ["ok"]})
            for item in context.get("coverage", [])
            if item["kind"] == "REQUIREMENT"
        ]
    return WorkerReport(
        task_id=context["task_id"],
        round_id=context["round_id"],
        spec_revision=context["task"]["specs"][-1]["revision"],
        input_digest=context["input_digest"],
        summary="round delivered",
        **kwargs,
    )


def children_report(context, names, retained=(), gap=None):
    owner = context["task_id"]
    capability = CapabilityNeed(
        id=f"cap-{context['round_id']}",
        owner_task_id=owner,
        description="required storage behavior",
        root_requirement_ids=["r1"],
    )
    candidates = [
        CandidateProposal(
            id=f"candidate-{name}",
            spec=spec(name, child=True),
            capability_ids=[capability.id],
            reason=f"implement independent {name} behavior",
        )
        for name in names
    ]
    plan = PlanProposal(
        id=f"proposal-{context['round_id']}",
        capability_ids=[capability.id],
        candidate_ids=[c.id for c in candidates],
        retained_child_ids=list(retained),
        rationale="complete required plan",
    )
    return DecompositionReport(
        task_id=context["task_id"],
        round_id=context["round_id"],
        spec_revision=context["task"]["specs"][-1]["revision"],
        input_digest=context["input_digest"],
        summary="allocate required responsibilities",
        capability_proposals=[capability],
        candidates=candidates,
        plan_proposals=[plan],
        gaps=[gap] if gap else [],
    )


def test_root_requires_real_confirmation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RunStore(tmp_path / "control", "unapproved")
    TaskRuntime.create(store, "task", workspace, ExecutionProfile())
    calls = []

    def worker(context, model, started, timeout):
        calls.append(context["kind"])
        if context["kind"] == "PREPARE":
            return preparation_report(context, workspace)
        return SpecProposal(
            task_id=context["task_id"], round_id=context["round_id"], proposed_spec=spec()
        )

    runtime = TaskRuntime(store, worker, FakeJudge(), local_command)
    runtime.step()
    assert runtime.state.status == RunStatus.ACTIVE and not runtime.state.root_approval
    runtime.step()
    assert runtime.state.status == RunStatus.AWAITING_APPROVAL
    runtime.step()
    assert calls == ["PREPARE", "SPECIFY"]
    assert len(runtime.state.nodes) == 1
    with pytest.raises(ValueError):
        runtime.approve_root("old-contract", "user")
    runtime.approve_root(
        runtime.state.nodes[runtime.state.root_task_id].spec.contract_hash(), "user typed approve"
    )
    assert select_next(runtime.state).work_ref


@pytest.mark.parametrize("invalid", ["optional-root", "optional-check"])
def test_invalid_specification_uses_fresh_worker_without_approving_or_resetting_budget(
    tmp_path, invalid
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RunStore(tmp_path / "control", "spec-retry")
    TaskRuntime.create(store, "task", workspace, ExecutionProfile())
    calls = []

    def worker(context, *_):
        if context["kind"] == "PREPARE":
            return preparation_report(context, workspace)
        calls.append(context)
        proposal = spec()
        if len(calls) == 1:
            if invalid == "optional-root":
                proposal.requirements[0].required = False
            else:
                proposal.acceptance[0].required = False
        return SpecProposal(
            task_id=context["task_id"], round_id=context["round_id"], proposed_spec=proposal
        )

    runtime = TaskRuntime(store, worker, FakeJudge(), local_command)
    runtime.step()
    runtime.step()
    assert runtime.state.status == RunStatus.ACTIVE
    assert runtime.state.root_approval is None
    failed = next(r for r in runtime.state.rounds.values() if r.phase == RoundPhase.FAILED)
    assert failed.worker_report_ref and failed.error
    elapsed = runtime.state.progress.active_elapsed_seconds
    runtime.step()
    assert runtime.state.status == RunStatus.AWAITING_APPROVAL
    assert len(calls) == 2 and runtime.state.progress.worker_attempts_started == 3
    assert calls[0]["round_id"] != calls[1]["round_id"]
    assert failed.error in calls[1]["purpose"]
    assert runtime.state.progress.active_elapsed_seconds >= elapsed
    assert runtime.state.root_approval is None and len(runtime.state.nodes) == 1
    assert [p for p in workspace.iterdir() if p.name != "background"] == []


@pytest.mark.parametrize("failure", ["actual-check", "completion-denied"])
def test_bfs_parent_failure_new_children_second_integration(setup, failure):
    runtime, judge, workspace = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    root.spec.acceptance[0].check_spec.command = [
        sys.executable,
        "-B",
        "-c",
        "from pathlib import Path; assert Path('Root.txt').read_text() == 'delivered'",
    ]
    root.spec.acceptance[0].adopted_check_revision_id = None
    adopt(runtime.state, runtime.store, root.id)
    sequence = []

    def routing_judge(kind, target, payload):
        decision = judge(kind, target, payload)
        if kind == "COMPLETE" and target == root.id:
            actual = {r["task_id"]: r for r in payload["worker_results"]}
            assert root.id in actual
            for child in runtime.state.children(root.id):
                assert actual[child]["delivery_report"]["task_id"] == child
                assert actual[child]["delivery_kind"] in {"EXECUTE", "RECONCILE"}
            assert any(
                a["current_content"].get("text") == "delivered"
                for a in payload["artifact_contents"]
            )
        if kind == "SHOULD_SPLIT":
            title = runtime.state.nodes[target].spec.title
            decision.score = int(title in {"Root", "A"})
        if (
            failure == "completion-denied"
            and kind == "COMPLETE"
            and target == root.id
            and not (workspace / "C.txt").exists()
        ):
            decision.score = 0
        return decision

    def worker(context, model, started, timeout):
        title, kind = context["task"]["specs"][-1]["title"], context["kind"]
        sequence.append((title, kind))
        if kind == "EXPLORE":
            return exploration_report(context)
        if kind == "SCOPE_REVIEW":
            return scope_review_report(context)
        if kind == "DECOMPOSE":
            if title == "A":
                return children_report(context, ["A1", "A2"])
            if not context["children"]:
                return children_report(context, ["A", "B"])
            return children_report(context, ["C"], [n["id"] for n in context["children"]])
        if title == "Root" and not (workspace / "C.txt").exists():
            if failure == "completion-denied":
                (workspace / "Root.txt").write_text("delivered")
                return report_for(context, deliverable_paths=["Root.txt"], completion_claim=True)
            (workspace / "Root.txt").write_text("restart behavior missing")
            return report_for(
                context,
                deliverable_paths=["Root.txt"],
                gaps=[
                    Gap(
                        id="restart-gap",
                        description="restart behavior missing",
                        root_requirement_ids=["r1"],
                    )
                ],
            )
        (workspace / (title + ".txt")).write_text("delivered")
        return report_for(context, deliverable_paths=[title + ".txt"], completion_claim=True)

    runtime.judge = routing_judge
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = worker
    for _ in range(60):
        for task in runtime.state.nodes.values():
            if not task.spec.acceptance[0].adopted_check_revision_id:
                adopt(runtime.state, runtime.store, task.id)
        runtime.save()
        if runtime.state.status != RunStatus.ACTIVE:
            break
        runtime.step()
    assert runtime.state.status == RunStatus.SUCCEEDED
    assert runtime.state.scheduler.wave is None
    assert [item for item in sequence if item[1] not in {"EXPLORE", "SCOPE_REVIEW"}] == [
        ("Root", "DECOMPOSE"),
        ("A", "DECOMPOSE"),
        ("B", "EXECUTE"),
        ("A1", "EXECUTE"),
        ("A2", "EXECUTE"),
        ("A", "RECONCILE"),
        ("Root", "RECONCILE"),
        ("Root", "DECOMPOSE"),
        ("C", "EXECUTE"),
        ("Root", "RECONCILE"),
    ]
    assert sequence.index(("B", "EXPLORE")) < sequence.index(("A1", "EXPLORE"))
    root = runtime.state.nodes[runtime.state.root_task_id]
    assert len(root.result_history) == 2 and root.depth == 0
    first_checks = [
        runtime.state.verifications[key] for key in root.result_history[0].verification_ids
    ]
    if failure == "actual-check":
        assert any(check.status == "FAIL" for check in first_checks)
    else:
        assert all(check.status == "PASS" for check in first_checks)
        assert any(
            d.kind == "COMPLETE" and d.target_id == root.id and not d.accepted
            for d in runtime.state.decisions.values()
        )
    assert len(runtime.state.children(root.id)) == 3
    assert all(n.depth <= 2 for n in runtime.state.nodes.values())


def test_actual_failure_overrides_jev_and_round_commit_is_not_success(setup):
    runtime, judge, _ = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    # Replacing the fixture check with a reviewed failing check models actual failure.
    root.spec.acceptance[0].check_spec.command = [sys.executable, "-c", "raise SystemExit(1)"]
    root.spec.acceptance[0].adopted_check_revision_id = None
    adopt(runtime.state, runtime.store, root.id)
    runtime.worker = lambda context, *_: report_for(context, completion_claim=True)
    runtime.save()
    runtime.step()
    assert root.id in runtime.state.nodes
    assert runtime.state.nodes[root.id].status != NodeStatus.SUCCEEDED
    assert any(
        r.phase == RoundPhase.COMMITTED and r.kind == RoundKind.EXECUTE
        for r in runtime.state.rounds.values()
    )
    assert not any(kind == "COMPLETE" for kind, _, _ in judge.calls)


def test_rejected_candidate_does_not_block_an_independent_admitted_child(setup):
    runtime, judge, _ = setup
    judge.denied.add("candidate-B")
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["A", "B"])
    runtime.step()
    assert [n.spec.title for n in runtime.state.nodes.values() if n.parent_id] == ["A"]
    assert any(r.phase == RoundPhase.QUEUED for r in runtime.state.rounds.values())
    previous = decomposition_view(runtime.state, runtime.state.root_task_id)
    assert previous["report"]["candidates"]
    assert any(
        d["kind"] == "NECESSARY" and d["target_id"] == "candidate-B" and d["score"] == 0
        for d in previous["decisions"]
    )
    assert not runtime.state.nodes[runtime.state.root_task_id].result_history
    assert not any(kind in {"PLAN_ACCEPTABLE", "COMPLETE"} for kind, _, _ in judge.calls)


def test_only_one_alternative_plan_is_selected(setup):
    runtime, judge, _ = setup

    def worker(context, *_):
        report = children_report(context, ["gzip"])
        alternative = report.candidates[0].model_copy(deep=True)
        alternative.id, alternative.spec.title = "zstd-option", "zstd"
        report.candidates.append(alternative)
        report.plan_proposals.append(
            PlanProposal(
                id="zstd-plan",
                recommended=False,
                capability_ids=report.plan_proposals[0].capability_ids,
                candidate_ids=[alternative.id],
                rationale="alternative",
            )
        )
        return report

    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = worker
    runtime.step()
    assert [n.spec.title for n in runtime.state.nodes.values() if n.parent_id] == ["gzip"]


def test_wait_graph_includes_parent_child_edges(setup):
    runtime, _, _ = setup
    state = runtime.state
    parent = state.root_task_id
    child = TaskNode(
        id="child",
        parent_id=parent,
        depth=1,
        created_seq=100,
        specs=[spec("child", child=True)],
        depends_on=[parent],
    )
    state.nodes[child.id] = child
    state.nodes[parent].depends_on = [child.id]
    with pytest.raises(ValueError, match="cycle"):
        validate_wait_graph(state)


def test_store_revision_and_artifact_atomicity(setup):
    runtime, _, _ = setup
    old = runtime.store.load()
    runtime.save()
    with pytest.raises(RuntimeError, match="stale"):
        runtime.store.save(old, old.revision)
    path = runtime.store.write_json("reports/x.json", {"value": 1})
    with pytest.raises(ValueError, match="immutable"):
        runtime.store.write_json("reports/x.json", {"value": 2})
    assert json.loads(open(path).read()) == {"value": 1}


def test_workspace_alias_overlap_and_unresolved_reservations(setup, tmp_path):
    runtime, _, workspace = setup
    other = RunStore(runtime.store.root, "other")
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    for path in (workspace, alias, workspace.parent):
        with pytest.raises((RuntimeError, ValueError)):
            other.acquire_workspace(path)
    runtime.store.execution_reservation(["unknown-worker"])
    with pytest.raises(RuntimeError, match="unresolved"):
        runtime.store.release_workspace()


def test_metadata_round_retains_products_and_historical_bytes(setup):
    runtime, judge, workspace = setup
    judge.denied.add("SHOULD_SPLIT")
    calls = []

    def worker(context, *_):
        if context["kind"] == "EXPLORE":
            return exploration_report(context)
        if context["kind"] == "SCOPE_REVIEW":
            return scope_review_report(context)
        calls.append(context)
        if len(calls) <= 2:
            (workspace / "product.txt").write_text(f"version {len(calls)}")
            return report_for(
                context,
                deliverable_paths=["product.txt"],
                gaps=[
                    Gap(id="unfinished", description="final review", root_requirement_ids=["r1"])
                ],
            )
        return report_for(context, completion_claim=True)

    runtime.worker = worker
    for _ in range(10):
        if runtime.state.status != RunStatus.ACTIVE:
            break
        runtime.step()
    node = runtime.state.nodes[runtime.state.root_task_id]
    assert runtime.state.status == RunStatus.SUCCEEDED
    assert len(node.result_history) == 3
    assert len(calls) == 3
    first, second, final = [r.artifact_ids for r in node.result_history]
    assert first != second and second == final
    artifacts = runtime.state.artifacts
    assert Path(artifacts[first[0]].snapshot_ref).read_text() == "version 1"
    assert Path(artifacts[second[0]].snapshot_ref).read_text() == "version 2"
    assert calls[-1]["artifacts"][0]["path"] == str(workspace / "product.txt")
    (workspace / "product.txt").write_text("external change")
    assert not certificate_valid(runtime.state, node.id)
    assert Path(artifacts[second[0]].snapshot_ref).read_text() == "version 2"


def test_new_product_expands_verification_inputs_before_reusing_pass(setup):
    runtime, _, workspace = setup
    root = runtime.state.root_task_id
    first = runtime.verify(root, "initial")
    (workspace / "extra.txt").write_text("delivered")
    second = runtime.verify(root, "new-product", ["extra.txt"])
    assert first != second
    record = runtime.state.verifications[second[0]]
    assert "extra.txt" in record.input_paths
    assert runtime.verify(root, "unchanged", ["extra.txt"]) == second
    (workspace / "extra.txt").write_text("changed")
    assert runtime.verify(root, "changed-product", ["extra.txt"]) != second


def test_claim_is_persistent_and_selection_cannot_skip_active(setup):
    runtime, _, _ = setup
    ref = runtime.claim_next_work()
    restored = runtime.store.load()
    assert restored.scheduler.active_work == ref
    assert restored.rounds[ref.id].phase == RoundPhase.PREPARING
    assert restored.scheduler.wave.next_index == 1
    with pytest.raises(ValueError, match="recover"):
        select_next(restored)


def test_report_recovery_does_not_repeat_worker_or_children(setup):
    runtime, _, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.claim_next_work()
    ref = runtime.state.scheduler.active_work
    work = runtime.state.rounds[ref.id]
    context = build_context(runtime.state, work.id)
    work.input_digest = digest(context)
    context["input_digest"] = work.input_digest
    report = children_report(context, ["A", "B"])
    work.context_manifest_ref = runtime.store.write_json(
        f"contexts/{work.id}.json", {"context": context}
    )
    work.worker_report_ref = runtime.store.write_json(
        f"reports/{work.id}/worker.json", report.model_dump(mode="json")
    )
    work.phase = RoundPhase.VERIFYING
    runtime.save()
    restored = TaskRuntime(
        runtime.store,
        lambda *_: pytest.fail("worker must not replay"),
        runtime.judge,
        local_command,
    )
    restored.step()
    assert len(restored.state.nodes) == 3
    restored._commit_decomposition(restored.state.rounds[work.id], report)
    assert len(restored.state.nodes) == 3


def test_unknown_old_worker_is_not_replayed(setup):
    runtime, _, _ = setup
    ref = runtime.claim_next_work()
    runtime.state.rounds[ref.id].phase = RoundPhase.WORKER_RUNNING
    runtime.save()
    with pytest.raises(RuntimeError, match="old worker"):
        runtime.step()
    assert runtime.state.status == RunStatus.BLOCKED


def test_unassessed_coverage_blocks_all_yes_decisions(setup):
    runtime, _, _ = setup
    root = runtime.state.root_task_id
    runtime.state.coverage["missing-aspect"] = CoverageItem(
        id="missing-aspect",
        kind="ASPECT",
        owner_task_id=root,
        root_requirement_ids=["r1"],
        subject="restart discovery",
        discovery_source_refs=["module inventory"],
    )
    runtime.worker = lambda context, *_: report_for(context, completion_claim=True)
    runtime.save()
    runtime.step()
    assert not hard_complete(runtime.state, root)


def test_required_root_coverage_cannot_be_excluded(setup):
    runtime, _, _ = setup
    root = runtime.state.root_task_id
    runtime.state.coverage["req"] = CoverageItem(
        id="req",
        kind="REQUIREMENT",
        owner_task_id=root,
        root_requirement_ids=["r1"],
        subject="usable storage",
    )
    item = runtime.state.coverage["req"].model_copy(deep=True)
    item.status, item.rationale = "NOT_REQUIRED", "skip"
    runtime.worker = lambda context, *_: report_for(context, coverage_updates=[item])
    runtime.save()
    runtime.step()
    assert runtime.state.coverage["req"].status == "UNASSESSED"
    assert any(
        r.phase == RoundPhase.FAILED and "cannot be invented or excluded" in (r.error or "")
        for r in runtime.state.rounds.values()
    )


def test_pure_reverification_does_not_invent_result_or_worker(setup):
    runtime, judge, workspace = setup
    runtime.worker = lambda context, *_: report_for(context, completion_claim=True)
    runtime.step()
    root = runtime.state.root_task_id
    assert certificate_valid(runtime.state, root)
    version = runtime.state.nodes[root].result_revision
    (workspace / "Root.txt").write_text("external change, behavior unchanged")
    assert not certificate_valid(runtime.state, root)
    runtime.worker = lambda *_: pytest.fail("freshness only must not start a worker")
    runtime.step()
    assert runtime.state.nodes[root].result_revision == version
    assert certificate_valid(runtime.state, root)
    assert len([kind for kind, _, _ in judge.calls if kind == "COMPLETE"]) == 2


def test_formal_bundle_cannot_be_replaced_by_workspace_test(setup):
    runtime, _, workspace = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    (workspace / "check.py").write_text("raise SystemExit(1)")
    proposal = CheckChangeProposal(
        id="new",
        criterion_id="ok",
        old_check_revision_id=root.spec.acceptance[0].adopted_check_revision_id,
        check_spec=CheckSpec(command=[sys.executable, "check.py"], artifact_paths=["Root.txt"]),
        file_paths=["check.py"],
        reason="independent test fixture",
    )
    check = propose_check(runtime.state, runtime.store, root.id, "proposer", proposal)
    check.status = "ADOPTED"
    root.spec.acceptance[0].adopted_check_revision_id = check.id
    (workspace / "check.py").write_text("pass")
    records = runtime.verify(root.id, "test")
    assert runtime.state.verifications[records[0]].status == "FAIL"


def test_check_reviewer_cannot_approve_own_proposal(setup):
    runtime, _, _ = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    ref = runtime.claim_next_work()
    work = runtime.state.rounds[ref.id]
    check = propose_check(
        runtime.state,
        runtime.store,
        root.id,
        work.id,
        CheckChangeProposal(
            id="new",
            criterion_id="ok",
            old_check_revision_id=root.spec.acceptance[0].adopted_check_revision_id,
            check_spec=CheckSpec(),
            reason="new",
        ),
    )
    report = WorkerReport(
        task_id=root.id,
        round_id=work.id,
        spec_revision=1,
        input_digest="",
        summary="review",
        check_review={check.id: "accept: mine"},
    )
    with pytest.raises(ValueError, match="independent"):
        runtime._review_checks(report)


def test_global_budget_survives_restart(setup):
    runtime, _, _ = setup
    runtime.state.execution_profile.max_worker_rounds = 1
    runtime.state.progress.worker_attempts_started = 1
    runtime.save()
    restored = TaskRuntime(
        runtime.store, lambda *_: pytest.fail("budget"), runtime.judge, local_command
    )
    restored.step()
    assert restored.state.status == RunStatus.BLOCKED
    assert restored.state.progress.worker_attempts_started == 1


def test_no_runnable_work_is_blocked_not_success(setup):
    runtime, _, _ = setup
    for work in runtime.state.rounds.values():
        work.phase = RoundPhase.COMMITTED
    runtime.step()
    assert runtime.state.status == RunStatus.BLOCKED


def test_external_edit_during_complete_decision_prevents_certificate(setup):
    runtime, judge, workspace = setup

    def editing_judge(kind, target, payload):
        if kind == "COMPLETE":
            (workspace / "Root.txt").write_text("changed after verification")
        return judge(kind, target, payload)

    runtime.judge = editing_judge
    runtime.worker = lambda context, *_: report_for(context, completion_claim=True)
    runtime.step()
    assert not certificate_valid(runtime.state, runtime.state.root_task_id)


def test_repeated_parent_notification_is_deduplicated(setup):
    runtime, _, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["A"])
    runtime.step()
    root = runtime.state.root_task_id
    child = runtime.state.children(root)[0]
    for work in runtime.state.rounds.values():
        if work.task_id == child:
            work.phase = RoundPhase.COMMITTED
    runtime.state.nodes[child].status = NodeStatus.SUCCEEDED
    runtime.refresh_parents()
    count = len(runtime.state.rounds)
    runtime.refresh_parents()
    assert len(runtime.state.rounds) == count
