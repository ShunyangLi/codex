from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from codex_task_tree.cli import main
from codex_task_tree.codex_adapter import CodexAdapter, instructions_for_kind
from codex_task_tree.models import (
    DecompositionReport,
    ExecutionProfile,
    ExplorationReport,
    Gap,
    NodeStatus,
    RoundKind,
    RoundPhase,
    RunStatus,
)
from codex_task_tree.runtime import TaskRuntime, certificate_valid, queue_round
from codex_task_tree.scheduler import select_next

from .conftest import authorize, exploration_report, exploration_worker
from .test_tree import children_report, report_for


def begin_exploration(runtime):
    for work in runtime.state.rounds.values():
        if work.phase == RoundPhase.QUEUED:
            work.phase = RoundPhase.CANCELLED
    node = runtime.state.nodes[runtime.state.root_task_id]
    node.status = NodeStatus.READY
    work = queue_round(runtime.state, node.id, RoundKind.EXPLORE, "actual-routing-test")
    runtime.save()
    return work


def next_kind(runtime):
    return runtime.state.rounds[select_next(runtime.state).work_ref.id].kind


def step_route(runtime):
    runtime.step()


def test_jev_decides_before_plan_generation_and_children_reenter_exploration(setup):
    runtime, judge, workspace = setup
    begin_exploration(runtime)
    calls = []

    def worker(context, *_):
        calls.append(context["kind"])
        if context["kind"] == "EXPLORE":
            return exploration_report(context)
        assert any(kind == "SHOULD_SPLIT" for kind, _, _ in judge.calls)
        assert context["exploration"][0]["report"]["findings"]
        return children_report(context, ["codec", "compression"])

    runtime.worker = worker
    step_route(runtime)
    assert calls == ["EXPLORE"] and list(workspace.iterdir()) == []
    assert next_kind(runtime) == RoundKind.DECOMPOSE
    assert not runtime.state.nodes[runtime.state.root_task_id].result_history
    assert [kind for kind, _, _ in judge.calls] == ["SHOULD_SPLIT"]
    assert judge.calls[0][2]["exploration_draft"]["split_opportunities"] == []
    step_route(runtime)
    assert calls == ["EXPLORE", "DECOMPOSE"]
    assert len(runtime.state.nodes) == 3 and list(workspace.iterdir()) == []
    assert all(
        work.kind == RoundKind.EXPLORE
        for work in runtime.state.rounds.values()
        if work.phase == RoundPhase.QUEUED
    )
    assert not any(
        node.result_history or node.completion_certificate_id
        for node in runtime.state.nodes.values()
    )


@pytest.mark.parametrize("blocker", [None, "declared", "missing-input", "undefined-deliverable"])
def test_no_split_uses_input_contract_checks_without_extra_model_gates(setup, blocker):
    runtime, judge, workspace = setup
    begin_exploration(runtime)
    judge.denied.add("SHOULD_SPLIT")

    def worker(context, *_):
        report = exploration_report(context)
        if blocker == "declared":
            report.execution_blockers = ["Essential source credential is unavailable"]
        elif blocker == "missing-input":
            report.input_paths = ["absent-source.csv"]
        return report

    if blocker == "undefined-deliverable":
        runtime.state.nodes[runtime.state.root_task_id].spec.deliverables = []
        runtime.state.root_approval.contract_hash = runtime.state.nodes[
            runtime.state.root_task_id
        ].spec.contract_hash()
    runtime.worker = worker
    step_route(runtime)
    assert [kind for kind, _, _ in judge.calls] == ["SHOULD_SPLIT"]
    assert next_kind(runtime) == (RoundKind.EXPLORE if blocker else RoundKind.EXECUTE)
    work = runtime.state.rounds[select_next(runtime.state).work_ref.id]
    if blocker:
        assert "execution blockers" in work.purpose
    else:
        assert runtime.authorization_current(work)
    assert not list(workspace.iterdir())
    assert not certificate_valid(runtime.state, runtime.state.root_task_id)


def test_missing_split_decision_cannot_authorize_leaf_execution(setup):
    runtime, _, _ = setup
    work = authorize(runtime)
    runtime.state.rounds[work.authorization_round_id].decision_ids = []
    runtime.worker = lambda *_: pytest.fail("incomplete routing must not execute")
    step_route(runtime)
    assert work.phase == RoundPhase.CANCELLED
    assert next_kind(runtime) == RoundKind.EXPLORE


@pytest.mark.parametrize("essential", [False, True, None])
def test_observed_missing_output_is_distinct_from_required_execution_input(setup, essential):
    runtime, judge, workspace = setup
    begin_exploration(runtime)
    judge.denied.add("SHOULD_SPLIT")

    def worker(context, *_):
        report = exploration_report(context)
        report.input_paths = ["future-output"]
        report.required_input_paths = (
            None if essential is None else (["future-output"] if essential else [])
        )
        return report

    runtime.worker = worker
    step_route(runtime)
    assert not (workspace / "future-output").exists()
    assert next_kind(runtime) == (RoundKind.EXECUTE if essential is False else RoundKind.EXPLORE)
    if essential is False:
        work = runtime.state.rounds[select_next(runtime.state).work_ref.id]
        assert runtime.authorization_current(work)
        (workspace / "future-output").mkdir()
        assert not runtime.authorization_current(
            work
        )  # An actual observed-input change still expires routing.


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("changed", ["sibling-output", "source"])
def test_precise_explorer_read_set_guards_only_real_dependencies(
    setup, split, changed
):
    runtime, judge, workspace = setup
    (workspace / "source").write_text("source v1")
    begin_exploration(runtime)
    if not split:
        judge.denied.add("SHOULD_SPLIT")

    def worker(context, *_):
        report = exploration_report(context)
        report.input_paths = ["source"]
        report.required_input_paths = ["source"]
        return report

    runtime.worker = worker
    step_route(runtime)
    work = runtime.state.rounds[select_next(runtime.state).work_ref.id]
    assert work.kind == (RoundKind.DECOMPOSE if split else RoundKind.EXECUTE)
    assert runtime.authorization_current(work)
    (workspace / changed).write_text("changed")
    assert runtime.authorization_current(work) == (changed == "sibling-output")


@pytest.mark.parametrize("original_input", [False, True])
def test_required_inputs_cannot_be_omitted_from_routing_read_set(setup, original_input):
    runtime, judge, workspace = setup
    (workspace / "observed").write_text("observed")
    if original_input:
        runtime.state.input_refs = [str(workspace / "essential")]
    begin_exploration(runtime)
    judge.denied.add("SHOULD_SPLIT")

    def worker(context, *_):
        report = exploration_report(context)
        report.input_paths = ["observed"]
        report.required_input_paths = [] if original_input else ["essential"]
        return report

    runtime.worker = worker
    step_route(runtime)
    node = runtime.state.nodes[runtime.state.root_task_id]
    source = runtime.state.rounds[node.exploration_round_ids[-1]]
    assert "MISSING" in source.routing_fingerprints.values()
    assert next_kind(runtime) == RoundKind.EXPLORE
    work = runtime.state.rounds[select_next(runtime.state).work_ref.id]
    assert "declared execution input is missing" in work.purpose


def test_atomic_execution_follows_jev_and_can_complete_without_artificial_children(setup):
    runtime, judge, workspace = setup
    begin_exploration(runtime)
    judge.denied.add("SHOULD_SPLIT")
    calls = []

    def worker(context, *_):
        calls.append(context["kind"])
        if context["kind"] == "EXPLORE":
            return exploration_report(context)
        assert [kind for kind, _, _ in judge.calls] == ["SHOULD_SPLIT"]
        (workspace / "Root.txt").write_text("actual delivery")
        return report_for(context, deliverable_paths=["Root.txt"], completion_claim=True)

    runtime.worker = worker
    step_route(runtime)
    step_route(runtime)
    step_route(runtime)
    assert calls == ["EXPLORE", "EXECUTE"]
    assert runtime.state.status == RunStatus.SUCCEEDED
    assert len(runtime.state.nodes) == 1 and certificate_valid(
        runtime.state, runtime.state.root_task_id
    )


@pytest.mark.parametrize("kind", [RoundKind.DECOMPOSE, RoundKind.EXECUTE])
def test_missing_authorization_returns_to_exploration_without_running_worker(setup, kind):
    runtime, _, _ = setup
    work = begin_exploration(runtime)
    work.kind = kind
    runtime.worker = lambda *_: pytest.fail("unapproved route must not execute")
    step_route(runtime)
    assert work.phase == RoundPhase.CANCELLED
    assert next_kind(runtime) == RoundKind.EXPLORE
    assert runtime.state.progress.worker_attempts_started == 0


@pytest.mark.parametrize("changed", ["files", "gap", "policy", "common-policy", "model", "effort"])
def test_changed_routing_inputs_require_new_exploration(setup, changed, monkeypatch):
    runtime, judge, workspace = setup
    begin_exploration(runtime)
    judge.denied.add("SHOULD_SPLIT")
    runtime.worker = exploration_worker
    step_route(runtime)
    if changed == "files":
        (workspace / "new-input.txt").write_text("changed after route approval")
    elif changed == "gap":
        runtime.state.nodes[runtime.state.root_task_id].open_gaps.append(
            Gap(id="new", description="new evidence", root_requirement_ids=["r1"])
        )
    elif changed == "policy":
        from codex_task_tree.jev import PREDICATES

        monkeypatch.setitem(PREDICATES, "SHOULD_SPLIT", "Updated split policy")
    elif changed == "common-policy":
        from codex_task_tree import jev

        monkeypatch.setattr(jev, "COMMON_INSTRUCTIONS", "Updated common decision policy")
    elif changed == "model":
        runtime.state.execution_profile.worker_model = "different-worker"
    else:
        runtime.state.execution_profile.worker_effort = "high"
    runtime.worker = lambda *_: pytest.fail("stale route must not execute")
    step_route(runtime)
    assert next_kind(runtime) == RoundKind.EXPLORE
    assert runtime.state.progress.worker_attempts_started == 1


def test_missing_focus_assessment_cannot_authorize_a_leaf(setup):
    runtime, judge, workspace = setup
    begin_exploration(runtime)
    judge.denied.add("SHOULD_SPLIT")

    def worker(context, *_):
        report = exploration_report(context)
        # Historical reports remain readable, but a packaging-only assessment cannot route anew.
        values = report.model_dump(exclude={"split_opportunities", "leaf_rationale"})
        values["execution_assessment"] = "One script shares one dataset and runs in 120 seconds."
        return ExplorationReport.model_validate(values)

    runtime.worker = worker
    step_route(runtime)
    failed = [r for r in runtime.state.rounds.values() if r.phase == RoundPhase.FAILED]
    assert len(failed) == 1 and "leaf_rationale" in failed[0].error
    assert not judge.calls and not list(workspace.iterdir())
    assert not any(
        r.kind == RoundKind.EXECUTE and r.phase == RoundPhase.QUEUED
        for r in runtime.state.rounds.values()
    )


def test_child_blueprints_are_rejected_before_jev_authorizes_decomposition(setup):
    from codex_task_tree.models import SplitOpportunity

    runtime, judge, workspace = setup
    begin_exploration(runtime)

    def worker(context, *_):
        report = exploration_report(context)
        report.split_opportunities = [
            SplitOpportunity(
                question_or_outcome="Does the association differ by recorded seniority code?",
                deliverable="Aggregate code-level comparison with denominators and limitations",
                verification="Recompute the group totals and compare with the parent sample",
                focus_benefit="Separate one subgroup question from occupation composition",
                expected_information_gain="Check whether a pooled association masks heterogeneity",
                effect_on_parent="A difference qualifies the pooled result; no difference validates its scope",
                evidence_refs=["schema.json:seniority", "current aggregate support probe"],
                feasibility="Supported codes; preserve undocumented labels",
                cost_and_overlap="Small aggregate query; no equivalent current tree result",
            )
        ]
        report.leaf_rationale = "Occupation and seniority questions have independent answers; this is not a focused leaf."
        return report

    runtime.worker = worker
    step_route(runtime)
    assert not judge.calls
    assert any(
        r.phase == RoundPhase.FAILED and "prior SHOULD_SPLIT authorization" in r.error
        for r in runtime.state.rounds.values()
    )
    assert not runtime.state.children(runtime.state.root_task_id)
    assert not list(workspace.iterdir())


def test_read_only_exploration_cannot_publish_product_even_if_worker_tries(setup):
    runtime, judge, workspace = setup
    begin_exploration(runtime)

    def worker(context, *_):
        (workspace / "premature-report.md").write_text("attempted parent delivery")
        return exploration_report(context)

    runtime.worker = worker
    step_route(runtime)
    failed = [work for work in runtime.state.rounds.values() if work.phase == RoundPhase.FAILED]
    assert len(failed) == 1 and "read-only" in failed[0].error
    assert not judge.calls
    assert not runtime.state.nodes[runtime.state.root_task_id].result_history


def test_decision_transport_failure_blocks_instead_of_becoming_direct_execution(setup):
    runtime, _, _ = setup
    begin_exploration(runtime)
    runtime.worker = exploration_worker

    def unavailable(*_):
        raise RuntimeError("decision transport unavailable")

    runtime.judge = unavailable
    with pytest.raises(RuntimeError, match="transport unavailable"):
        step_route(runtime)
    assert runtime.state.status == RunStatus.BLOCKED
    assert not any(
        work.kind == RoundKind.EXECUTE and work.phase == RoundPhase.QUEUED
        for work in runtime.state.rounds.values()
    )


def test_input_change_during_routing_decision_cannot_authorize_stale_exploration(setup):
    runtime, judge, workspace = setup
    begin_exploration(runtime)
    (workspace / "source.txt").write_text("original evidence")

    def worker(context, *_):
        report = exploration_report(context)
        report.input_paths = ["source.txt"]
        return report

    runtime.worker = worker

    def concurrent_edit(kind, target, payload):
        (workspace / "source.txt").write_text("changed during decision")
        return judge(kind, target, payload)

    runtime.judge = concurrent_edit
    step_route(runtime)
    assert next_kind(runtime) == RoundKind.EXPLORE
    assert not runtime.state.nodes[runtime.state.root_task_id].exploration_round_ids[-1:] == [
        work.id for work in runtime.state.rounds.values() if work.phase == RoundPhase.FAILED
    ]
    assert not any(
        work.kind == RoundKind.DECOMPOSE and work.phase == RoundPhase.QUEUED
        for work in runtime.state.rounds.values()
    )


def test_execution_cannot_create_children_without_returning_through_jev(setup):
    runtime, _, _ = setup

    def worker(context, *_):
        values = children_report(context, ["illicit-child"]).model_dump(
            mode="json", exclude={"unsplittable_reason"}
        )
        from codex_task_tree.models import WorkerReport

        return WorkerReport(**values)

    runtime.worker = worker
    step_route(runtime)
    assert len(runtime.state.nodes) == 1
    assert any(
        work.phase == RoundPhase.FAILED and "DECOMPOSE" in work.error
        for work in runtime.state.rounds.values()
    )


def test_parent_cannot_enter_integration_without_child_products(setup):
    runtime, _, _ = setup
    work = begin_exploration(runtime)
    work.kind = RoundKind.RECONCILE
    runtime.worker = lambda *_: pytest.fail(
        "parent must not bypass decomposition by calling integration"
    )
    with pytest.raises(ValueError, match="child deliveries"):
        step_route(runtime)
    assert runtime.state.progress.worker_attempts_started == 0


@pytest.mark.parametrize(
    ("break_child", "change_identity"), [(False, False), (True, False), (False, True)]
)
def test_integration_can_change_child_inputs_but_must_reverify_current_products(
    setup, break_child, change_identity
):
    import sys

    from codex_task_tree.checks import latest_record

    from .conftest import adopt

    runtime, judge, workspace = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["child"])
    step_route(runtime)
    root = runtime.state.root_task_id
    child = runtime.state.nodes[runtime.state.children(root)[0]]
    item = child.spec.acceptance[0]
    item.check_spec.command = [
        sys.executable,
        "-B",
        "-c",
        "from pathlib import Path; assert Path('child.txt').read_text().strip() == 'valid'",
    ]
    adopt(runtime.state, runtime.store, child.id)
    judge.denied.add("SHOULD_SPLIT")

    def worker(context, *_):
        if context["kind"] == "EXPLORE":
            return exploration_report(context)
        if context["kind"] == "EXECUTE":
            (workspace / "child.txt").write_text("valid")
            return report_for(context, deliverable_paths=["child.txt"])
        assert context["kind"] == "RECONCILE"
        assert certificate_valid(runtime.state, child.id)
        if change_identity:
            child.result_history[-1].revision += 1
        (workspace / "child.txt").write_text("broken" if break_child else "valid\n")
        (workspace / "Root.txt").write_text("integrated")
        return report_for(context, deliverable_paths=["Root.txt", "child.txt"])

    runtime.worker = worker
    step_route(runtime)
    step_route(runtime)
    assert certificate_valid(runtime.state, child.id)
    step_route(runtime)
    integration = next(r for r in runtime.state.rounds.values() if r.kind == RoundKind.RECONCILE)
    if change_identity:
        assert integration.phase == RoundPhase.FAILED
        assert "delivery identity changed" in integration.error
        assert not certificate_valid(runtime.state, root)
        return
    assert integration.phase == RoundPhase.COMMITTED, integration.error
    current = latest_record(runtime.state, child.id, item.id)
    assert current and current.source_ref == integration.id
    assert current.status == ("FAIL" if break_child else "PASS")
    assert certificate_valid(runtime.state, root) is not break_child


def test_approved_routing_preserves_global_no_progress_history(setup):
    runtime, _, _ = setup
    begin_exploration(runtime)
    runtime.state.progress.rounds_without_substantive_progress = 2
    runtime.worker = exploration_worker
    step_route(runtime)
    assert next_kind(runtime) == RoundKind.DECOMPOSE
    assert runtime.state.progress.rounds_without_substantive_progress == 2


def test_saved_exploration_resumes_decisions_without_repeating_worker(setup):
    runtime, judge, _ = setup
    begin_exploration(runtime)
    runtime.worker = exploration_worker
    runtime.judge = lambda *_: (_ for _ in ()).throw(RuntimeError("temporary decision outage"))
    with pytest.raises(RuntimeError):
        step_route(runtime)
    runtime.state.status = RunStatus.ACTIVE
    runtime.save()
    restored = TaskRuntime(
        runtime.store,
        lambda *_: pytest.fail("do not replay saved exploration"),
        judge,
        runtime.command_exec,
    )
    step_route(restored)
    assert next_kind(restored) == RoundKind.DECOMPOSE
    assert restored.state.progress.worker_attempts_started == 1


def test_leaf_can_be_reclassified_after_execution_discovers_hidden_complexity(setup):
    runtime, judge, workspace = setup

    def execution(context, *_):
        (workspace / "partial.txt").write_text("partial work retained")
        return report_for(
            context,
            deliverable_paths=["partial.txt"],
            gaps=[
                Gap(
                    id="hidden",
                    description="two independent formats need separate acceptance",
                    root_requirement_ids=["r1"],
                )
            ],
        )

    runtime.worker = execution
    step_route(runtime)
    assert next_kind(runtime) == RoundKind.EXPLORE
    runtime.worker = exploration_worker
    step_route(runtime)
    assert next_kind(runtime) == RoundKind.DECOMPOSE
    assert [kind for kind, _, _ in judge.calls][-1:] == ["SHOULD_SPLIT"]
    assert (workspace / "partial.txt").read_text() == "partial work retained"


@pytest.mark.parametrize("kind", ["EXPLORE", "SCOPE_REVIEW", "DECOMPOSE", "REVIEW", "SPECIFY"])
def test_planning_and_review_have_native_read_only_sandbox(kind):
    adapter = object.__new__(CodexAdapter)
    adapter.profile = ExecutionProfile()
    from codex_task_tree.runtime import REPORT_MODELS

    assert adapter.report_contract(kind, REPORT_MODELS[RoundKind(kind)])["sandbox"] == "read-only"
    assert instructions_for_kind(kind)


def test_exploration_schema_cannot_smuggle_completion_or_children(setup):
    runtime, _, _ = setup
    work = begin_exploration(runtime)
    from codex_task_tree.context import build_context

    context = build_context(runtime.state, work.id)
    context["input_digest"] = "test"
    values = exploration_report(context).model_dump(mode="json")
    for field, value in [
        ("completion_claim", True),
        ("candidates", []),
        ("deliverable_paths", ["report.md"]),
    ]:
        with pytest.raises(ValidationError, match="Extra inputs"):
            ExplorationReport.model_validate({**values, field: value})
    decomposition = children_report(context, ["child"]).model_dump(mode="json")
    with pytest.raises(ValidationError, match="Extra inputs"):
        DecompositionReport.model_validate({**decomposition, "completion_claim": True})


def test_reviewer_sees_the_exact_environment_used_by_formal_runner(setup):
    from codex_task_tree.context import build_context

    runtime, _, _ = setup
    captured = []
    execute = runtime.command_exec

    def recording_command(argv, cwd, timeout, env):
        captured.append(env)
        return execute(argv, cwd, timeout, env)

    runtime.command_exec = recording_command
    work = next(work for work in runtime.state.rounds.values() if work.phase == RoundPhase.QUEUED)
    context = build_context(runtime.state, work.id)
    declared = context["checks"][0]["runner_environment"]
    runtime.worker = lambda context, *_: report_for(context)
    step_route(runtime)
    assert captured and all(env == declared for env in captured)
    assert declared["CODEX_CHECK_WORKSPACE"] == str(runtime.state.workspace)
    assert declared["PYTHONPATH"].startswith(declared["CODEX_CHECK_BUNDLE"])


def test_legacy_runs_are_readable_but_resume_cannot_silently_change_protocol(setup, capsys):
    runtime, _, _ = setup
    runtime.state.schema_version = 1
    runtime.state.status = RunStatus.PAUSED
    runtime.save()
    before = runtime.store.load().model_dump(mode="json")
    assert main(["--home", str(runtime.store.root), "status", runtime.store.run_id]) == 0
    assert not json.loads(capsys.readouterr().out)["execution_supported"]
    assert (
        main(
            [
                "--home",
                str(runtime.store.root),
                "resume",
                runtime.store.run_id,
                "--reason",
                "inspect",
            ]
        )
        != 0
    )
    assert "inspection-only" in capsys.readouterr().err
    assert runtime.store.load().model_dump(mode="json") == before


def test_execution_context_does_not_offer_a_new_spec_revision_to_echo(setup):
    from codex_task_tree.context import build_context

    runtime, _, _ = setup
    work = begin_exploration(runtime)
    context = build_context(runtime.state, work.id)
    assert context["spec_revision"] == runtime.state.nodes[work.task_id].spec.revision
    assert context["next_spec_revision"] is None
    work.kind = RoundKind.SPECIFY
    context = build_context(runtime.state, work.id)
    assert context["next_spec_revision"] == context["spec_revision"] + 1


def test_fresh_exploration_preserves_errors_and_reports_after_stale_retry_rerouting(setup):
    from codex_task_tree.context import build_context

    runtime, _, workspace = setup
    failed = next(r for r in runtime.state.rounds.values() if r.phase == RoundPhase.QUEUED)
    failed.phase, failed.error = RoundPhase.FAILED, "assertion was also listed as product input"
    failed.worker_report_ref = runtime.store.write_json("reports/failed.json", {"partial": True})
    (workspace / "partial.txt").write_text("side effect from rejected report")
    probe = begin_exploration(runtime)
    context = build_context(runtime.state, probe.id)
    diagnostic = next(r for r in context["failed_rounds"] if r["id"] == failed.id)
    assert diagnostic["error"] == failed.error
    assert diagnostic["worker_report_ref"] == failed.worker_report_ref
    assert not runtime.state.nodes[failed.task_id].result_history


def test_invalid_child_reports_wake_parent_for_diagnosis_instead_of_waiting_forever(setup):
    runtime, _, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["child"])
    step_route(runtime)
    parent = runtime.state.root_task_id
    child = runtime.state.nodes[runtime.state.children(parent)[0]]
    child.no_progress_rounds = runtime.state.execution_profile.max_node_no_progress - 1

    def invalid_report(context, *_):
        report = exploration_report(context)
        report.spec_revision += 1
        return report

    runtime.worker = invalid_report
    step_route(runtime)
    assert child.status == NodeStatus.BLOCKED
    next_work = runtime.state.rounds[select_next(runtime.state).work_ref.id]
    assert next_work.task_id == parent and next_work.kind == RoundKind.EXPLORE
    assert next_work.purpose == "repair_or_replace_blocked_child"


@pytest.mark.parametrize("invalid", ["constraint", "child-requirement"])
def test_invalid_read_only_plan_can_be_corrected_under_the_existing_split_authorization(
    setup, invalid
):
    runtime, _, _ = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    root.spec.constraints = {"dependencies": "stdlib only"}
    runtime.state.root_approval.contract_hash = root.spec.contract_hash()
    authorize(runtime, RoundKind.DECOMPOSE)
    calls = []

    def worker(context, *_):
        calls.append(context["round_id"])
        report = children_report(context, ["child"])
        if len(calls) == 1:
            if invalid == "constraint":
                report.candidates[0].spec.constraints["dependencies"] = "third party dependency"
            else:
                report.candidates[0].spec.requirements = [
                    root.spec.requirements[0].model_copy(deep=True)
                ]
        return report

    runtime.worker = worker
    step_route(runtime)
    retry = runtime.state.rounds[select_next(runtime.state).work_ref.id]
    assert retry.kind == RoundKind.DECOMPOSE and retry.authorization_round_id
    assert retry.retry_of_round_id == calls[0] and retry.attempt == 2
    step_route(runtime)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert len(runtime.state.nodes) == 2
