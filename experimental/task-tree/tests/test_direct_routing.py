from __future__ import annotations

import pytest

from codex_task_tree.cli import parser
from codex_task_tree.models import ExecutionProfile, RoundKind, RoundPhase, RunStatus, SpecProposal
from codex_task_tree.runtime import TaskRuntime
from codex_task_tree.scheduler import select_next

from .conftest import exploration_report, exploration_worker, preparation_report, spec
from .test_preparation import new_runtime


@pytest.fixture
def scoped(tmp_path):
    runtime, workspace, judge = new_runtime(tmp_path)

    def initial(context, *_):
        if context["kind"] == "PREPARE":
            return preparation_report(context, workspace)
        return SpecProposal(
            task_id=context["task_id"], round_id=context["round_id"], proposed_spec=spec()
        )

    runtime.worker = initial
    runtime.step()
    runtime.step()
    root = runtime.state.nodes[runtime.state.root_task_id]
    runtime.approve_root(root.spec.contract_hash(), "fixture approval")
    runtime.worker = exploration_worker
    return runtime, workspace, judge


def next_work(runtime):
    return runtime.state.rounds[select_next(runtime.state).work_ref.id]


def test_new_workers_default_to_the_requested_model_and_effort():
    profile = ExecutionProfile()
    args = parser().parse_args(["start", "fixture task"])
    assert profile.worker_model == args.model == "gpt-6-sol"
    assert profile.worker_effort == args.effort == "xhigh"


def test_exploration_routes_directly_to_jev_without_scope_review(scoped):
    runtime, _, judge = scoped
    calls = []

    def worker(context, model, started, timeout):
        calls.append(context["kind"])
        started(context["kind"] + "-thread")
        return exploration_worker(context)

    runtime.worker = worker
    runtime.step()
    source = next(r for r in runtime.state.rounds.values() if r.kind == RoundKind.EXPLORE)
    assert calls == ["EXPLORE"]
    assert source.phase == RoundPhase.COMMITTED
    assert source.worker_model == "gpt-6-sol" and source.worker_effort == "xhigh"
    assert source.scope_review_round_id is None
    assert [kind for kind, _, _ in judge.calls] == ["SHOULD_SPLIT"]
    assert judge.calls[0][2]["exploration_draft"]["findings"]
    assert next_work(runtime).kind == RoundKind.DECOMPOSE
    assert runtime.authorization_current(next_work(runtime))
    assert not runtime.state.nodes[source.task_id].result_history
    assert not any(r.kind == RoundKind.SCOPE_REVIEW for r in runtime.state.rounds.values())


@pytest.mark.parametrize("invalid", ["omitted", "duplicate", "relabelled", "missing-opportunity"])
def test_inventory_dispositions_are_checked_before_jev(scoped, invalid):
    runtime, _, judge = scoped

    def worker(context, *_):
        report = exploration_report(context)
        if invalid == "omitted":
            report.aspect_assessments = []
        elif invalid == "duplicate":
            report.aspect_assessments.append(report.aspect_assessments[0].model_copy())
        elif invalid == "relabelled":
            report.aspect_assessments[0].subject = "Different subject hiding the original obligation"
        else:
            report.aspect_assessments[0].disposition = "SPLIT_OPPORTUNITY"
        return report

    runtime.worker = worker
    runtime.step()
    assert not judge.calls
    assert any(r.kind == RoundKind.EXPLORE and r.phase == RoundPhase.FAILED for r in runtime.state.rounds.values())
    assert next_work(runtime).kind == RoundKind.EXPLORE


def test_declined_split_with_missing_essential_input_returns_to_exploration(scoped):
    runtime, _, judge = scoped
    judge.denied.add("SHOULD_SPLIT")

    def worker(context, *_):
        report = exploration_report(context)
        report.execution_blockers = ["Required comparison input is inaccessible"]
        return report

    runtime.worker = worker
    runtime.step()
    assert [kind for kind, _, _ in judge.calls] == ["SHOULD_SPLIT"]
    followup = next_work(runtime)
    assert followup.kind == RoundKind.EXPLORE
    assert "inaccessible" in followup.purpose
    assert not runtime.state.nodes[followup.task_id].completion_certificate_id


def test_source_change_revokes_direct_routing(scoped):
    runtime, workspace, judge = scoped
    (workspace / "source.txt").write_text("original")

    def worker(context, *_):
        report = exploration_report(context)
        report.input_paths = ["source.txt"]
        return report

    runtime.worker = worker
    runtime.step()
    authorized = next_work(runtime)
    assert authorized.kind == RoundKind.DECOMPOSE
    (workspace / "source.txt").write_text("changed")
    runtime.worker = lambda *_: pytest.fail("stale route must not execute")
    runtime.step()
    assert authorized.phase == RoundPhase.CANCELLED
    assert next_work(runtime).kind == RoundKind.EXPLORE
    assert len(judge.calls) == 1


def test_saved_exploration_retries_decision_without_repeating_worker(scoped):
    runtime, _, judge = scoped
    runtime.judge = lambda *_: (_ for _ in ()).throw(RuntimeError("temporary decision outage"))
    with pytest.raises(RuntimeError, match="outage"):
        runtime.step()
    attempts = runtime.state.progress.worker_attempts_started
    runtime.state.status = RunStatus.ACTIVE
    runtime.save()
    restored = TaskRuntime(
        runtime.store,
        lambda *_: pytest.fail("saved exploration must be recovered"),
        judge,
        runtime.command_exec,
    )
    restored.step()
    assert restored.state.progress.worker_attempts_started == attempts
    assert next_work(restored).kind == RoundKind.DECOMPOSE
