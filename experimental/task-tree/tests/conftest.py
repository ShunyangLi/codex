from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from codex_task_tree.checks import propose_check
from codex_task_tree.context import build_context, fingerprints
from codex_task_tree.models import (
    AcceptanceCriterion,
    AspectAssessment,
    CheckChangeProposal,
    CheckSpec,
    CoverageItem,
    DecisionRecord,
    ExecutionProfile,
    ExplorationReport,
    PreparationReport,
    Requirement,
    RootApproval,
    RoundKind,
    RoundPhase,
    ScopeReviewReport,
    TaskSpec,
    digest,
)
from codex_task_tree.runtime import TaskRuntime, queue_round
from codex_task_tree.store import RunStore


def spec(title="Root", goal=None, *, child=False, paths=None):
    check = AcceptanceCriterion(
        id="ok",
        description=f"{title} behavior works",
        root_requirement_ids=["r1"],
        check_spec=CheckSpec(
            command=[sys.executable, "-c", "print('verified')"],
            artifact_paths=paths or [title + ".txt"],
        ),
    )
    return TaskSpec(
        title=title,
        goal=goal or title + " complete behavior",
        requirements=[]
        if child
        else [Requirement(id="r1", description="usable storage", acceptance_ids=["ok"])],
        root_requirement_ids=["r1"],
        deliverables=[title + ".txt"],
        parent_criterion_ids=["ok"] if child else [],
        acceptance=[check],
    )


class FakeJudge:
    def __init__(self, denied=()):
        self.denied, self.calls = set(denied), []

    def __call__(self, kind, target, payload):
        self.calls.append((kind, target, payload))
        return DecisionRecord(
            id="d_" + digest([kind, target, payload]),
            kind=kind,
            target_id=target,
            input_digest=digest(payload),
            score=0 if kind in self.denied or target in self.denied else 1,
            threshold=0.5,
            model="fake",
        )


def local_command(argv, cwd, timeout, env=None):
    p = subprocess.run(
        argv,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def preparation_report(context, workspace):
    (workspace / "background").mkdir(exist_ok=True)
    (workspace / "background/index.json").write_text(
        json.dumps(
            {
                "sources": [{"path": "fixture-source", "status": "inspected"}],
                "known": "CSV storage",
                "investigation_inventory": [
                    {
                        "id": "csv-roundtrip",
                        "subject": "CSV roundtrip behavior",
                        "source_refs": ["fixture-source"],
                        "possible_questions": ["Do supported rows roundtrip without loss?"],
                        "meaning_and_limits": "Fixture behavior; no production claim",
                    }
                ],
            }
        )
    )
    return PreparationReport(
        task_id=context["task_id"],
        round_id=context["round_id"],
        input_digest=context["input_digest"],
        summary="inspected fixture background",
        artifact_paths=["background/index.json"],
        inspected_sources=["fixture-source"],
        facts={"verified": "CSV storage"},
    )


def exploration_report(context):
    return ExplorationReport(
        task_id=context["task_id"],
        round_id=context["round_id"],
        spec_revision=context["task"]["specs"][-1]["revision"],
        input_digest=context["input_digest"],
        summary="inspected task boundaries",
        inspected_sources=["fixture source"],
        findings=["inputs and current task obligations inspected"],
        responsibilities=[context["task"]["specs"][-1]["goal"]],
        execution_assessment="assess the bounded responsibility and acceptance using this evidence",
        split_opportunities=[],
        leaf_rationale="The fixture has one stated behavioral outcome; its queries and checks support that outcome. No independent outcome or material unanswered question was found in the fixture sources.",
        aspect_assessments=[
            AspectAssessment(
                aspect_id=item["id"],
                subject=item["subject"],
                disposition="SUPPORTING_STEP",
                rationale="The fixture scope and sources assign this to the same behavioral outcome",
                evidence_refs=item["source_refs"],
            )
            for item in (context.get("preparation") or {}).get("investigation_inventory", [])
        ],
    )


def scope_review_report(context):
    explored = context["exploration"][0]["report"]
    return ScopeReviewReport(
        task_id=context["task_id"],
        round_id=context["round_id"],
        spec_revision=context["spec_revision"],
        input_digest=context["input_digest"],
        reviewed_exploration_round_id=explored["round_id"],
        summary="Independent fixture source and scope review",
        evidence=["fixture sources inspected independently"],
        aspect_assessments=explored.get("aspect_assessments", []),
    )


def exploration_worker(context, *_):
    return (
        scope_review_report(context)
        if context["kind"] == "SCOPE_REVIEW"
        else exploration_report(context)
    )


def authorize(runtime, kind=RoundKind.EXECUTE):
    """Downstream unit fixtures start after routing; test_routing exercises the real gate."""
    state = runtime.state
    node = state.nodes[state.root_task_id]
    for work in state.rounds.values():
        if work.task_id == node.id and work.phase == RoundPhase.QUEUED:
            work.phase = RoundPhase.CANCELLED
    source = queue_round(state, node.id, RoundKind.EXPLORE, f"fixture-route:{state.sequence()}")
    context = build_context(state, source.id)
    source.input_digest = digest(context)
    context["input_digest"] = source.input_digest
    report = exploration_report(context)
    source.worker_report_ref = runtime.store.write_json(
        f"reports/{source.id}/worker.json", report.model_dump(mode="json")
    )
    for decision_kind, score in [("SHOULD_SPLIT", int(kind == RoundKind.DECOMPOSE))]:
        decision = DecisionRecord(
            id="fixture_" + digest([source.id, decision_kind]),
            kind=decision_kind,
            target_id=node.id,
            input_digest=digest(report),
            score=score,
            threshold=0.5,
            model="fixture",
        )
        state.decisions[decision.id] = decision
        source.decision_ids.append(decision.id)
    source.routing_basis = runtime.routing_basis(node.id)
    # Routing freshness is isolated from product-mutation tests in this fixture.
    source.routing_fingerprints = fingerprints(state.workspace, ["__fixture_route_input__"])
    source.phase = RoundPhase.COMMITTED
    node.exploration_round_ids.append(source.id)
    result = queue_round(
        state, node.id, kind, f"fixture-authorized:{source.id}", authorization_round_id=source.id
    )
    runtime.save()
    return result


def adopt(state, store, task_id):
    for item in state.nodes[task_id].spec.acceptance:
        check = propose_check(
            state,
            store,
            task_id,
            "fixture-proposer",
            CheckChangeProposal(
                id="initial",
                criterion_id=item.id,
                check_spec=item.check_spec,
                reason="test fixture",
            ),
        )
        check.status, check.review_refs = "ADOPTED", ["fixture-independent-review"]
        item.adopted_check_revision_id = check.id


@pytest.fixture
def setup(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RunStore(tmp_path / "controller", "test")
    state = TaskRuntime.create(store, "build storage", workspace, ExecutionProfile())
    # These downstream fixtures begin from an explicitly preapproved contract.
    state.preparation_required = False
    root = state.nodes[state.root_task_id]
    root.specs = [spec()]
    state.coverage["requirement:r1"] = CoverageItem(
        id="requirement:r1",
        kind="REQUIREMENT",
        owner_task_id=root.id,
        root_requirement_ids=["r1"],
        subject="usable storage",
    )
    for work in state.rounds.values():
        work.phase = RoundPhase.COMMITTED
    state.root_approval = RootApproval(
        approved_spec_revision=1,
        contract_hash=root.spec.contract_hash(),
        user_confirmation_ref="fixture-user",
    )
    adopt(state, store, root.id)
    store.save(state, state.revision)
    judge = FakeJudge()
    runtime = TaskRuntime(store, None, judge, local_command)
    authorize(runtime)
    return runtime, judge, workspace
