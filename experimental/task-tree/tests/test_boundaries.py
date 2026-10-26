from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from codex_task_tree.checks import latest_record, propose_check
from codex_task_tree.codex_adapter import CodexAdapter
from codex_task_tree.models import (
    CapabilityNeed,
    CheckChangeProposal,
    ExecutionProfile,
    Gap,
    RoundKind,
    RoundPhase,
    WorkerReport,
)

from .conftest import authorize
from .test_tree import children_report, report_for


def test_task_tree_worker_disables_skill_instruction_injection(monkeypatch, tmp_path):
    from codex_task_tree import codex_adapter

    captured = {}

    class NativeClient:
        def __init__(self, config):
            captured["config"] = config
            self._client = SimpleNamespace(_proc=SimpleNamespace(pid=12345))

    monkeypatch.setattr(codex_adapter, "Codex", NativeClient)
    monkeypatch.setattr(codex_adapter, "process_start", lambda _: "fixture-process")
    CodexAdapter(ExecutionProfile(), tmp_path)
    assert "skills.include_instructions=false" in captured["config"].config_overrides
    assert "memories.use_memories=false" in captured["config"].config_overrides


def test_remote_execution_inputs_are_not_reported_as_missing_local_files(setup):
    runtime, _, workspace = setup
    (workspace / "source.json").write_text("{}")
    explored = SimpleNamespace(
        execution_blockers=[],
        execution_assessment="remote source inspected successfully",
        required_input_paths=["source.json", "ssh hk:/mnt/data/job.parquet"],
        input_paths=["source.json"],
    )
    reviewed = SimpleNamespace(execution_blockers=[])
    root = runtime.state.root_task_id

    assert runtime.execution_blockers(root, explored, reviewed) == []

    explored.required_input_paths.append("missing-local.json")
    assert runtime.execution_blockers(root, explored, reviewed) == [
        "declared execution input is missing: missing-local.json"
    ]


@pytest.mark.parametrize("explicit_membership", [True, False])
def test_parent_integration_capability_does_not_require_an_artificial_child(
    setup, explicit_membership
):
    runtime, _, _ = setup

    def worker(context, *_):
        report = children_report(context, ["CSV", "gzip"])
        integration = CapabilityNeed(
            id="parent-integrates",
            owner_task_id=context["task_id"],
            description="Integrate the actual codec products in parent reconciliation",
            root_requirement_ids=["r1"],
        )
        report.capability_proposals.append(integration)
        if explicit_membership:
            report.plan_proposals[0].capability_ids.append(integration.id)
        report.plan_proposals[0].parent_capability_ids = [integration.id]
        return report

    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = worker
    runtime.step()
    assert len(runtime.state.nodes) == 3
    plan = runtime.state.plans[
        runtime.state.nodes[runtime.state.root_task_id].current_plan_revision_id
    ]
    assert plan.parent_capability_ids == ["parent-integrates"]
    assert "parent-integrates" in plan.capability_ids


def test_declined_capability_does_not_block_other_children(setup):
    runtime, judge, _ = setup
    judge.denied.add("cap-optional")

    def worker(context, *_):
        report = children_report(context, ["required", "optional"])
        optional = CapabilityNeed(
            id="cap-optional",
            owner_task_id=context["task_id"],
            description="useful but not necessary extension",
            root_requirement_ids=["r1"],
        )
        report.capability_proposals.append(optional)
        report.candidates[1].capability_ids = [optional.id]
        report.plan_proposals[0].capability_ids.append(optional.id)
        return report

    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = worker
    runtime.step()
    root = runtime.state.root_task_id
    children = runtime.state.children(root)
    assert len(children) == 1
    assert runtime.state.nodes[children[0]].spec.title == "required"
    assert not any(
        kind == "NECESSARY" and target == "candidate-optional"
        for kind, target, _ in judge.calls
    )
    plan = runtime.state.plans[runtime.state.nodes[root].current_plan_revision_id]
    assert "cap-optional" not in plan.capability_ids


def test_interrupted_plan_rolls_back_uncommitted_gaps(setup):
    runtime, judge, _ = setup
    root = runtime.state.root_task_id
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(
        context,
        ["child"],
        gap=Gap(
            id="candidate-gap",
            description="unresolved candidate work",
            root_requirement_ids=["r1"],
            evidence_refs=["background"],
        ),
    )

    def interrupted_judge(kind, target, payload):
        if kind == "CAPABILITY_NEEDED":
            raise KeyboardInterrupt
        return judge(kind, target, payload)

    runtime.judge = interrupted_judge
    with pytest.raises(KeyboardInterrupt):
        runtime.step()
    assert runtime.state.nodes[root].open_gaps == []
    assert runtime.store.load().nodes[root].open_gaps == []


def test_constraints_are_inherited_before_judgment(setup):
    runtime, judge, _ = setup
    runtime.state.nodes[runtime.state.root_task_id].spec.constraints = {
        "dependencies": "standard library only"
    }
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["child"])
    runtime.state.root_approval.contract_hash = runtime.state.nodes[
        runtime.state.root_task_id
    ].spec.contract_hash()
    runtime.step()
    children = runtime.state.children(runtime.state.root_task_id)
    assert (
        runtime.state.nodes[children[0]].spec.constraints["dependencies"] == "standard library only"
    )
    members = [payload for kind, _, payload in judge.calls if kind == "NECESSARY"]
    assert members[0]["candidate"]["spec"]["constraints"]["dependencies"] == "standard library only"
    assert not any(kind in {"PLAN_ACCEPTABLE", "COMPLETE"} for kind, _, _ in judge.calls)


def test_evidence_acceptance_asks_only_about_the_criterion_not_whole_task_completion(setup):
    from .conftest import adopt

    runtime, judge, workspace = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    item = root.spec.acceptance[0]
    item.check_kind = "evidence"
    item.check_spec.command = []
    item.check_spec.artifact_paths = ["evidence.md"]
    item.adopted_check_revision_id = None
    (workspace / "evidence.md").write_text("Actual observations supporting this one criterion")
    runtime.state.root_approval.contract_hash = root.spec.contract_hash()
    adopt(runtime.state, runtime.store, root.id)
    runtime.verify(root.id, "criterion-probe")
    assert [kind for kind, _, _ in judge.calls] == ["EVIDENCE_SUPPORTED"]
    assert judge.calls[0][2]["criterion"]["id"] == item.id
    assert not root.completion_certificate_id


@pytest.mark.parametrize(
    "mode", ["pass", "exit-failure", "stdout-failure", "timeout", "model-failure"]
)
def test_evidence_command_must_run_and_pass_before_model_judgment(setup, mode):
    from codex_task_tree.checks import record_current, verification_profile

    from .conftest import adopt

    runtime, judge, workspace = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    item = root.spec.acceptance[0]
    item.check_kind = "evidence"
    item.check_spec.command = ["test-command"]
    item.check_spec.expected_stdout = "actual checked evidence"
    item.check_spec.artifact_paths = ["evidence.md"]
    item.adopted_check_revision_id = None
    (workspace / "evidence.md").write_text("Actual evidence")
    runtime.state.root_approval.contract_hash = root.spec.contract_hash()
    adopt(runtime.state, runtime.store, root.id)
    calls = []

    def command(argv, cwd, timeout, env):
        calls.append((argv, cwd, env))
        if mode == "timeout":
            raise TimeoutError("check timed out")
        return {
            "exit_code": 1 if mode == "exit-failure" else 0,
            "stdout": "wrong output" if mode == "stdout-failure" else "actual checked evidence",
            "stderr": "",
        }

    runtime.command_exec = command
    if mode == "model-failure":
        judge.denied.add("EVIDENCE_SUPPORTED")
    runtime.verify(root.id, "evidence-command-probe")
    record = latest_record(runtime.state, root.id, item.id)
    assert len(calls) == 1 and calls[0][0] == ["test-command"]
    assert calls[0][1] == workspace
    assert calls[0][2]["CODEX_CHECK_WORKSPACE"] == str(workspace)
    assert record.method == "command+model"
    assert record.status == ("ERROR" if mode == "timeout" else "PASS" if mode == "pass" else "FAIL")
    if mode in {"pass", "model-failure"}:
        assert [kind for kind, _, _ in judge.calls] == ["EVIDENCE_SUPPORTED"]
        proof = judge.calls[0][2]["behavioral_check"]
        assert proof["exit_code"] == 0 and proof["stdout"] == "actual checked evidence"
        assert "actual checked evidence" in Path(record.output_ref).read_text()
    else:
        assert not judge.calls
    legacy = record.model_copy(
        update={
            "method": "model",
            "execution_profile_digest": verification_profile(runtime.state, "model"),
        }
    )
    assert not record_current(runtime.state, legacy)


def test_explicit_constraint_override_is_quarantined_without_partial_plan(setup):
    runtime, _, _ = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    root.spec.constraints = {"dependencies": "standard library only"}
    runtime.state.root_approval.contract_hash = root.spec.contract_hash()

    def worker(context, *_):
        report = children_report(context, ["child"])
        report.candidates[0].spec.constraints["dependencies"] = "install a new dependency"
        return report

    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = worker
    runtime.step()
    assert len(runtime.state.nodes) == 1
    assert any(r.phase == RoundPhase.FAILED for r in runtime.state.rounds.values())


def test_changed_root_contract_blocks_before_execution(setup):
    runtime, _, _ = setup
    runtime.state.nodes[runtime.state.root_task_id].spec.goal = "different goal"
    runtime.worker = lambda *_: pytest.fail("unapproved scope cannot execute")
    runtime.step()
    assert runtime.state.status.value == "BLOCKED"


def test_actual_user_acceptance_is_distinct_from_command_override(setup):
    runtime, _, workspace = setup
    root = runtime.state.root_task_id
    with pytest.raises(ValueError, match="cannot override"):
        runtime.accept_user_criterion(root, "ok", "user says yes")
    runtime.state.nodes[root].spec.acceptance[0].check_kind = "user"
    runtime.state.root_approval.contract_hash = runtime.state.nodes[root].spec.contract_hash()
    runtime.accept_user_criterion(root, "ok", "human reviewed this exact output")
    record = latest_record(runtime.state, root, "ok")
    assert record.method == "user" and record.status == "PASS"
    (workspace / "Root.txt").write_text("changed after acceptance")
    assert latest_record(runtime.state, root, "ok") is None


def test_worker_timeout_interrupts_before_returning():
    event = Event()

    class Turn:
        id = "turn"

        def run(self):
            event.wait(3)

        def interrupt(self):
            event.set()

    adapter = object.__new__(CodexAdapter)
    adapter.profile = ExecutionProfile(worker_model="test")
    adapter.workspace = "."
    adapter.codex = SimpleNamespace(
        _client=SimpleNamespace(config=SimpleNamespace(rpc_timeout_seconds=None)),
        thread_start=lambda **_: SimpleNamespace(id="thread", turn=lambda *_, **__: Turn()),
    )
    with pytest.raises(TimeoutError, match="interruption acknowledged"):
        adapter({"kind": "EXECUTE"}, WorkerReport, lambda *_: None, 0.01)
    assert event.is_set()


def test_run_process_lock_rejects_second_controller(setup):
    runtime, _, _ = setup
    with runtime.store.locked():
        with pytest.raises(RuntimeError, match="another controller"):
            with runtime.store.locked():
                pass


def test_parent_wait_edge_rejects_child_waiting_for_ancestor_completion(setup):
    runtime, _, _ = setup

    def worker(context, *_):
        report = children_report(context, ["child"])
        report.candidates[0].depends_on = [context["task_id"]]
        return report

    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = worker
    runtime.step()
    assert len(runtime.state.nodes) == 1
    assert any(
        r.phase == RoundPhase.FAILED and "cycle" in (r.error or "")
        for r in runtime.state.rounds.values()
    )


def test_crash_after_decisions_does_not_publish_half_plan_or_duplicate_children(setup, monkeypatch):
    from codex_task_tree.runtime import TaskRuntime

    runtime, _, _ = setup
    authorize(runtime, RoundKind.DECOMPOSE)
    runtime.worker = lambda context, *_: children_report(context, ["A", "B"])
    original = runtime.store.save

    def crashing_save(state, revision):
        if len(state.nodes) > 1:
            raise RuntimeError("simulated crash before final commit")
        original(state, revision)

    monkeypatch.setattr(runtime.store, "save", crashing_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        runtime.step()
    assert len(runtime.store.load().nodes) == 1
    monkeypatch.setattr(runtime.store, "save", original)
    restored = TaskRuntime(
        runtime.store,
        lambda *_: pytest.fail("saved worker must not replay"),
        runtime.judge,
        runtime.command_exec,
    )
    restored.state.status = "ACTIVE"
    restored.save()
    restored.step()
    assert len(restored.state.nodes) == 3


@pytest.mark.parametrize("qualified", [True, False])
def test_check_pending_gap_resolves_only_from_its_current_official_pass(setup, qualified):
    runtime, _, _ = setup
    root = runtime.state.root_task_id
    runtime.worker = lambda context, *_: report_for(
        context,
        gaps=[
            Gap(
                id="pending",
                kind="CHECK_PENDING",
                criterion_refs=[f"{root}:ok" if qualified else "ok"],
                description="formal check pending",
                root_requirement_ids=["r1"],
            )
        ],
    )
    runtime.step()
    assert not runtime.state.nodes[root].open_gaps
    assert runtime.state.nodes[root].status.value == "SUCCEEDED"


@pytest.mark.parametrize("invalid_reference", ["unknown", "unrelated-task:ok"])
def test_check_gap_ids_are_validated_before_decisions_and_repaired_in_session(
    setup, invalid_reference
):
    runtime, judge, _ = setup
    calls_before = len(judge.calls)
    errors = []

    class Worker:
        def run_with_feedback(self, context, *_, validator, on_feedback):
            report = report_for(
                context,
                gaps=[
                    Gap(
                        id="pending",
                        kind="CHECK_PENDING",
                        criterion_refs=[invalid_reference],
                        description="formal check pending",
                        root_requirement_ids=["r1"],
                    )
                ],
            )
            with pytest.raises(ValueError, match="effective acceptance") as error:
                validator(report)
            errors.append(str(error.value))
            assert len(judge.calls) == calls_before
            report.gaps[0].criterion_refs = ["ok"]
            validator(report)
            return report

    runtime.worker = Worker()
    runtime.step()
    assert len(errors) == 1
    assert runtime.state.nodes[runtime.state.root_task_id].status.value == "SUCCEEDED"


def test_pending_check_from_failed_report_still_gets_independent_review(setup):
    runtime, _, _ = setup
    root = runtime.state.root_task_id
    item = runtime.state.nodes[root].spec.acceptance[0]
    pending = propose_check(
        runtime.state,
        runtime.store,
        root,
        "failed-prior-round",
        CheckChangeProposal(
            id="new",
            criterion_id="ok",
            old_check_revision_id=item.adopted_check_revision_id,
            check_spec=item.check_spec,
            reason="new independently reviewable definition",
        ),
    )
    runtime.worker = lambda context, *_: report_for(context)
    runtime.step()
    assert any(
        r.phase == RoundPhase.QUEUED and pending.id in r.purpose
        for r in runtime.state.rounds.values()
    )


def test_pending_proposal_is_not_the_adopted_revision_for_a_replacement(setup):
    runtime, _, _ = setup
    root = runtime.state.root_task_id
    item = runtime.state.nodes[root].spec.acceptance[0]
    proposal = CheckChangeProposal(
        id="pending",
        criterion_id=item.id,
        old_check_revision_id=item.adopted_check_revision_id,
        check_spec=item.check_spec,
        reason="replacement requiring independent review",
    )
    pending = propose_check(runtime.state, runtime.store, root, "first", proposal)
    with pytest.raises(ValueError, match="adopted_check_revision_id"):
        propose_check(
            runtime.state,
            runtime.store,
            root,
            "second",
            proposal.model_copy(update={"old_check_revision_id": pending.id}),
        )
    replacement = propose_check(runtime.state, runtime.store, root, "second", proposal)
    assert replacement.status == "PROPOSED"
    assert item.adopted_check_revision_id != replacement.id


def test_decision_context_keeps_current_results_and_discloses_omitted_history(setup):
    import json

    from codex_task_tree.models import NodeResult

    runtime, _, _ = setup
    root = runtime.state.nodes[runtime.state.root_task_id]
    root.result_history = [
        NodeResult(revision=1, round_id="old", summary="obsolete" * 10000),
        NodeResult(revision=2, round_id="current", summary="current delivered result"),
    ]
    context = runtime.decision_context(root.id)
    assert context["task"]["current_result"]["revision"] == 2
    assert context["context_selection"]["historical_result_revisions"] == [1]
    assert "obsolete" not in json.dumps(context)
    assert len(context["task_catalog"]) == len(runtime.state.nodes)


def test_context_manifest_binds_actual_execution_contract(tmp_path):
    import json

    from codex_task_tree.models import SpecProposal, digest
    from codex_task_tree.runtime import TaskRuntime
    from codex_task_tree.store import RunStore

    from .conftest import preparation_report

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RunStore(tmp_path / "controller", "manifest")
    TaskRuntime.create(store, "understand task", workspace, ExecutionProfile())

    class Worker:
        def report_contract(self, *_):
            return {
                "developer_instructions": "read only; propose scope for approval",
                "output_schema": SpecProposal.model_json_schema(),
            }

        def __call__(self, context, *_):
            return preparation_report(context, workspace)

    runtime = TaskRuntime(store, Worker(), None, None)
    runtime.step()
    work = next(iter(runtime.state.rounds.values()))
    from pathlib import Path

    manifest = json.loads(Path(work.context_manifest_ref).read_text())
    assert "output_schema" in manifest["execution_contract"]
    assert manifest["context"].pop("input_digest") == work.input_digest
    assert digest(manifest) == work.input_digest


def test_valid_formal_support_is_not_poisoned_by_supplementary_citations(setup):
    runtime, _, _ = setup
    root = runtime.state.root_task_id
    item = runtime.state.coverage["requirement:r1"].model_copy(deep=True)
    item.status, item.evidence_refs = (
        "SUPPORTED",
        ["ok", "exploratory local execution passed", "Root.txt"],
    )
    runtime.worker = lambda context, *_: report_for(context, coverage_updates=[item])
    runtime.step()
    assert runtime.state.nodes[root].status.value == "SUCCEEDED"
    stored = runtime.state.coverage[item.id]
    assert all(key in runtime.state.verifications for key in stored.evidence_refs)


def test_supplementary_claims_alone_never_become_formal_support(setup):
    runtime, _, _ = setup
    root = runtime.state.root_task_id
    item = runtime.state.coverage["requirement:r1"].model_copy(deep=True)
    item.status, item.evidence_refs = (
        "SUPPORTED",
        ["exploratory local execution passed", "Root.txt"],
    )
    runtime.worker = lambda context, *_: report_for(context, coverage_updates=[item])
    runtime.step()
    assert runtime.state.nodes[root].status.value != "SUCCEEDED"
    assert runtime.state.coverage[item.id].status == "IN_PROGRESS"


def test_frozen_assertion_helper_is_used_instead_of_mutable_workspace_copy(setup):
    import sys

    runtime, _, workspace = setup
    root = runtime.state.root_task_id
    item = runtime.state.nodes[root].spec.acceptance[0]
    (workspace / "product.py").write_text("value = 0\n")
    (workspace / "assertion_helper.py").write_text("expected = 1\n")
    (workspace / "official.py").write_text(
        "from assertion_helper import expected\nfrom product import value\nassert value == expected\n"
    )
    from codex_task_tree.models import CheckSpec

    check = propose_check(
        runtime.state,
        runtime.store,
        root,
        "independently-reviewed-fixture",
        CheckChangeProposal(
            id="helper",
            criterion_id="ok",
            old_check_revision_id=item.adopted_check_revision_id,
            check_spec=CheckSpec(
                command=[sys.executable, "official.py"], artifact_paths=["product.py"]
            ),
            file_paths=["official.py", "assertion_helper.py"],
            reason="test helper freezing",
        ),
    )
    check.status = "ADOPTED"
    item.adopted_check_revision_id = check.id
    (workspace / "assertion_helper.py").write_text("expected = 0\n")
    records = runtime.verify(root, "helper-fixture")
    assert runtime.state.verifications[records[0]].status == "FAIL"
    (workspace / "product.py").write_text("value = 1\n")
    records = runtime.verify(root, "helper-fixture-corrected")
    assert runtime.state.verifications[records[0]].status == "PASS"


def test_check_cannot_freeze_the_product_instead_of_checking_current_product(setup):
    import sys

    from codex_task_tree.models import CheckSpec

    runtime, _, workspace = setup
    root = runtime.state.root_task_id
    item = runtime.state.nodes[root].spec.acceptance[0]
    (workspace / "product.py").write_text("value = 1\n")
    with pytest.raises(ValueError, match="mutable product"):
        propose_check(
            runtime.state,
            runtime.store,
            root,
            "wrong-role",
            CheckChangeProposal(
                id="bad",
                criterion_id="ok",
                old_check_revision_id=item.adopted_check_revision_id,
                check_spec=CheckSpec(
                    command=[sys.executable, "product.py"], artifact_paths=["product.py"]
                ),
                file_paths=["product.py"],
                reason="invalid product snapshot",
            ),
        )


def test_frozen_files_are_not_adopted_when_command_still_discovers_workspace_tests(setup):
    from codex_task_tree.models import CheckSpec

    runtime, _, workspace = setup
    root = runtime.state.root_task_id
    item = runtime.state.nodes[root].spec.acceptance[0]
    (workspace / "test_official.py").write_text("assert False\n")
    with pytest.raises(ValueError, match="frozen entrypoint"):
        propose_check(
            runtime.state,
            runtime.store,
            root,
            "wrong-entry",
            CheckChangeProposal(
                id="bad",
                criterion_id="ok",
                old_check_revision_id=item.adopted_check_revision_id,
                check_spec=CheckSpec(command=["pytest"], artifact_paths=["Root.txt"]),
                file_paths=["test_official.py"],
                reason="invalid test discovery",
            ),
        )
