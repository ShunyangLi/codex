from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .checks import (
    latest_record,
    propose_check,
    record_current,
    review_file,
    verification_paths,
    verify_one,
)
from .codex_adapter import WorkerReportError
from .context import (
    active_subtree,
    build_context,
    check_environment,
    coverage_view,
    criterion,
    decomposition_view,
    effective_criteria,
    external_input_ref,
    fingerprints,
    input_path,
    local_inputs,
    node_input,
    parse_investigation_inventory,
    preparation_view,
    read_roots,
    scope_review_view,
    workspace_path,
)
from .decision_context import REVIEWABLE_SECTIONS
from .jev import DecisionContextTooLarge, DecisionWorkerUnavailable
from .luna_decision import CONTEXT_ENCODING_VERSION, decision_policy_digest
from .models import (
    TERMINAL_ROUNDS,
    AcceptanceCriterion,
    ArtifactRef,
    CheckChangeProposal,
    CheckReviewReport,
    CompletionCertificate,
    ContextReviewReport,
    CoverageItem,
    DecisionJob,
    DecompositionReport,
    ExecutionProfile,
    ExplorationReport,
    Gap,
    NodeResult,
    NodeStatus,
    MustSolveItem,
    PlanProposal,
    PlanRevision,
    PreparationRecord,
    PreparationReport,
    Requirement,
    RootApproval,
    RoundKind,
    RoundPhase,
    RunState,
    RunStatus,
    ScopeReviewReport,
    SpecProposal,
    Suggestion,
    TaskNode,
    TaskRound,
    TaskSpec,
    VerificationJob,
    VerificationRecord,
    WorkerReport,
    WorkRef,
    digest,
    identifier,
)
from .scheduler import select_next
from .store import RunStore, file_hash

REPORT_MODELS = {
    RoundKind.PREPARE: PreparationReport,
    RoundKind.SPECIFY: SpecProposal,
    RoundKind.EXPLORE: ExplorationReport,
    RoundKind.SCOPE_REVIEW: ScopeReviewReport,
    RoundKind.DECOMPOSE: DecompositionReport,
    RoundKind.EXECUTE: WorkerReport,
    RoundKind.REVIEW: CheckReviewReport,
    RoundKind.CONTEXT_REVIEW: ContextReviewReport,
    RoundKind.RECONCILE: WorkerReport,
}


def saved_decision_context(decision) -> dict:
    directory = Path(decision.request_ref).parent
    for name in ("selected-context.json", "full-context.json", "source.json"):
        source = directory / name
        if source.is_file():
            return json.loads(source.read_text())
    raise FileNotFoundError(f"decision evidence context missing: {directory}")


@dataclass(frozen=True)
class NodeEvaluation:
    completed: bool = False
    next_kind: RoundKind | None = None
    purpose: str = ""
    waiting_children: bool = False
    decision_ids: tuple[str, ...] = ()


def spec_for_decision(spec) -> dict:
    value = spec.model_dump(mode="json")
    for item in value["acceptance"]:
        item.pop("check_spec")
    return value


def candidate_for_decision(candidate) -> dict:
    value = candidate.model_dump(mode="json")
    value["spec"] = spec_for_decision(candidate.spec)
    return value


def duplicate_signature(spec) -> str:
    return digest(
        {
            "goal": spec.goal.strip().casefold(),
            "scope": sorted(spec.scope),
            "constraints": spec.constraints,
            "acceptance": sorted(c.description for c in spec.acceptance if c.required),
        }
    )


def reuse_scope_covers(target, source) -> bool:
    needed = set(target.root_requirement_ids) | {
        r for c in target.acceptance if c.required for r in c.root_requirement_ids
    }
    available = set(source.root_requirement_ids) & {
        r for c in source.acceptance if c.required for r in c.root_requirement_ids
    }
    # Full-result replacement is conservative; related work remains a usable dependency.
    return (
        bool(needed)
        and needed <= available
        and target.goal.strip().casefold() == source.goal.strip().casefold()
        and set(target.scope) <= set(source.scope)
        and all(source.constraints.get(k) == v for k, v in target.constraints.items())
        and {c.description for c in target.acceptance if c.required}
        <= {c.description for c in source.acceptance if c.required}
    )


def inherited_root_criteria(state: RunState, task_id: str) -> list[str] | None:
    """Resolve root-branch obligations in code; local criterion IDs may collide."""
    if task_id == state.root_task_id:
        return None
    node = state.nodes[task_id]
    while node.parent_id and node.parent_id != state.root_task_id:
        node = state.nodes[node.parent_id]
    return list(node.spec.parent_criterion_ids) if node.parent_id == state.root_task_id else None


def queue_round(
    state: RunState,
    task_id: str,
    kind: RoundKind,
    source: str,
    *,
    purpose: str = "execute",
    authorization_round_id: str | None = None,
    decision_job_id: str | None = None,
) -> TaskRound:
    for work in state.rounds.values():
        if work.task_id == task_id and work.kind == kind and work.source_event_id == source:
            return work
    if any(
        r.task_id == task_id
        and r.phase not in TERMINAL_ROUNDS
        and not (kind == RoundKind.CONTEXT_REVIEW and r.phase == RoundPhase.WAITING_CONTEXT)
        for r in state.rounds.values()
    ):
        raise ValueError("task already has unfinished work")
    work = TaskRound(
        id=identifier("round"),
        task_id=task_id,
        created_seq=state.sequence(),
        source_event_id=source,
        kind=kind,
        purpose=purpose,
        decision_job_id=decision_job_id,
        authorization_round_id=authorization_round_id,
    )
    state.rounds[work.id] = work
    return work


def validate_spec(state: RunState, task_id: str, spec) -> None:
    node = state.nodes[task_id]
    root = state.nodes[state.root_task_id]
    if node.parent_id is None:
        if not spec.requirements or not any(r.required for r in spec.requirements):
            raise ValueError("root specification needs required Requirement IDs")
        ids = {r.id for r in spec.requirements}
        criteria = {c.id for c in spec.acceptance}
        for requirement in spec.requirements:
            if requirement.required and (
                not requirement.acceptance_ids or not set(requirement.acceptance_ids) <= criteria
            ):
                raise ValueError("every required root requirement needs acceptance mapping")
            if requirement.required and any(
                not next(c for c in spec.acceptance if c.id == cid).required
                for cid in requirement.acceptance_ids
            ):
                raise ValueError(
                    "required root obligations cannot rely on optional acceptance criteria"
                )
            if any(
                requirement.id
                not in next(c for c in spec.acceptance if c.id == cid).root_requirement_ids
                for cid in requirement.acceptance_ids
            ):
                raise ValueError("root requirement and criterion mappings disagree")
    else:
        ids = {r.id for r in root.spec.requirements}
        parent = state.nodes[node.parent_id].spec
        if not spec.root_requirement_ids or not set(spec.root_requirement_ids) <= ids:
            raise ValueError("child must map to approved root requirements")
        if not spec.parent_criterion_ids or not set(spec.parent_criterion_ids) <= {
            c.id for c in parent.acceptance
        }:
            raise ValueError("child must map to parent acceptance")
        if spec.requirements:
            raise ValueError("only root defines Requirement IDs")
        for ancestor in ancestors(state, task_id):
            for key, value in state.nodes[ancestor].spec.constraints.items():
                if spec.constraints.get(key) != value:
                    raise ValueError("child must inherit ancestor constraints")
    for item in spec.acceptance:
        if not item.root_requirement_ids or not set(item.root_requirement_ids) <= ids:
            raise ValueError("criterion needs valid root requirement IDs")
        if item.adopted_check_revision_id:
            adopted = state.check_definitions.get(item.adopted_check_revision_id)
            if not adopted or (
                adopted.task_id,
                adopted.criterion_id,
                adopted.semantic_revision,
            ) != (task_id, item.id, item.revision):
                raise ValueError("cannot borrow another task's formal check pointer")


def ancestors(state: RunState, task_id: str) -> list[str]:
    result = []
    parent = state.nodes[task_id].parent_id
    while parent:
        result.append(parent)
        parent = state.nodes[parent].parent_id
    return result


def validate_wait_graph(state: RunState) -> None:
    active = {key for key, node in state.nodes.items() if node.status != NodeStatus.SUPERSEDED}
    visiting, visited = set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError("active plan/dependency waiting cycle")
        if key in visited:
            return
        visiting.add(key)
        for target in state.children(key) + state.nodes[key].depends_on:
            if target not in active:
                raise ValueError("active dependency references a retired/missing task")
            visit(target)
        visiting.remove(key)
        visited.add(key)

    for key in active:
        visit(key)


def coverage_ok(state: RunState, task_id: str) -> bool:
    active = set(active_subtree(state, task_id))
    required = {r.id for r in state.nodes[state.root_task_id].spec.requirements if r.required}
    if task_id == state.root_task_id:
        for key in required:
            items = [
                c
                for c in state.coverage.values()
                if c.owner_task_id == task_id
                and c.kind == "REQUIREMENT"
                and c.root_requirement_ids == [key]
            ]
            if len(items) != 1 or items[0].status != "SUPPORTED":
                return False
    for item in state.coverage.values():
        if item.owner_task_id not in active:
            continue
        if (
            item.kind == "REQUIREMENT"
            and required.intersection(item.root_requirement_ids)
            and item.status == "NOT_REQUIRED"
        ):
            return False
        if item.status not in {"SUPPORTED", "NOT_REQUIRED"}:
            return False
        if item.status == "SUPPORTED":
            if not item.evidence_refs:
                return False
            for ref in item.evidence_refs:
                record = state.verifications.get(ref)
                if record is None or record.status != "PASS" or not record_current(state, record):
                    return False
        elif not item.rationale or not any(
            state.decisions[d].accepted for d in item.decision_ids if d in state.decisions
        ):
            return False
    return True


def certification_input(state: RunState, task_id: str) -> str:
    node = state.nodes[task_id]
    records = [latest_record(state, *ref) for ref in effective_criteria(state, task_id)]
    return digest(
        {
            "decision_policy": decision_policy_digest(),
            "decision_context_encoding": CONTEXT_ENCODING_VERSION,
            "preparation": state.current_preparation_id,
            "current_artifacts": {
                state.artifacts[key].path: file_hash(Path(state.artifacts[key].path))
                if Path(state.artifacts[key].path).is_file()
                else "MISSING"
                for task in active_subtree(state, task_id)
                for key in (
                    state.nodes[task].result_history[-1].artifact_ids
                    if state.nodes[task].result_history
                    else []
                )
            },
            "spec": node.spec,
            "result": node.result_revision,
            "plan": node.current_plan_revision_id,
            "children": {key: state.nodes[key].result_revision for key in state.children(task_id)},
            "coverage": coverage_view(state, task_id),
            "checks": [r.model_dump(mode="json") if r else None for r in records],
            "gaps": [g.model_dump(mode="json") for g in node.open_gaps],
        }
    )


def hard_complete(state: RunState, task_id: str) -> bool:
    if state.preparation_required:
        try:
            if not preparation_view(state):
                return False
        except (OSError, ValueError):
            return False
    if (
        not state.root_approval
        or state.root_approval.contract_hash != state.nodes[state.root_task_id].spec.contract_hash()
    ):
        return False
    node = state.nodes[task_id]
    if not any(
        state.rounds[result.round_id].kind in {RoundKind.EXECUTE, RoundKind.RECONCILE}
        for result in node.result_history
    ):
        return False
    if any(not certificate_valid(state, dep) for dep in node.depends_on):
        return False
    if node.open_gaps or not node.result_history or not coverage_ok(state, task_id):
        return False
    if any(
        state.nodes[key].status != NodeStatus.SUCCEEDED
        for key in active_subtree(state, task_id)[1:]
    ):
        return False
    if any(not certificate_valid(state, key) for key in state.children(task_id)):
        return False
    return all(
        (record := latest_record(state, *ref)) is not None and record.status == "PASS"
        for ref in effective_criteria(state, task_id)
    )


def certificate_valid(state: RunState, task_id: str) -> bool:
    if state.schema_version == 3:
        from .action_runtime import completion_current
        return completion_current(state, task_id)
    node = state.nodes[task_id]
    cert = state.completion_certificates.get(node.completion_certificate_id or "")
    return bool(
        cert
        and hard_complete(state, task_id)
        and cert.input_digest == certification_input(state, task_id)
    )


def install_plan(
    state: RunState,
    round_id: str,
    plan: PlanProposal,
    decision_ids: list[str],
    duplicate_sources: dict | None = None,
) -> PlanRevision:
    """Apply to a draft snapshot; caller commits the whole revision or discards it."""
    work = state.rounds[round_id]
    parent = state.nodes[work.task_id]
    previous = set(state.children(parent.id))
    retired = set(plan.supersedes_task_ids)
    if not retired <= previous or set(plan.retained_child_ids) != previous - retired:
        raise ValueError("plan must explicitly retain or replace every current child")
    if len(plan.candidate_ids) != len(set(plan.candidate_ids)):
        raise ValueError("duplicate plan member")
    if set(plan.replacement_map) != retired:
        raise ValueError("every superseded child needs an obligation transfer")
    for old in retired:
        for key in active_subtree(state, old):
            if state.nodes[key].status == NodeStatus.RUNNING or any(
                r.task_id == key and r.phase not in TERMINAL_ROUNDS | {RoundPhase.QUEUED}
                for r in state.rounds.values()
            ):
                raise ValueError("cannot retire unresolved execution")
        replacements = plan.replacement_map[old]
        if not replacements or not set(replacements) <= set(plan.candidate_ids) | set(
            plan.retained_child_ids
        ):
            raise ValueError(
                "replacement map must refer to selected tasks or verified retained results"
            )
        old_requirements = set(state.nodes[old].spec.root_requirement_ids)
        new_requirements = set()
        for key in replacements:
            if key in state.candidates:
                new_requirements.update(state.candidates[key].spec.root_requirement_ids)
            else:
                if not certificate_valid(state, key):
                    raise ValueError("retained replacement result must be currently verified")
                new_requirements.update(state.nodes[key].spec.root_requirement_ids)
        if not old_requirements <= new_requirements:
            raise ValueError("replacement loses a root obligation")
    mapping = {key: identifier("task") for key in plan.candidate_ids}
    for candidate_id, task_id in mapping.items():
        candidate = state.candidates[candidate_id]
        if (
            not set(candidate.capability_ids) <= set(plan.capability_ids)
            or not candidate.capability_ids
        ):
            raise ValueError("selected member lacks selected capability")
        if candidate.spec.goal.strip().casefold() in {
            state.nodes[a].spec.goal.strip().casefold()
            for a in [parent.id] + ancestors(state, parent.id)
        }:
            raise ValueError(
                "renaming/redelegating the unchanged ancestor goal is not decomposition"
            )
        node = TaskNode(
            id=task_id,
            parent_id=parent.id,
            depth=parent.depth + 1,
            created_seq=state.sequence(),
            created_by_round_id=round_id,
            specs=[candidate.spec.model_copy(deep=True)],
            depends_on=[mapping.get(dep, dep) for dep in candidate.depends_on],
            duplicate_of_task_id=(duplicate_sources or {}).get(candidate_id),
        )
        if node.duplicate_of_task_id and node.duplicate_of_task_id not in node.depends_on:
            node.depends_on.append(node.duplicate_of_task_id)
        state.nodes[task_id] = node
        validate_spec(state, task_id, node.spec)
        parent.child_ids.append(task_id)
        queue_round(state, task_id, RoundKind.EXPLORE, f"created:{round_id}:{candidate_id}")
    revision = PlanRevision(
        id=identifier("plan"),
        owner_task_id=parent.id,
        previous_revision_id=parent.current_plan_revision_id,
        source_round_id=round_id,
        capability_ids=plan.capability_ids,
        parent_capability_ids=plan.parent_capability_ids,
        selected_candidate_ids=plan.candidate_ids,
        required_child_ids=plan.retained_child_ids + list(mapping.values()),
        superseded_task_ids=list(retired),
        replacement_map={
            old: [mapping.get(k, k) for k in values] for old, values in plan.replacement_map.items()
        },
        reason=plan.rationale,
        decision_ids=decision_ids,
    )
    retired_all = set()
    for old in retired:
        retired_all.update(active_subtree(state, old))
    for key in retired_all:
        node = state.nodes[key]
        node.status = NodeStatus.SUPERSEDED
        node.supersession_plan_revision_id = revision.id
        for work in state.rounds.values():
            if work.task_id == key and work.phase == RoundPhase.QUEUED:
                work.phase, work.error = RoundPhase.CANCELLED, f"superseded by {revision.id}"
        for job in state.verification_jobs.values():
            if job.owner_task_id == key and job.phase == "QUEUED":
                job.phase = "CANCELLED"
    # A branch replacement transfers dependencies on its direct root. References to
    # internal retired nodes need an explicit parent replan rather than a guessed mapping.
    for node in state.nodes.values():
        if node.status == NodeStatus.SUPERSEDED:
            continue
        updated = []
        owned_scope = set(active_subtree(state, parent.id)) | set(mapping.values())
        for dep in node.depends_on:
            if dep in retired:
                if node.id not in owned_scope:
                    raise ValueError("common ancestor must own cross-branch dependency migration")
                updated.extend(revision.replacement_map[dep])
            elif dep in retired_all:
                raise ValueError("dependency on a retired internal task needs explicit migration")
            else:
                updated.append(dep)
        node.depends_on = list(dict.fromkeys(updated))
    state.plans[revision.id] = revision
    parent.current_plan_revision_id = revision.id
    validate_wait_graph(state)
    return revision


class TaskRuntime:
    protocol_version = 2

    def __init__(self, store: RunStore, worker, judge, command_exec: Callable):
        self.store, self.worker, self.judge, self.command_exec = store, worker, judge, command_exec
        self.state = store.load()
        validate_wait_graph(self.state)
        self._started = time.monotonic()
        self._charged_at = self._started
        self._was_active = self.state.status == RunStatus.ACTIVE
        checkpoint = self.state.progress.last_active_checkpoint
        if self._was_active and (self.state.scheduler.active_work or self.state.scheduler.active_siblings) and checkpoint:
            self.state.progress.active_elapsed_seconds += min(
                max(0, time.time() - checkpoint),
                self.state.execution_profile.worker_timeout_seconds,
            )

    @classmethod
    def create(
        cls,
        store: RunStore,
        request: str,
        workspace: Path,
        profile: ExecutionProfile,
        *,
        input_refs: list[str] | None = None,
    ) -> RunState:
        try:
            store.load()
        except FileNotFoundError:
            pass
        else:
            raise ValueError("run already exists; resume it rather than recreating it")
        workspace = workspace.expanduser().resolve(strict=True)
        if not workspace.is_dir() or not request.strip():
            raise ValueError("nonempty request and workspace directory required")
        root_id = identifier("task")
        state = RunState(
            schema_version=cls.protocol_version,
            run_id=store.run_id,
            user_request=request,
            root_task_id=root_id,
            workspace=workspace,
            execution_profile=profile,
            input_refs=input_refs or [],
            preparation_required=cls.protocol_version != 3,
        )
        state.nodes[root_id] = TaskNode(
            id=root_id, parent_id=None, depth=0, created_seq=state.sequence()
        )
        if cls.protocol_version == 3:
            root = state.nodes[root_id]
            root.specs = [
                TaskSpec(
                    title=" ".join(request.split())[:120],
                    goal=request,
                    requirements=[
                        Requirement(
                            id="root_goal",
                            description=request,
                            acceptance_ids=["root_goal"],
                        )
                    ],
                    root_requirement_ids=["root_goal"],
                    acceptance=[
                        AcceptanceCriterion(
                            id="root_goal",
                            description=(
                                "Answer every required part of the original user request with "
                                "actual evidence or checks, and explain any unresolved limits."
                            ),
                            check_kind="evidence",
                            root_requirement_ids=["root_goal"],
                        )
                    ],
                )
            ]
            validate_spec(state, root_id, root.spec)
            root.must_solve = [
                MustSolveItem(
                    problem=request,
                    verification_method="Compare the final deliverable with every required part of the original request and its actual evidence or checks.",
                    expected_output="A complete, evidence-backed answer to the original request with unresolved limits stated.",
                )
            ]
            root.root_contribution = request
            state.root_approval = RootApproval(
                approved_spec_revision=root.spec.revision,
                contract_hash=root.spec.contract_hash(),
                user_confirmation_ref="original user request at start",
            )
            queue_round(state, root_id, RoundKind.EXPLORE, "initial")
        else:
            queue_round(state, root_id, RoundKind.PREPARE, "initial")
        state.workspace_lease = store.acquire_workspace(workspace)
        try:
            store.save(state, None)
        except Exception:
            store.release_workspace()
            raise
        return state

    def save(self) -> None:
        now = time.monotonic()
        if self._was_active:
            self.state.progress.active_elapsed_seconds += now - self._charged_at
        self._was_active = self.state.status == RunStatus.ACTIVE
        self.state.progress.last_active_checkpoint = time.time() if self._was_active else None
        self._charged_at = now
        self.store.save(self.state, self.state.revision)

    def time_left(self) -> float:
        elapsed = self.state.progress.active_elapsed_seconds + (
            time.monotonic() - self._charged_at if self.state.status == RunStatus.ACTIVE else 0
        )
        return max(0.0, self.state.execution_profile.max_active_seconds - elapsed)

    def block(self, reason: str) -> None:
        self.state.status, self.state.block_reason = RunStatus.BLOCKED, reason
        self.save()

    def approve_root(self, contract_hash: str, confirmation_ref: str) -> None:
        if self.state.schema_version not in {2, 3}:
            raise ValueError("legacy task-tree protocol is inspection-only; start a fresh run")
        root = self.state.nodes[self.state.root_task_id]
        if (
            self.state.status != RunStatus.AWAITING_APPROVAL
            or root.spec.contract_hash() != contract_hash
        ):
            raise ValueError("approval must refer to the current proposed contract")
        if not confirmation_ref.strip():
            raise ValueError("real user confirmation reference required")
        if self.state.preparation_required and not preparation_view(self.state):
            raise ValueError("shared background preparation must finish before root approval")
        self.state.root_approval = RootApproval(
            approved_spec_revision=root.spec.revision,
            contract_hash=contract_hash,
            user_confirmation_ref=confirmation_ref,
        )
        root.status, self.state.status = NodeStatus.READY, RunStatus.ACTIVE
        queue_round(self.state, root.id, RoundKind.EXPLORE, f"approved:{contract_hash}")
        self._charged_at = time.monotonic()
        self.save()

    def accept_user_criterion(self, task_id: str, criterion_id: str, confirmation: str) -> None:
        if (
            self.state.scheduler.active_work
            or not self.state.root_approval
            or not confirmation.strip()
        ):
            raise ValueError(
                "user acceptance requires approved scope, stopped work and a real confirmation"
            )
        if (
            self.state.root_approval.contract_hash
            != self.state.nodes[self.state.root_task_id].spec.contract_hash()
        ):
            raise ValueError("root contract changed; reconfirmation is required")
        item = criterion(self.state, task_id, criterion_id)
        if item.check_kind != "user":
            raise ValueError("human acceptance cannot override a command/evidence check")
        key = identifier("user_acceptance")
        check = propose_check(
            self.state,
            self.store,
            task_id,
            key,
            CheckChangeProposal(
                id=key,
                criterion_id=criterion_id,
                old_check_revision_id=item.adopted_check_revision_id,
                check_spec=item.check_spec,
                reason="explicit human acceptance of fixed criterion",
            ),
        )
        if item.adopted_check_revision_id:
            self.state.check_definitions[item.adopted_check_revision_id].status = "RETIRED"
        check.status, check.review_refs = "ADOPTED", [confirmation]
        item.adopted_check_revision_id = check.id
        paths = verification_paths(self.state, task_id, criterion_id)
        proof = {
            "task_id": task_id,
            "criterion": item.model_dump(mode="json"),
            "confirmation": confirmation,
            "inputs": fingerprints(self.state.workspace, paths, self.state.input_refs),
        }
        output_ref = self.store.write_json(f"checks/{key}/user.json", proof)
        record = VerificationRecord(
            id=key,
            task_id=task_id,
            criterion_id=criterion_id,
            criterion_revision=item.revision,
            check_revision_id=check.id,
            check_bundle_hash=check.bundle_hash,
            status="PASS",
            method="user",
            input_fingerprints=proof["inputs"],
            input_paths=paths,
            execution_profile_digest=digest(self.state.execution_profile),
            output_ref=output_ref,
            source_ref=confirmation,
        )
        self.state.verifications[key] = record
        self.save()

    def decide(self, kind: str, target: str, payload: dict):
        if self.time_left() <= 0:
            raise TimeoutError("run time budget exhausted")
        active = self.state.scheduler.active_work
        waiting = next(
            (
                item
                for item in self.state.decision_jobs.values()
                if item.source_ref == active
                and item.kind == kind
                and item.target_id == target
                and item.status != "SUPERSEDED"
            ),
            None,
        )
        if (
            waiting
            and waiting.brief_ref
            and waiting.brief_fingerprints
            == fingerprints(
                self.state.workspace, list(waiting.brief_fingerprints), read_roots(self.state)
            )
        ):
            review = ContextReviewReport.model_validate_json(Path(waiting.brief_ref).read_text())
            brief = review.decision_brief.model_dump(mode="json")
            for claim in brief["claims"]:
                refs = list(claim["evidence_refs"])
                for ref in refs:
                    if ref not in self.state.artifacts and ref not in self.state.verifications:
                        path = input_path(self.state.workspace, ref, read_roots(self.state))
                        if path.is_file() and str(path) not in claim["evidence_refs"]:
                            claim["evidence_refs"].append(str(path))
            payload = {
                **payload,
                "decision_brief_override": {
                    "decision_brief": brief,
                    "review_ref": waiting.brief_ref,
                    "reviewed_sections": review.reviewed_sections,
                    "scope": "Independent read-only inspection for this predicate; not a completion claim.",
                },
            }
        try:
            decision = self.judge(kind, target, payload)
        except (DecisionContextTooLarge, DecisionWorkerUnavailable) as error:
            if active:
                work = (
                    self.state.rounds[active.id]
                    if active.kind == "ROUND"
                    else self.state.verification_jobs[active.id]
                )
                owner = work.task_id if active.kind == "ROUND" else work.owner_task_id
                job = (
                    waiting.model_copy(deep=True)
                    if waiting
                    else DecisionJob(
                        id=identifier("decision_job"),
                        owner_task_id=owner,
                        source_ref=active,
                        resume_phase=RoundPhase.DECIDING
                        if work.phase == RoundPhase.COMMITTED
                        else work.phase,
                        kind=kind,
                        target_id=target,
                        context_ref="",
                        prior_node_status=self.state.nodes[owner].status,
                    )
                )
                job.context_ref = self.store.write_json(
                    f"decision_jobs/{job.id}/context-{digest(payload)}.json", payload
                )
                job.status, job.error = "WAITING_CONTEXT", str(error)
                error.pending_job = job
            raise
        self.state.decisions[decision.id] = decision
        if waiting:
            waiting.status, waiting.decision_id = "RESOLVED", decision.id
        return decision

    def _park_decision(self, error: DecisionContextTooLarge) -> None:
        job = error.pending_job
        if job is None:
            self.block(str(error))
            raise error
        work = (
            self.state.rounds[job.source_ref.id]
            if job.source_ref.kind == "ROUND"
            else self.state.verification_jobs[job.source_ref.id]
        )
        if work.phase not in {
            RoundPhase.QUEUED,
            RoundPhase.PREPARING,
            RoundPhase.VERIFYING,
            RoundPhase.DECIDING,
            "RUNNING",
            "DECIDING",
        }:
            self.block("decision failure with unresolved execution: " + str(error))
            raise error
        work.phase, work.error = "WAITING_CONTEXT", str(error)
        node = self.state.nodes[job.owner_task_id]
        node.status, node.block_reason = NodeStatus.NEEDS_CONTEXT, str(error)
        self.state.decision_jobs[job.id] = job
        self.state.scheduler.active_work = None
        self.store.execution_reservation([])
        if isinstance(error, DecisionWorkerUnavailable):
            if job.source_ref.kind == "ROUND":
                work.phase, work.finished_at = RoundPhase.FAILED, time.time()
            else:
                work.phase = "ERROR"
            node.status = NodeStatus.BLOCKED
            self._queue_parent_diagnosis(node)
            self.save()
            return
        if job.repair_attempts < self.state.execution_profile.max_context_repairs:
            job.repair_attempts += 1
            queue_round(
                self.state,
                node.id,
                RoundKind.CONTEXT_REVIEW,
                f"context-repair:{job.id}:{job.repair_attempts}",
                purpose="prepare a concise independently inspected decision brief; " + str(error),
                decision_job_id=job.id,
            )
        self.save()

    def _resume_decision_work(self, ref: WorkRef) -> None:
        for job in self.state.decision_jobs.values():
            if job.source_ref == ref and job.status == "QUEUED":
                work = (
                    self.state.rounds[ref.id]
                    if ref.kind == "ROUND"
                    else self.state.verification_jobs[ref.id]
                )
                work.phase = job.resume_phase
                job.status = "RUNNING"
                self.save()
                return

    def latest_exploration(self, task_id: str) -> dict | None:
        ids = self.state.nodes[task_id].exploration_round_ids
        if not ids:
            return None
        work = self.state.rounds[ids[-1]]
        return {
            "report": json.loads(Path(work.worker_report_ref).read_text()),
            "decisions": [
                self.state.decisions[key].model_dump(mode="json") for key in work.decision_ids
            ],
        }

    def routing_basis(self, task_id: str) -> str:
        node = self.state.nodes[task_id]
        basis = {
                "spec": node.spec.contract_hash(),
                "plan": node.current_plan_revision_id,
                "children": {
                    key: self.state.nodes[key].result_revision
                    for key in self.state.children(task_id)
                },
                "dependencies": {
                    key: self.state.nodes[key].result_revision for key in node.depends_on
                },
                "gaps": node.open_gaps,
                "preparation": self.state.current_preparation_id,
                "input_refs": self.state.input_refs,
                "decision_policy": decision_policy_digest(),
                "decision_model": self.state.execution_profile.decision_model,
                "worker_model": self.state.execution_profile.worker_model,
                "worker_effort": self.state.execution_profile.worker_effort,
                "native_prompt_profile": self.state.execution_profile.native_prompt_profile,
            }
        return digest(basis)

    def authorization_current(self, work: TaskRound, *, check_files: bool = True) -> bool:
        source = self.state.rounds.get(work.authorization_round_id or "")
        node = self.state.nodes[work.task_id]
        if not (
            source
            and source.task_id == work.task_id
            and source.kind == RoundKind.EXPLORE
            and source.phase == RoundPhase.COMMITTED
            and node.exploration_round_ids
            and node.exploration_round_ids[-1] == source.id
            and source.routing_basis == self.routing_basis(node.id)
            and (
                not check_files
                or source.routing_fingerprints
                == fingerprints(
                    self.state.workspace, list(source.routing_fingerprints), read_roots(self.state)
                )
            )
        ):
            return False
        decisions = {
            self.state.decisions[key].kind: self.state.decisions[key] for key in source.decision_ids
        }
        review = self.state.rounds.get(source.scope_review_round_id or "")
        if source.scope_review_round_id and not (
            review
            and review.task_id == work.task_id
            and review.kind == RoundKind.SCOPE_REVIEW
            and review.phase == RoundPhase.COMMITTED
            and review.authorization_round_id == source.id
            and (
                not check_files
                or not review.routing_fingerprints
                or review.routing_fingerprints
                == fingerprints(
                    self.state.workspace, list(review.routing_fingerprints), read_roots(self.state)
                )
            )
        ):
            return False
        split = decisions.get("SHOULD_SPLIT")
        if work.kind == RoundKind.DECOMPOSE:
            return bool(split and split.accepted)
        if work.kind not in {RoundKind.EXECUTE, RoundKind.RECONCILE}:
            return False
        explored = ExplorationReport.model_validate_json(Path(source.worker_report_ref).read_text())
        reviewed = (
            ScopeReviewReport.model_validate_json(Path(review.worker_report_ref).read_text())
            if review
            else None
        )
        return bool(
            split
            and not split.accepted
            and not self.execution_blockers(node.id, explored, reviewed)
            and (
                not self.state.children(node.id)
                if work.kind == RoundKind.EXECUTE
                else all(certificate_valid(self.state, key) for key in self.state.children(node.id))
            )
        )

    def execution_blockers(
        self, task_id: str, explored: ExplorationReport, reviewed: ScopeReviewReport | None = None
    ) -> list[str]:
        node = self.state.nodes[task_id]
        blockers = [
            *explored.execution_blockers,
            *(reviewed.execution_blockers if reviewed else []),
        ]
        if not node.spec.deliverables or any(not path.strip() for path in node.spec.deliverables):
            blockers.append("task needs explicit deliverables before focused execution")
        if not explored.execution_assessment.strip():
            blockers.append("task needs a nonblank execution assessment")
        required_paths = (
            explored.required_input_paths
            if explored.required_input_paths is not None
            else explored.input_paths
        )
        # Remote source references are checked by the exploration worker.
        # Fingerprinting them as workspace paths would create false
        # MISSING blockers and contaminate the decision context.
        required_paths = [
            *[path for path in required_paths if not external_input_ref(path)],
            *local_inputs(self.state.input_refs),
        ]
        required_fingerprints = (
            fingerprints(self.state.workspace, required_paths, self.state.input_refs)
            if required_paths
            else {}
        )
        for path, value in required_fingerprints.items():
            if value == "MISSING":
                blockers.append(f"declared execution input is missing: {path}")
        if any(not certificate_valid(self.state, key) for key in self.state.children(task_id)):
            blockers.append("required child deliveries are not currently verified")
        return blockers

    def integration_basis(self, task_id: str) -> str:
        return digest(
            {
                "routing": self.routing_basis(task_id),
                "children": {
                    key: {
                        "certificate": self.state.nodes[key].completion_certificate_id,
                        "spec": self.state.nodes[key].spec.contract_hash(),
                    }
                    for key in self.state.children(task_id)
                },
            }
        )

    def _validate_read_only_report(self, work, report) -> None:
        node = self.state.nodes[work.task_id]
        if (report.task_id, report.round_id, report.spec_revision, report.input_digest) != (
            node.id,
            work.id,
            node.spec.revision,
            work.input_digest,
        ):
            raise ValueError("read-only report identity/spec/context mismatch")
        if work.kind != RoundKind.CONTEXT_REVIEW:
            self._validate_brief(work.task_id, getattr(report, "decision_brief", None))
        manifest = json.loads(Path(work.context_manifest_ref).read_text())
        if (
            fingerprints(
                self.state.workspace,
                list(manifest["context"]["input_fingerprints"]),
                self.state.input_refs,
            )
            != manifest["context"]["input_fingerprints"]
        ):
            raise ValueError("read-only round inputs changed; inspect actual side effects")

    def _validate_brief(self, task_id, brief, extra_criteria=()) -> None:
        if brief is None:
            return
        allowed = {item.id for item in self.state.nodes[task_id].spec.acceptance} | set(
            extra_criteria
        )
        ids = [item.id for item in brief.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("decision brief has duplicate claim IDs")
        for claim in brief.claims:
            if not set(claim.criterion_ids) <= allowed:
                raise ValueError(
                    f"decision brief refers to an unknown criterion; allowed IDs: {sorted(allowed)}"
                )
            for ref in claim.evidence_refs:
                if ref in self.state.artifacts or ref in self.state.verifications:
                    continue
                if not input_path(self.state.workspace, ref, read_roots(self.state)).is_file():
                    raise ValueError(
                        f"claim {claim.id!r} evidence_refs contains nonexistent file or unregistered ID {ref!r}; cite an actual workspace/authorized file or registered artifact/verification ID, not a command, criterion ID or task ID"
                    )

    def _check_gap_references(self, task_id: str, references: list[str]) -> list[str]:
        refs = [ref if ":" in ref else f"{task_id}:{ref}" for ref in references]
        allowed = {f"{owner}:{cid}" for owner, cid in effective_criteria(self.state, task_id)}
        if not set(refs) <= allowed:
            raise ValueError(
                f"check gap references a criterion outside this task's effective acceptance; allowed: {sorted(allowed)}"
            )
        return refs

    def _validate_aspect_assessments(
        self, assessments, opportunities, required_assessments=()
    ) -> None:
        prepared = preparation_view(self.state)
        inventory = (
            {entry["id"]: entry for entry in prepared["investigation_inventory"]}
            if prepared
            else {}
        )
        inventory.update(
            {item.aspect_id: {"subject": item.subject} for item in required_assessments}
        )
        ids = [item.aspect_id for item in assessments]
        if len(ids) != len(set(ids)) or not set(inventory) <= set(ids):
            raise ValueError("aspect assessments must uniquely account for every inventory aspect")
        for item in assessments:
            if any(
                not value.strip()
                for value in [item.aspect_id, item.subject, item.rationale, *item.evidence_refs]
            ):
                raise ValueError("aspect assessments need nonblank reasons and evidence")
            if item.aspect_id in inventory and item.subject != inventory[item.aspect_id]["subject"]:
                raise ValueError("inventory aspect subjects cannot be relabelled")
            if item.disposition == "SPLIT_OPPORTUNITY" and not (
                item.opportunity_index and item.opportunity_index <= len(opportunities)
            ):
                raise ValueError("split aspect must reference an actual 1-based opportunity_index")
            if item.opportunity_index and item.opportunity_index > len(opportunities):
                raise ValueError("aspect opportunity_index is outside the exploration")

    def _scope_source_current(self, work: TaskRound) -> bool:
        source = self.state.rounds.get(work.authorization_round_id or "")
        node = self.state.nodes[work.task_id]
        return bool(
            source
            and source.kind == RoundKind.EXPLORE
            and source.task_id == work.task_id
            and source.phase == RoundPhase.COMMITTED
            and node.exploration_round_ids
            and node.exploration_round_ids[-1] == source.id
            and source.routing_basis == self.routing_basis(node.id)
            and source.routing_fingerprints
            == fingerprints(
                self.state.workspace, list(source.routing_fingerprints), read_roots(self.state)
            )
        )

    def _commit_exploration(self, work: TaskRound, report: ExplorationReport) -> None:
        if work.phase == RoundPhase.COMMITTED:
            return
        self._validate_read_only_report(work, report)
        if any(
            not value.strip()
            for value in report.inspected_sources
            + report.findings
            + report.responsibilities
            + report.execution_blockers
            + report.input_paths
            + (report.required_input_paths or [])
        ):
            raise ValueError("exploration needs nonblank sources, findings and responsibilities")
        if not report.leaf_rationale.strip():
            raise ValueError("exploration needs a focus and information-gain leaf_rationale")
        if report.split_opportunities:
            raise ValueError("child planning requires a prior SHOULD_SPLIT authorization")
        self._validate_aspect_assessments(report.aspect_assessments, report.split_opportunities)
        before = self.state.model_copy(deep=True)
        work.routing_fingerprints = fingerprints(
            self.state.workspace,
            [
                *report.input_paths,
                *(report.required_input_paths or []),
                *local_inputs(self.state.input_refs),
            ],
            read_roots(self.state),
        )
        work.routing_basis = self.routing_basis(work.task_id)
        payload = {
            **self.decision_context(work.task_id),
            "exploration_draft": report.model_dump(mode="json"),
        }
        payload.pop("exploration", None)
        payload.pop("scope_review", None)
        evaluation = self.evaluate_node(work.task_id, "ROUTING", payload, explored=report)
        self._validate_read_only_report(work, report)
        if work.routing_basis != self.routing_basis(work.task_id) or work.routing_fingerprints != fingerprints(
            self.state.workspace, list(work.routing_fingerprints), read_roots(self.state)
        ):
            raise ValueError("exploration inputs changed during model routing")
        work.decision_ids = list(evaluation.decision_ids)
        work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
        self.state.nodes[work.task_id].exploration_round_ids.append(work.id)
        self.state.nodes[work.task_id].status = NodeStatus.READY
        self.state.scheduler.active_work = None
        queue_round(
            self.state,
            work.task_id,
            evaluation.next_kind,
            f"route:{work.id}",
            purpose=evaluation.purpose,
            authorization_round_id=work.id if evaluation.next_kind != RoundKind.EXPLORE else None,
        )
        if evaluation.next_kind == RoundKind.EXPLORE:
            self._record_progress(work.task_id, before)
        self.save()

    def _commit_scope_review(self, work: TaskRound, report: ScopeReviewReport) -> None:
        if work.phase == RoundPhase.COMMITTED:
            return
        self._validate_read_only_report(work, report)
        if not self._scope_source_current(work):
            raise ValueError("scope review source exploration changed")
        source = self.state.rounds[work.authorization_round_id]
        if report.reviewed_exploration_round_id != source.id:
            raise ValueError("scope review must identify the exact exploration round")
        explored = ExplorationReport.model_validate_json(Path(source.worker_report_ref).read_text())
        self._validate_aspect_assessments(
            report.aspect_assessments, explored.split_opportunities, explored.aspect_assessments
        )
        if any(
            not value.strip()
            for value in [
                report.summary,
                *report.evidence,
                *report.omissions,
                *report.required_followups,
                *report.execution_blockers,
                *report.input_paths,
            ]
        ):
            raise ValueError("scope review needs nonblank substantive evidence")
        before = self.state.model_copy(deep=True)
        node = self.state.nodes[work.task_id]
        work.routing_fingerprints = (
            fingerprints(self.state.workspace, report.input_paths, read_roots(self.state))
            if report.input_paths
            else {}
        )
        payload = {
            **self.decision_context(node.id),
            "exploration_draft": explored.model_dump(mode="json"),
            "scope_review": report.model_dump(mode="json"),
        }
        # The same committed report is already supplied as exploration_draft.
        payload.pop("exploration", None)
        evaluation = self.evaluate_node(
            node.id, "ROUTING", payload, explored=explored, reviewed=report
        )
        work.decision_ids = list(evaluation.decision_ids)
        self._validate_read_only_report(work, report)
        if not self._scope_source_current(work):
            raise ValueError("scope review source changed during decision")
        if report.input_paths and work.routing_fingerprints != fingerprints(
            self.state.workspace, report.input_paths, read_roots(self.state)
        ):
            raise ValueError("scope review inputs changed during decision")
        source.scope_review_round_id = work.id
        source.decision_ids = list(work.decision_ids)
        work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
        node.status = NodeStatus.READY
        self.state.scheduler.active_work = None
        queue_round(
            self.state,
            node.id,
            evaluation.next_kind,
            f"route:{work.id}",
            purpose=evaluation.purpose,
            authorization_round_id=source.id if evaluation.next_kind != RoundKind.EXPLORE else None,
        )
        if evaluation.next_kind == RoundKind.EXPLORE:
            self._record_progress(node.id, before)
        self.save()

    def _commit_decomposition(self, work: TaskRound, report: DecompositionReport) -> None:
        if work.phase == RoundPhase.COMMITTED:
            return
        self._validate_read_only_report(work, report)
        if not self.authorization_current(work):
            raise ValueError("decomposition requires a current SHOULD_SPLIT authorization")
        before = self.state.model_copy(deep=True)
        if not report.candidates:
            work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
            node = self.state.nodes[work.task_id]
            node.status = NodeStatus.READY
            self.state.scheduler.active_work = None
            queue_round(
                self.state,
                node.id,
                RoundKind.EXPLORE,
                f"no-valid-refinement:{work.id}",
                purpose="planning found no legitimate narrower work; reassess using the saved planning evidence: "
                + report.unsplittable_reason,
            )
            self._record_progress(node.id, before)
            self.save()
            return
        normalized = WorkerReport(**report.model_dump(mode="json", exclude={"unsplittable_reason"}))
        node = self.state.nodes[work.task_id]
        inherited = {}
        for key in reversed([node.id] + ancestors(self.state, node.id)):
            inherited.update(self.state.nodes[key].spec.constraints)
        for candidate in normalized.candidates:
            for key, value in inherited.items():
                if key in candidate.spec.constraints and candidate.spec.constraints[key] != value:
                    raise ValueError(
                        "candidate explicitly overrides an approved ancestor constraint"
                    )
                candidate.spec.constraints[key] = value
        try:
            for gap in normalized.gaps:
                if not gap.root_requirement_ids:
                    raise ValueError("decomposition gap must map to root requirements")
                if gap.id not in {item.id for item in node.open_gaps}:
                    node.open_gaps.append(gap)
            accepted = self._select_plan(normalized)
            self._validate_read_only_report(work, report)
            work.decision_ids = [key for key in self.state.decisions if key not in before.decisions]
            work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
            self.state.scheduler.active_work = None
            if accepted:
                node.status = NodeStatus.WAITING_CHILDREN
            else:
                node.status = NodeStatus.READY
                queue_round(
                    self.state,
                    node.id,
                    RoundKind.EXPLORE,
                    f"rejected-plan:{work.id}",
                    purpose="reassess rejected decomposition; preserve necessary obligations",
                )
            self._record_progress(node.id, before)
            self.save()
        except BaseException:
            # Ctrl-C can arrive while the model is judging proposed capabilities.
            # Restore the last committed state before the outer controller
            # records PAUSED, so uncommitted gaps cannot stale their own route.
            self.state = self.store.load()
            raise

    def _commit_report(self, work: TaskRound, report) -> None:
        if not isinstance(report, REPORT_MODELS[work.kind]):
            raise ValueError(f"wrong report schema for {work.kind.value}")
        if work.kind == RoundKind.CONTEXT_REVIEW:
            self._validate_read_only_report(work, report)
            job = self.state.decision_jobs.get(work.decision_job_id or "")
            if not job or report.decision_job_id != job.id or job.owner_task_id != work.task_id:
                raise ValueError("context review must identify its pending decision")
            allowed_criteria = {
                item.id
                for key in {work.task_id, job.target_id}
                if key in self.state.nodes
                for item in self.state.nodes[key].spec.acceptance
            }
            self._validate_brief(work.task_id, report.decision_brief, allowed_criteria)
            if not set(report.reviewed_sections) <= REVIEWABLE_SECTIONS:
                raise ValueError("context review cannot replace protected goals or actual checks")
            if report.reviewed_sections and not report.decision_brief.claims:
                raise ValueError("context section summaries need independently evidenced claims")
            job.brief_ref = work.worker_report_ref
            paths = {*report.input_paths, job.context_ref}
            for claim in report.decision_brief.claims:
                for ref in claim.evidence_refs:
                    if ref in self.state.artifacts:
                        paths.add(self.state.artifacts[ref].path)
                    elif ref in self.state.verifications:
                        record = self.state.verifications[ref]
                        paths.update(record.input_fingerprints)
                        if record.output_ref:
                            paths.add(record.output_ref)
                    else:
                        paths.add(ref)
            job.brief_fingerprints = fingerprints(
                self.state.workspace, list(paths), read_roots(self.state)
            )
            job.status = "QUEUED"
            node = self.state.nodes[work.task_id]
            node.status = (
                NodeStatus.SUCCEEDED
                if job.prior_node_status == NodeStatus.SUCCEEDED
                else NodeStatus.READY
            )
            node.block_reason = None
            work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
            self.state.scheduler.active_work = None
            self.save()
        elif work.kind == RoundKind.PREPARE:
            self._commit_preparation(work, report)
        elif work.kind == RoundKind.SPECIFY:
            self._commit_spec(work, report)
        elif work.kind == RoundKind.EXPLORE:
            self._commit_exploration(work, report)
        elif work.kind == RoundKind.SCOPE_REVIEW:
            self._commit_scope_review(work, report)
        elif work.kind == RoundKind.DECOMPOSE:
            self._commit_decomposition(work, report)
        elif work.kind == RoundKind.REVIEW:
            self._validate_read_only_report(work, report)
            self.apply_report(WorkerReport(**report.model_dump(mode="json")))
        else:
            self.apply_report(report)

    def decision_context(self, task_id: str, report: WorkerReport | None = None) -> dict:
        state = self.state
        node = state.nodes[task_id]
        task = {
            "id": node.id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "status": node.status.value,
            "spec": {"same_as_root": True}
            if task_id == state.root_task_id
            else spec_for_decision(node.spec),
            "current_result": node.result_history[-1].model_dump(mode="json")
            if node.result_history and not report
            else None,
            "open_gaps": [gap.model_dump(mode="json") for gap in node.open_gaps],
            "depends_on": node.depends_on,
        }
        return {
            "root": spec_for_decision(state.nodes[state.root_task_id].spec),
            "workspace_root": str(state.workspace),
            "inherited_root_criterion_ids": inherited_root_criteria(state, task_id),
            "task": task,
            "context_selection": {
                "historical_result_revisions": [
                    result.revision for result in node.result_history[:-1]
                ],
                "history_policy": "Current result, all task summaries, full coverage and current evidence are included. Superseded result bodies and duplicate root specification are omitted; original reports remain in the immutable run artifacts.",
                "check_policy": "Acceptance semantics and current verification results are included. Check implementations are independently reviewed in CHECK_CHANGE and retained in frozen bundles.",
            },
            "background_inventory": state.background,
            "preparation": preparation_view(state),
            "scope_review": scope_review_view(state, task_id),
            "last_decomposition": decomposition_view(state, task_id),
            "exploration": self.latest_exploration(task_id),
            "coverage": coverage_view(state, task_id),
            "active_plan": state.plans[state.nodes[task_id].current_plan_revision_id].model_dump(
                mode="json"
            )
            if state.nodes[task_id].current_plan_revision_id
            else None,
            "task_catalog": [
                {
                    "id": n.id,
                    "parent_id": n.parent_id,
                    "status": n.status.value,
                    "title": n.spec.title,
                    "question": n.spec.goal,
                    "scope": n.spec.scope,
                }
                for n in state.nodes.values()
                if n.specs
            ],
            "verification": [
                {
                    **r.model_dump(mode="json"),
                    "actual_output": Path(r.output_ref).read_text() if r.output_ref else "",
                }
                for ref in effective_criteria(state, task_id)
                if (r := latest_record(state, *ref))
            ],
            "artifacts": [
                {
                    **state.artifacts[key].model_dump(mode="json"),
                    "current_workspace_hash": file_hash(Path(state.artifacts[key].path))
                    if Path(state.artifacts[key].path).is_file()
                    else "MISSING",
                }
                for task in active_subtree(state, task_id)
                for key in (
                    state.nodes[task].result_history[-1].artifact_ids
                    if state.nodes[task].result_history
                    else []
                )
            ],
            "result_draft": {
                key: report.model_dump(mode="json")[key]
                for key in (
                    "task_id",
                    "round_id",
                    "summary",
                    "deliverable_paths",
                    "evidence",
                    "completion_claim",
                    "decision_brief",
                )
            }
            if report
            else None,
        }

    def claim_next_work(self) -> WorkRef | None:
        while True:
            choice = select_next(self.state)
            if choice.work_ref is None:
                return None
            ref = choice.work_ref
            if ref.kind == "ROUND":
                work = self.state.rounds[ref.id]
                requires_route = work.kind in {RoundKind.DECOMPOSE, RoundKind.EXECUTE} or (
                    work.kind == RoundKind.RECONCILE and bool(work.authorization_round_id)
                )
                if requires_route and not self.authorization_current(work):
                    work.phase, work.error = RoundPhase.CANCELLED, "routing authorization is stale"
                    work.finished_at = time.time()
                    node = self.state.nodes[work.task_id]
                    node.status = NodeStatus.READY
                    queue_round(
                        self.state,
                        node.id,
                        RoundKind.EXPLORE,
                        f"stale-authorization:{work.id}",
                        purpose="reassess task under the current decision policy and inputs",
                    )
                    self.save()
                    continue
            break
        self.state.scheduler.wave = choice.wave_after
        self.state.scheduler.active_work = ref
        if ref.kind == "ROUND":
            work = self.state.rounds[ref.id]
            if work.phase == RoundPhase.WAITING_CONTEXT:
                self._resume_decision_work(ref)
            else:
                work.phase, work.started_at = RoundPhase.PREPARING, time.time()
            self.state.nodes[work.task_id].status = NodeStatus.RUNNING
        else:
            if self.state.verification_jobs[ref.id].phase == "WAITING_CONTEXT":
                self._resume_decision_work(ref)
            else:
                self.state.verification_jobs[ref.id].phase = "RUNNING"
            if hasattr(self.worker, "execution_owner"):
                self.state.verification_jobs[ref.id].execution_owner = self.worker.execution_owner()
        self.save()
        return ref

    def queue_verification(self, task_id: str) -> None:
        refs = [f"{t}:{c}" for t, c in effective_criteria(self.state, task_id)]
        input_digest = digest(
            [
                refs,
                fingerprints(
                    self.state.workspace,
                    [".", *local_inputs(self.state.input_refs)],
                    self.state.input_refs,
                ),
            ]
        )
        if any(
            j.owner_task_id == task_id
            and j.input_digest == input_digest
            and j.phase in {"QUEUED", "RUNNING", "DECIDING"}
            for j in self.state.verification_jobs.values()
        ):
            return
        job = VerificationJob(
            id=identifier("verify"),
            created_seq=self.state.sequence(),
            owner_task_id=task_id,
            criterion_refs=refs,
            input_digest=input_digest,
        )
        self.state.verification_jobs[job.id] = job

    def refresh_parents(self) -> None:
        for node in sorted(self.state.nodes.values(), key=lambda n: -n.depth):
            if (
                node.status in {NodeStatus.SUPERSEDED, NodeStatus.RUNNING}
                or (
                    node.status == NodeStatus.BLOCKED
                    and node.block_reason != "node made no substantive progress"
                )
                or not self.state.children(node.id)
            ):
                continue
            if any(
                self.state.nodes[key].status != NodeStatus.SUCCEEDED
                for key in self.state.children(node.id)
            ):
                continue
            current = digest(node_input(self.state, node.id))
            if current == node.last_reconciled_input_digest:
                continue
            if any(
                r.task_id == node.id and r.phase not in TERMINAL_ROUNDS
                for r in self.state.rounds.values()
            ):
                continue
            node.status = NodeStatus.READY
            node.block_reason = None
            previous_children = (
                node.result_history[-1].child_result_versions if node.result_history else {}
            )
            if any(
                self.state.nodes[key].result_revision > previous_children.get(key, 0)
                for key in self.state.children(node.id)
            ):
                # Actual new child delivery gives the parent a fresh integration attempt.
                node.no_progress_rounds = 0
            queue_round(
                self.state,
                node.id,
                RoundKind.RECONCILE,
                f"reconcile:{node.current_plan_revision_id}:{current}",
            )

    def recover_active(self) -> None:
        ref = self.state.scheduler.active_work
        if not ref:
            return
        if ref.kind == "VERIFICATION":
            job = self.state.verification_jobs[ref.id]
            if job.execution_owner:
                self.worker.ensure_stopped(job.execution_owner)
                self.store.execution_reservation([])
                job.execution_owner = self.worker.execution_owner()
                self.save()
            return
        work = self.state.rounds[ref.id]
        if work.phase != RoundPhase.WORKER_RUNNING:
            return
        if not hasattr(self.worker, "recover"):
            raise RuntimeError(
                "recover and confirm the old worker stopped before another invocation"
            )
        model = REPORT_MODELS[work.kind]
        report = self.worker.recover(work, model)
        self.store.execution_reservation([])
        if report is not None:
            work.worker_report_ref = self.store.write_json(
                f"reports/{work.id}/worker.json", report.model_dump(mode="json")
            )
            work.phase = RoundPhase.VERIFYING
            self.save()
            return
        work.phase, work.finished_at, work.error = (
            RoundPhase.INTERRUPTED,
            time.time(),
            "executor confirmed stopped; inspect actual side effects in new worker",
        )
        self.state.scheduler.active_work = None
        node = self.state.nodes[work.task_id]
        node.status = NodeStatus.READY
        new = queue_round(
            self.state,
            work.task_id,
            work.kind,
            f"recover:{work.id}",
            purpose="inspect_side_effects_then_" + work.purpose,
            decision_job_id=work.decision_job_id,
        )
        new.retry_of_round_id, new.attempt = work.id, work.attempt + 1
        new.authorization_round_id = work.authorization_round_id
        self.save()

    def begin_preparation(self) -> None:
        if self.state.schema_version not in {2, 3}:
            raise ValueError("legacy task-tree protocol is inspection-only; start a fresh run")
        if self.state.root_approval or self.state.scheduler.active_work:
            raise ValueError("preparation restart requires an unapproved, idle run")
        for work in self.state.rounds.values():
            if work.phase == RoundPhase.QUEUED:
                work.phase, work.error = RoundPhase.CANCELLED, "replaced by background preparation"
        self.state.preparation_required = True
        self.state.current_preparation_id = None
        self.state.status, self.state.block_reason = RunStatus.ACTIVE, None
        self.state.nodes[self.state.root_task_id].status = NodeStatus.READY
        queue_round(
            self.state, self.state.root_task_id, RoundKind.PREPARE, f"prepare:{self.state.revision}"
        )
        self.save()

    def _commit_preparation(self, work: TaskRound, report: PreparationReport) -> None:
        if (report.task_id, report.round_id, report.input_digest) != (
            work.task_id,
            work.id,
            work.input_digest,
        ):
            raise ValueError("preparation report identity/context mismatch")
        if work.phase == RoundPhase.COMMITTED:
            return
        if "background/index.json" not in report.artifact_paths:
            raise ValueError(
                "preparation artifact_paths must use workspace-relative paths and include 'background/index.json'; absolute paths are not accepted"
            )
        if any(not source.strip() for source in report.inspected_sources):
            raise ValueError("preparation inspected_sources cannot contain blank values")
        for relative in report.artifact_paths:
            if (
                not Path(relative).parts
                or Path(relative).is_absolute()
                or Path(relative).parts[0] != "background"
            ):
                raise ValueError("preparation deliverables must stay under background/")
            if not workspace_path(self.state.workspace, relative).is_file():
                raise ValueError(f"preparation deliverable does not exist: {relative}")
        index = json.loads(
            workspace_path(self.state.workspace, "background/index.json").read_text()
        )
        if (
            not isinstance(index, dict)
            or not isinstance(index.get("sources"), list)
            or not index["sources"]
        ):
            raise ValueError("background index must describe the inspected sources")
        parse_investigation_inventory(index, required=True)
        artifacts, index_id = [], None
        for relative in dict.fromkeys(report.artifact_paths):
            path = workspace_path(self.state.workspace, relative)
            sha, snapshot = self.store.snapshot(path)
            artifact = ArtifactRef(
                id=identifier("artifact"),
                path=str(path),
                sha256=sha,
                snapshot_ref=snapshot,
                task_id=work.task_id,
                round_id=work.id,
            )
            self.state.artifacts[artifact.id] = artifact
            artifacts.append(artifact.id)
            if relative == "background/index.json":
                index_id = artifact.id
        record = PreparationRecord(
            round_id=work.id,
            summary=report.summary,
            artifact_ids=artifacts,
            index_artifact_id=index_id,
            inspected_sources=report.inspected_sources,
            facts=report.facts,
            limitations=report.limitations,
        )
        self.state.preparations[work.id] = record
        self.state.current_preparation_id = work.id
        work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
        self.state.scheduler.active_work = None
        self.state.nodes[work.task_id].status = NodeStatus.READY
        queue_round(self.state, work.task_id, RoundKind.SPECIFY, f"prepared:{work.id}")
        self.save()

    def _commit_spec(self, work: TaskRound, report: SpecProposal) -> None:
        if (report.task_id, report.round_id) != (work.task_id, work.id):
            raise ValueError("specification proposal identity mismatch")
        validate_spec(self.state, work.task_id, report.proposed_spec)
        root = self.state.nodes[work.task_id]
        if self.state.preparation_required and not preparation_view(self.state):
            raise ValueError("specification requires completed shared background preparation")
        if report.proposed_spec.revision != (root.spec.revision + 1 if root.specs else 1):
            raise ValueError("specification revision must follow the existing proposal")
        self.state.coverage.clear()
        root.coverage_ids.clear()
        root.specs.append(report.proposed_spec)
        for requirement in root.spec.requirements:
            item = CoverageItem(
                id=f"requirement:{requirement.id}",
                kind="REQUIREMENT",
                owner_task_id=root.id,
                root_requirement_ids=[requirement.id],
                subject=requirement.description,
                discovery_source_refs=[work.worker_report_ref],
            )
            self.state.coverage[item.id] = item
            root.coverage_ids.append(item.id)
        root.status, self.state.status = NodeStatus.AWAITING_APPROVAL, RunStatus.AWAITING_APPROVAL
        work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
        self.state.scheduler.active_work = None
        self.save()

    def verify(
        self, task_id: str, source_ref: str, extra_paths: list[str] | None = None
    ) -> list[str]:
        ids = []
        for owner, item_id in effective_criteria(self.state, task_id):
            if self.time_left() <= 0:
                raise TimeoutError("run time budget exhausted")
            record = latest_record(self.state, owner, item_id)
            paths = verification_paths(
                self.state, owner, item_id, extra_paths if owner == task_id else None
            )
            if (
                not record
                or record.status != "PASS"
                or ("." not in record.input_paths and not set(paths) <= set(record.input_paths))
            ):
                record = verify_one(
                    self.state,
                    self.store,
                    owner,
                    item_id,
                    source_ref,
                    lambda argv, cwd, timeout, env: self._checked_command(
                        argv, cwd, timeout, source_ref, env
                    ),
                    self.decide,
                    self.time_left,
                    extra_paths if owner == task_id else None,
                )
                self.state.verifications[record.id] = record
                self.save()
            ids.append(record.id)
        return ids

    def _checked_command(self, argv, cwd, timeout, source_ref, env):
        self.store.execution_reservation([source_ref])
        result = self.command_exec(argv, cwd, timeout, env)
        self.store.execution_reservation([])
        return result

    def refresh_coverage_refs(self) -> None:
        for item in self.state.coverage.values():
            if item.status == "SUPPORTED":
                refs = []
                for key in item.evidence_refs:
                    old = self.state.verifications.get(key)
                    new = latest_record(self.state, old.task_id, old.criterion_id) if old else None
                    refs.append(new.id if new and new.status == "PASS" else key)
                item.evidence_refs = refs

    def completion_context(self, task_id: str, payload: dict) -> dict:
        # The result has already been committed; a REVIEW draft may still say checks are pending.
        payload = {key: value for key, value in payload.items() if key != "result_draft"}
        worker_results = []
        for key in active_subtree(self.state, task_id):
            node = self.state.nodes[key]
            latest = node.result_history[-1]
            delivered = next(
                result
                for result in reversed(node.result_history)
                if self.state.rounds[result.round_id].kind
                in {RoundKind.EXECUTE, RoundKind.RECONCILE}
            )
            work = self.state.rounds[delivered.round_id]
            latest_work = self.state.rounds[latest.round_id]
            worker_results.append(
                {
                    "task_id": key,
                    "parent_id": node.parent_id,
                    "spec": spec_for_decision(node.spec),
                    "current_result": latest.model_dump(mode="json"),
                    "delivery_kind": work.kind.value,
                    "delivery_report": json.loads(Path(work.worker_report_ref).read_text()),
                    "latest_round_report": json.loads(
                        Path(latest_work.worker_report_ref).read_text()
                    )
                    if latest.round_id != delivered.round_id
                    else None,
                }
            )
        materials = []
        for artifact in payload["artifacts"]:
            if (
                artifact["snapshot_ref"]
                and file_hash(Path(artifact["snapshot_ref"])) != artifact["sha256"]
            ):
                raise RuntimeError("delivered artifact snapshot changed; inspect before completion")
            if artifact["task_id"] != task_id:
                materials.append(
                    {
                        "artifact_id": artifact["id"],
                        "task_id": artifact["task_id"],
                        "path": artifact["path"],
                        "delivered_sha256": artifact["sha256"],
                        "current_content": {
                            "sha256": artifact["current_workspace_hash"],
                            "content": "Descendant contents are not supplied here. Use the actual direct-child results and their current scoped completion proofs; full artifacts remain in the evidence store.",
                        },
                    }
                )
                continue
            sha, snapshot = self.store.snapshot(Path(artifact["path"]))
            if sha != artifact["current_workspace_hash"]:
                raise RuntimeError("completion material changed; reverify current inputs")
            material = review_file(Path(snapshot))
            if material["sha256"] != sha:
                raise RuntimeError("completion material snapshot changed")
            reviews = []
            for record in payload["verification"]:
                if record["status"] == "PASS":
                    definition = self.state.check_definitions.get(record["check_revision_id"])
                    relative = Path(artifact["path"]).relative_to(self.state.workspace)
                    for ref in definition.review_refs if definition else []:
                        accepted = self.state.decisions.get(ref)
                        if not (
                            accepted
                            and accepted.kind == "CHECK_CHANGE"
                            and accepted.accepted
                            and accepted.target_id == definition.id
                            and accepted.request_ref
                        ):
                            continue
                        frozen = saved_decision_context(accepted).get("frozen_files", {}).get(
                            "files/" + str(relative), {}
                        )
                        frozen_path = Path(definition.bundle_ref) / "files" / relative
                        if (
                            isinstance(material.get("text"), str)
                            and frozen.get("sha256") == sha
                            and frozen.get("text") == material["text"]
                            and frozen_path.is_file()
                            and file_hash(frozen_path) == sha
                        ):
                            reviews.append(
                                {
                                    "criterion_id": record["criterion_id"],
                                    "verification_id": record["id"],
                                    "decision_id": accepted.id,
                                    "score": accepted.score,
                                    "scope": "This exact frozen check implementation was independently accepted for this criterion and the actual check passed; this supports only the check's stated scope, not every property of the product.",
                                }
                            )
                if record["status"] != "PASS" or record["method"] not in {"model", "command+model"}:
                    continue
                decision_id = record["actual_output"].strip().splitlines()[-1]
                decision = self.state.decisions.get(decision_id)
                if (
                    not decision
                    or decision.kind != "EVIDENCE_SUPPORTED"
                    or not decision.accepted
                    or not decision.request_ref
                ):
                    continue
                checked = saved_decision_context(decision).get("review_material", {})
                if not any(
                    isinstance(text, str)
                    and text == material.get("text")
                    and input_path(self.state.workspace, path, read_roots(self.state))
                    == Path(artifact["path"])
                    and record["input_fingerprints"].get(path) == sha
                    for path, text in checked.items()
                ):
                    continue
                reviews.append(
                    {
                        "criterion_id": record["criterion_id"],
                        "description": criterion(
                            self.state, record["task_id"], record["criterion_id"]
                        ).description,
                        "verification_id": record["id"],
                        "method": record["method"],
                        "actual_output": record["actual_output"],
                        "decision_id": decision.id,
                        "score": decision.score,
                        "scope": "This content was actual material for this criterion only; this is not a guarantee of every property of the artifact.",
                    }
                )
            materials.append(
                {
                    "artifact_id": artifact["id"],
                    "task_id": artifact["task_id"],
                    "path": artifact["path"],
                    "delivered_sha256": artifact["sha256"],
                    "current_snapshot_ref": snapshot,
                    "current_content": material,
                    "evidence_reviews": reviews,
                }
            )
        return {
            **payload,
            "workspace_root": str(self.state.workspace),
            "inherited_root_criterion_ids": inherited_root_criteria(self.state, task_id),
            "worker_results": worker_results,
            "artifact_contents": materials,
            "child_completion_proofs": [
                {
                    "task_id": key,
                    "currently_valid": certificate_valid(self.state, key),
                    "certificate": self.state.completion_certificates[
                        self.state.nodes[key].completion_certificate_id
                    ].model_dump(mode="json")
                    if self.state.nodes[key].completion_certificate_id
                    else None,
                }
                for key in self.state.children(task_id)
            ],
            "completion_material_policy": "Current task and required descendants include their actual delivery reports, latest result/review, current verification outputs and current artifact contents. Text is included in full from stable snapshots. Binary contents are explicitly unread and require behavioral or independently reviewed evidence. References alone do not establish completion; changed-but-reverified files are distinguished from the original delivered hash.",
        }

    def evaluate_node(
        self,
        task_id: str,
        stage: Literal["ROUTING", "DELIVERY"],
        payload: dict,
        *,
        explored: ExplorationReport | None = None,
        reviewed: ScopeReviewReport | None = None,
        pending_checks: list[str] | None = None,
    ) -> NodeEvaluation:
        """One lifecycle entry; each model call still evaluates one proposition.

        DELIVERY follows actual verification/integration. Failure returns to
        investigation; only a fresh ROUTING decision may authorize decomposition.
        """
        if stage == "ROUTING":
            if explored is None:
                raise ValueError("routing needs exploration")
            blockers = self.execution_blockers(task_id, explored, reviewed)
            split = self.decide(
                "SHOULD_SPLIT",
                task_id,
                {**payload, "controller_facts": {"focused_execution_blockers": blockers}},
            )
            decisions = (split.id,)
            if split.accepted:
                return NodeEvaluation(
                    next_kind=RoundKind.DECOMPOSE,
                    purpose="allocate the whole scope into focused children",
                    decision_ids=decisions,
                )
            if blockers:
                return NodeEvaluation(
                    next_kind=RoundKind.EXPLORE,
                    purpose="resolve focused execution blockers: " + "; ".join(blockers),
                    decision_ids=decisions,
                )
            return NodeEvaluation(
                next_kind=RoundKind.RECONCILE
                if self.state.children(task_id)
                else RoundKind.EXECUTE,
                purpose="execute only the scoped responsibility after the judge declined further splitting",
                decision_ids=decisions,
            )
        if stage != "DELIVERY":
            raise ValueError("unknown node evaluation stage")
        if pending_checks:
            return NodeEvaluation(
                next_kind=RoundKind.REVIEW,
                purpose="review_checks:" + ",".join(pending_checks),
            )
        if self.publish_certificate(task_id, payload):
            return NodeEvaluation(completed=True)
        if any(
            self.state.nodes[key].status != NodeStatus.SUCCEEDED
            for key in self.state.children(task_id)
        ):
            return NodeEvaluation(waiting_children=True)
        return NodeEvaluation(
            next_kind=RoundKind.EXPLORE,
            purpose="repair_or_replan: inspect actual failures, unmet goals or missing evidence; do not assume another split is necessary",
        )

    def publish_certificate(self, task_id: str, payload: dict) -> bool:
        if not hard_complete(self.state, task_id):
            return False
        payload = self.completion_context(task_id, payload)
        payload["controller_facts"] = {
            "actual_required_checks_passed": True,
            "necessary_children_have_current_certificates": True,
            "registered_required_coverage_supported": True,
            "explicit_open_gaps_remain": False,
            "scope": "These are code-checked preconditions; semantic fulfillment of the goal is the single remaining judgment.",
        }
        complete = self.decide("COMPLETE", task_id, payload)
        if not complete.accepted:
            return False
        # Re-read actual inputs after potentially slow model calls.
        if not hard_complete(self.state, task_id):
            return False
        node = self.state.nodes[task_id]
        cert = CompletionCertificate(
            id=identifier("certificate"),
            task_id=task_id,
            result_revision=node.result_revision,
            spec_revision=node.spec.revision,
            plan_revision_id=node.current_plan_revision_id,
            verification_ids=[
                latest_record(self.state, *ref).id
                for ref in effective_criteria(self.state, task_id)
            ],
            coverage_digest=digest(coverage_view(self.state, task_id)),
            input_digest=certification_input(self.state, task_id),
            decision_ids=[complete.id],
        )
        self.state.completion_certificates[cert.id] = cert
        node.completion_certificate_id, node.status = cert.id, NodeStatus.SUCCEEDED
        node.last_reconciled_input_digest = digest(node_input(self.state, task_id))
        return True

    def _apply_coverage(self, report: WorkerReport) -> None:
        root_requirements = {
            r.id: r for r in self.state.nodes[self.state.root_task_id].spec.requirements
        }
        for proposed in report.coverage_updates:
            item = proposed.model_copy(deep=True)
            old = self.state.coverage.get(item.id)
            if (
                item.owner_task_id != report.task_id
                or not item.root_requirement_ids
                or not set(item.root_requirement_ids) <= set(root_requirements)
            ):
                raise ValueError("coverage has invalid ownership or root mapping")
            if old and (item.kind, item.subject, item.root_requirement_ids) != (
                old.kind,
                old.subject,
                old.root_requirement_ids,
            ):
                raise ValueError("cannot rewrite a coverage obligation")
            if item.kind == "REQUIREMENT" and (
                not old
                or item.status == "NOT_REQUIRED"
                and any(root_requirements[r].required for r in item.root_requirement_ids)
            ):
                raise ValueError("root requirement coverage cannot be invented or excluded")
            if item.kind == "ASPECT" and not item.discovery_source_refs:
                raise ValueError("new aspect needs an inventory/discovery source")
            refs, supplementary = [], []
            for ref in item.evidence_refs:
                if item.status != "SUPPORTED":
                    refs.append(ref)
                    continue
                record = self.state.verifications.get(ref)
                if record is not None and not record_current(self.state, record):
                    record = latest_record(self.state, record.task_id, record.criterion_id)
                if record is None:
                    owner, cid = ref.split(":", 1) if ":" in ref else (report.task_id, ref)
                    try:
                        record = latest_record(self.state, owner, cid)
                    except (KeyError, StopIteration):
                        record = None
                if not record or record.status != "PASS" or not record_current(self.state, record):
                    supplementary.append(ref)
                else:
                    refs.append(record.id)
            if item.status == "SUPPORTED":
                supported_roots = {
                    root
                    for key in refs
                    for root in criterion(
                        self.state,
                        self.state.verifications[key].task_id,
                        self.state.verifications[key].criterion_id,
                    ).root_requirement_ids
                }
                if not refs or not set(item.root_requirement_ids) <= supported_roots:
                    item.status = "IN_PROGRESS"
                    item.rationale += (
                        "; missing current formal support for the declared requirement IDs"
                    )
                if supplementary:
                    item.rationale += "; supplementary citations remain in the raw report and are not formal proof"
            item.evidence_refs = list(dict.fromkeys(refs))
            item.decision_ids = []
            if item.status == "NOT_REQUIRED":
                if not item.rationale:
                    raise ValueError("excluding an aspect needs a reason")
                decision = self.decide(
                    "COVERAGE",
                    item.id,
                    {
                        **self.decision_context(report.task_id, report),
                        "proposed_exclusion": item.model_dump(mode="json"),
                    },
                )
                item.decision_ids = [decision.id]
                if not decision.accepted:
                    item.status = "UNASSESSED"
            item.revision = old.revision + 1 if old else 1
            self.state.coverage[item.id] = item
            if item.id not in self.state.nodes[report.task_id].coverage_ids:
                self.state.nodes[report.task_id].coverage_ids.append(item.id)

    def _select_plan(self, report: WorkerReport) -> bool:
        node = self.state.nodes[report.task_id]
        payload = self.decision_context(report.task_id, report)
        needed = set()
        allocated = set()
        root_ids = {r.id for r in self.state.nodes[self.state.root_task_id].spec.requirements}
        for capability in report.capability_proposals:
            if (
                capability.owner_task_id != report.task_id
                or not capability.root_requirement_ids
                or not set(capability.root_requirement_ids) <= root_ids
            ):
                raise ValueError("capability has invalid owner or root mapping")
            decision = self.decide(
                "CAPABILITY_NEEDED",
                capability.id,
                {**payload, "capability": capability.model_dump(mode="json")},
            )
            capability.necessity_decision_id = decision.id
            self.state.capabilities[capability.id] = capability
            if decision.accepted:
                needed.add(capability.id)
        for candidate in report.candidates:
            if (
                candidate.id in self.state.candidates
                and self.state.candidates[candidate.id] != candidate
            ):
                raise ValueError("candidate ID reused for different content")
            self.state.candidates[candidate.id] = candidate
        recommended = [p for p in report.plan_proposals if p.recommended]
        if len(recommended) > 1:
            raise ValueError("recommend exactly one complete plan")
        chosen = recommended[0] if recommended else None
        signatures = (
            [duplicate_signature(self.state.candidates[key].spec) for key in chosen.candidate_ids]
            if chosen
            else []
        )
        if len(signatures) != len(set(signatures)):
            raise ValueError(
                "selected plan contains identical scoped tasks; merge them and preserve their obligations"
            )
        accepted = False
        if chosen and needed:
            # A plan is proposed before the judge rules on each responsibility. Keep
            # the necessary portion when it declines an optional capability.
            chosen.capability_ids = [key for key in chosen.capability_ids if key in needed]
            chosen.parent_capability_ids = [
                key for key in chosen.parent_capability_ids if key in needed
            ]
            # Parent assignments are explicit obligations even when omitted from the redundant list.
            chosen.capability_ids = list(
                dict.fromkeys(chosen.capability_ids + chosen.parent_capability_ids)
            )
            if not needed <= set(chosen.capability_ids):
                raise ValueError("recommended plan omits a necessary capability")
            if any(key not in self.state.candidates for key in chosen.candidate_ids):
                raise ValueError("unknown plan candidate")
            member_capabilities = {
                cap
                for key in chosen.candidate_ids
                for cap in self.state.candidates[key].capability_ids
            }
            retained_capabilities = set()
            for retained in chosen.retained_child_ids:
                if retained in self.state.nodes and certificate_valid(self.state, retained):
                    requirements = set(self.state.nodes[retained].spec.root_requirement_ids)
                    retained_capabilities.update(
                        key
                        for key in chosen.capability_ids
                        if key in self.state.capabilities
                        and set(self.state.capabilities[key].root_requirement_ids) <= requirements
                    )
            member_capabilities.update(retained_capabilities)
            if not set(chosen.parent_capability_ids) <= needed:
                raise ValueError(
                    "parent capability assignment must refer to a necessary capability"
                )
            if not needed <= member_capabilities | set(chosen.parent_capability_ids):
                raise ValueError("plan members do not cover every necessary capability")
            necessary_capabilities = {
                key
                for key, cap in self.state.capabilities.items()
                if cap.necessity_decision_id
                and self.state.decisions[cap.necessity_decision_id].accepted
            }
            if not set(chosen.capability_ids) <= necessary_capabilities:
                raise ValueError("plan contains a capability not judged necessary")
            groups = [
                self.state.candidates[key].alternative_group_id
                for key in chosen.candidate_ids
                if self.state.candidates[key].alternative_group_id
            ]
            if len(groups) != len(set(groups)):
                raise ValueError("mutually alternative members cannot be cumulative requirements")
            ids, admitted = [], set()
            duplicate_sources = {}
            for key in chosen.candidate_ids:
                candidate = self.state.candidates[key]
                if not needed.intersection(candidate.capability_ids):
                    continue
                excluded = {node.id, *ancestors(self.state, node.id)}
                exact = [
                    existing.id
                    for existing in self.state.nodes.values()
                    if existing.id not in excluded
                    and existing.specs
                    and existing.status not in {NodeStatus.SUPERSEDED, NodeStatus.BLOCKED}
                    and duplicate_signature(existing.spec) == duplicate_signature(candidate.spec)
                ]
                for existing_id in dict.fromkeys([*exact, *candidate.possible_duplicate_task_ids]):
                    existing = self.state.nodes.get(existing_id)
                    if (
                        not existing
                        or existing_id in excluded
                        or existing.status in {NodeStatus.SUPERSEDED, NodeStatus.BLOCKED}
                    ):
                        continue
                    if not set(candidate.spec.root_requirement_ids) <= set(
                        existing.spec.root_requirement_ids
                    ):
                        continue
                    duplicate = self.decide(
                        "DUPLICATE",
                        f"{key}:{existing_id}",
                        {
                            "candidate": candidate_for_decision(candidate),
                            "existing_task": {
                                "id": existing.id,
                                "spec": spec_for_decision(existing.spec),
                                "status": existing.status.value,
                            },
                        },
                    )
                    ids.append(duplicate.id)
                    if duplicate.accepted:
                        duplicate_sources[key] = existing_id
                        break
                member = self.decide(
                    "NECESSARY",
                    key,
                    {
                        **payload,
                        "selected_plan": chosen.model_dump(mode="json"),
                        "candidate": candidate_for_decision(self.state.candidates[key]),
                        "duplicate_handling": {
                            "existing_task_id": duplicate_sources.get(key),
                            "policy": "When equivalent, this node only waits for, adapts and verifies the existing delivery; it does not repeat the investigation.",
                        },
                        "members": [
                            candidate_for_decision(self.state.candidates[member])
                            for member in chosen.candidate_ids
                        ],
                    },
                )
                ids.append(member.id)
                if member.accepted:
                    admitted.add(key)
            # A declined prerequisite must not become a dangling dependency.
            while True:
                ready = {
                    key
                    for key in admitted
                    if not set(self.state.candidates[key].depends_on)
                    & (set(chosen.candidate_ids) - admitted)
                }
                if ready == admitted:
                    break
                admitted = ready
            admitted_plan = chosen.model_copy(deep=True)
            admitted_plan.candidate_ids = [key for key in chosen.candidate_ids if key in admitted]
            replacement_safe = True
            if chosen.supersedes_task_ids:
                replacement_safe = len(admitted) == len(chosen.candidate_ids)
                if replacement_safe:
                    supersede = self.decide(
                        "SUPERSEDE",
                        chosen.id,
                        {**payload, "selected_plan": chosen.model_dump(mode="json")},
                    )
                    ids.append(supersede.id)
                    replacement_safe = supersede.accepted
            if (
                admitted_plan.candidate_ids or admitted_plan.supersedes_task_ids
            ) and replacement_safe:
                install_plan(self.state, report.round_id, admitted_plan, ids, duplicate_sources)
                allocated = retained_capabilities | set(admitted_plan.parent_capability_ids)
                allocated.update(
                    cap
                    for key in admitted_plan.candidate_ids
                    for cap in self.state.candidates[key].capability_ids
                )
                accepted = True
        selected = set(admitted_plan.candidate_ids) if accepted else set()
        for candidate in report.candidates:
            if candidate.id not in selected and not needed.intersection(candidate.capability_ids):
                suggestion = self.decide(
                    "SUGGEST",
                    candidate.id,
                    {**payload, "candidate": candidate.model_dump(mode="json")},
                )
                if suggestion.accepted and all(
                    s.candidate_id != candidate.id for s in self.state.suggestions
                ):
                    self.state.suggestions.append(
                        Suggestion(
                            candidate_id=candidate.id,
                            description=candidate.reason,
                            decision_id=suggestion.id,
                        )
                    )
        if needed - allocated:
            node = self.state.nodes[report.task_id]
            for key in needed - allocated:
                capability = self.state.capabilities[key]
                gap = Gap(
                    id=(capability.gap_ids or ["unplanned:" + key])[0],
                    description=capability.description,
                    root_requirement_ids=capability.root_requirement_ids,
                    evidence_refs=[capability.necessity_decision_id],
                )
                if gap.id not in {g.id for g in node.open_gaps}:
                    node.open_gaps.append(gap)
        return accepted

    def _stage_checks(self, report: WorkerReport) -> list[str]:
        for proposal in report.check_change_proposals:
            propose_check(self.state, self.store, report.task_id, report.round_id, proposal)
        latest = {}
        for check in self.state.check_definitions.values():
            if check.task_id == report.task_id and (
                check.criterion_id not in latest
                or check.implementation_revision
                > latest[check.criterion_id].implementation_revision
            ):
                latest[check.criterion_id] = check
        return [c.id for c in latest.values() if c.status == "PROPOSED" and not c.review_refs]

    def _review_checks(self, report: WorkerReport) -> None:
        for key, reason in report.check_review.items():
            check = self.state.check_definitions.get(key)
            if check is None:
                matches = [
                    item
                    for item in self.state.check_definitions.values()
                    if item.task_id == report.task_id
                    and item.status == "PROPOSED"
                    and json.loads((Path(item.bundle_ref) / "definition.json").read_text())[
                        "proposal"
                    ]["id"]
                    == key
                ]
                if len(matches) != 1:
                    raise ValueError(f"check review ID is unknown or ambiguous: {key}")
                check = matches[0]
            if report.round_id in check.review_refs:
                continue
            if (
                check.task_id != report.task_id
                or check.proposed_by_round_id == report.round_id
                or check.status != "PROPOSED"
            ):
                raise ValueError(
                    "check review must be independent and refer to a pending definition"
                )
            if not reason.strip():
                raise ValueError("independent review needs evidence and a rationale")
            decision = self.decide(
                "CHECK_CHANGE",
                key,
                {
                    "root": self.state.nodes[self.state.root_task_id].spec.model_dump(mode="json"),
                    "criterion": criterion(
                        self.state, report.task_id, check.criterion_id
                    ).model_dump(mode="json"),
                    "proposal": json.loads(
                        (Path(check.bundle_ref) / "definition.json").read_text()
                    ),
                    "frozen_files": {
                        str(p.relative_to(check.bundle_ref)): review_file(p)
                        for p in Path(check.bundle_ref).rglob("*")
                        if p.is_file()
                    },
                    "independent_review": reason,
                    "runner_environment": check_environment(
                        self.state.workspace, Path(check.bundle_ref)
                    ),
                },
            )
            check.review_refs = [report.round_id, decision.id]
            if reason.lower().startswith("accept") and decision.accepted:
                item = criterion(self.state, report.task_id, check.criterion_id)
                old = self.state.check_definitions.get(item.adopted_check_revision_id or "")
                if old:
                    old.status = "RETIRED"
                check.status, item.adopted_check_revision_id = "ADOPTED", check.id

    def apply_report(self, report: WorkerReport) -> None:
        work = self.state.rounds[report.round_id]
        node = self.state.nodes[work.task_id]
        if work.phase == RoundPhase.COMMITTED:
            return
        if work.kind not in {RoundKind.EXECUTE, RoundKind.RECONCILE, RoundKind.REVIEW}:
            raise ValueError(
                "only authorized execution, integration or check review can publish a product result"
            )
        if report.candidates or report.capability_proposals or report.plan_proposals:
            raise ValueError("children can only be proposed in DECOMPOSE after SHOULD_SPLIT approval")
        if work.kind == RoundKind.EXECUTE and not self.authorization_current(
            work, check_files=False
        ):
            raise ValueError("execution requires a current split judgment")
        if work.kind == RoundKind.RECONCILE and (
            not node.current_plan_revision_id
            or not self.state.children(node.id)
            or work.integration_basis != self.integration_basis(node.id)
        ):
            raise ValueError("integration input plan or child delivery identity changed")
        if work.kind == RoundKind.REVIEW and not node.result_history:
            raise ValueError("check review cannot substitute for an actual task delivery")
        if (report.task_id, report.spec_revision, report.input_digest) != (
            node.id,
            node.spec.revision,
            work.input_digest,
        ):
            raise ValueError("worker report identity/spec/context mismatch")
        if work.phase == RoundPhase.COMMITTED:
            return
        if work.phase not in {RoundPhase.VERIFYING, RoundPhase.DECIDING}:
            raise ValueError("report not durably staged")
        report = report.model_copy(deep=True)
        self._validate_brief(work.task_id, report.decision_brief)
        for gap in report.gaps:
            if gap.kind in {"CHECK_PENDING", "CHECK_ERROR"} and gap.criterion_refs:
                gap.criterion_refs = self._check_gap_references(node.id, gap.criterion_refs)
        inherited = {}
        for key in reversed([node.id] + ancestors(self.state, node.id)):
            inherited.update(self.state.nodes[key].spec.constraints)
        for candidate in report.candidates:
            for key, value in inherited.items():
                if key in candidate.spec.constraints and candidate.spec.constraints[key] != value:
                    raise ValueError(
                        "candidate explicitly overrides an approved ancestor constraint"
                    )
                candidate.spec.constraints[key] = value
        before = self.state.model_copy(deep=True)
        try:
            self._review_checks(report)
            pending = self._stage_checks(report)
            work.verification_ids = self.verify(node.id, work.id, report.deliverable_paths)
            work.phase = RoundPhase.DECIDING
            draft = {
                "report": report.model_dump(mode="json"),
                "verifications": work.verification_ids,
            }
            self.store.write_json(f"reports/{work.id}/draft-{digest(draft)}.json", draft)
            self.save()
            current_artifacts = {
                self.state.artifacts[key].path: key
                for result in node.result_history
                for key in result.artifact_ids
            }
            for key in current_artifacts.values():
                artifact = self.state.artifacts[key]
                file = Path(artifact.path)
                if (
                    not artifact.snapshot_ref
                    and file.is_file()
                    and file_hash(file) == artifact.sha256
                ):
                    sha, snapshot = self.store.snapshot(file)
                    if sha != artifact.sha256:
                        raise RuntimeError("historical artifact changed while capturing snapshot")
                    artifact.snapshot_ref = snapshot
            for path in report.deliverable_paths:
                file = workspace_path(self.state.workspace, path)
                if not file.is_file():
                    raise ValueError(f"deliverable does not exist: {path}")
                sha, snapshot = self.store.snapshot(file)
                artifact = ArtifactRef(
                    id=identifier("artifact"),
                    path=str(file),
                    sha256=sha,
                    task_id=node.id,
                    round_id=work.id,
                    snapshot_ref=snapshot,
                )
                self.state.artifacts[artifact.id] = artifact
                current_artifacts[str(file)] = artifact.id
            node.open_gaps = report.gaps
            for gap in list(node.open_gaps):
                if gap.kind in {"CHECK_PENDING", "CHECK_ERROR"} and gap.criterion_refs:
                    records = [
                        latest_record(self.state, *ref.split(":", 1)) for ref in gap.criterion_refs
                    ]
                    if all(record and record.status == "PASS" for record in records):
                        node.open_gaps.remove(gap)
            for gap in node.open_gaps:
                if not gap.root_requirement_ids:
                    raise ValueError("gap must be tied to a root requirement")
                chain = gap.origin_gap_id or gap.id
                self.state.progress.gap_history.setdefault(chain, []).append(node.id)
            self.state.background.update(
                {f"{node.id}:{key}": value for key, value in report.background_updates.items()}
            )
            self._apply_coverage(report)
            self.refresh_coverage_refs()
            self._select_plan(report)
            for child_id in reversed(active_subtree(self.state, node.id)[1:]):
                child = self.state.nodes[child_id]
                if (
                    child.status == NodeStatus.SUCCEEDED
                    and not certificate_valid(self.state, child_id)
                    and hard_complete(self.state, child_id)
                ):
                    if not self.evaluate_node(
                        child_id, "DELIVERY", self.decision_context(child_id)
                    ).completed:
                        child.status = NodeStatus.READY
                        if not any(
                            r.task_id == child_id and r.phase not in TERMINAL_ROUNDS
                            for r in self.state.rounds.values()
                        ):
                            queue_round(
                                self.state,
                                child_id,
                                RoundKind.EXPLORE,
                                f"integration-reassess:{work.id}",
                                purpose="inspect current product after parent integration",
                            )
            result = NodeResult(
                revision=node.result_revision + 1,
                round_id=work.id,
                summary=report.summary,
                artifact_ids=list(current_artifacts.values()),
                verification_ids=work.verification_ids,
                decision_ids=[
                    d.id for d in self.state.decisions.values() if d.id not in before.decisions
                ],
                child_result_versions={
                    key: self.state.nodes[key].result_revision
                    for key in self.state.children(node.id)
                },
                gaps=node.open_gaps,
            )
            node.result_history.append(result)
            node.reuse_from = report.reuse_from
            if report.reuse_from and not any(
                d.kind == "REUSE" and d.target_id == f"{node.id}:{report.reuse_from}" and d.accepted
                for d in self.state.decisions.values()
            ):
                raise ValueError("reuse lacks an accepted source decision")
            work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
            evaluation = self.evaluate_node(
                node.id,
                "DELIVERY",
                self.decision_context(node.id, report),
                pending_checks=pending,
            )
            self.state.scheduler.active_work = None
            if not evaluation.completed:
                if evaluation.waiting_children:
                    node.status = NodeStatus.WAITING_CHILDREN
                else:
                    node.status = NodeStatus.READY
                    queue_round(
                        self.state,
                        node.id,
                        evaluation.next_kind,
                        f"followup:{work.id}",
                        purpose=evaluation.purpose,
                    )
            if work.kind == RoundKind.RECONCILE:
                node.last_reconciled_input_digest = digest(node_input(self.state, node.id))
            self._record_progress(node.id, before)
            if any(
                c.check_kind == "user"
                and (
                    not (record := latest_record(self.state, node.id, c.id))
                    or record.status != "PASS"
                )
                for c in node.spec.acceptance
                if c.required
            ):
                self.state.status, self.state.block_reason = (
                    RunStatus.BLOCKED,
                    f"human acceptance required for task {node.id}; use accept then resume",
                )
            self.refresh_parents()
            self.save()
        except Exception:
            # The persisted report/check stages remain recoverable; publishing the
            # plan and all descendant changes happens only in the final snapshot.
            self.state = self.store.load()
            raise

    def _record_progress(self, task_id: str, before: RunState) -> None:
        node, previous = self.state.nodes[task_id], before.nodes[task_id]
        old_files = {(a.path, a.sha256) for a in before.artifacts.values()}
        new_files = {
            (a.path, a.sha256) for a in self.state.artifacts.values() if a.task_id == task_id
        }
        passed = {
            (r.task_id, r.criterion_id, r.input_fingerprints.__repr__())
            for r in self.state.verifications.values()
            if r.status == "PASS"
        }
        old_passed = {
            (r.task_id, r.criterion_id, r.input_fingerprints.__repr__())
            for r in before.verifications.values()
            if r.status == "PASS"
        }
        covered = {
            key
            for key, c in self.state.coverage.items()
            if c.status in {"SUPPORTED", "NOT_REQUIRED"}
        }
        old_covered = {
            key for key, c in before.coverage.items() if c.status in {"SUPPORTED", "NOT_REQUIRED"}
        }
        resolved = {g.id for g in previous.open_gaps} - {g.id for g in node.open_gaps}
        substantive = bool(
            new_files - old_files or passed - old_passed or covered - old_covered or resolved
        )
        plan = self.state.plans.get(node.current_plan_revision_id or "")
        groups = (
            (
                [
                    set(self.state.candidates[key].capability_ids)
                    for key in plan.selected_candidate_ids
                ]
                + [set(plan.parent_capability_ids)]
            )
            if plan
            else []
        )
        assigned = set().union(*groups) if groups else set()
        refined = bool(
            plan
            and plan.id != previous.current_plan_revision_id
            and set(self.state.children(node.id)) - set(before.children(node.id))
            and len(assigned) > 1
            and len([group for group in groups if group]) > 1
            and all(group != assigned for group in groups)
        )
        # Verified scope allocation is planning, not delivery or a failed execution attempt.
        node.no_progress_rounds = 0 if substantive else node.no_progress_rounds + (not refined)
        progress = self.state.progress
        progress.rounds_without_substantive_progress = (
            0 if substantive else progress.rounds_without_substantive_progress + (not refined)
        )
        if substantive:
            progress.last_substantive_progress_digest = digest(
                [sorted(new_files), sorted(covered), sorted(resolved)]
            )
        if (
            node.status != NodeStatus.WAITING_CHILDREN
            and not certificate_valid(self.state, task_id)
            and node.no_progress_rounds >= self.state.execution_profile.max_node_no_progress
        ):
            node.status, node.block_reason = NodeStatus.BLOCKED, "node made no substantive progress"
            for work in self.state.rounds.values():
                if work.task_id == task_id and work.phase == RoundPhase.QUEUED:
                    work.phase, work.error = RoundPhase.CANCELLED, node.block_reason
            self._queue_parent_diagnosis(node)
        if (
            progress.rounds_without_substantive_progress
            >= self.state.execution_profile.max_run_no_progress
        ):
            self.state.status, self.state.block_reason = (
                RunStatus.BLOCKED,
                "run made no substantive progress across tasks",
            )

    def _queue_parent_diagnosis(self, node: TaskNode) -> None:
        if not node.parent_id:
            return
        parent = self.state.nodes[node.parent_id]
        for work in self.state.rounds.values():
            if (
                work.task_id == parent.id
                and work.kind == RoundKind.RECONCILE
                and work.phase == RoundPhase.QUEUED
            ):
                work.phase, work.error = (
                    RoundPhase.CANCELLED,
                    "child blocked; parent diagnosis required",
                )
        if any(
            work.task_id == parent.id and work.phase not in TERMINAL_ROUNDS
            for work in self.state.rounds.values()
        ):
            return
        parent.status, parent.block_reason = NodeStatus.READY, None
        queue_round(
            self.state,
            parent.id,
            RoundKind.EXPLORE,
            f"blocked-child:{node.id}:{node.result_revision}",
            purpose="repair_or_replace_blocked_child",
        )

    def _run_verification_job(self, job_id: str) -> None:
        job = self.state.verification_jobs[job_id]
        if job.phase == "WAITING_CONTEXT":
            self._resume_decision_work(WorkRef(kind="VERIFICATION", id=job_id))
        job.verification_ids = self.verify(job.owner_task_id, job.id)
        job.phase = "DECIDING"
        self.refresh_coverage_refs()
        self.save()
        failures = [
            self.state.verifications[key]
            for key in job.verification_ids
            if self.state.verifications[key].status != "PASS"
        ]
        if failures:
            if any(r.status == "ERROR" for r in failures):
                job.phase, job.error = (
                    "ERROR",
                    "verification environment/check error; inspect before retrying",
                )
                self.state.status, self.state.block_reason = RunStatus.BLOCKED, job.error
            else:
                job.phase = "DONE"
                node = self.state.nodes[job.owner_task_id]
                node.status = NodeStatus.READY
                for failure in failures:
                    gap = Gap(
                        id=f"check-failure:{failure.task_id}:{failure.criterion_id}",
                        description=f"Current verification is {failure.status}; inspect {failure.output_ref}",
                        root_requirement_ids=criterion(
                            self.state, failure.task_id, failure.criterion_id
                        ).root_requirement_ids,
                        evidence_refs=[failure.id],
                    )
                    if gap.id not in {g.id for g in node.open_gaps}:
                        node.open_gaps.append(gap)
                if not any(
                    r.task_id == node.id and r.phase not in TERMINAL_ROUNDS
                    for r in self.state.rounds.values()
                ):
                    queue_round(
                        self.state,
                        node.id,
                        RoundKind.EXPLORE,
                        f"verification-failure:{job.id}",
                        purpose="repair_or_replan",
                    )
        else:
            # Refresh child certificates bottom up without inventing a new result version.
            for key in reversed(active_subtree(self.state, job.owner_task_id)):
                if self.state.nodes[key].status == NodeStatus.SUCCEEDED:
                    if not self.evaluate_node(
                        key, "DELIVERY", self.decision_context(key)
                    ).completed:
                        self.state.nodes[key].status = NodeStatus.READY
                        if not any(
                            r.task_id == key and r.phase not in TERMINAL_ROUNDS
                            for r in self.state.rounds.values()
                        ):
                            queue_round(
                                self.state,
                                key,
                                RoundKind.EXPLORE,
                                f"verification-reassess:{job.id}",
                                purpose="inspect_unresolved_completion_or_coverage",
                            )
            job.phase = "DONE"
        self.state.scheduler.active_work = None
        self.save()

    def _run_round(self, round_id: str) -> None:
        work = self.state.rounds[round_id]
        if work.phase == RoundPhase.WAITING_CONTEXT:
            self._resume_decision_work(WorkRef(kind="ROUND", id=round_id))
        if work.phase in {RoundPhase.VERIFYING, RoundPhase.DECIDING} and work.worker_report_ref:
            report = REPORT_MODELS[work.kind].model_validate_json(
                Path(work.worker_report_ref).read_text()
            )
            self._commit_report(work, report)
            return
        if work.kind == RoundKind.RECONCILE and (
            not self.state.nodes[work.task_id].current_plan_revision_id
            or not self.state.children(work.task_id)
            or not all(
                certificate_valid(self.state, key) for key in self.state.children(work.task_id)
            )
        ):
            raise ValueError("integration requires actual current required child deliveries")
        if work.kind in {RoundKind.SCOPE_REVIEW, RoundKind.DECOMPOSE, RoundKind.EXECUTE} or (
            work.kind == RoundKind.RECONCILE and work.authorization_round_id
        ):
            current = (
                self._scope_source_current(work)
                if work.kind == RoundKind.SCOPE_REVIEW
                else self.authorization_current(work)
            )
            if not current:
                refresh_review = (
                    work.kind != RoundKind.SCOPE_REVIEW
                    and bool(
                        (source := self.state.rounds.get(work.authorization_round_id or ""))
                        and source.scope_review_round_id
                    )
                    and self._scope_source_current(work)
                    and self.authorization_current(work, check_files=False)
                )
                work.phase, work.error, work.finished_at = (
                    RoundPhase.CANCELLED,
                    "missing or stale split routing authorization",
                    time.time(),
                )
                self.state.scheduler.active_work = None
                self.state.nodes[work.task_id].status = NodeStatus.READY
                queue_round(
                    self.state,
                    work.task_id,
                    RoundKind.SCOPE_REVIEW if refresh_review else RoundKind.EXPLORE,
                    f"stale-route:{work.id}",
                    purpose=(
                        "refresh affected independent review and split routing; reuse still-current exploration facts without repeating source scans"
                        if refresh_review
                        else work.error
                    ),
                    authorization_round_id=work.authorization_round_id if refresh_review else None,
                )
                self.save()
                return
        if work.phase == RoundPhase.WORKER_RUNNING:
            raise RuntimeError(
                "recover and confirm the old worker stopped before another invocation"
            )
        if work.kind == RoundKind.RECONCILE:
            # Child proofs are current before execution; integration writes are reverified below.
            work.integration_basis = self.integration_basis(work.task_id)
        context = build_context(self.state, work.id)
        context["decision_support"]["controller_run_directory"] = str(self.store.directory)
        if work.kind in {RoundKind.DECOMPOSE, RoundKind.EXECUTE, RoundKind.RECONCILE}:
            source = self.state.rounds.get(work.authorization_round_id or "")
            if source and source.kind == RoundKind.EXPLORE and source.thread_id:
                context["continuation_thread_id"] = source.thread_id
        if work.kind in {RoundKind.EXECUTE, RoundKind.RECONCILE}:
            for source in self.state.nodes.values():
                if (
                    source.id != work.task_id
                    and source.status == NodeStatus.SUCCEEDED
                    and certificate_valid(self.state, source.id)
                    and reuse_scope_covers(self.state.nodes[work.task_id].spec, source.spec)
                    and (
                        not self.state.nodes[work.task_id].duplicate_of_task_id
                        or source.id == self.state.nodes[work.task_id].duplicate_of_task_id
                    )
                ):
                    self.decide(
                        "REUSE",
                        f"{work.task_id}:{source.id}",
                        {
                            "target": spec_for_decision(self.state.nodes[work.task_id].spec),
                            "source": {
                                "id": source.id,
                                "spec": spec_for_decision(source.spec),
                                "result": source.result_history[-1].model_dump(mode="json"),
                                "artifacts": [
                                    self.state.artifacts[key].model_dump(mode="json")
                                    for key in source.result_history[-1].artifact_ids
                                ],
                            },
                            "source_currently_verified": True,
                        },
                    )
            context["reuse_candidates"] = [
                d.model_dump(mode="json")
                for d in self.state.decisions.values()
                if d.kind == "REUSE" and d.target_id.startswith(work.task_id + ":") and d.accepted
            ]
        report_type = REPORT_MODELS[work.kind]
        manifest = {
            "context": context,
            "execution_contract": self.worker.report_contract(work.kind.value, report_type)
            if hasattr(self.worker, "report_contract")
            else {"worker": "externally supplied worker callable"},
        }
        work.input_digest = digest(manifest)
        context["input_digest"] = work.input_digest
        work.context_manifest_ref = self.store.write_json(
            f"contexts/{work.id}-{work.input_digest}.json", manifest
        )
        work.phase = RoundPhase.WORKER_RUNNING
        work.worker_model = self.state.execution_profile.worker_model
        work.worker_effort = self.state.execution_profile.worker_effort
        if hasattr(self.worker, "execution_owner"):
            work.execution_owner = self.worker.execution_owner()
        self.state.progress.worker_attempts_started += 1
        self.store.execution_reservation([work.id])
        self.save()

        def on_started(thread_id, turn_id=None):
            work.thread_id, work.turn_id = thread_id, turn_id
            self.save()

        arguments = (
            context,
            report_type,
            on_started,
            min(self.state.execution_profile.worker_timeout_seconds, self.time_left()),
        )
        if hasattr(self.worker, "run_with_feedback"):

            def validate(report):
                if work.kind not in {RoundKind.PREPARE, RoundKind.SPECIFY}:
                    self._validate_brief(work.task_id, getattr(report, "decision_brief", None))
                if getattr(report, "input_paths", None):
                    fingerprints(self.state.workspace, report.input_paths, read_roots(self.state))
                for relative in getattr(report, "deliverable_paths", []):
                    if not workspace_path(self.state.workspace, relative).is_file():
                        raise ValueError(
                            f"declared deliverable is not an actual workspace file: {relative}"
                        )
                for proposal in getattr(report, "check_change_proposals", []):
                    own = {c.id: c for c in self.state.nodes[work.task_id].spec.acceptance}
                    if proposal.criterion_id not in own:
                        raise ValueError(f"unknown check criterion; allowed: {sorted(own)}")
                    if (
                        proposal.old_check_revision_id
                        != own[proposal.criterion_id].adopted_check_revision_id
                    ):
                        raise ValueError(
                            f"old_check_revision_id must be {own[proposal.criterion_id].adopted_check_revision_id!r} for {proposal.criterion_id}"
                        )
                for gap in getattr(report, "gaps", []):
                    if gap.kind in {"CHECK_PENDING", "CHECK_ERROR"} and gap.criterion_refs:
                        self._check_gap_references(work.task_id, gap.criterion_refs)
                if work.kind in {
                    RoundKind.EXPLORE,
                    RoundKind.SCOPE_REVIEW,
                    RoundKind.DECOMPOSE,
                    RoundKind.REVIEW,
                    RoundKind.CONTEXT_REVIEW,
                }:
                    self._validate_read_only_report(work, report)
                if work.kind == RoundKind.EXPLORE:
                    if report.split_opportunities:
                        raise ValueError(
                            "split_opportunities must be []; the judge has not authorized planning"
                        )
                    self._validate_aspect_assessments(report.aspect_assessments, [])
                if work.kind == RoundKind.DECOMPOSE:
                    parent = self.state.nodes[work.task_id]
                    root_ids = {
                        r.id for r in self.state.nodes[self.state.root_task_id].spec.requirements
                    }
                    for candidate in report.candidates:
                        if candidate.spec.requirements:
                            raise ValueError(
                                "child requirements must be []; use root_requirement_ids"
                            )
                        if not set(candidate.spec.root_requirement_ids) <= root_ids:
                            raise ValueError(
                                f"unknown root requirement mapping; allowed: {sorted(root_ids)}"
                            )
                        if not set(candidate.spec.parent_criterion_ids) <= {
                            c.id for c in parent.spec.acceptance
                        }:
                            raise ValueError(
                                f"unknown parent criterion mapping; allowed: {[c.id for c in parent.spec.acceptance]}"
                            )
                        for key, value in parent.spec.constraints.items():
                            if (
                                key in candidate.spec.constraints
                                and candidate.spec.constraints[key] != value
                            ):
                                raise ValueError(
                                    f"inherited constraint {key!r} must match exactly or be omitted"
                                )

            def feedback(diagnostics, raw):
                self.store.write_json(
                    f"reports/{work.id}/feedback/{diagnostics['attempt']}.json",
                    {"diagnostics": diagnostics, "raw_report": raw},
                )

            report = self.worker.run_with_feedback(
                *arguments, validator=validate, on_feedback=feedback
            )
        else:
            report = self.worker(*arguments)
        self.store.execution_reservation([])
        if getattr(self.worker, "last_usage", None):
            self.state.progress.recorded_usage.append(
                {"round_id": work.id, "usage": self.worker.last_usage}
            )
        work.worker_report_ref = self.store.write_json(
            f"reports/{work.id}/worker.json", report.model_dump(mode="json")
        )
        work.phase = RoundPhase.VERIFYING
        self.save()
        self._commit_report(work, report)

    def _claim_same_thread_continuation(self, source: TaskRound | None) -> WorkRef | None:
        if not (
            source
            and source.kind == RoundKind.EXPLORE
            and source.phase == RoundPhase.COMMITTED
            and source.thread_id
            and getattr(self.worker, "supports_continuation", False)
            and self.state.status == RunStatus.ACTIVE
            and self.state.scheduler.active_work is None
        ):
            return None
        followups = [
            work
            for work in self.state.rounds.values()
            if work.task_id == source.task_id
            and work.authorization_round_id == source.id
            and work.kind in {RoundKind.DECOMPOSE, RoundKind.EXECUTE, RoundKind.RECONCILE}
            and work.phase == RoundPhase.QUEUED
        ]
        if not followups:
            return None
        if len(followups) != 1:
            raise ValueError("one exploration must authorize exactly one immediate continuation")
        if self.state.progress.worker_attempts_started >= self.state.execution_profile.max_worker_rounds:
            self.block("worker round budget exhausted")
            return None
        if self.time_left() <= 0:
            self.block("run active time budget exhausted")
            return None
        work = followups[0]
        ref = WorkRef(kind="ROUND", id=work.id)
        self.state.scheduler.active_work = ref
        work.phase, work.started_at = RoundPhase.PREPARING, time.time()
        self.state.nodes[work.task_id].status = NodeStatus.RUNNING
        self.save()
        return ref

    def step(self) -> RunState:
        if self.state.schema_version != 2:
            raise ValueError(
                "legacy task-tree protocol is inspection-only; start a fresh run for model-gated decomposition"
            )
        self.state.workspace_lease = self.store.acquire_workspace(self.state.workspace)
        unresolved = self.state.workspace_lease.unresolved_execution_refs
        if unresolved:
            try:
                for key in unresolved:
                    work = self.state.rounds.get(key) or self.state.verification_jobs.get(key)
                    if (
                        not work
                        or not work.execution_owner
                        or not hasattr(self.worker, "ensure_stopped")
                    ):
                        raise RuntimeError(
                            "unresolved execution identity; inspect it before any new work"
                        )
                    self.worker.ensure_stopped(work.execution_owner)
                self.store.execution_reservation([])
            except Exception as error:
                self.block(str(error))
                raise
        if (
            self.state.root_approval
            and self.state.root_approval.contract_hash
            != self.state.nodes[self.state.root_task_id].spec.contract_hash()
        ):
            self.block(
                "root contract changed; user reconfirmation and replan required before execution"
            )
            return self.state
        if self.state.status == RunStatus.SUCCEEDED and not certificate_valid(
            self.state, self.state.root_task_id
        ):
            self.state.status = RunStatus.ACTIVE
            self.save()
        if self.state.scheduler.active_work:
            try:
                self.recover_active()
            except (DecisionContextTooLarge, DecisionWorkerUnavailable) as error:
                self._park_decision(error)
                return self.state
            except Exception as error:
                self.block(str(error))
                raise
            if not self.state.scheduler.active_work:
                return self.state
            ref = self.state.scheduler.active_work
        else:
            if self.state.status != RunStatus.ACTIVE:
                return self.state
            if self.state.root_approval:
                root_id = self.state.root_task_id
                if (
                    certificate_valid(self.state, root_id)
                    and not any(r.phase not in TERMINAL_ROUNDS for r in self.state.rounds.values())
                    and not any(
                        j.phase in {"QUEUED", "RUNNING", "DECIDING"}
                        for j in self.state.verification_jobs.values()
                    )
                ):
                    self.state.status = RunStatus.SUCCEEDED
                    self.state.scheduler.wave = None
                    self.save()
                    self.store.release_workspace()
                    return self.state
                # Historical delivery is retained; stale evidence is refreshed without workers.
                stale = [
                    n
                    for n in self.state.nodes.values()
                    if n.status == NodeStatus.SUCCEEDED and not certificate_valid(self.state, n.id)
                ]
                for node in sorted(stale, key=lambda n: n.depth):
                    if not any(
                        ancestor
                        in {
                            j.owner_task_id
                            for j in self.state.verification_jobs.values()
                            if j.phase == "QUEUED"
                        }
                        for ancestor in ancestors(self.state, node.id)
                    ):
                        self.queue_verification(node.id)
            if self.time_left() <= 0:
                self.block("run active time budget exhausted")
                return self.state
            self.refresh_parents()
            choice = select_next(self.state)
            if (
                choice.work_ref
                and choice.work_ref.kind == "ROUND"
                and (
                    self.state.rounds[choice.work_ref.id].phase != RoundPhase.WAITING_CONTEXT
                    or any(
                        job.source_ref == choice.work_ref
                        and job.resume_phase in {"QUEUED", "PREPARING"}
                        for job in self.state.decision_jobs.values()
                    )
                )
                and self.state.progress.worker_attempts_started
                >= self.state.execution_profile.max_worker_rounds
            ):
                self.block("worker round budget exhausted")
                return self.state
            ref = self.claim_next_work()
            if ref is None:
                self.block(
                    "no runnable work; root is not currently complete (inspect blocked branches/dependencies)"
                )
                return self.state
        try:
            if self.time_left() <= 0:
                raise TimeoutError("run active time budget exhausted")
            while True:
                source = self.state.rounds[ref.id] if ref.kind == "ROUND" else None
                if source:
                    self._run_round(ref.id)
                else:
                    self._run_verification_job(ref.id)
                continuation = self._claim_same_thread_continuation(source)
                if continuation is None:
                    break
                ref = continuation
        except WorkerReportError as error:
            work = self.state.rounds[ref.id]
            work.phase, work.error, work.finished_at = RoundPhase.FAILED, str(error), time.time()
            node = self.state.nodes[work.task_id]
            node.status, node.block_reason = NodeStatus.BLOCKED, str(error)
            self.state.scheduler.active_work = None
            self.store.execution_reservation([])
            self._queue_parent_diagnosis(node)
            self.save()
        except ValueError as error:
            work = self.state.rounds.get(ref.id) if ref.kind == "ROUND" else None
            if (
                work
                and work.worker_report_ref
                and work.phase in {RoundPhase.VERIFYING, RoundPhase.DECIDING}
            ):
                work.phase, work.error, work.finished_at = (
                    RoundPhase.FAILED,
                    str(error),
                    time.time(),
                )
                self.state.scheduler.active_work = None
                node = self.state.nodes[work.task_id]
                node.no_progress_rounds += 1
                self.state.progress.rounds_without_substantive_progress += 1
                if node.no_progress_rounds >= self.state.execution_profile.max_node_no_progress:
                    node.status, node.block_reason = NodeStatus.BLOCKED, str(error)
                    self._queue_parent_diagnosis(node)
                else:
                    node.status = NodeStatus.READY
                    retry = queue_round(
                        self.state,
                        node.id,
                        work.kind,
                        f"invalid-report:{work.id}",
                        purpose="inspect_actual_changes_and_correct_report: " + str(error),
                        authorization_round_id=work.authorization_round_id,
                        decision_job_id=work.decision_job_id,
                    )
                    retry.retry_of_round_id, retry.attempt = work.id, work.attempt + 1
                if (
                    self.state.progress.rounds_without_substantive_progress
                    >= self.state.execution_profile.max_run_no_progress
                ):
                    self.state.status, self.state.block_reason = (
                        RunStatus.BLOCKED,
                        "repeated invalid reports without progress",
                    )
                self.save()
                return self.state
            self.block(str(error))
            raise
        except (DecisionContextTooLarge, DecisionWorkerUnavailable) as error:
            self._park_decision(error)
        except Exception as error:
            self.block(str(error))
            raise
        return self.state
