from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")

    def encode(item):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot hash {type(item)}")

    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=encode,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunStatus(str, Enum):
    ACTIVE = "ACTIVE"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    BLOCKED = "BLOCKED"
    PAUSED = "PAUSED"
    SUCCEEDED = "SUCCEEDED"
    CANCELLED = "CANCELLED"


class NodeStatus(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING_CHILDREN = "WAITING_CHILDREN"
    WAITING_DEPENDENCIES = "WAITING_DEPENDENCIES"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    SUPERSEDED = "SUPERSEDED"


class RoundKind(str, Enum):
    PREPARE = "PREPARE"
    SPECIFY = "SPECIFY"
    EXPLORE = "EXPLORE"
    SCOPE_REVIEW = "SCOPE_REVIEW"
    DECOMPOSE = "DECOMPOSE"
    EXECUTE = "EXECUTE"
    REVIEW = "REVIEW"
    CONTEXT_REVIEW = "CONTEXT_REVIEW"
    # Retained only so old runs can still be inspected and exported.
    ADVANCE = "ADVANCE"
    RECONCILE = "RECONCILE"


class RoundPhase(str, Enum):
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    WORKER_RUNNING = "WORKER_RUNNING"
    VERIFYING = "VERIFYING"
    DECIDING = "DECIDING"
    WAITING_CONTEXT = "WAITING_CONTEXT"
    COMMITTED = "COMMITTED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_ROUNDS = {
    RoundPhase.COMMITTED,
    RoundPhase.INTERRUPTED,
    RoundPhase.FAILED,
    RoundPhase.CANCELLED,
}


class Requirement(Model):
    id: str
    description: str = Field(min_length=1)
    required: bool = True
    acceptance_ids: list[str] = Field(default_factory=list)


class CheckSpec(Model):
    command: list[str] = Field(
        default_factory=list,
        description="Behavioral command argv. When freezing assertion files, explicitly execute the frozen entrypoint (the controller rewrites matching file argv), or explicitly load CODEX_CHECK_BUNDLE; mutable workspace test discovery is not authoritative.",
    )
    cwd: str = "."
    timeout_seconds: int = Field(default=120, gt=0, le=600)
    expected_exit_code: int = 0
    expected_stdout: str | None = None
    artifact_paths: list[str] = Field(
        default_factory=list,
        description="Current product, dataset and runtime input paths under test, within the workspace or explicitly declared read-only input roots. Exclude the assertion files/helpers/fixtures frozen via CheckChangeProposal.file_paths; their contents are covered by the check bundle hash. Use ['.'] if the product read set is unknown.",
    )


class AcceptanceCriterion(Model):
    id: str
    revision: int = Field(default=1, ge=1)
    description: str = Field(min_length=1)
    required: bool = True
    check_kind: Literal["command", "artifact", "evidence", "user"] = "command"
    check_spec: CheckSpec = Field(default_factory=CheckSpec)
    adopted_check_revision_id: str | None = None
    root_requirement_ids: list[str] = Field(default_factory=list)
    parent_criterion_ids: list[str] = Field(default_factory=list)


class TaskSpec(Model):
    revision: int = Field(default=1, ge=1)
    title: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    scope: list[str] = Field(default_factory=list)
    constraints: dict[str, str] = Field(default_factory=dict)
    deliverables: list[str] = Field(default_factory=list)
    requirements: list[Requirement] = Field(
        default_factory=list,
        description="Only the root defines Requirement IDs. Every child specification MUST use an empty list; map existing root IDs via root_requirement_ids and acceptance mappings.",
    )
    root_requirement_ids: list[str] = Field(default_factory=list)
    parent_criterion_ids: list[str] = Field(default_factory=list)
    acceptance: list[AcceptanceCriterion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> TaskSpec:
        for items in (self.acceptance, self.requirements):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate criterion or requirement ID")
        return self

    def contract_hash(self) -> str:
        value = self.model_dump(mode="json")
        value.pop("revision")
        for check in value["acceptance"]:
            for key in ("revision", "check_spec", "adopted_check_revision_id"):
                check.pop(key)
        return digest(value)


class RootApproval(Model):
    approved_spec_revision: int
    contract_hash: str
    user_confirmation_ref: str = Field(min_length=1)
    confirmed_at: float = Field(default_factory=time.time)


class ExecutionProfile(Model):
    codex_bin: str = "codex"
    worker_model: str | None = "gpt-6-sol"
    worker_effort: str = "xhigh"
    native_prompt_profile: Literal["default", "open-task-v1"] = "default"
    jev_url: str = "http://127.0.0.1:11434"
    jev_model: str = "nimble:9b-q4_K_M"
    decision_key_env: str | None = None
    decision_env_file: str | None = None
    decision_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    split_decision_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    max_worker_rounds: int = Field(default=100, gt=0)
    max_active_seconds: float = Field(default=7200, gt=0)
    worker_timeout_seconds: float = Field(default=600, gt=0)
    max_node_no_progress: int = Field(default=3, gt=0)
    max_run_no_progress: int = Field(default=8, gt=0)
    codex_version: str = ""
    codex_binary_sha256: str = ""
    code_mode_host_sha256: str = ""
    permission_profile: str = "workspace-write"
    jev_timeout_seconds: float = Field(default=120, gt=0)
    max_jev_input_bytes: int = Field(default=48000, gt=1000)
    jev_context_target_bytes: int = Field(default=48000, gt=1000)
    max_context_repairs: int = Field(default=2, ge=0)


class ArtifactRef(Model):
    id: str
    path: str
    sha256: str
    task_id: str
    round_id: str
    snapshot_ref: str | None = None


class PreparationRecord(Model):
    round_id: str
    summary: str
    artifact_ids: list[str]
    index_artifact_id: str
    inspected_sources: list[str]
    facts: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class Gap(Model):
    id: str
    description: str
    root_requirement_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    origin_gap_id: str | None = None
    kind: Literal["CAPABILITY", "CHECK_PENDING", "CHECK_ERROR", "EVIDENCE", "USER_INPUT"] = (
        "CAPABILITY"
    )
    criterion_refs: list[str] = Field(default_factory=list)


class DecisionRecord(Model):
    id: str
    kind: Literal[
        "SHOULD_SPLIT",
        "INFORMATION_GAIN",
        "EXPLORATION_ADEQUATE",
        "EXECUTABLE",
        "REUSE",
        "CAPABILITY_NEEDED",
        "PLAN_ACCEPTABLE",
        "NECESSARY",
        "SUPERSEDE",
        "COVERAGE",
        "CHECK_CHANGE",
        "SUGGEST",
        "DUPLICATE",
        "COMPLETE",
        "EVIDENCE_SUPPORTED",
    ]
    target_id: str
    input_digest: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    exclusive: bool = False
    model: str
    prompt_version: str = "1"
    request_ref: str = ""
    response_ref: str = ""
    worker_thread_id: str = ""
    worker_turn_id: str = ""

    @property
    def accepted(self) -> bool:
        return self.score > self.threshold if self.exclusive else self.score >= self.threshold


class CapabilityNeed(Model):
    id: str
    owner_task_id: str
    description: str
    root_requirement_ids: list[str]
    parent_criterion_ids: list[str] = Field(default_factory=list)
    gap_ids: list[str] = Field(default_factory=list)
    necessity_decision_id: str | None = None


class CandidateProposal(Model):
    id: str
    spec: TaskSpec
    capability_ids: list[str]
    addresses_gap_ids: list[str] = Field(default_factory=list)
    alternative_group_id: str | None = None
    reason: str
    depends_on: list[str] = Field(default_factory=list)
    possible_duplicate_task_ids: list[str] = Field(default_factory=list)


class PlanProposal(Model):
    id: str
    capability_ids: list[str] = Field(
        description="All responsibilities covered by this complete plan, including parent integration."
    )
    parent_capability_ids: list[str] = Field(
        default_factory=list,
        description="Responsibilities reserved for parent integration; included in the complete plan's capability_ids.",
    )
    candidate_ids: list[str]
    recommended: bool = True
    rationale: str
    retained_child_ids: list[str] = Field(default_factory=list)
    supersedes_task_ids: list[str] = Field(default_factory=list)
    replacement_map: dict[str, list[str]] = Field(default_factory=dict)


class PlanRevision(Model):
    id: str
    owner_task_id: str
    previous_revision_id: str | None
    source_round_id: str
    capability_ids: list[str]
    parent_capability_ids: list[str] = Field(default_factory=list)
    selected_candidate_ids: list[str]
    required_child_ids: list[str]
    superseded_task_ids: list[str] = Field(default_factory=list)
    replacement_map: dict[str, list[str]] = Field(default_factory=dict)
    reason: str
    decision_ids: list[str]


class CoverageItem(Model):
    id: str
    kind: Literal["REQUIREMENT", "ASPECT"]
    owner_task_id: str
    root_requirement_ids: list[str]
    subject: str
    discovery_source_refs: list[str] = Field(default_factory=list)
    status: Literal["UNASSESSED", "IN_PROGRESS", "GAP", "SUPPORTED", "NOT_REQUIRED"] = "UNASSESSED"
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    revision: int = 1


class CheckDefinition(Model):
    id: str
    task_id: str
    criterion_id: str
    semantic_revision: int
    implementation_revision: int
    check_spec: CheckSpec
    bundle_ref: str
    bundle_hash: str
    status: Literal["PROPOSED", "ADOPTED", "RETIRED"] = "PROPOSED"
    proposed_by_round_id: str
    review_refs: list[str] = Field(default_factory=list)


class VerificationRecord(Model):
    id: str
    task_id: str
    criterion_id: str
    criterion_revision: int
    check_revision_id: str
    check_bundle_hash: str
    status: Literal["PASS", "FAIL", "ERROR", "NOT_RUN"]
    input_fingerprints: dict[str, str]
    execution_profile_digest: str
    output_ref: str = ""
    exit_code: int | None = None
    method: Literal["command", "artifact", "model", "command+model", "user"] = "command"
    source_ref: str = ""
    input_paths: list[str] = Field(default_factory=list)


class NodeResult(Model):
    revision: int
    round_id: str
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    verification_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    child_result_versions: dict[str, int] = Field(default_factory=dict)
    gaps: list[Gap] = Field(default_factory=list)


class CompletionCertificate(Model):
    id: str
    task_id: str
    result_revision: int
    spec_revision: int
    plan_revision_id: str | None
    verification_ids: list[str]
    coverage_digest: str
    input_digest: str
    decision_ids: list[str]


class TaskNode(Model):
    id: str
    parent_id: str | None
    depth: int
    created_seq: int
    created_by_round_id: str | None = None
    specs: list[TaskSpec] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)
    current_plan_revision_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.READY
    result_history: list[NodeResult] = Field(default_factory=list)
    completion_certificate_id: str | None = None
    coverage_ids: list[str] = Field(default_factory=list)
    open_gaps: list[Gap] = Field(default_factory=list)
    supersession_plan_revision_id: str | None = None
    last_reconciled_input_digest: str | None = None
    last_progress_digest: str | None = None
    no_progress_rounds: int = 0
    block_reason: str | None = None
    reuse_from: str | None = None
    duplicate_of_task_id: str | None = None
    exploration_round_ids: list[str] = Field(default_factory=list)

    @property
    def spec(self) -> TaskSpec:
        if not self.specs:
            raise ValueError(f"task {self.id} has no specification")
        return self.specs[-1]

    @property
    def result_revision(self) -> int:
        return self.result_history[-1].revision if self.result_history else 0


class TaskRound(Model):
    id: str
    task_id: str
    created_seq: int
    source_event_id: str
    kind: RoundKind
    phase: RoundPhase = RoundPhase.QUEUED
    purpose: str = "execute"
    attempt: int = 1
    retry_of_round_id: str | None = None
    input_digest: str = ""
    context_manifest_ref: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    worker_report_ref: str | None = None
    verification_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    execution_owner: dict[str, str] = Field(default_factory=dict)
    authorization_round_id: str | None = None
    routing_basis: str = ""
    routing_fingerprints: dict[str, str] = Field(default_factory=dict)
    integration_basis: str | None = None
    scope_review_round_id: str | None = None
    worker_model: str | None = None
    worker_effort: str | None = None
    decision_job_id: str | None = None


class VerificationJob(Model):
    id: str
    created_seq: int
    owner_task_id: str
    criterion_refs: list[str]
    input_digest: str
    source_round_id: str | None = None
    phase: Literal[
        "QUEUED", "RUNNING", "DECIDING", "WAITING_CONTEXT", "DONE", "ERROR", "CANCELLED"
    ] = "QUEUED"
    verification_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    execution_owner: dict[str, str] = Field(default_factory=dict)


class WorkRef(Model):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["ROUND", "VERIFICATION"]
    id: str


class WaveCursor(Model):
    depth: int = Field(description="Minimum task depth in this finite ready-work batch.")
    work_refs: list[WorkRef] = Field(default_factory=list)
    next_index: int = 0

    @model_validator(mode="before")
    @classmethod
    def legacy_round_batch(cls, value):
        if isinstance(value, dict) and "round_ids" in value:
            value = dict(value)
            ids = value.pop("round_ids")
            value.setdefault("work_refs", [{"kind": "ROUND", "id": key} for key in ids])
        return value

    @property
    def round_ids(self) -> list[str]:
        return [ref.id for ref in self.work_refs if ref.kind == "ROUND"]


class SchedulerState(Model):
    active_work: WorkRef | None = None
    wave: WaveCursor | None = None


class RunProgress(Model):
    worker_attempts_started: int = 0
    active_elapsed_seconds: float = 0
    last_substantive_progress_digest: str | None = None
    rounds_without_substantive_progress: int = 0
    gap_history: dict[str, list[str]] = Field(default_factory=dict)
    recorded_usage: list[dict[str, Any]] = Field(default_factory=list)
    last_active_checkpoint: float | None = None


class WorkspaceLease(Model):
    run_id: str
    canonical_write_roots: list[str]
    owner_ref: str
    acquired_at: float = Field(default_factory=time.time)
    unresolved_execution_refs: list[str] = Field(default_factory=list)


class Suggestion(Model):
    candidate_id: str
    description: str
    decision_id: str


class DecisionJob(Model):
    id: str
    owner_task_id: str
    source_ref: WorkRef
    resume_phase: str
    kind: str
    target_id: str
    context_ref: str
    prior_node_status: NodeStatus
    status: Literal["WAITING_CONTEXT", "QUEUED", "RUNNING", "RESOLVED", "SUPERSEDED"] = (
        "WAITING_CONTEXT"
    )
    error: str = ""
    decision_id: str | None = None
    repair_attempts: int = 0
    brief_ref: str | None = None
    brief_fingerprints: dict[str, str] = Field(default_factory=dict)


class RunState(Model):
    schema_version: int = 1
    run_id: str
    revision: int = 0
    user_request: str
    status: RunStatus = RunStatus.ACTIVE
    root_task_id: str
    root_approval: RootApproval | None = None
    workspace: Path
    workspace_lease: WorkspaceLease | None = None
    input_refs: list[str] = Field(default_factory=list)
    execution_profile: ExecutionProfile = Field(default_factory=ExecutionProfile)
    nodes: dict[str, TaskNode] = Field(default_factory=dict)
    rounds: dict[str, TaskRound] = Field(default_factory=dict)
    candidates: dict[str, CandidateProposal] = Field(default_factory=dict)
    capabilities: dict[str, CapabilityNeed] = Field(default_factory=dict)
    plans: dict[str, PlanRevision] = Field(default_factory=dict)
    coverage: dict[str, CoverageItem] = Field(default_factory=dict)
    decisions: dict[str, DecisionRecord] = Field(default_factory=dict)
    decision_jobs: dict[str, DecisionJob] = Field(default_factory=dict)
    verifications: dict[str, VerificationRecord] = Field(default_factory=dict)
    verification_jobs: dict[str, VerificationJob] = Field(default_factory=dict)
    check_definitions: dict[str, CheckDefinition] = Field(default_factory=dict)
    completion_certificates: dict[str, CompletionCertificate] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    background: dict[str, str] = Field(default_factory=dict)
    preparation_required: bool = False
    preparations: dict[str, PreparationRecord] = Field(default_factory=dict)
    current_preparation_id: str | None = None
    suggestions: list[Suggestion] = Field(default_factory=list)
    scheduler: SchedulerState = Field(default_factory=SchedulerState)
    progress: RunProgress = Field(default_factory=RunProgress)
    next_sequence: int = 0
    block_reason: str | None = None

    def sequence(self) -> int:
        self.next_sequence += 1
        return self.next_sequence

    def children(self, task_id: str) -> list[str]:
        plan_id = self.nodes[task_id].current_plan_revision_id
        return self.plans[plan_id].required_child_ids if plan_id else []


class PreparationReport(Model):
    task_id: str
    round_id: str
    input_digest: str
    summary: str = Field(min_length=1)
    artifact_paths: list[str] = Field(
        min_length=1,
        description="Workspace-relative paths under background/, including exactly 'background/index.json' and every referenced supporting file. Never absolute paths, even when the file is inside the workspace.",
    )
    inspected_sources: list[str] = Field(min_length=1)
    facts: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class SpecProposal(Model):
    task_id: str
    round_id: str
    proposed_spec: TaskSpec
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


class CheckChangeProposal(Model):
    id: str
    criterion_id: str
    old_check_revision_id: str | None = Field(
        default=None,
        description="Copy this criterion's adopted_check_revision_id exactly, including null when none is adopted. A pending or rejected proposal ID is not the adopted revision.",
    )
    check_spec: CheckSpec
    file_paths: list[str] = Field(
        default_factory=list,
        description="Assertion scripts and their assertion-affecting helpers, fixtures and configuration to freeze. Never the mutable product/data being tested, and do not also list these assertions as CheckSpec.artifact_paths.",
    )
    reason: str


class DecisionClaim(Model):
    id: str
    statement: str = Field(min_length=1)
    conditions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)
    counter_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    criterion_ids: list[str] = Field(default_factory=list)


class DecisionBrief(Model):
    summary: str = Field(min_length=1)
    claims: list[DecisionClaim] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class WorkerReport(Model):
    task_id: str
    round_id: str
    spec_revision: int
    input_digest: str
    summary: str
    deliverable_paths: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    capability_proposals: list[CapabilityNeed] = Field(default_factory=list)
    plan_proposals: list[PlanProposal] = Field(default_factory=list)
    candidates: list[CandidateProposal] = Field(default_factory=list)
    coverage_updates: list[CoverageItem] = Field(default_factory=list)
    check_change_proposals: list[CheckChangeProposal] = Field(default_factory=list)
    check_review: dict[str, str] = Field(default_factory=dict)
    background_updates: dict[str, str] = Field(default_factory=dict)
    reuse_from: str | None = Field(
        default=None,
        description="Source TASK ID from an accepted context.reuse_candidates decision, or null. Never an artifact ID, this task's own prior result, or merely a background/code dependency.",
    )
    completion_claim: bool = False
    changed_paths: list[str] = Field(default_factory=list)
    decision_brief: DecisionBrief | None = None


class InvestigationAspect(Model):
    id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)
    possible_questions: list[str] = Field(min_length=1)
    meaning_and_limits: str = Field(min_length=1)


class AspectAssessment(Model):
    aspect_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    disposition: Literal[
        "SUPPORTING_STEP",
        "SPLIT_OPPORTUNITY",
        "ANSWERED",
        "OUT_OF_SCOPE",
        "UNSUPPORTED",
        "NEEDS_PROBE",
    ]
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    opportunity_index: int | None = Field(default=None, ge=1)


class SplitOpportunity(Model):
    question_or_outcome: str = Field(min_length=1)
    deliverable: str = Field(min_length=1)
    verification: str = Field(min_length=1)
    focus_benefit: str = Field(
        min_length=1,
        description="Which independently deliverable responsibility would become focused; explain if none.",
    )
    expected_information_gain: str = Field(
        min_length=1,
        description="What relevant uncertainty, heterogeneity, explanation or delivery risk this could resolve; a qualitative expectation, not a fabricated entropy estimate.",
    )
    effect_on_parent: str = Field(
        min_length=1,
        description="How different possible answers change, qualify or validate the parent result or acceptance.",
    )
    evidence_refs: list[str] = Field(min_length=1)
    feasibility: str = Field(min_length=1)
    cost_and_overlap: str = Field(
        min_length=1,
        description="Execution/integration cost, actual coupling, and any equivalent current tree result; cite it or state none.",
    )


class ExplorationReport(Model):
    task_id: str
    round_id: str
    spec_revision: int
    input_digest: str
    summary: str = Field(min_length=1)
    inspected_sources: list[str] = Field(min_length=1)
    findings: list[str] = Field(min_length=1)
    responsibilities: list[str] = Field(min_length=1)
    execution_assessment: str = Field(min_length=1)
    execution_blockers: list[str] = Field(
        default_factory=list,
        description="Concrete missing essential inputs or unresolved task/acceptance definitions preventing focused execution; record unanswered outcome questions as findings or uncertainties, without planning children.",
    )
    split_opportunities: list[SplitOpportunity] = Field(
        default_factory=list,
        description="Historical field; MUST be [] in new exploration reports. Child planning starts only after Jev SHOULD_SPLIT authorizes DECOMPOSE.",
    )
    leaf_rationale: str = Field(
        default="",
        description="Evidence-backed assessment of task workload, complexity, cohesion and uncertainty for Jev; do not propose children or choose the route. Historical field name retained for readability.",
    )
    uncertainties: list[str] = Field(default_factory=list)
    aspect_assessments: list[AspectAssessment] = Field(default_factory=list)
    input_paths: list[str] = Field(
        default_factory=lambda: ["."],
        min_length=1,
        description="Actual files or bounded component directories whose contents affect routing. Do not include the whole workspace or output directories merely because they were listed; explain any genuine whole-workspace dependency.",
    )
    required_input_paths: list[str] | None = Field(
        default=None,
        description="Inputs that must exist before execution, distinct from observed paths and future outputs. New reports provide an explicit list, possibly empty; null retains legacy input_paths existence checks.",
    )
    decision_brief: DecisionBrief | None = None


class ScopeReviewReport(Model):
    task_id: str
    round_id: str
    spec_revision: int
    input_digest: str
    reviewed_exploration_round_id: str
    summary: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    aspect_assessments: list[AspectAssessment] = Field(default_factory=list)
    omissions: list[str] = Field(default_factory=list)
    required_followups: list[str] = Field(default_factory=list)
    execution_blockers: list[str] = Field(
        default_factory=list,
        description="Missing essential inputs or undefined task/acceptance preventing execution. Bounded questions that can be delegated are omissions/followups, not execution prerequisites for the parent.",
    )
    input_paths: list[str] = Field(
        default_factory=list,
        description="Actual files or bounded directories independently inspected for this review; these remain fingerprinted until the authorized work starts. Avoid unrelated output directories and whole-workspace fallbacks.",
    )
    decision_brief: DecisionBrief | None = None


class DecompositionReport(Model):
    task_id: str
    round_id: str
    spec_revision: int
    input_digest: str
    summary: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    capability_proposals: list[CapabilityNeed] = Field(default_factory=list)
    plan_proposals: list[PlanProposal] = Field(default_factory=list)
    candidates: list[CandidateProposal] = Field(default_factory=list)
    unsplittable_reason: str = ""

    @model_validator(mode="after")
    def valid_plan_or_explained_limit(self):
        if self.candidates:
            if not self.capability_proposals or not self.plan_proposals:
                raise ValueError("candidate tasks need capability and plan proposals")
        elif (
            not self.unsplittable_reason.strip() or self.capability_proposals or self.plan_proposals
        ):
            raise ValueError(
                "no candidates requires a concrete unsplittable_reason and empty proposals; do not invent children"
            )
        return self


class CheckReviewReport(Model):
    task_id: str
    round_id: str
    spec_revision: int
    input_digest: str
    summary: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    coverage_updates: list[CoverageItem] = Field(default_factory=list)
    check_review: dict[str, str] = Field(default_factory=dict)
    decision_brief: DecisionBrief | None = None


class ContextReviewReport(Model):
    task_id: str
    round_id: str
    spec_revision: int
    input_digest: str
    decision_job_id: str
    decision_brief: DecisionBrief
    reviewed_sections: list[str] = Field(
        default_factory=list,
        description="Optional context section paths independently inspected in the saved pending request and covered by the brief. Only controller-allowlisted background/history sections may be replaced; never goals, criteria, verification, coverage, current delivery, or gaps.",
    )
    input_paths: list[str] = Field(min_length=1)
