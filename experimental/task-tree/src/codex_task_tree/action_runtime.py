"""Breadth-first visits with isolated root branches and serial tree commits."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from pathlib import Path

from .action_context import bounded_context
from .action_protocol import (
    ACTION_JUDGE_INSTRUCTIONS,
    VISIT_INSTRUCTIONS,
    DrillDownReport,
    ExecutionReport,
    ExploreBrief,
    VisitReport,
)
from .codex_adapter import WorkerReportError
from .context import (
    external_input_ref,
    fingerprints,
    input_path,
    preparation_view,
    read_roots,
    workspace_path,
)
from .models import (
    TERMINAL_ROUNDS,
    ActionDecision,
    ArtifactRef,
    CompletionCertificate,
    MustSolveItem,
    NodeResult,
    NodeStatus,
    RoundKind,
    RoundPhase,
    RunStatus,
    Suggestion,
    TaskNode,
    WorkRef,
    digest,
    identifier,
)
from .runtime import (
    REPORT_MODELS,
    TaskRuntime,
    ancestors,
    queue_round,
    validate_spec,
    validate_wait_graph,
)

REPORTS = {
    **REPORT_MODELS,
    RoundKind.EXPLORE: VisitReport,
    RoundKind.DECOMPOSE: DrillDownReport,
    RoundKind.EXECUTE: ExecutionReport,
}


def file_versions(state, paths, cache=None):
    local = [p for p in paths if not external_input_ref(p)]
    return fingerprints(state.workspace, local, read_roots(state), cache=cache) if local else {}


def judge_context(context, brief):
    """Give Luna observations and contracts without the worker's routing opinions."""
    packet = deepcopy(context)
    packet.pop("purpose", None)
    packet.pop("last_drill_down", None)
    packet["catalog"] = [
        {key: task[key] for key in ("id", "parent_id", "status", "title", "goal")}
        for task in packet["catalog"]
    ]
    for node in [packet["root"], packet["task"], *packet["ancestors"],
                 *packet["children"], *packet["dependencies"]]:
        node.pop("planning_reason", None)
        for answer in (node.get("result") or {}).get("must_solve_answers", []):
            answer.pop("verification_status", None)
        for answer in node.get("delivery_assessment", {}).get("inline_work_answers", []):
            answer.pop("verification_status", None)
        latest = node.get("latest_exploration")
        if latest:
            latest["brief"].pop("granularity_assessment", None)
        for assessment in node.get("delivery_assessment", {}).get("acceptance", []):
            assessment.pop("status", None)  # Worker labels are claims, not Luna verdicts.
    if packet.get("last_visit"):
        packet["last_visit"].pop("decision", None)
        packet["last_visit"]["brief"].pop("granularity_assessment", None)
    packet["exploration"] = brief.model_dump(mode="json", exclude={"granularity_assessment"})
    return packet


def responsibility(spec, must_solve, contribution):
    # A vague shared title/goal is not proof that two tasks have the same obligations.
    contract = spec.model_dump(mode="json")
    contract.pop("title", None)
    contract.pop("revision", None)
    return digest([contract, sorted(digest(item) for item in must_solve), contribution])


def must_solve_data(items):
    return [item.model_dump(mode="json") if isinstance(item, MustSolveItem) else item for item in items]


def result_versions(state, ids):
    return {key: state.nodes[key].result_revision for key in ids}


def result_basis(state, task_id):
    node = state.nodes[task_id]
    result = node.result_history[-1] if node.result_history else None
    result_data = result.model_dump(
        mode="json", exclude={"verification_ids", "decision_ids", "gaps"}
    ) if result else None
    if (result_data and all(isinstance(item, str) for item in node.must_solve)
            and all(item.verification_status is None
                    and not item.verification_result and not item.evidence_refs
                    for item in result.must_solve_answers)):
        # Keep historical certificates stable after adding item-level answer fields.
        result_data["must_solve_answers"] = [
            {"must_solve": item.must_solve, "answer": item.answer}
            for item in result.must_solve_answers
        ]
    return digest(
        {
            "spec": node.spec.contract_hash(),
            "must_solve": node.must_solve,
            "root_contribution": node.root_contribution,
            "root_goal": state.user_request,
            # Protocol 3 does not populate these legacy fields; a reload marks their defaults as set.
            "result": result_data,
            "children": result_versions(state, state.children(task_id)),
            "dependencies": result_versions(state, node.depends_on),
            "files": result.input_fingerprints if result else {},
        }
    )


def completion_current(state, task_id, *, check_files=True, memo=None, cache=None):
    memo, cache = ({} if memo is None else memo), ({} if cache is None else cache)
    if task_id in memo:
        return memo[task_id]
    # Also fail closed on malformed cycles when inspecting historical state.
    memo[task_id] = False
    node = state.nodes[task_id]
    certificate = state.completion_certificates.get(node.completion_certificate_id or "")
    if not certificate or node.status != NodeStatus.SUCCEEDED or not node.result_history:
        return False
    if (
        not state.root_approval
        or state.root_approval.contract_hash != state.nodes[state.root_task_id].spec.contract_hash()
    ):
        return False
    if any(
        not completion_current(state, key, check_files=check_files, memo=memo, cache=cache)
        for key in state.children(task_id) + node.depends_on
    ):
        return False
    try:
        decision = state.action_decisions[certificate.decision_ids[0]]
        checked = state.rounds[decision.round_id].routing_fingerprints
        current = certificate.input_digest == digest([result_basis(state, task_id), checked])
        if current and check_files:
            saved = node.result_history[-1].input_fingerprints
            current = (
                file_versions(state, saved, cache) == saved
                and file_versions(state, checked, cache) == checked
            )
        memo[task_id] = current
        return current
    except (OSError, ValueError, KeyError, IndexError):
        return False


def answer_issues(node, goal_answer, answers):
    issues = []
    if not goal_answer.strip():
        issues.append("goal_answer must answer the current task goal")
    if [item.must_solve for item in answers] != [
        item.problem if isinstance(item, MustSolveItem) else item for item in node.must_solve
    ]:
        issues.append("must_solve_answers must match every current must_solve in order")
    if any(not item.answer.strip() for item in answers):
        issues.append("each must_solve answer must contain an actual answer")
    if any(
        isinstance(contract, MustSolveItem)
        and (answer.verification_status is None or not answer.verification_result.strip())
        for contract, answer in zip(node.must_solve, answers)
    ):
        issues.append("each structured must_solve answer needs a verification status and result")
    return issues


class ActionRuntime(TaskRuntime):
    protocol_version = 3

    def __init__(self, *args, worker_factory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._needs_file_audit = True
        self._checked_preparation = None
        self.worker_factory = worker_factory
        self._parallel_lock = threading.RLock()

    def _root_branch(self, task_id):
        node = self.state.nodes[task_id]
        while node.parent_id and node.parent_id != self.state.root_task_id:
            node = self.state.nodes[node.parent_id]
        return node.id if node.parent_id else None

    def _session_owner(self, task_id):
        node = self.state.nodes[task_id]
        while node.parent_id and node.session_mode == "inherit":
            node = self.state.nodes[node.parent_id]
        return node

    def _session_thread(self, task_id):
        return self._session_owner(task_id).worker_thread_id

    def _inline_work(self, work):
        if not work.context_manifest_ref:
            return []
        return json.loads(Path(work.context_manifest_ref).read_text()).get("inline_work", [])

    def _remember_thread(self, task_id, thread_id):
        self._session_owner(task_id).worker_thread_id = thread_id

    def _ensure_branch_workspaces(self):
        if self.worker_factory is None or self.state.execution_profile.max_parallel_siblings < 2:
            return
        for branch_id in self.state.children(self.state.root_task_id):
            if self.state.nodes[branch_id].session_mode == "inherit":
                continue
            if branch_id in self.state.branch_workspaces:
                continue
            if self.state.nodes[branch_id].status == NodeStatus.SUCCEEDED:
                continue
            target = (
                self.state.workspace.parent / f"{self.state.workspace.name}-branches"
                / self.state.run_id / branch_id
            )
            self.store.reserve_branch_workspace(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            marker = target / ".task-tree-branch.json"
            if target.exists():
                if not marker.is_file() or json.loads(marker.read_text()) != {
                    "run_id": self.state.run_id, "branch_id": branch_id
                }:
                    raise RuntimeError(f"unregistered branch workspace needs inspection: {target}")
            else:
                pending = target.parent / f".pending-{branch_id}"
                shutil.rmtree(pending, ignore_errors=True)
                if sys.platform == "darwin":
                    cloned = subprocess.run(
                        ["cp", "-cRP", str(self.state.workspace), str(pending)],
                        capture_output=True,
                    )
                    if cloned.returncode:
                        shutil.rmtree(pending, ignore_errors=True)
                        shutil.copytree(self.state.workspace, pending, symlinks=True)
                else:
                    shutil.copytree(self.state.workspace, pending, symlinks=True)
                (pending / marker.name).write_text(json.dumps({
                    "run_id": self.state.run_id, "branch_id": branch_id
                }))
                os.replace(pending, target)
            self.state.branch_workspaces[branch_id] = str(target.resolve())
            # A migrated run retains saved results; new work starts in the isolated checkout.
            for node in self.state.nodes.values():
                if self._root_branch(node.id) == branch_id:
                    node.worker_thread_id = None
            self.save()

    def _branch_ref(self, ref, workspace):
        if ref in self.state.artifacts or external_input_ref(ref):
            return ref
        path = Path(ref)
        return str(workspace / path) if not path.is_absolute() else ref

    def _branch_report_paths(self, report, workspace):
        if isinstance(report, ExploreBrief):
            for name in ("evidence_refs", "input_paths", "required_input_paths"):
                setattr(report, name, [
                    self._branch_ref(ref, workspace) for ref in getattr(report, name)
                ])
            for observation in report.remote_inputs:
                observation.evidence_refs = [
                    self._branch_ref(ref, workspace) for ref in observation.evidence_refs
                ]
        elif isinstance(report, ExecutionReport):
            for name in ("evidence_refs", "input_paths"):
                setattr(report, name, [
                    self._branch_ref(ref, workspace) for ref in getattr(report, name)
                ])
            for item in [*report.acceptance, *report.must_solve_answers,
                         *report.inline_work_answers]:
                item.evidence_refs = [
                    self._branch_ref(ref, workspace) for ref in item.evidence_refs
                ]
            for name in ("changed_paths", "deliverable_paths"):
                setattr(report, name, [
                    str(Path(ref).relative_to(workspace))
                    if Path(ref).is_absolute() and Path(ref).is_relative_to(workspace)
                    else ref for ref in getattr(report, name)
                ])
        return report

    def _publish_branch_files(self, work, report):
        branch = self._root_branch(work.task_id)
        private = Path(self.state.branch_workspaces[branch])
        pending = []
        for relative in dict.fromkeys(report.changed_paths + report.deliverable_paths):
            source = workspace_path(private, relative)
            target = workspace_path(self.state.workspace, relative)
            key = str(target.relative_to(self.state.workspace))
            owner = self.state.branch_output_owners.get(key)
            if owner and owner != branch:
                raise ValueError(f"parallel branches changed the same file: {key}")
            if not source.is_file() and (
                relative in report.deliverable_paths or source.exists()
            ):
                raise ValueError(f"parallel branch output must be a file: {key}")
            pending.append((source, target, key))
        for source, target, key in pending:
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".branch-", dir=target.parent)
                os.close(fd)
                try:
                    shutil.copy2(source, temporary)
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            elif target.is_file():
                target.unlink()
            self.state.branch_output_owners[key] = branch
        if pending:
            self.save()

    def _sync_cross_branch_inputs(self, task_id, private):
        branch = self._root_branch(task_id)
        for dependency_id in self.state.nodes[task_id].depends_on:
            if self._root_branch(dependency_id) == branch:
                continue
            dependency = self.state.nodes[dependency_id]
            if not dependency.result_history:
                continue
            for artifact_id in dependency.result_history[-1].artifact_ids:
                source = Path(self.state.artifacts[artifact_id].path)
                relative = source.relative_to(self.state.workspace)
                target = workspace_path(private, str(relative))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    @classmethod
    def create(cls, store, *args, **kwargs):
        state = super().create(store, *args, **kwargs)
        state.background["root_goal"] = state.user_request
        store.save(state, state.revision)
        return state

    def pending(self, task_id):
        return any(
            r.task_id == task_id and r.phase not in TERMINAL_ROUNDS
            for r in self.state.rounds.values()
        )

    def _commit_preparation(self, work, report):
        path = workspace_path(self.state.workspace, "background/index.json")
        index = json.loads(path.read_text())
        index["root_goal"] = self.state.user_request
        path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
        report.facts["root_goal"] = self.state.user_request
        super()._commit_preparation(work, report)

    def _commit_spec(self, work, report):
        super()._commit_spec(work, report)
        root = self.state.nodes[self.state.root_task_id]
        root.must_solve = [
            MustSolveItem(
                problem=r.description,
                verification_method="Check the delivered answer against its required acceptance criteria and cited evidence.",
                expected_output="A supported answer to this requirement with explicit unresolved limits.",
            )
            for r in root.spec.requirements if r.required
        ]
        root.root_contribution = self.state.user_request
        self.save()

    def enqueue(self, task_id, event, purpose="inspect_current_results_and_changes"):
        if not self.pending(task_id):
            node = self.state.nodes[task_id]
            node.status, node.block_reason = NodeStatus.READY, None
            queue_round(self.state, task_id, RoundKind.EXPLORE, event, purpose=purpose)

    def _node_view(self, task_id):
        node = self.state.nodes[task_id]
        result = node.result_history[-1] if node.result_history else None
        report_ref = self.state.rounds[result.round_id].worker_report_ref if result else None
        report = json.loads(Path(report_ref).read_text()) if report_ref else {}
        delivery_ref = self.state.rounds[result.round_id].delivery_ref if result else None
        delivery = json.loads(Path(delivery_ref).read_text()) if delivery_ref else {}
        return {
            "id": node.id,
            "session_mode": node.session_mode,
            "is_root": node.id == self.state.root_task_id,
            "spec": node.spec.model_dump(mode="json"),
            "must_solve": must_solve_data(node.must_solve),
            "root_contribution": node.root_contribution,
            "planning_reason": node.planning_reason,
            "status": node.status.value,
            "block_reason": node.block_reason,
            "latest_exploration": self._exploration(node),
            "result": result.model_dump(mode="json") if result else None,
            "artifacts": [
                self.state.artifacts[key].model_dump(mode="json") for key in result.artifact_ids
            ]
            if result
            else [],
            "result_report_ref": self.state.rounds[result.round_id].worker_report_ref
            if result
            else None,
            "remote_inputs_at_delivery": delivery.get("remote_inputs", []),
            "reuse_links": node.reuse_links,
            "delivery_assessment": {
                key: report.get(key, []) for key in (
                    "acceptance", "remaining_work", "blocked_on", "inline_work_answers"
                )
            },
        }

    def _exploration(self, node, exclude=None):
        for key in reversed(node.exploration_round_ids):
            work = self.state.rounds[key]
            if key == exclude:
                continue
            if work.exploration_ref:
                saved = json.loads(Path(work.exploration_ref).read_text())
                return {
                    "round_id": key,
                    "ref": work.exploration_ref,
                    "brief": saved["exploration"],
                    "judge_error": work.error,
                }
            if work.decision_ids:
                record = self.state.action_decisions[work.decision_ids[0]]
                return {
                    "round_id": key,
                    "ref": record.request_ref,
                    "brief": json.loads(Path(record.request_ref).read_text())["exploration"],
                    "judge_error": work.error,
                }
        return None

    def context_for(self, work):
        state, node = self.state, self.state.nodes[work.task_id]
        previous = next(
            (
                state.rounds[key]
                for key in reversed(node.exploration_round_ids)
                if key != work.id
                and (state.rounds[key].exploration_ref or state.rounds[key].decision_ids)
            ),
            None,
        )
        watched = dict(previous.routing_fingerprints) if previous else {}
        if node.result_history:
            watched.update(node.result_history[-1].input_fingerprints)
        current = file_versions(state, watched)
        last_visit = None
        previous_request = {}
        if previous:
            last_visit = self._exploration(node, exclude=work.id)
            if previous.decision_ids:
                decision = state.action_decisions[previous.decision_ids[0]]
                previous_request = json.loads(Path(decision.request_ref).read_text())
                last_visit["decision"] = decision.model_dump(mode="json")
        last_drill_down = next(
            (
                r
                for r in reversed(list(state.rounds.values()))
                if r.task_id == node.id
                and r.kind == RoundKind.DECOMPOSE
                and r.phase == RoundPhase.COMMITTED
                and r.worker_report_ref
            ),
            None,
        )
        drill_report = (
            json.loads(Path(last_drill_down.worker_report_ref).read_text())
            if last_drill_down else None
        )
        return {
            "protocol_version": 3,
            "run_id": state.run_id,
            "kind": work.kind.value,
            "root_goal": state.user_request,
            "root_approval": state.root_approval.model_dump(mode="json")
            if state.root_approval
            else None,
            "root": self._node_view(state.root_task_id),
            "task_id": node.id,
            "round_id": work.id,
            "spec_revision": node.spec.revision,
            "task": self._node_view(node.id),
            "purpose": work.purpose,
            "workspace": str(state.workspace),
            "input_refs": state.input_refs,
            "background": state.background,
            "preparation": preparation_view(state, include_documents=False),
            "ancestors": [self._node_view(key) for key in ancestors(state, node.id)],
            "children": [self._node_view(key) for key in state.children(node.id)],
            "dependencies": [self._node_view(key) for key in node.depends_on],
            "catalog": [
                {
                    "id": n.id,
                    "parent_id": n.parent_id,
                    "status": n.status.value,
                    "title": n.spec.title,
                    "goal": n.spec.goal,
                    "scope": n.spec.scope,
                    "must_solve": must_solve_data(n.must_solve),
                    "exploration_ref": state.rounds[n.exploration_round_ids[-1]].exploration_ref
                    if n.exploration_round_ids
                    else None,
                }
                for n in state.nodes.values()
                if n.specs
            ],
            "last_visit": last_visit,
            "last_drill_down": drill_report,
            "inline_work": (drill_report or {}).get("inline_work", []),
            "result_changes": {
                "previous_own_revision": (previous_request.get("task", {}).get("result") or {}).get(
                    "revision", 0
                ),
                "current_own_revision": node.result_revision,
                "previous_children": previous_request.get("controller_facts", {}).get(
                    "child_result_versions", {}
                ),
                "current_children": result_versions(state, state.children(node.id)),
                "previous_dependencies": previous_request.get("controller_facts", {}).get(
                    "dependency_result_versions", {}
                ),
                "current_dependencies": result_versions(state, node.depends_on),
            },
            "changes_since_visit": {
                path: {"before": watched[path], "now": value}
                for path, value in current.items()
                if value != watched[path]
            },
            "parent_integration_required": bool(state.children(node.id) or node.depends_on),
            "user_acceptance": [
                v.model_dump(mode="json")
                for v in state.verifications.values()
                if v.task_id == node.id and v.method == "user"
            ],
            "failed_rounds": [
                {
                    "kind": r.kind.value,
                    "error": r.error,
                    "report_ref": r.worker_report_ref,
                    "exploration_ref": r.exploration_ref,
                }
                for r in state.rounds.values()
                if r.task_id == node.id and r.error
            ],
        }

    def _local_evidence(self, refs):
        paths = []
        for ref in refs:
            if ref in self.state.artifacts:
                ref = self.state.artifacts[ref].path
            if external_input_ref(ref):
                continue
            path = input_path(self.state.workspace, ref, read_roots(self.state))
            if not path.is_file():
                raise ValueError(f"evidence must reference an actual file or source URL: {ref}")
            paths.append(str(path))
        return paths

    def _finish_issues(self, task_id, verdict, *, cache=None, memo=None):
        node, state = self.state.nodes[task_id], self.state
        cache, memo = ({} if cache is None else cache), ({} if memo is None else memo)
        issues = list(verdict.remaining_work) + list(verdict.blocked_on)
        assessments = {item.criterion_id: item for item in verdict.acceptance}
        for criterion in node.spec.acceptance:
            if not criterion.required:
                continue
            checked = assessments.get(criterion.id)
            if not checked or checked.status != "PASS" or not checked.evidence_refs:
                issues.append(f"required criterion lacks current PASS evidence: {criterion.id}")
            if criterion.check_kind == "user" and not any(
                v.task_id == node.id
                and v.criterion_id == criterion.id
                and v.status == "PASS"
                and v.method == "user"
                and v.criterion_revision == criterion.revision
                and file_versions(state, v.input_fingerprints, cache) == v.input_fingerprints
                for v in state.verifications.values()
            ):
                issues.append(f"user confirmation missing: {criterion.id}")
        if not node.result_history:
            issues.append("task has no own executed delivery")
            return issues
        result = node.result_history[-1]
        issues.extend(answer_issues(node, result.goal_answer, result.must_solve_answers))
        checked_items = {item.must_solve: item for item in verdict.must_solve_assessments}
        for contract in node.must_solve:
            problem = contract.problem if isinstance(contract, MustSolveItem) else contract
            checked = checked_items.get(problem)
            if not checked or checked.status != "PASS" or not checked.evidence_refs:
                issues.append(f"must_solve lacks current PASS evidence: {problem}")
        for contract, answer in zip(node.must_solve, result.must_solve_answers):
            if isinstance(contract, MustSolveItem):
                if answer.verification_status != "PASS":
                    issues.append(f"must_solve Worker verification did not pass: {contract.problem}")
                if not answer.evidence_refs:
                    issues.append(f"must_solve answer lacks actual evidence: {contract.problem}")
        if result.child_result_versions != result_versions(state, state.children(node.id)):
            issues.append("parent must integrate current children into its own delivery")
        if result.dependency_result_versions != result_versions(state, node.depends_on):
            issues.append("task must consume current dependency results")
        if any(
            not completion_current(state, key, memo=memo, cache=cache)
            for key in state.children(node.id) + node.depends_on
        ):
            issues.append("necessary child/dependency not currently finished")
        if file_versions(state, result.input_fingerprints, cache) != result.input_fingerprints:
            issues.append("delivery inputs or products changed since execution")
        result_round = state.rounds[result.round_id]
        report = json.loads(Path(result_round.worker_report_ref).read_text())
        inline = report.get("inline_work_answers", [])
        if [item.get("must_solve") for item in inline] != self._inline_work(result_round):
            issues.append("inline work lacks an answer for every planned item")
        for item in inline:
            if item.get("verification_status") != "PASS" or not item.get("evidence_refs"):
                issues.append(f"inline work lacks PASS evidence: {item.get('must_solve')}")
        if result_round.delivery_ref:
            delivery = json.loads(Path(result_round.delivery_ref).read_text())
            source = state.rounds.get(result_round.authorization_round_id or "")
            required_remote = set()
            if source and source.exploration_ref:
                original = json.loads(Path(source.exploration_ref).read_text())["exploration"]
                required_remote = {
                    ref for ref in original.get("required_input_paths", []) if external_input_ref(ref)
                }
            latest = self._exploration(node)
            observed = {
                o["ref"]: o for o in (latest or {}).get("brief", {}).get("remote_inputs", [])
            }
            for old in delivery.get("remote_inputs", []):
                if old["ref"] not in required_remote:
                    continue  # A cited public source is evidence, not a required live remote input.
                current = observed.get(old["ref"])
                if (
                    not current
                    or not current["accessible"]
                    or current["observed_version"] != old["observed_version"]
                ):
                    issues.append(
                        "remote input must be rechecked and match the delivered version: "
                        + old["ref"]
                    )
        for path in node.spec.deliverables:
            if not workspace_path(state.workspace, path).is_file():
                issues.append(f"required deliverable missing: {path}")
        return issues

    def _decide(self, work, context, brief):
        # A retry of the tool returns the same verdict; it cannot shop for another answer.
        if work.decision_ids:
            record = self.state.action_decisions[work.decision_ids[0]]
            return {"ok": True, "decision_id": record.id, **record.verdict.model_dump(mode="json")}
        if work.error and work.error.startswith("Luna unavailable:"):
            raise RuntimeError(work.error)
        self._local_evidence(brief.evidence_refs)
        paths = list(
            dict.fromkeys(
                brief.input_paths
                + brief.required_input_paths
                + self._local_evidence(brief.evidence_refs)
            )
        )
        observations = {o.ref: o for o in brief.remote_inputs}
        if len(observations) != len(brief.remote_inputs):
            raise ValueError("remote input observations must have unique references")
        for observation in observations.values():
            if not external_input_ref(observation.ref):
                raise ValueError("remote observation needs an ssh or URL reference")
            paths.extend(self._local_evidence(observation.evidence_refs))
        watched = file_versions(self.state, paths)
        for ref in brief.required_input_paths:
            if external_input_ref(ref) and ref not in observations:
                raise ValueError(
                    "required remote input needs accessibility/version probe evidence: " + ref
                )
        # Save the exploration before calling the provider, including failed calls.
        work.exploration_ref = self.store.write_json(
            f"explorations/{work.id}-{digest([brief, watched])}.json",
            {"exploration": brief.model_dump(mode="json"), "input_fingerprints": watched},
        )
        work.routing_fingerprints = watched
        node = self.state.nodes[work.task_id]
        if work.kind == RoundKind.EXPLORE and work.id not in node.exploration_round_ids:
            node.exploration_round_ids.append(work.id)
        self.save()
        request = {
            **judge_context(context, brief),
            "kind": "JUDGE",
            "controller_facts": {
                "child_result_versions": result_versions(
                    self.state, self.state.children(work.task_id)
                ),
                "dependency_result_versions": result_versions(
                    self.state, self.state.nodes[work.task_id].depends_on
                ),
                "input_fingerprints": watched,
                "missing_required_inputs": [
                    p
                    for p in brief.required_input_paths
                    if (
                        not observations[p].accessible
                        if external_input_ref(p)
                        else watched[p] == "MISSING"
                    )
                ],
                "remote_inputs": [o.model_dump(mode="json") for o in brief.remote_inputs],
            },
        }
        key = "action_" + digest(
            [
                work.id,
                request,
                ACTION_JUDGE_INSTRUCTIONS,
                self.state.execution_profile.decision_model,
            ]
        )
        request_ref = self.store.write_json(f"actions/{key}/request.json", request)
        sent_request = bounded_context(request, request_ref)
        sent_request["context_policy"] = (
            "Decide from this packet only. Do not open full_context_ref, omitted_to references "
            "or evidence files. If omitted evidence is material to THIS node's obligation, mark "
            "it UNKNOWN and choose execute or drill_down. An omitted catalog is not evidence "
            "that peer tasks do not exist; the planner checks duplicates."
        )
        self.store.write_json(f"actions/{key}/sent-request.json", sent_request)
        trace = {}

        def started(thread_id, turn_id=None):
            trace.update(thread_id=thread_id, turn_id=turn_id or "")
            self.store.write_json(
                f"actions/{key}/traces/{thread_id}-{turn_id or 'start'}.json", trace
            )

        def owner(value):
            work.decision_execution_owner = value
            self.save()

        def validate_verdict(verdict):
            allowed = {c.id for c in self.state.nodes[work.task_id].spec.acceptance}
            ids = [item.criterion_id for item in verdict.acceptance]
            if len(ids) != len(set(ids)) or not set(ids) <= allowed:
                raise ValueError("Luna acceptance must use unique current-task criterion IDs")
            problems = [
                item.problem if isinstance(item, MustSolveItem) else item
                for item in self.state.nodes[work.task_id].must_solve
            ]
            assessed = [item.must_solve for item in verdict.must_solve_assessments]
            if assessed != problems:
                raise ValueError("Luna must_solve assessments must cover every current-task item in order")
            if any(item.status == "PASS" and not item.evidence_refs
                   for item in verdict.must_solve_assessments):
                raise ValueError("Luna must_solve PASS needs actual evidence")
            if verdict.action == "finish" and (verdict.remaining_work or verdict.blocked_on):
                raise ValueError("finish must have no current-task remaining_work or blocked_on")
            for item in verdict.acceptance:
                self._local_evidence(item.evidence_refs)
            for item in verdict.must_solve_assessments:
                self._local_evidence(item.evidence_refs)

        response_path = self.store.directory / "actions" / key / "response.json"
        error_path = response_path.with_name("error.json")
        if error_path.exists():
            raise RuntimeError(
                "Luna failed in this visit; preserve evidence and retry in a fresh visit"
            )
        if response_path.exists():
            from .models import ActionVerdict

            saved = json.loads(response_path.read_text())
            verdict, trace = ActionVerdict.model_validate(saved["verdict"]), saved["trace"]
        else:
            try:
                verdict = self.worker.run_action_decision(
                    sent_request,
                    started,
                    min(self.state.execution_profile.decision_timeout_seconds, self.time_left()),
                    owner,
                    validator=validate_verdict,
                )
            except Exception as error:
                self.store.write_json(f"actions/{key}/error.json", {"error": str(error)})
                work.error = f"Luna unavailable: {type(error).__name__}: {error}"
                self.save()
                raise
        validate_verdict(verdict)
        for item in verdict.acceptance:
            paths.extend(self._local_evidence(item.evidence_refs))
        for item in verdict.must_solve_assessments:
            paths.extend(self._local_evidence(item.evidence_refs))
        after = file_versions(self.state, paths)
        if any(after[path] != value for path, value in watched.items()):
            raise ValueError("evidence changed while Luna was checking; start a fresh visit")
        watched = after
        response_ref = self.store.write_json(
            f"actions/{key}/response.json",
            {
                "verdict": verdict.model_dump(mode="json"),
                "trace": trace,
            },
        )
        record = ActionDecision(
            id=key,
            task_id=work.task_id,
            round_id=work.id,
            input_digest=digest(request),
            verdict=verdict,
            request_ref=request_ref,
            response_ref=response_ref,
            model=self.state.execution_profile.decision_model,
            **trace,
        )
        self.state.action_decisions[key] = record
        work.decision_ids = [key]
        work.routing_fingerprints = watched
        work.decision_execution_owner = {}
        self.save()
        return {"ok": True, "decision_id": key, **verdict.model_dump(mode="json")}

    def _commit_visit(self, work, report):
        if report.decision_id not in work.decision_ids:
            raise ValueError("Explore must return the real bound task_tree_decide result")
        decision = self.state.action_decisions[report.decision_id]
        node = self.state.nodes[work.task_id]
        if work.id not in node.exploration_round_ids:
            node.exploration_round_ids.append(work.id)
        self._commit_round(work)
        self._route_action(work, decision)

    def _route_action(self, work, decision):
        node = self.state.nodes[work.task_id]
        verdict = decision.verdict
        request = json.loads(Path(decision.request_ref).read_text())
        missing = request["controller_facts"]["missing_required_inputs"]
        if file_versions(self.state, work.routing_fingerprints) != work.routing_fingerprints:
            self._retry_visit(
                node, work, "referenced evidence changed after decision; inspect changes"
            )
        elif verdict.action != "drill_down" and (verdict.blocked_on or missing):
            node.status = NodeStatus.BLOCKED
            node.block_reason = "; ".join(verdict.blocked_on + missing)
        elif verdict.action == "finish":
            cache, memo = self.audit_files()
            issues = self._finish_issues(node.id, verdict, cache=cache, memo=memo)
            if issues:
                self._retry_visit(node, work, "finish rejected: " + "; ".join(issues))
            else:
                certificate = CompletionCertificate(
                    id=identifier("completion"),
                    task_id=node.id,
                    result_revision=node.result_revision,
                    spec_revision=node.spec.revision,
                    plan_revision_id=None,
                    verification_ids=[],
                    coverage_digest=digest([verdict.acceptance, verdict.must_solve_assessments]),
                    input_digest=digest(
                        [
                            result_basis(self.state, node.id),
                            work.routing_fingerprints,
                        ]
                    ),
                    decision_ids=[decision.id],
                )
                self.state.completion_certificates[certificate.id] = certificate
                node.completion_certificate_id, node.status = certificate.id, NodeStatus.SUCCEEDED
                if node.id == self.state.root_task_id:
                    self.state.status = RunStatus.SUCCEEDED
        else:
            if verdict.action == "execute" and any(
                not completion_current(self.state, key, check_files=False)
                for key in self.state.children(node.id) + node.depends_on
            ):
                self._retry_visit(
                    node,
                    work,
                    "children remain blocked; drill_down to replace an invalid approach, or report the external blocker; do not absorb their unfinished work",
                )
                self.save()
                return
            node.status = NodeStatus.READY
            queue_round(
                self.state,
                node.id,
                RoundKind.EXECUTE if verdict.action == "execute" else RoundKind.DECOMPOSE,
                f"action:{decision.id}",
                authorization_round_id=work.id,
                purpose=verdict.reason,
            )
        self.save()

    def _commit_round(self, work):
        work.phase, work.finished_at = RoundPhase.COMMITTED, time.time()
        self.state.scheduler.active_work = None

    def _retry_visit(self, node, work, reason):
        node.no_progress_rounds += 1
        work.error = reason
        if node.no_progress_rounds >= self.state.execution_profile.max_node_no_progress:
            node.status, node.block_reason = NodeStatus.BLOCKED, reason
        else:
            incomplete = any(
                not completion_current(self.state, key, check_files=False)
                for key in self.state.children(node.id) + node.depends_on
            )
            self.enqueue(
                node.id,
                f"retry:{work.id}",
                "replan_blocked_inputs" if incomplete and self._settled_inputs(node.id) else reason,
            )

    def _commit_split(self, work, report):
        original = self.state
        self.state = draft = original.model_copy(deep=True)
        try:
            work = draft.rounds[work.id]
            parent = draft.nodes[work.task_id]
            previous_inputs = set(parent.child_ids + parent.depends_on)
            proposed = [child.id for child in report.children]
            if len(proposed) != len(set(proposed)) or set(proposed) & set(draft.nodes):
                raise ValueError(
                    "child proposal IDs must be unique and distinct from existing task IDs"
                )
            if not report.children and not report.no_deeper_question_reason.strip():
                raise ValueError("an empty drill_down needs its actual no_deeper_question_reason")
            mapping, additions = {}, []
            replaced = {item.task_id: item for item in report.replacements}
            if len(replaced) != len(report.replacements) or len(
                {r.replacement_id for r in report.replacements}
            ) != len(report.replacements):
                raise ValueError("replacement mappings must be one-to-one")
            for old_id, replacement in replaced.items():
                if old_id not in parent.child_ids or replacement.replacement_id not in proposed:
                    raise ValueError("replace only a direct child with a new child proposal")
                old = draft.nodes[old_id]
                if old.status != NodeStatus.BLOCKED:
                    raise ValueError("only a blocked child may be replaced")
                proposal = next(c for c in report.children if c.id == replacement.replacement_id)
                candidate_spec = proposal.spec.model_copy(deep=True)
                candidate_spec.constraints = {
                    **parent.spec.constraints,
                    **candidate_spec.constraints,
                }
                if proposal.reuse_task_id or responsibility(
                    old.spec, old.must_solve, old.root_contribution
                ) != responsibility(
                    candidate_spec, proposal.must_solve, proposal.root_contribution
                ):
                    raise ValueError(
                        "replacement must preserve the full original responsibility, deliverables and acceptance"
                    )
            for child in report.children:
                spec = child.spec.model_copy(deep=True)
                for key, value in parent.spec.constraints.items():
                    if key in spec.constraints and spec.constraints[key] != value:
                        raise ValueError("child cannot change inherited constraints")
                    spec.constraints[key] = value
                source = child.reuse_task_id
                if source is None:
                    source = next(
                        (
                            n.id
                            for n in draft.nodes.values()
                            if n.specs
                            and n.status != NodeStatus.SUPERSEDED
                            and n.id not in replaced
                            and responsibility(n.spec, n.must_solve, n.root_contribution)
                            == responsibility(spec, child.must_solve, child.root_contribution)
                        ),
                        None,
                    )
                if source:
                    if source not in draft.nodes or source in {
                        parent.id,
                        *ancestors(draft, parent.id),
                    }:
                        raise ValueError("cannot reuse an ancestor, self, or missing task")
                    existing = draft.nodes[source]
                    if existing.status == NodeStatus.SUPERSEDED or responsibility(
                        existing.spec, existing.must_solve, existing.root_contribution
                    ) != responsibility(spec, child.must_solve, child.root_contribution):
                        raise ValueError(
                            "reuse requires matching full responsibility and acceptance; partial overlap needs a separate task"
                        )
                    mapping[child.id] = source
                    parent.reuse_links[child.id] = source
                    if source not in parent.depends_on and source not in draft.children(parent.id):
                        parent.depends_on.append(source)
                    continue
                new_id = identifier("task")
                node = TaskNode(
                    id=new_id,
                    parent_id=parent.id,
                    depth=parent.depth + 1,
                    created_seq=draft.sequence(),
                    session_mode=child.session_mode,
                    created_by_round_id=work.id,
                    specs=[spec],
                    must_solve=child.must_solve,
                    root_contribution=child.root_contribution,
                    planning_reason=child.reason,
                )
                draft.nodes[new_id] = node
                validate_spec(draft, new_id, spec)
                parent.child_ids.append(new_id)
                mapping[child.id] = new_id
                additions.append((child, node))
            for child, node in additions:
                node.depends_on = list(
                    dict.fromkeys(mapping.get(key, key) for key in child.depends_on)
                )
                queue_round(draft, node.id, RoundKind.EXPLORE, f"child:{work.id}")
            added_ids = {child.id for child, _ in additions}
            for child in report.children:
                if child.id not in added_ids:
                    parent.depends_on = list(
                        dict.fromkeys(
                            parent.depends_on + [mapping.get(key, key) for key in child.depends_on]
                        )
                    )
            for old_id, replacement in replaced.items():
                new_id = mapping[replacement.replacement_id]
                retired, todo = set(), [old_id]
                while todo:
                    key = todo.pop()
                    if key not in retired:
                        retired.add(key)
                        todo.extend(draft.children(key))
                for other in draft.nodes.values():
                    if other.id in retired:
                        continue
                    if set(other.depends_on) & (retired - {old_id}):
                        raise ValueError(
                            "replacement would strand a dependency on a descendant; replan that consumer explicitly"
                        )
                    other.depends_on = list(
                        dict.fromkeys(new_id if key == old_id else key for key in other.depends_on)
                    )
                    other.reuse_links = {
                        key: new_id if target == old_id else target
                        for key, target in other.reuse_links.items()
                    }
                for key in retired:
                    old = draft.nodes[key]
                    old.status, old.completion_certificate_id = NodeStatus.SUPERSEDED, None
                    old.replaced_by_task_id = new_id
                    for pending in draft.rounds.values():
                        if pending.task_id == key and pending.phase not in TERMINAL_ROUNDS:
                            if pending.phase != RoundPhase.QUEUED:
                                raise ValueError("cannot replace a child with live work")
                            pending.phase, pending.error = RoundPhase.CANCELLED, replacement.reason
                parent.child_ids.remove(old_id)
            validate_wait_graph(draft)
            for text in report.suggestions:
                draft.suggestions.append(
                    Suggestion(
                        candidate_id=identifier("suggestion"),
                        description=text,
                        decision_id=work.authorization_round_id,
                    )
                )
            self._commit_round(work)
            if set(parent.child_ids + parent.depends_on) != previous_inputs:
                parent.status, parent.no_progress_rounds = NodeStatus.WAITING_CHILDREN, 0
            elif any(
                not completion_current(draft, key, check_files=False)
                for key in parent.child_ids + parent.depends_on
            ):
                parent.status = NodeStatus.WAITING_CHILDREN
            else:
                parent.status = NodeStatus.READY
                followup = queue_round(
                    draft, parent.id, RoundKind.EXECUTE, f"inline:{work.id}",
                    authorization_round_id=work.authorization_round_id,
                    purpose=report.no_deeper_question_reason,
                )
                followup.routing_basis = "inline_after_empty_drill"
            # Other branch workers still hold these round objects while the draft is committed.
            for ref in original.scheduler.active_siblings:
                if ref.id != work.id:
                    draft.rounds[ref.id] = original.rounds[ref.id]
            self.save()
        except BaseException:
            self.state = original
            raise

    def _freeze_delivery(self, work, report, context):
        """Persist one immutable receipt before state advances; never restamp it on replay."""
        node, state = self.state.nodes[work.task_id], self.state
        artifacts = []
        for relative in dict.fromkeys(report.deliverable_paths):
            path = workspace_path(state.workspace, relative)
            sha, snapshot = self.store.snapshot(path)
            artifact = ArtifactRef(
                id=identifier("artifact"),
                path=str(path),
                sha256=sha,
                snapshot_ref=snapshot,
                task_id=node.id,
                round_id=work.id,
            )
            artifacts.append(artifact.model_dump(mode="json"))
        evidence = self._local_evidence(
            report.evidence_refs
            + [ref for a in report.acceptance for ref in a.evidence_refs]
            + [ref for a in report.inline_work_answers for ref in a.evidence_refs]
        )
        paths = list(dict.fromkeys(report.input_paths + report.deliverable_paths + evidence))
        # Retain routing dependencies too; own writes are fingerprinted AFTER execution.
        source = state.rounds[work.authorization_round_id]
        paths = list(dict.fromkeys(paths + list(source.routing_fingerprints)))
        versions = file_versions(state, paths)
        for artifact in artifacts:
            if file_versions(state, [artifact["path"]])[artifact["path"]] != artifact["sha256"]:
                raise ValueError("deliverable changed while freezing the delivery receipt")
        # Versions handed to the worker are the consumed versions, even after recovery.
        receipt = {
            "report": report.model_dump(mode="json"),
            "artifacts": artifacts,
            "input_fingerprints": versions,
            "child_result_versions": {
                n["id"]: (n["result"] or {}).get("revision", 0) for n in context["children"]
            },
            "dependency_result_versions": {
                n["id"]: (n["result"] or {}).get("revision", 0) for n in context["dependencies"]
            },
            "remote_inputs": json.loads(Path(source.exploration_ref).read_text())[
                "exploration"
            ].get("remote_inputs", [])
            if source.exploration_ref
            else [],
        }
        # A changed read-only input during execution cannot be certified as consumed.
        own_changes = {
            str(workspace_path(state.workspace, p))
            for p in report.changed_paths + report.deliverable_paths
        }
        branch = self._root_branch(work.task_id)
        if branch in state.branch_workspaces:
            own_changes.update(
                str(workspace_path(Path(state.branch_workspaces[branch]), p))
                for p in report.changed_paths + report.deliverable_paths
            )
        for path, old in source.routing_fingerprints.items():
            if (
                versions[path] != old
                and str(input_path(state.workspace, path, read_roots(state))) not in own_changes
            ):
                receipt["input_fingerprints"][path] = old
        work.delivery_ref = self.store.write_json(f"reports/{work.id}/delivery.json", receipt)

    def _commit_execution(self, work, report):
        node, state = self.state.nodes[work.task_id], self.state
        if not work.delivery_ref:
            # Historical/unreceipted outputs are still available as partial work, not new evidence.
            work.phase, work.error = (
                RoundPhase.INTERRUPTED,
                "delivery has no frozen input receipt; inspect saved partial result",
            )
            state.scheduler.active_work = None
            self.enqueue(node.id, f"unreceipted:{work.id}", work.error)
            self.save()
            return
        receipt = json.loads(Path(work.delivery_ref).read_text())
        if ExecutionReport.model_validate(receipt["report"]).model_dump(mode="json") != report.model_dump(mode="json"):
            raise ValueError("delivery receipt does not match persisted report")
        artifacts = [ArtifactRef.model_validate(a) for a in receipt["artifacts"]]
        for artifact in artifacts:
            state.artifacts[artifact.id] = artifact
        result = NodeResult(
            revision=node.result_revision + 1,
            round_id=work.id,
            summary=report.summary,
            goal_answer=report.goal_answer,
            must_solve_answers=report.must_solve_answers,
            artifact_ids=[a.id for a in artifacts],
            child_result_versions=receipt["child_result_versions"],
            dependency_result_versions=receipt["dependency_result_versions"],
            input_fingerprints=receipt["input_fingerprints"],
            evidence=report.evidence_refs,
        )
        substantive = digest(
            [
                result.input_fingerprints,
                result.child_result_versions,
                result.dependency_result_versions,
                [(a.criterion_id, a.status) for a in report.acceptance],
                # New source evidence is progress even without a local product change.
                # Rewording summaries/reasons or minting result IDs is not progress.
                sorted(
                    set(
                        report.evidence_refs
                        + [ref for a in report.acceptance for ref in a.evidence_refs]
                        + [ref for a in report.inline_work_answers for ref in a.evidence_refs]
                    )
                ),
            ]
        )
        already_recorded = bool(node.result_history and node.result_history[-1].round_id == work.id)
        unchanged = node.last_progress_digest == substantive and not already_recorded
        if not already_recorded:
            node.last_progress_digest = substantive
            node.result_history.append(result)
            node.completion_certificate_id = None
            self.save()  # A recovered Luna call must not append this delivery twice.
        self.audit_files()
        if unchanged:
            self._commit_round(work)
            self._retry_visit(
                node, work, "execution repeated without changed evidence, outcomes or inputs"
            )
        else:
            node.no_progress_rounds = 0
            source = state.rounds[work.authorization_round_id]
            prior = ExploreBrief.model_validate(
                json.loads(Path(source.exploration_ref).read_text())["exploration"]
            )
            if any(external_input_ref(ref) for ref in prior.required_input_paths) or (
                file_versions(state, result.input_fingerprints) != result.input_fingerprints
            ):
                self._commit_round(work)
                self.enqueue(node.id, f"changed:{work.id}", "recheck current inputs")
            else:
                brief = ExploreBrief(
                    summary=report.summary,
                    findings=[
                        f"{item.criterion_id}: {item.reason}"
                        for item in report.acceptance
                    ],
                    remaining_work=report.remaining_work + report.blocked_on,
                    evidence_refs=list(dict.fromkeys(
                        report.evidence_refs
                        + report.deliverable_paths
                        + [ref for item in report.acceptance for ref in item.evidence_refs]
                    )),
                    input_paths=list(dict.fromkeys(
                        report.input_paths + report.changed_paths + report.deliverable_paths
                    )),
                    required_input_paths=prior.required_input_paths,
                    remote_inputs=prior.remote_inputs,
                )
                decision_id = self._decide(work, self.context_for(work), brief)["decision_id"]
                self._commit_round(work)
                self._route_action(work, state.action_decisions[decision_id])
        self.save()

    def _validate_report(self, work, report, workspace=None):
        workspace = workspace or self.state.workspace
        for name, value in {
            "task_id": work.task_id,
            "round_id": work.id,
            "spec_revision": self.state.nodes[work.task_id].spec.revision,
            "input_digest": work.input_digest,
        }.items():
            if getattr(report, name) != value:
                raise ValueError(f"report identity mismatch: {name}")
        if isinstance(report, VisitReport):
            if report.decision_id not in work.decision_ids:
                if work.error and work.error.startswith("Luna unavailable:"):
                    raise WorkerReportError(work.error)
                raise ValueError("call task_tree_decide and return its actual decision_id")
        elif isinstance(report, ExecutionReport):
            issues = answer_issues(
                self.state.nodes[work.task_id], report.goal_answer, report.must_solve_answers
            )
            inline = self._inline_work(work)
            if [item.must_solve for item in report.inline_work_answers] != inline:
                issues.append("inline_work_answers must match every planned inline item in order")
            if any(not item.answer.strip() or item.verification_status is None
                   or not item.verification_result.strip()
                   for item in report.inline_work_answers):
                issues.append("each inline work answer needs a result and verification")
            if issues:
                raise ValueError("; ".join(issues))
            for path in report.deliverable_paths:
                if not workspace_path(workspace, path).is_file():
                    raise ValueError(f"missing declared deliverable: {path}")
            self._local_evidence(report.evidence_refs)
            for assessment in report.acceptance:
                self._local_evidence(assessment.evidence_refs)
            for answer in report.must_solve_answers:
                self._local_evidence(answer.evidence_refs)
            for answer in report.inline_work_answers:
                self._local_evidence(answer.evidence_refs)
            file_versions(self.state, report.input_paths)
            for path in report.changed_paths:
                workspace_path(workspace, path)
            ids = [a.criterion_id for a in report.acceptance]
            if len(ids) != len(set(ids)) or not set(ids) <= {
                c.id for c in self.state.nodes[work.task_id].spec.acceptance
            }:
                raise ValueError("execution acceptance must map current task criteria")

    def _commit_report(self, work, report):
        if work.kind in {RoundKind.PREPARE, RoundKind.SPECIFY}:
            return super()._commit_report(work, report)
        # EXECUTE was validated before freezing; files may legitimately change before replay.
        if work.kind != RoundKind.EXECUTE:
            self._validate_report(work, report)
        if work.kind == RoundKind.EXPLORE:
            self._commit_visit(work, report)
        elif work.kind == RoundKind.DECOMPOSE:
            self._commit_split(work, report)
        elif work.kind == RoundKind.EXECUTE:
            self._commit_execution(work, report)
        else:
            raise ValueError("legacy review/reconcile phases cannot run in protocol 3")

    def _run_round(self, round_id):
        work = self.state.rounds[round_id]
        if work.kind in {RoundKind.PREPARE, RoundKind.SPECIFY}:
            return super()._run_round(round_id)
        model = REPORTS[work.kind]
        if work.worker_report_ref and work.phase in {RoundPhase.VERIFYING, RoundPhase.DECIDING}:
            return self._commit_report(
                work, model.model_validate_json(Path(work.worker_report_ref).read_text())
            )
        inputs = self.state.children(work.task_id) + self.state.nodes[work.task_id].depends_on
        if work.kind == RoundKind.EXECUTE and inputs:
            self.audit_files()
            if any(not completion_current(self.state, key, check_files=False) for key in inputs):
                work.phase, work.error = (
                    RoundPhase.CANCELLED,
                    "child/dependency evidence changed before integration",
                )
                self.state.scheduler.active_work = None
                self.state.nodes[work.task_id].status = NodeStatus.WAITING_CHILDREN
                self.save()
                return
        context = self.context_for(work)
        node = self.state.nodes[work.task_id]
        thread_id = self._session_thread(work.task_id)
        if thread_id:
            context["continuation_thread_id"] = thread_id
        if work.authorization_round_id:
            source = self.state.rounds[work.authorization_round_id]
            record = self.state.action_decisions[source.decision_ids[0]]
            expected = (
                "drill_down" if work.routing_basis == "inline_after_empty_drill"
                or work.kind == RoundKind.DECOMPOSE else "execute"
            )
            if record.verdict.action != expected:
                raise ValueError("phase does not match its bound Luna action")
            if (
                file_versions(self.state, source.routing_fingerprints)
                != source.routing_fingerprints
            ):
                work.phase, work.error = (
                    RoundPhase.CANCELLED,
                    "referenced inputs changed after Luna decision",
                )
                self.state.scheduler.active_work = None
                self.enqueue(work.task_id, f"changed:{work.id}", work.error)
                self.save()
                return
            if thread_id:
                context["continuation_thread_id"] = thread_id
            context["action_decision"] = record.model_dump(mode="json")
        work.input_digest = digest([context, VISIT_INSTRUCTIONS, model.model_json_schema()])
        context["input_digest"] = work.input_digest
        work.context_manifest_ref = self.store.write_json(f"contexts/{work.id}.json", context)
        sent_context = bounded_context(context, work.context_manifest_ref)
        self.store.write_json(f"contexts/{work.id}-sent.json", sent_context)
        work.phase = RoundPhase.WORKER_RUNNING
        work.worker_model, work.worker_effort = (
            self.state.execution_profile.worker_model,
            self.state.execution_profile.worker_effort,
        )
        work.execution_owner = self.worker.execution_owner()
        self.state.progress.worker_attempts_started += 1
        self.store.execution_reservation([work.id])
        self.save()

        def started(thread_id, turn_id=None):
            work.thread_id, work.turn_id = thread_id, turn_id
            self._remember_thread(work.task_id, thread_id)
            self.save()

        def feedback(diagnostics, raw):
            self.store.write_json(
                f"reports/{work.id}/feedback/{diagnostics['attempt']}.json",
                {"diagnostics": diagnostics, "raw_report": raw},
            )

        args = (
            sent_context,
            model,
            started,
            min(self.state.execution_profile.worker_timeout_seconds, self.time_left()),
        )
        kwargs = {
            "validator": lambda report: self._validate_report(work, report),
            "on_feedback": feedback,
        }
        try:
            if work.kind == RoundKind.EXPLORE:
                report = self.worker.run_action_visit(
                    *args, submit=lambda brief: self._decide(work, context, brief), **kwargs
                )
            else:
                report = self.worker.run_with_feedback(*args, **kwargs)
        except (WorkerReportError, TimeoutError):
            # The adapter confirms interruption for TimeoutError; unconfirmed work uses a different error.
            self.store.execution_reservation([])
            work.phase = RoundPhase.FAILED
            raise
        self.store.execution_reservation([])
        if getattr(self.worker, "last_usage", None):
            self.state.progress.recorded_usage.append(
                {"round_id": work.id, "usage": self.worker.last_usage}
            )
        work.worker_report_ref = self.store.write_json(
            f"reports/{work.id}/worker.json", report.model_dump(mode="json")
        )
        if work.kind == RoundKind.EXECUTE:
            self._freeze_delivery(work, report, context)
        work.phase = RoundPhase.VERIFYING
        self.save()
        self._commit_report(work, report)

    def refresh_parents(self):
        state = self.state
        memo = {}
        for node in sorted(state.nodes.values(), key=lambda n: -n.depth):
            if node.status in {NodeStatus.SUPERSEDED, NodeStatus.BLOCKED, NodeStatus.RUNNING}:
                continue
            inputs = state.children(node.id) + node.depends_on
            ready = all(
                completion_current(state, key, check_files=False, memo=memo) for key in inputs
            )
            stale = node.status == NodeStatus.SUCCEEDED and not completion_current(
                state, node.id, check_files=False, memo=memo
            )
            if stale:
                node.completion_certificate_id = None
                node.status = NodeStatus.READY if ready else NodeStatus.WAITING_CHILDREN
            if ready and (
                stale
                or node.status in {NodeStatus.WAITING_CHILDREN, NodeStatus.WAITING_DEPENDENCIES}
            ):
                self.enqueue(
                    node.id,
                    f"inputs:{digest([result_versions(state, inputs), node.result_revision, state.revision])}",
                )
            elif (
                inputs and self._settled_inputs(node.id) and not ready and not self.pending(node.id)
            ):
                signature = digest(
                    [
                        (
                            key,
                            state.nodes[key].status,
                            state.nodes[key].block_reason,
                            state.nodes[key].result_revision,
                        )
                        for key in inputs
                    ]
                )
                if node.blocked_inputs_digest != signature:
                    node.blocked_inputs_digest = signature
                    if node.replan_attempts >= state.execution_profile.max_node_no_progress:
                        node.status, node.block_reason = (
                            NodeStatus.BLOCKED,
                            "blocked-child replanning limit reached; external help needed",
                        )
                    else:
                        node.replan_attempts += 1
                        self.enqueue(
                            node.id, f"blocked-inputs:{signature}", "replan_blocked_inputs"
                        )

    def _settled_inputs(self, task_id):
        """A blocked subtree can wake its parent, but runnable siblings finish first."""

        def settled(key):
            node = self.state.nodes[key]
            if node.status in {NodeStatus.BLOCKED, NodeStatus.SUCCEEDED}:
                return True
            inputs = self.state.children(key) + node.depends_on
            return (
                bool(inputs) and not self.pending(key) and all(settled(child) for child in inputs)
            )

        return all(
            settled(key)
            for key in self.state.children(task_id) + self.state.nodes[task_id].depends_on
        )

    def audit_files(self):
        """One shared snapshot at recovery/write boundaries, not on every queue refresh."""
        cache, memo = {}, {}
        preparation = self.state.preparations.get(self.state.current_preparation_id or "")
        if preparation and preparation.round_id != self._checked_preparation:
            for key in preparation.artifact_ids:
                artifact = self.state.artifacts[key]
                if (
                    not artifact.snapshot_ref
                    or file_versions(self.state, [artifact.snapshot_ref], cache).get(
                        artifact.snapshot_ref
                    )
                    != artifact.sha256
                ):
                    raise ValueError(
                        "shared preparation snapshot is missing or changed; prepare again"
                    )
            self._checked_preparation = preparation.round_id
        invalid = [
            node.id
            for node in self.state.nodes.values()
            if node.status == NodeStatus.SUCCEEDED
            and not completion_current(self.state, node.id, cache=cache, memo=memo)
        ]
        for key in invalid:
            node = self.state.nodes[key]
            node.completion_certificate_id = None
            node.status = NodeStatus.WAITING_CHILDREN
        if invalid and self.state.status == RunStatus.SUCCEEDED:
            self.state.status = RunStatus.ACTIVE
        self.refresh_parents()
        return cache, memo

    def recover_active(self):
        ref = self.state.scheduler.active_work
        if not ref:
            return
        work = self.state.rounds[ref.id]
        if work.decision_execution_owner:
            self.worker.ensure_stopped(work.decision_execution_owner)
        if work.phase != RoundPhase.WORKER_RUNNING:
            return
        receipt_path = self.store.directory / f"reports/{work.id}/delivery.json"
        if work.kind == RoundKind.EXECUTE and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            work.delivery_ref = str(receipt_path)
            work.worker_report_ref = self.store.write_json(
                f"reports/{work.id}/worker.json", receipt["report"]
            )
            work.phase = RoundPhase.VERIFYING
            self.store.execution_reservation([])
            self.save()
            return
        report = self.worker.recover(work, REPORTS[work.kind])
        self.store.execution_reservation([])
        if report is not None:
            work.worker_report_ref = self.store.write_json(
                f"reports/{work.id}/worker.json", report.model_dump(mode="json")
            )
            if work.kind != RoundKind.EXECUTE:
                work.phase = RoundPhase.VERIFYING
        if report is None or work.kind == RoundKind.EXECUTE:
            work.phase, work.finished_at, work.error = (
                RoundPhase.INTERRUPTED,
                time.time(),
                "stopped worker; inspect partial effects and any unreceipted recovered report",
            )
            self.state.scheduler.active_work = None
            if work.kind in {RoundKind.PREPARE, RoundKind.SPECIFY}:
                queue_round(self.state, work.task_id, work.kind, f"recover:{work.id}")
            else:
                self.enqueue(work.task_id, f"recover:{work.id}", work.error)
        self.save()

    def _ready_rounds(self):
        return [
            r
            for r in self.state.rounds.values()
            if r.phase == RoundPhase.QUEUED
            and self.state.nodes[r.task_id].status
            not in {NodeStatus.BLOCKED, NodeStatus.SUPERSEDED}
            and (r.kind in {RoundKind.PREPARE, RoundKind.SPECIFY} or self.state.root_approval)
            and (
                all(
                    completion_current(self.state, key, check_files=False)
                    for key in self.state.children(r.task_id)
                    + self.state.nodes[r.task_id].depends_on
                )
                or (
                    r.kind == RoundKind.EXPLORE
                    and r.purpose == "replan_blocked_inputs"
                    and self._settled_inputs(r.task_id)
                )
            )
        ]

    def claim_next_work(self):
        ready = self._ready_rounds()
        if not ready:
            return None
        work = min(ready, key=lambda r: (self.state.nodes[r.task_id].depth, r.created_seq))
        ref = WorkRef(kind="ROUND", id=work.id)
        self.state.scheduler.active_work = ref
        work.phase, work.started_at = RoundPhase.PREPARING, time.time()
        self.state.nodes[work.task_id].status = NodeStatus.RUNNING
        self.save()
        return ref

    def _parallel_candidates(self):
        slots = min(
            self.state.execution_profile.max_parallel_siblings,
            self.state.execution_profile.max_worker_rounds
            - self.state.progress.worker_attempts_started,
        )
        if slots < 1:
            return []
        ready = sorted(self._ready_rounds(), key=lambda r: (
            self.state.nodes[r.task_id].depth, r.created_seq
        ))
        if not ready:
            return []
        if self.state.branch_workspaces:
            if ready[0].task_id == self.state.root_task_id:
                return []
            busy = {
                self._root_branch(self.state.rounds[ref.id].task_id)
                for ref in self.state.scheduler.active_siblings
            }
            chosen = []
            for work in ready:
                branch = self._root_branch(work.task_id)
                if (branch not in self.state.branch_workspaces or branch in busy
                    or self._session_owner(work.task_id).id == self.state.root_task_id):
                    continue
                chosen.append(work)
                busy.add(branch)
                if len(chosen) == slots:
                    break
            return chosen
        if slots < 2:
            return []
        first = ready[0]
        parent_id = self.state.nodes[first.task_id].parent_id
        if first.kind != RoundKind.EXPLORE or parent_id is None:
            return []
        siblings = [
            work for work in ready
            if work.kind == RoundKind.EXPLORE
            and work.purpose != "replan_blocked_inputs"
            and self.state.nodes[work.task_id].parent_id == parent_id
            and self.state.nodes[work.task_id].session_mode == "isolated"
            and work.task_id != self.state.root_task_id
        ]
        if len(siblings) < 2:
            return []
        if self.worker_factory is None:
            raise ValueError("parallel siblings require a factory for independent Codex adapters")
        return siblings[:slots]

    def _run_parallel_branches(self):
        limit = self.state.execution_profile.max_parallel_siblings
        futures = {}

        def run_one(work, worker, context, sent, private):
            def started(thread_id, turn_id=None):
                with self._parallel_lock:
                    work.thread_id, work.turn_id = thread_id, turn_id
                    self._remember_thread(work.task_id, thread_id)
                    self.save()

            def validate(report):
                with self._parallel_lock:
                    self._branch_report_paths(report, private)
                    self._validate_report(work, report, workspace=private)

            def feedback(diagnostics, raw):
                self.store.write_json(
                    f"reports/{work.id}/feedback/{diagnostics['attempt']}.json",
                    {"diagnostics": diagnostics, "raw_report": raw},
                )

            args = (
                sent, REPORTS[work.kind], started,
                min(self.state.execution_profile.worker_timeout_seconds, self.time_left()),
            )
            if work.kind == RoundKind.EXPLORE:
                def submit(brief):
                    with self._parallel_lock:
                        self._branch_report_paths(brief, private)
                        return self._decide(work, context, brief)

                report = worker.run_action_visit(
                    *args, submit=submit, validator=validate, on_feedback=feedback,
                )
            else:
                report = worker.run_with_feedback(
                    *args, validator=validate, on_feedback=feedback,
                )
            return report, getattr(worker, "last_usage", None)

        def launch(pool):
            while len(futures) < limit:
                if self.state.status != RunStatus.ACTIVE or self.time_left() <= 0:
                    return
                candidates = self._parallel_candidates()
                if not candidates:
                    return
                work = candidates[0]
                node = self.state.nodes[work.task_id]
                if work.kind == RoundKind.EXECUTE:
                    inputs = self.state.children(work.task_id) + node.depends_on
                    if inputs:
                        self.audit_files()
                        if any(not completion_current(self.state, key, check_files=False)
                               for key in inputs):
                            work.phase, work.error = (
                                RoundPhase.CANCELLED,
                                "child/dependency evidence changed before integration",
                            )
                            node.status = NodeStatus.WAITING_CHILDREN
                            self.save()
                            continue
                private = Path(self.state.branch_workspaces[self._root_branch(work.task_id)])
                self._sync_cross_branch_inputs(work.task_id, private)
                context = self.context_for(work)
                if work.authorization_round_id:
                    source = self.state.rounds[work.authorization_round_id]
                    record = self.state.action_decisions[source.decision_ids[0]]
                    expected = (
                        "drill_down" if work.routing_basis == "inline_after_empty_drill"
                        or work.kind == RoundKind.DECOMPOSE else "execute"
                    )
                    if record.verdict.action != expected:
                        raise ValueError("phase does not match its bound Luna action")
                    if file_versions(self.state, source.routing_fingerprints) != source.routing_fingerprints:
                        work.phase, work.error = (
                            RoundPhase.CANCELLED,
                            "referenced inputs changed after Luna decision",
                        )
                        self.enqueue(work.task_id, f"changed:{work.id}", work.error)
                        self.save()
                        continue
                    context["action_decision"] = record.model_dump(mode="json")
                context["workspace"] = str(private)
                thread_id = self._session_thread(work.task_id)
                if thread_id:
                    context["continuation_thread_id"] = thread_id
                work.input_digest = digest([
                    context, VISIT_INSTRUCTIONS, REPORTS[work.kind].model_json_schema()
                ])
                context["input_digest"] = work.input_digest
                work.context_manifest_ref = self.store.write_json(
                    f"contexts/{work.id}.json", context
                )
                sent = bounded_context(context, work.context_manifest_ref)
                self.store.write_json(f"contexts/{work.id}-sent.json", sent)
                worker = self.worker_factory(private)
                work.phase, work.started_at = RoundPhase.WORKER_RUNNING, time.time()
                work.worker_model, work.worker_effort = (
                    self.state.execution_profile.worker_model,
                    self.state.execution_profile.worker_effort,
                )
                work.execution_owner = worker.execution_owner()
                node.status = NodeStatus.RUNNING
                self.state.progress.worker_attempts_started += 1
                self.state.scheduler.active_siblings.append(WorkRef(kind="ROUND", id=work.id))
                self.store.execution_reservation(
                    [ref.id for ref in self.state.scheduler.active_siblings]
                )
                self.save()
                futures[pool.submit(run_one, work, worker, context, sent, private)] = (
                    work, worker, context, private
                )

        with ThreadPoolExecutor(max_workers=limit) as pool:
            with self._parallel_lock:
                launch(pool)
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    work, worker, context, private = futures.pop(future)
                    settled = False
                    try:
                        report, usage = future.result()
                        with self._parallel_lock:
                            self._branch_report_paths(report, private)
                            if work.kind == RoundKind.EXECUTE:
                                self._publish_branch_files(work, report)
                                self._validate_report(work, report)
                            work.worker_report_ref = self.store.write_json(
                                f"reports/{work.id}/worker.json", report.model_dump(mode="json")
                            )
                            if work.kind == RoundKind.EXECUTE:
                                self._freeze_delivery(work, report, context)
                            work.phase = RoundPhase.VERIFYING
                            if usage:
                                self.state.progress.recorded_usage.append(
                                    {"round_id": work.id, "usage": usage}
                                )
                            self.save()
                            self._commit_report(work, report)
                        settled = True
                    except (WorkerReportError, TimeoutError, ValueError) as error:
                        with self._parallel_lock:
                            work.phase, work.finished_at, work.error = (
                                RoundPhase.FAILED, time.time(), str(error)
                            )
                            self._retry_visit(self.state.nodes[work.task_id], work, str(error))
                        settled = True
                    finally:
                        if settled:
                            with self._parallel_lock:
                                self.state.scheduler.active_siblings = [
                                    ref for ref in self.state.scheduler.active_siblings if ref.id != work.id
                                ]
                                self.store.execution_reservation(
                                    [ref.id for ref in self.state.scheduler.active_siblings]
                                )
                                self.refresh_parents()
                                self.save()
                        close = getattr(worker, "close", None)
                        if close:
                            close()
                with self._parallel_lock:
                    launch(pool)

    def _run_parallel_explores(self, works):
        workers = []
        contexts = {}
        try:
            for _ in works:
                workers.append(self.worker_factory())
            if len({id(worker) for worker in workers}) != len(workers) or any(
                worker is self.worker for worker in workers
            ):
                raise ValueError("parallel siblings require independent Codex adapters")
            self.state.scheduler.active_siblings = [
                WorkRef(kind="ROUND", id=work.id) for work in works
            ]
            for work, worker in zip(works, workers):
                node = self.state.nodes[work.task_id]
                work.phase, work.started_at = RoundPhase.PREPARING, time.time()
                node.status = NodeStatus.RUNNING
                context = self.context_for(work)
                thread_id = self._session_thread(work.task_id)
                if thread_id:
                    context["continuation_thread_id"] = thread_id
                work.input_digest = digest([
                    context, VISIT_INSTRUCTIONS, VisitReport.model_json_schema()
                ])
                context["input_digest"] = work.input_digest
                work.context_manifest_ref = self.store.write_json(
                    f"contexts/{work.id}.json", context
                )
                sent = bounded_context(context, work.context_manifest_ref)
                self.store.write_json(f"contexts/{work.id}-sent.json", sent)
                work.phase = RoundPhase.WORKER_RUNNING
                work.worker_model = self.state.execution_profile.worker_model
                work.worker_effort = self.state.execution_profile.worker_effort
                work.execution_owner = worker.execution_owner()
                self.state.progress.worker_attempts_started += 1
                contexts[work.id] = (context, sent)
            self.store.execution_reservation([work.id for work in works])
            self.save()

            def run_one(work, worker):
                context, sent = contexts[work.id]

                def started(thread_id, turn_id=None):
                    with self._parallel_lock:
                        work.thread_id, work.turn_id = thread_id, turn_id
                        self._remember_thread(work.task_id, thread_id)
                        self.save()

                def submit(brief):
                    with self._parallel_lock:
                        return self._decide(work, context, brief)

                def validate(report):
                    with self._parallel_lock:
                        self._validate_report(work, report)

                def feedback(diagnostics, raw):
                    self.store.write_json(
                        f"reports/{work.id}/feedback/{diagnostics['attempt']}.json",
                        {"diagnostics": diagnostics, "raw_report": raw},
                    )

                report = worker.run_action_visit(
                    sent, VisitReport, started,
                    timeout,
                    submit, validator=validate, on_feedback=feedback,
                )
                return report, getattr(worker, "last_usage", None)

            errors = {}
            timeout = min(self.state.execution_profile.worker_timeout_seconds, self.time_left())
            with ThreadPoolExecutor(max_workers=len(works)) as pool:
                futures = {
                    pool.submit(run_one, work, worker): work
                    for work, worker in zip(works, workers)
                }
                for future in as_completed(futures):
                    work = futures[future]
                    try:
                        report, usage = future.result()
                        with self._parallel_lock:
                            self._validate_report(work, report)
                            work.worker_report_ref = self.store.write_json(
                                f"reports/{work.id}/worker.json", report.model_dump(mode="json")
                            )
                            work.phase = RoundPhase.VERIFYING
                            if usage:
                                self.state.progress.recorded_usage.append(
                                    {"round_id": work.id, "usage": usage}
                                )
                            self.save()
                    except (WorkerReportError, TimeoutError) as error:
                        errors[work.id] = str(error)
                    except BaseException:
                        # Keep all reservations for recovery until every executor is confirmed stopped.
                        raise
            self.store.execution_reservation([])
            for work in works:
                if work.id in errors:
                    work.phase, work.finished_at, work.error = (
                        RoundPhase.FAILED, time.time(), errors[work.id]
                    )
                    self._retry_visit(self.state.nodes[work.task_id], work, errors[work.id])
                else:
                    report = VisitReport.model_validate_json(Path(work.worker_report_ref).read_text())
                    self._commit_report(work, report)
                self.state.scheduler.active_siblings = [
                    ref for ref in self.state.scheduler.active_siblings if ref.id != work.id
                ]
                self.save()
        finally:
            for worker in workers:
                close = getattr(worker, "close", None)
                if close:
                    close()

    def _recover_siblings(self):
        for ref in list(self.state.scheduler.active_siblings):
            work = self.state.rounds[ref.id]
            branch = self._root_branch(work.task_id)
            private = Path(self.state.branch_workspaces[branch]) if branch in self.state.branch_workspaces else None
            model = REPORTS[work.kind]
            report_path = self.store.directory / "reports" / work.id / "worker.json"
            delivery_path = report_path.with_name("delivery.json")
            if work.phase == RoundPhase.COMMITTED:
                pass
            else:
                report = (
                    model.model_validate_json(report_path.read_text())
                    if report_path.exists()
                    else self.worker.recover(work, model) if work.thread_id else None
                )
                if report is not None:
                    if private:
                        self._branch_report_paths(report, private)
                    if work.kind == RoundKind.EXECUTE and not work.delivery_ref:
                        if not private:
                            raise ValueError("parallel execution has no branch workspace")
                        if delivery_path.exists():
                            work.delivery_ref = str(delivery_path)
                        else:
                            self._publish_branch_files(work, report)
                            self._validate_report(work, report)
                            context = json.loads(Path(work.context_manifest_ref).read_text())
                            self._freeze_delivery(work, report, context)
                    work.worker_report_ref = (
                        str(report_path) if report_path.exists() else self.store.write_json(
                            f"reports/{work.id}/worker.json", report.model_dump(mode="json")
                        )
                    )
                    work.phase = RoundPhase.VERIFYING
                    self._commit_report(work, report)
                else:
                    work.phase, work.finished_at, work.error = (
                        RoundPhase.INTERRUPTED, time.time(),
                        "parallel branch stopped without a recoverable report",
                    )
                    self.enqueue(work.task_id, f"recover:{work.id}", work.error)
            self.state.scheduler.active_siblings = [
                item for item in self.state.scheduler.active_siblings if item.id != ref.id
            ]
            self.save()
        self.store.execution_reservation([])

    def step(self):
        try:
            return self._step()
        except Exception as error:
            self.block(str(error))
            raise

    def _step(self):
        state = self.state
        if state.schema_version != 3:
            raise ValueError(
                "three-action runtime requires protocol 3; old runs retain their protocol"
            )
        state.workspace_lease = self.store.acquire_workspace(state.workspace)
        for key in state.workspace_lease.unresolved_execution_refs:
            work = state.rounds[key]
            self.worker.ensure_stopped(work.execution_owner)
            if work.decision_execution_owner:
                self.worker.ensure_stopped(work.decision_execution_owner)
        if state.workspace_lease.unresolved_execution_refs:
            self.store.execution_reservation([])
        if state.scheduler.active_siblings:
            self._recover_siblings()
            self.refresh_parents()
            self.save()
            return state
        if self._needs_file_audit or state.status == RunStatus.SUCCEEDED:
            self.audit_files()
            self._needs_file_audit = False
        if (
            state.root_approval
            and state.root_approval.contract_hash
            != state.nodes[state.root_task_id].spec.contract_hash()
        ):
            self.block("root contract changed; user reconfirmation required")
            return self.state
        if state.scheduler.active_work:
            self.recover_active()
            if not state.scheduler.active_work:
                return self.state
            ref = state.scheduler.active_work
        else:
            if state.status != RunStatus.ACTIVE:
                return state
            self.refresh_parents()
            if (
                self.time_left() <= 0
                or state.progress.worker_attempts_started
                >= state.execution_profile.max_worker_rounds
            ):
                self.block("run time or worker round budget exhausted; progress is resumable")
                return self.state
            self._ensure_branch_workspaces()
            siblings = self._parallel_candidates()
            if siblings:
                if state.branch_workspaces:
                    self._run_parallel_branches()
                else:
                    self._run_parallel_explores(siblings)
                self.refresh_parents()
                self.save()
                if self.state.status == RunStatus.SUCCEEDED:
                    self.store.release_workspace()
                return self.state
            ref = self.claim_next_work()
            if ref is None:
                self.block("no runnable work; inspect blocked tasks/dependencies (not completion)")
                return self.state
        try:
            while ref:
                source = self.state.rounds[ref.id]
                self._run_round(ref.id)
                source = self.state.rounds[ref.id]
                ref = self._claim_same_thread_continuation(source)
            self.refresh_parents()
            self.save()
            if self.state.status == RunStatus.SUCCEEDED:
                self.store.release_workspace()
        except (ValueError, RuntimeError, OSError, TimeoutError) as error:
            work = self.state.rounds[ref.id] if ref else source
            # Interrupted live executors remain reserved until recovery confirms termination.
            if work.phase == RoundPhase.WORKER_RUNNING:
                self.block(str(error))
                raise
            work.phase, work.finished_at = RoundPhase.FAILED, time.time()
            self.state.scheduler.active_work = None
            self._retry_visit(self.state.nodes[work.task_id], work, str(error))
            self.save()
        return self.state
