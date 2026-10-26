from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .decision_skill import SKILL_DIRECTORY, support_context
from .models import InvestigationAspect, RunState, digest
from .store import canonical, file_hash


def workspace_path(workspace: Path, relative: str) -> Path:
    path = canonical(workspace / relative)
    if path != workspace and workspace not in path.parents:
        raise ValueError(f"path outside workspace: {relative}")
    return path


def local_inputs(input_refs: list[str]) -> list[str]:
    return [ref for ref in input_refs if Path(ref).is_absolute()]


def external_input_ref(ref: str) -> bool:
    """Whether an input names a remote resource rather than a local file."""
    return bool(
        re.fullmatch(r"ssh\s+[^\s:]+:/\S+", ref)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*://\S+", ref)
    )


def read_roots(state: RunState) -> list[str]:
    """Authorized inputs include controller-registered evidence, never arbitrary parent directories."""
    return list(
        dict.fromkeys(
            [
                *state.input_refs,
                str(SKILL_DIRECTORY),
                *(item.snapshot_ref for item in state.artifacts.values() if item.snapshot_ref),
                *(item.bundle_ref for item in state.check_definitions.values()),
                *(
                    item.worker_report_ref
                    for item in state.rounds.values()
                    if item.worker_report_ref
                ),
                *(
                    item.context_manifest_ref
                    for item in state.rounds.values()
                    if item.context_manifest_ref
                ),
                *(item.context_ref for item in state.decision_jobs.values()),
                *(item.output_ref for item in state.verifications.values() if item.output_ref),
                *(
                    ref
                    for item in state.decisions.values()
                    for ref in (item.request_ref, item.response_ref)
                    if ref
                ),
            ]
        )
    )


def input_path(workspace: Path, relative: str, input_refs: list[str]) -> Path:
    path = canonical(workspace / relative)
    roots = [workspace, *(canonical(Path(ref)) for ref in local_inputs(input_refs))]
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError(f"path outside workspace and declared read-only inputs: {relative}")
    return path


def check_environment(workspace: Path, bundle: Path) -> dict[str, str]:
    return {
        "PYTHONPATH": str(bundle / "files") + os.pathsep + str(workspace),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CODEX_CHECK_BUNDLE": str(bundle / "files"),
        "CODEX_CHECK_WORKSPACE": str(workspace),
    }


def fingerprints(
    workspace: Path, paths: list[str], input_refs: list[str] | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in sorted(set(paths or ["."])):
        path = input_path(workspace, relative, input_refs or [])
        if not path.exists():
            result[relative] = "MISSING"
        elif path.is_file():
            result[relative] = file_hash(path)
        elif path.is_dir():
            entries = {}
            for directory, dirs, files in os.walk(path, followlinks=False):
                # Build/cache outputs are excluded only from an undeclared, conservative read set.
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if d
                    not in {
                        ".git",
                        ".venv",
                        "__pycache__",
                        "node_modules",
                        ".pytest_cache",
                        ".ruff_cache",
                    }
                )
                for name in sorted(files):
                    entry = Path(directory) / name
                    if entry.is_symlink():
                        input_path(workspace, str(entry), input_refs or [])
                    key = (
                        str(entry.relative_to(workspace))
                        if entry.is_relative_to(workspace)
                        else str(entry)
                    )
                    entries[key] = file_hash(entry)
            result[relative] = digest(entries)
        else:
            raise ValueError(f"unsupported input: {relative}")
    return result


def active_subtree(state: RunState, task_id: str) -> list[str]:
    result = [task_id]
    for child in state.children(task_id):
        result.extend(active_subtree(state, child))
    return result


def effective_criteria(state: RunState, task_id: str) -> list[tuple[str, str]]:
    return [
        (key, c.id)
        for key in active_subtree(state, task_id)
        for c in state.nodes[key].spec.acceptance
        if c.required
    ]


def criterion(state: RunState, task_id: str, criterion_id: str):
    return next(c for c in state.nodes[task_id].spec.acceptance if c.id == criterion_id)


def parse_investigation_inventory(index: dict, *, required: bool = False) -> list[dict]:
    entries = index.get("investigation_inventory", [])
    if not isinstance(entries, list) or (required and not entries):
        raise ValueError("background index needs a nonempty investigation_inventory")
    inventory = [InvestigationAspect.model_validate(value) for value in entries]
    ids = [value.id for value in inventory]
    if len(ids) != len(set(ids)):
        raise ValueError("investigation inventory IDs must be unique")
    for value in inventory:
        if any(
            not text.strip()
            for text in [
                value.id,
                value.subject,
                value.meaning_and_limits,
                *value.source_refs,
                *value.possible_questions,
            ]
        ):
            raise ValueError("investigation inventory needs nonblank source and question evidence")
    return [value.model_dump(mode="json") for value in inventory]


def preparation_view(state: RunState) -> dict | None:
    record = state.preparations.get(state.current_preparation_id or "")
    if record is None:
        return None
    artifacts = [state.artifacts[key] for key in record.artifact_ids]
    for artifact in artifacts:
        if not artifact.snapshot_ref or file_hash(Path(artifact.snapshot_ref)) != artifact.sha256:
            raise ValueError("shared preparation snapshot is missing or changed; prepare again")
    index = state.artifacts[record.index_artifact_id]
    content = json.loads(Path(index.snapshot_ref).read_text())
    return {
        **record.model_dump(mode="json"),
        "index": content,
        "investigation_inventory": parse_investigation_inventory(content),
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        "knowledge_documents": {
            artifact.path: Path(artifact.snapshot_ref).read_text()
            for artifact in artifacts
            if Path(artifact.path).suffix.lower() in {".md", ".txt"}
        },
        "detail_policy": "The index, facts and verified Markdown/text knowledge snapshots are included in full. Other file contents remain references for worker inspection and are not assumed to have been read by the decision model.",
    }


def scope_review_view(state: RunState, task_id: str) -> dict | None:
    ids = state.nodes[task_id].exploration_round_ids
    if not ids:
        return None
    review_id = state.rounds[ids[-1]].scope_review_round_id
    if not review_id:
        return None
    work = state.rounds[review_id]
    return {
        "round_id": review_id,
        "report": json.loads(Path(work.worker_report_ref).read_text()),
        "decisions": [state.decisions[key].model_dump(mode="json") for key in work.decision_ids],
    }


def coverage_view(state: RunState, task_id: str) -> list[dict]:
    active = set(active_subtree(state, task_id))
    return [c.model_dump(mode="json") for c in state.coverage.values() if c.owner_task_id in active]


def decomposition_view(state: RunState, task_id: str) -> dict | None:
    rounds = [
        work
        for work in state.rounds.values()
        if work.task_id == task_id
        and work.kind.value == "DECOMPOSE"
        and work.phase.value == "COMMITTED"
        and work.worker_report_ref
    ]
    if not rounds:
        return None
    work = max(rounds, key=lambda item: item.created_seq)
    return {
        "report": json.loads(Path(work.worker_report_ref).read_text()),
        "decisions": [state.decisions[key].model_dump(mode="json") for key in work.decision_ids],
        "policy": "Historical proposal and judgments, not proof of execution or completion.",
    }


def node_input(state: RunState, task_id: str) -> dict:
    node = state.nodes[task_id]
    paths = [
        path
        for key in active_subtree(state, task_id)
        for c in state.nodes[key].spec.acceptance
        for path in c.check_spec.artifact_paths
    ]
    return {
        "spec": node.spec.model_dump(mode="json"),
        "plan": node.current_plan_revision_id,
        "children": {key: state.nodes[key].result_revision for key in state.children(task_id)},
        "gaps": [gap.model_dump(mode="json") for gap in node.open_gaps],
        "background": {
            key: value
            for key, value in state.background.items()
            if key.split(":", 1)[0] in set(active_subtree(state, task_id)) | {task_id}
            or key.startswith("global:")
        },
        "files": fingerprints(state.workspace, paths or ["."], state.input_refs),
    }


def build_context(state: RunState, round_id: str) -> dict:
    work = state.rounds[round_id]
    node = state.nodes[work.task_id]
    ancestors = []
    parent = node.parent_id
    while parent:
        ancestors.append(state.nodes[parent].model_dump(mode="json"))
        parent = state.nodes[parent].parent_id
    return {
        "user_request": state.user_request,
        "decision_support": support_context(state.execution_profile),
        "root": state.nodes[state.root_task_id].model_dump(mode="json"),
        "background": state.background,
        "preparation": preparation_view(state),
        "scope_review": scope_review_view(state, node.id),
        "last_decomposition": decomposition_view(state, node.id),
        "ancestor_scope_reviews": [
            {"task_id": parent["id"], "review": reviewed}
            for parent in ancestors
            if (reviewed := scope_review_view(state, parent["id"])) is not None
        ],
        "run_id": state.run_id,
        "state_revision": state.revision,
        "task_id": node.id,
        "round_id": work.id,
        "kind": work.kind.value,
        "spec_revision": node.spec.revision if node.specs else None,
        "next_spec_revision": (node.spec.revision + 1 if node.specs else 1)
        if work.kind.value == "SPECIFY"
        else None,
        "purpose": work.purpose,
        "pending_decision": state.decision_jobs[work.decision_job_id].model_dump(mode="json")
        if work.decision_job_id
        else None,
        "authorization_round_id": work.authorization_round_id,
        "exploration": [
            {
                "report": json.loads(Path(state.rounds[key].worker_report_ref).read_text()),
                "decisions": [
                    state.decisions[d].model_dump(mode="json")
                    for d in state.rounds[key].decision_ids
                ],
            }
            for key in node.exploration_round_ids[-1:]
        ],
        "workspace": str(state.workspace),
        "root_approval": bool(state.root_approval),
        "task": node.model_dump(mode="json"),
        "report_criterion_ids": [item.id for item in node.spec.acceptance] if node.specs else [],
        "report_evidence_policy": "Claim criterion_ids use only report_criterion_ids. Claim evidence_refs use actual files or registered artifact/verification IDs, never inline commands, task IDs or criterion IDs. Coverage evidence_refs follow their separate formal-check contract.",
        "ancestors": ancestors,
        "catalog": [
            {
                "id": n.id,
                "title": n.spec.title if n.specs else "unspecified",
                "status": n.status.value,
                "goal": n.spec.goal if n.specs else state.user_request,
                "scope": n.spec.scope if n.specs else [],
                "parent_id": n.parent_id,
            }
            for n in state.nodes.values()
        ],
        "children": [state.nodes[key].model_dump(mode="json") for key in state.children(node.id)],
        "dependencies": [state.nodes[key].model_dump(mode="json") for key in node.depends_on],
        "failed_rounds": [
            {
                "id": r.id,
                "task_id": r.task_id,
                "kind": r.kind.value,
                "phase": r.phase.value,
                "error": r.error,
                "worker_report_ref": r.worker_report_ref,
            }
            for r in state.rounds.values()
            if r.task_id in {node.id, *state.children(node.id), *node.depends_on} and r.error
        ],
        "coverage": coverage_view(state, node.id) if node.specs else [],
        "checks": [
            {
                **c.model_dump(mode="json"),
                "runner_environment": check_environment(state.workspace, Path(c.bundle_ref)),
            }
            for c in state.check_definitions.values()
            if c.task_id == node.id
        ],
        "artifacts": [
            state.artifacts[key].model_dump(mode="json")
            for task in state.nodes.values()
            if task.result_history
            for key in task.result_history[-1].artifact_ids
        ],
        "verification_history": [
            v.model_dump(mode="json")
            for v in state.verifications.values()
            if v.task_id in active_subtree(state, node.id)
        ],
        "input_fingerprints": fingerprints(
            state.workspace, [".", *local_inputs(state.input_refs)], state.input_refs
        ),
        "input_refs": state.input_refs,
        "omitted": [],
    }
