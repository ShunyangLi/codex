from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .context import (
    check_environment,
    criterion,
    fingerprints,
    input_path,
    local_inputs,
    workspace_path,
)
from .jev import DecisionContextTooLarge, DecisionWorkerUnavailable
from .models import (
    CheckChangeProposal,
    CheckDefinition,
    RunState,
    VerificationRecord,
    digest,
    identifier,
)
from .store import RunStore, atomic_write, file_hash


def bundle_hash(directory: Path) -> str:
    return digest(
        {
            str(p.relative_to(directory)): file_hash(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        }
    )


def review_file(path: Path) -> dict:
    try:
        return {"sha256": file_hash(path), "text": path.read_text()}
    except UnicodeDecodeError:
        return {
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
            "content": "binary; requires independent worker inspection, not inferred from this hash",
        }


def verification_profile(state: RunState, method: str) -> str:
    if method in {"command", "artifact", "command+model"}:
        return digest({"execution": state.execution_profile, "command_policy": 2})
    return digest(state.execution_profile)


def validate_bundle_command(
    workspace: Path, spec, files: list[str], directory: Path, input_refs: list[str]
) -> None:
    for relative in files:
        source = workspace_path(workspace, relative)
        for subject in spec.artifact_paths:
            product = input_path(workspace, subject, input_refs)
            if source == product or (
                subject != "." and product.is_dir() and product in source.parents
            ):
                raise ValueError(
                    "formal assertion files cannot freeze the mutable product under test; separate checker files from product input paths"
                )
    if files and not any(
        str(directory / "files") in arg or "CODEX_CHECK_BUNDLE" in arg for arg in spec.command
    ):
        raise ValueError(
            "formal command must execute a frozen entrypoint or explicitly load CODEX_CHECK_BUNDLE; workspace test discovery is not authoritative"
        )


def propose_check(
    state: RunState, store: RunStore, task_id: str, round_id: str, proposal: CheckChangeProposal
) -> CheckDefinition:
    item = criterion(state, task_id, proposal.criterion_id)
    if proposal.old_check_revision_id != item.adopted_check_revision_id:
        raise ValueError(
            f"check proposal refers to an obsolete definition: {item.id} adopted_check_revision_id is {item.adopted_check_revision_id!r}; pending proposal IDs cannot replace it"
        )
    key = "check_" + digest([task_id, round_id, proposal])
    if key in state.check_definitions:
        return state.check_definitions[key]
    directory = store.directory / "check_definitions" / key
    directory.mkdir(parents=True, exist_ok=True)
    spec = proposal.check_spec.model_copy(deep=True)
    replacements = {}
    for relative in proposal.file_paths:
        source = workspace_path(state.workspace, relative)
        if not source.is_file():
            raise ValueError(f"check source is not a file: {relative}")
        target = directory / "files" / relative
        if ".." in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError("check bundle paths must be workspace relative")
        atomic_write(target, source.read_bytes())
        replacements[relative] = str(target)
        replacements[str(source)] = str(target)
    spec.command = [replacements.get(arg, arg) for arg in spec.command]
    validate_bundle_command(state.workspace, spec, proposal.file_paths, directory, state.input_refs)
    atomic_write(
        directory / "definition.json",
        json.dumps(
            {
                "spec": spec.model_dump(mode="json"),
                "proposal": proposal.model_dump(mode="json"),
                "criterion": item.model_dump(mode="json"),
            },
            sort_keys=True,
        ).encode(),
    )
    versions = [
        c.implementation_revision
        for c in state.check_definitions.values()
        if c.task_id == task_id and c.criterion_id == item.id
    ]
    check = CheckDefinition(
        id=key,
        task_id=task_id,
        criterion_id=item.id,
        semantic_revision=item.revision,
        implementation_revision=max(versions, default=0) + 1,
        check_spec=spec,
        bundle_ref=str(directory),
        bundle_hash=bundle_hash(directory),
        proposed_by_round_id=round_id,
    )
    state.check_definitions[key] = check
    return check


def record_current(state: RunState, record: VerificationRecord) -> bool:
    item = criterion(state, record.task_id, record.criterion_id)
    check = state.check_definitions.get(item.adopted_check_revision_id or "")
    if (
        not check
        or check.status != "ADOPTED"
        or (check.task_id, check.criterion_id, check.semantic_revision)
        != (record.task_id, record.criterion_id, item.revision)
        or record.check_revision_id != check.id
        or item.revision != record.criterion_revision
        or (
            item.check_kind == "evidence"
            and check.check_spec.command
            and record.method != "command+model"
        )
    ):
        return False
    if record.execution_profile_digest != verification_profile(state, record.method):
        return False
    if not set(local_inputs(state.input_refs)) <= set(record.input_paths):
        return False
    try:
        return record.check_bundle_hash == check.bundle_hash == bundle_hash(
            Path(check.bundle_ref)
        ) and record.input_fingerprints == fingerprints(
            state.workspace, record.input_paths, state.input_refs
        )
    except (OSError, ValueError):
        return False


def latest_record(state: RunState, task_id: str, criterion_id: str) -> VerificationRecord | None:
    for record in reversed(list(state.verifications.values())):
        if (
            record.task_id == task_id
            and record.criterion_id == criterion_id
            and record_current(state, record)
        ):
            return record
    return None


def verification_paths(
    state: RunState, task_id: str, criterion_id: str, extra_paths: list[str] | None = None
) -> list[str]:
    item = criterion(state, task_id, criterion_id)
    check = state.check_definitions.get(item.adopted_check_revision_id or "")
    paths = check.check_spec.artifact_paths if check else item.check_spec.artifact_paths
    node = state.nodes[task_id]
    outputs = [
        str(Path(state.artifacts[key].path).relative_to(state.workspace))
        for result in node.result_history
        for key in result.artifact_ids
    ]
    return sorted(
        set((paths or ["."]) + outputs + (extra_paths or []) + local_inputs(state.input_refs))
    )


def verify_one(
    state: RunState,
    store: RunStore,
    task_id: str,
    criterion_id: str,
    source_ref: str,
    command_exec: Callable,
    judge: Callable,
    time_left: Callable[[], float],
    extra_paths: list[str] | None = None,
) -> VerificationRecord:
    item = criterion(state, task_id, criterion_id)
    check = state.check_definitions.get(item.adopted_check_revision_id or "")
    key = identifier("verification")
    paths = verification_paths(state, task_id, criterion_id, extra_paths)
    before = fingerprints(state.workspace, paths, state.input_refs)
    status, method, exit_code, output = (
        "NOT_RUN",
        "command",
        None,
        "formal check has not been adopted",
    )
    if check:
        if (check.task_id, check.criterion_id, check.semantic_revision) != (
            task_id,
            criterion_id,
            item.revision,
        ):
            raise ValueError("formal check belongs to a different acceptance criterion")
        if bundle_hash(Path(check.bundle_ref)) != check.bundle_hash:
            raise RuntimeError("formal check bundle was modified")
        spec = check.check_spec
        if item.check_kind == "evidence":
            method = "command+model" if spec.command else "model"
        try:
            frozen = json.loads((Path(check.bundle_ref) / "definition.json").read_text())
            validate_bundle_command(
                state.workspace,
                spec,
                frozen["proposal"]["file_paths"],
                Path(check.bundle_ref),
                state.input_refs,
            )
            if item.check_kind in ("command", "artifact") or (
                item.check_kind == "evidence" and spec.command
            ):
                if not spec.command:
                    output = "artifact/command criteria require a behavioral check command"
                else:
                    response = command_exec(
                        spec.command,
                        workspace_path(state.workspace, spec.cwd),
                        min(spec.timeout_seconds, time_left()),
                        check_environment(state.workspace, Path(check.bundle_ref)),
                    )
                    exit_code = response["exit_code"]
                    output = response["stdout"] + response["stderr"]
                    status = (
                        "PASS"
                        if exit_code == spec.expected_exit_code
                        and (
                            spec.expected_stdout is None
                            or spec.expected_stdout in response["stdout"]
                        )
                        else "FAIL"
                    )
            if item.check_kind == "evidence" and (not spec.command or status == "PASS"):
                materials = {}
                reviewable = False
                for path in spec.artifact_paths:
                    file = input_path(state.workspace, path, state.input_refs)
                    if file.is_file():
                        try:
                            materials[path] = file.read_text()
                            reviewable = reviewable or bool(materials[path].strip())
                        except UnicodeDecodeError:
                            materials[path] = (
                                "Binary material is not directly reviewable; require a reproducible behavioral check or textual derivation."
                            )
                if not reviewable:
                    status, output = "NOT_RUN", "no reviewable evidence material"
                else:
                    payload = {
                        "criterion": item.model_dump(mode="json"),
                        "review_material": materials,
                        "purpose": "Assess this evidence criterion only; distinguish supported findings from unsupported claims.",
                    }
                    if spec.command:
                        payload["behavioral_check"] = {
                            "command": spec.command,
                            "status": status,
                            "exit_code": exit_code,
                            "stdout": response["stdout"],
                            "stderr": response["stderr"],
                        }
                    decision = judge(
                        "EVIDENCE_SUPPORTED",
                        f"{task_id}:{item.id}",
                        payload,
                    )
                    status = "PASS" if decision.accepted else "FAIL"
                    output = (output + "\n" if spec.command else "") + decision.id
            elif item.check_kind not in {"command", "artifact", "evidence"}:
                method, output = "user", "awaiting explicit criterion acceptance"
        except (DecisionContextTooLarge, DecisionWorkerUnavailable):
            raise
        except (TimeoutError, OSError, RuntimeError, ValueError) as error:
            status, output = "ERROR", str(error)
    after = fingerprints(state.workspace, paths, state.input_refs)
    if after != before:
        status, output = (
            "ERROR",
            output + "\nCheck changed its declared inputs; reverify stable inputs.",
        )
    output_ref = store.write_bytes(f"checks/{key}/output.txt", output.encode())
    return VerificationRecord(
        id=key,
        task_id=task_id,
        criterion_id=criterion_id,
        criterion_revision=item.revision,
        check_revision_id=check.id if check else "",
        check_bundle_hash=check.bundle_hash if check else "",
        status=status,
        input_fingerprints=after,
        input_paths=paths,
        execution_profile_digest=verification_profile(state, method),
        output_ref=output_ref,
        exit_code=exit_code,
        method=method,
        source_ref=source_ref,
    )
