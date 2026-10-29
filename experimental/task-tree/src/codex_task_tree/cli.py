from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .action_protocol import ACTION_JUDGE_INSTRUCTIONS
from .action_runtime import ActionRuntime
from .checks import latest_record
from .codex_adapter import CodexAdapter
from .luna_decision import LunaDecisionMaker
from .models import (
    TERMINAL_ROUNDS,
    ActionVerdict,
    ExecutionProfile,
    NodeStatus,
    RoundKind,
    RoundPhase,
    RunStatus,
    TaskSpec,
    identifier,
)
from .runtime import TaskRuntime, certificate_valid, queue_round, validate_spec
from .store import RunStore, atomic_write, file_hash


def runtime_type(state):
    return ActionRuntime if state.schema_version == 3 else TaskRuntime


def add_decision_arguments(command) -> None:
    command.add_argument("--decision-model", default=None)


def decision_settings(args, current=None) -> dict:
    return {"decision_model": args.decision_model or (current.decision_model if current else "gpt-6-luna")}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="codex task-tree", description="Persistent Codex task tree with independent Luna decisions"
    )
    cli.add_argument(
        "--home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser() / "task-tree",
    )
    sub = cli.add_subparsers(dest="command", required=True)
    start = sub.add_parser(
        "start", help="use the input as the root task and begin exploring its background"
    )
    start.add_argument("task")
    start.add_argument("--workspace", type=Path, default=Path.cwd())
    start.add_argument("--codex-bin", default=os.environ.get("CODEX_TASK_TREE_CODEX_BIN", "codex"))
    start.add_argument("--model", default=ExecutionProfile().worker_model)
    start.add_argument("--effort", default="xhigh", choices=["low", "medium", "high", "xhigh"])
    start.add_argument(
        "--worker-prompt-profile",
        choices=["default", "open-task-v1"],
        default="default",
        help="Select the compiled native base-instruction profile; developer role instructions remain unchanged",
    )
    add_decision_arguments(start)
    start.add_argument("--max-rounds", type=int, default=100)
    start.add_argument("--parallel-branches", "--parallel-siblings", dest="parallel_siblings",
                       type=int, default=ExecutionProfile().max_parallel_siblings,
                       help="Concurrent isolated root-child subtrees (default: 3)")
    start.add_argument("--max-seconds", type=float, default=7200)
    start.add_argument("--input", action="append", default=[])
    start.add_argument(
        "--allow-worker-network",
        action="store_true",
        help="Allow task-authorized remote access; retain each worker role's filesystem sandbox",
    )
    prepare = sub.add_parser(
        "prepare",
        help="legacy protocol-2 background preparation only",
    )
    prepare.add_argument("run_id")
    add_decision_arguments(prepare)
    for name in ("status", "run", "resume", "export", "release", "cancel"):
        command = sub.add_parser(name)
        command.add_argument("run_id")
        if name == "resume":
            command.add_argument(
                "--reason",
                required=True,
                help="what resolved the previous blocker; counters are preserved",
            )
    approve = sub.add_parser(
        "approve", help="approve a root contract for an older run awaiting confirmation"
    )
    approve.add_argument("run_id")
    approve.add_argument("--contract", required=True)
    approve.add_argument("--confirmation", required=True)
    accept = sub.add_parser(
        "accept", help="record a real human acceptance for an explicitly human criterion"
    )
    accept.add_argument("run_id")
    accept.add_argument("--task-id", required=True)
    accept.add_argument("--criterion", required=True)
    accept.add_argument("--confirmation", required=True)
    edit = sub.add_parser(
        "spec", help="edit an unapproved root proposal in an older run"
    )
    edit.add_argument("run_id")
    edit.add_argument("--file", required=True, type=Path)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--codex-bin", default=os.environ.get("CODEX_TASK_TREE_CODEX_BIN", "codex"))
    add_decision_arguments(doctor)
    doctor.add_argument("--probe-decision", action="store_true")
    return cli


def status(state) -> dict:
    root = state.nodes[state.root_task_id]

    def root_branch(task_id):
        node = state.nodes[task_id]
        while node.parent_id and node.parent_id != state.root_task_id:
            node = state.nodes[node.parent_id]
        return node.id if node.parent_id else None

    proposals = [
        work
        for work in state.rounds.values()
        if work.kind == RoundKind.SPECIFY and work.worker_report_ref
    ]
    proposal = (
        json.loads(Path(proposals[-1].worker_report_ref).read_text())
        if proposals and state.status == RunStatus.AWAITING_APPROVAL
        else {}
    )
    return {
        "run_id": state.run_id,
        "schema_version": state.schema_version,
        "execution_supported": state.schema_version in {2, 3},
        "status": state.status.value,
        "completion_current": certificate_valid(state, root.id) if state.root_approval else False,
        "revision": state.revision,
        "block_reason": state.block_reason,
        "workspace": str(state.workspace),
        "worker_attempts": state.progress.worker_attempts_started,
        "active_seconds": round(state.progress.active_elapsed_seconds, 2),
        "subtask_nodes": len(state.nodes) - 1,
        "preparation_required": state.preparation_required,
        "current_preparation_id": state.current_preparation_id,
        "preparation_summary": state.preparations[state.current_preparation_id].summary
        if state.current_preparation_id
        else None,
        "preparation_rounds_started": sum(
            r.kind == RoundKind.PREPARE and r.started_at is not None for r in state.rounds.values()
        ),
        "task_rounds_started": sum(
            r.kind not in {RoundKind.PREPARE, RoundKind.SPECIFY} and r.started_at is not None
            for r in state.rounds.values()
        ),
        "decision_backend": "codex",
        "decision_model": state.execution_profile.decision_model,
        "action_decisions": len(state.action_decisions),
        "latest_actions": [{"task_id": d.task_id, "action": d.verdict.action,
                            "reason": d.verdict.reason} for d in list(state.action_decisions.values())[-10:]],
        "native_prompt_profile": state.execution_profile.native_prompt_profile,
        "nodes": [
            {
                "id": n.id,
                "parent_id": n.parent_id,
                "title": n.spec.title if n.specs else "Unspecified",
                "status": n.status.value,
                "session_mode": n.session_mode,
                "result_revision": n.result_revision,
                "depth": n.depth,
            }
            for n in state.nodes.values()
        ],
        "active_work": state.scheduler.active_work.model_dump()
        if state.scheduler.active_work
        else None,
        "active_siblings": [ref.model_dump() for ref in state.scheduler.active_siblings],
        "active_branches": [
            {"root_child_id": root_branch(state.rounds[ref.id].task_id),
             "task_id": state.rounds[ref.id].task_id,
             "kind": state.rounds[ref.id].kind.value,
             "phase": state.rounds[ref.id].phase.value}
            for ref in state.scheduler.active_siblings
        ],
        "pending_decisions": [
            job.model_dump(mode="json")
            for job in state.decision_jobs.values()
            if job.status not in {"RESOLVED", "SUPERSEDED"}
        ],
        "root_contract_hash": root.spec.contract_hash() if root.specs else None,
        "specification_questions": proposal.get("questions", []),
        "specification_assumptions": proposal.get("assumptions", []),
        "root_specification": root.spec.model_dump(mode="json")
        if state.status == RunStatus.AWAITING_APPROVAL
        else None,
        "suggestions": [s.model_dump() for s in state.suggestions],
    }


def export(store: RunStore, state) -> Path:
    directory = store.directory / "exports"
    atomic_write(directory / "tree.json", state.model_dump_json(indent=2).encode(), immutable=False)
    lines = [f"# Task tree: {state.run_id}", "", f"Status: {state.status.value}", ""]
    for node in sorted(state.nodes.values(), key=lambda n: n.created_seq):
        lines.append(
            f"{'  ' * node.depth}- {node.spec.title if node.specs else 'Unspecified'} `{node.id}` — {node.status.value}; result v{node.result_revision}"
        )
        if node.result_history:
            lines.append(f"{'  ' * (node.depth + 1)}{node.result_history[-1].summary}")
    lines += ["", "## Rounds", ""]
    lines += [
        f"- {r.created_seq}: {r.task_id} {r.kind.value} {r.phase.value}; purpose={r.purpose}; thread={r.thread_id}"
        for r in sorted(state.rounds.values(), key=lambda r: r.created_seq)
    ]
    lines += ["", "## Suggestions", ""] + [f"- {s.description}" for s in state.suggestions]
    lines += ["", "## Actions", ""] + [
        f"- {d.task_id}: {d.verdict.action} — {d.verdict.reason} ([request]({d.request_ref}), [response]({d.response_ref}))"
        for d in state.action_decisions.values()
    ]
    atomic_write(directory / "tree.md", "\n".join(lines).encode(), immutable=False)
    return directory


def run(store: RunStore) -> None:
    state = store.load()
    current_binary = os.environ.get("CODEX_TASK_TREE_CODEX_BIN")
    if (
        current_binary
        and Path(current_binary).resolve() != Path(state.execution_profile.codex_bin).resolve()
    ):
        raise ValueError("this run is pinned to a different Codex binary; use its recorded fork")
    if state.schema_version not in {2, 3}:
        raise ValueError(
            "legacy task-tree protocol is inspection-only; start a fresh run for model-gated decomposition"
        )
    adapter = CodexAdapter(state.execution_profile, state.workspace)
    if store.load().execution_profile.worker_model != state.execution_profile.worker_model:
        store.save(state, state.revision)
    runtime = runtime_type(state)(
        store, adapter, None, adapter.command_exec,
        **({"worker_factory": lambda workspace=None: CodexAdapter(
            state.execution_profile, workspace or state.workspace
        )}
           if state.schema_version == 3 else {}),
    )
    if state.schema_version == 2:
        runtime.judge = LunaDecisionMaker(adapter, store, lambda: runtime.state, runtime.time_left)
    try:
        if runtime.state.status == RunStatus.SUCCEEDED and not certificate_valid(
            runtime.state, runtime.state.root_task_id
        ):
            runtime.state.status = RunStatus.ACTIVE
            runtime.save()
        while runtime.state.status == RunStatus.ACTIVE:
            runtime.step()
            print(
                json.dumps(
                    {
                        "run_id": state.run_id,
                        "status": runtime.state.status.value,
                        "rounds_started": runtime.state.progress.worker_attempts_started,
                        "active_work": runtime.state.scheduler.active_work.model_dump()
                        if runtime.state.scheduler.active_work
                        else None,
                        "active_siblings": [ref.model_dump() for ref in runtime.state.scheduler.active_siblings],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        runtime.state.status, runtime.state.block_reason = (
            RunStatus.PAUSED,
            "interrupted by user; recover active execution before continuing",
        )
        runtime.save()
        raise
    finally:
        adapter.close()
        export(store, store.load())
    print(json.dumps(status(runtime.state), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            binary = shutil.which(args.codex_bin)
            if not binary:
                raise ValueError("Codex binary not found")
            version = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=20, check=True
            ).stdout.strip()
            profile = ExecutionProfile(
                codex_bin=str(Path(binary).resolve()), **decision_settings(args)
            )
            probe = None
            if args.probe_decision:
                probe_store = RunStore(args.home, identifier("decision_probe"))
                source_ref = probe_store.write_json(
                    "probe-source.json", {"actual_required_check_status": "FAIL"}
                )
                adapter = CodexAdapter(profile, Path.cwd())
                try:
                    verdict = adapter.run_boolean_decision(
                        {
                            "kind": "JUDGE",
                            "root_goal": "Fix one known failing required check in a bounded fixture",
                            "question": "Choose execute, split or finish for a task whose required check currently FAILs and needs one focused repair.",
                            "source_context_ref": source_ref,
                            "protected_context": {"actual_required_check_status": "FAIL"},
                            "scope": "Connectivity probe only; do not update any task.",
                        },
                        ActionVerdict,
                        ACTION_JUDGE_INSTRUCTIONS,
                        lambda *_: None,
                        60,
                    )
                    probe = {"action": verdict.action, "record": probe_store.write_json("probe.json", verdict.model_dump(mode="json"))}
                finally:
                    adapter.close()
            print(
                json.dumps(
                    {
                        "python": sys.executable,
                        "codex_bin": str(Path(binary).resolve()),
                        "codex_version": version,
                        "decision_backend": "codex",
                        "decision_model": profile.decision_model,
                        "probe": probe,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "start":
            binary = shutil.which(args.codex_bin)
            if not binary:
                raise ValueError("Codex binary not found")
            version = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=20, check=True
            ).stdout.strip()
            store = RunStore(args.home, identifier("run"))
            profile = ExecutionProfile(
                codex_bin=str(Path(binary).resolve()),
                codex_version=version,
                codex_binary_sha256=file_hash(Path(binary).resolve()),
                code_mode_host_sha256=file_hash(
                    Path(binary).resolve().parent / "codex-code-mode-host"
                ),
                worker_model=args.model,
                worker_effort=args.effort,
                native_prompt_profile=args.worker_prompt_profile,
                permission_profile="workspace-write-network"
                if args.allow_worker_network
                else "workspace-write",
                **decision_settings(args),
                max_worker_rounds=args.max_rounds,
                max_parallel_siblings=args.parallel_siblings,
                max_active_seconds=args.max_seconds,
            )
            with store.locked():
                ActionRuntime.create(store, args.task, args.workspace, profile, input_refs=args.input)
                print(f"run_id: {store.run_id}", flush=True)
                run(store)
            return 0
        store = RunStore(args.home, args.run_id)
        if args.command in {"status", "export"}:
            state = store.load()
            if args.command == "status":
                print(json.dumps(status(state), ensure_ascii=False, indent=2))
            else:
                print(export(store, state))
            return 0
        with store.locked():
            state = store.load()
            if state.schema_version not in {2, 3} and args.command in {
                "prepare",
                "spec",
                "approve",
                "accept",
                "run",
                "resume",
            }:
                raise ValueError(
                    "legacy task-tree protocol is inspection-only; start a fresh run for model-gated decomposition"
                )
            if args.command == "status":
                print(json.dumps(status(state), ensure_ascii=False, indent=2))
            elif args.command == "export":
                print(export(store, state))
            elif args.command == "release":
                if state.scheduler.active_work or state.scheduler.active_siblings or state.status not in {
                    RunStatus.SUCCEEDED,
                    RunStatus.CANCELLED,
                    RunStatus.AWAITING_APPROVAL,
                    RunStatus.BLOCKED,
                }:
                    raise ValueError("stop/recover active execution before releasing the workspace")
                store.release_workspace()
            elif args.command == "prepare":
                if state.schema_version == 3:
                    raise ValueError("protocol 3 uses EXPLORE for background; prepare is unavailable")
                if state.root_approval or state.scheduler.active_work:
                    raise ValueError("prepare requires an unapproved, idle run")
                state.execution_profile = state.execution_profile.model_copy(
                    update=decision_settings(args, state.execution_profile)
                )
                state.execution_profile = ExecutionProfile.model_validate(
                    state.execution_profile.model_dump()
                )
                store.save(state, state.revision)
                runtime = runtime_type(state)(store, None, None, None)
                runtime.begin_preparation()
                run(store)
            elif args.command == "spec":
                if state.status != RunStatus.AWAITING_APPROVAL or state.root_approval:
                    raise ValueError("spec editing requires an unapproved root proposal")
                proposal = TaskSpec.model_validate_json(args.file.read_text())
                root = state.nodes[state.root_task_id]
                proposal.revision = root.spec.revision + 1
                validate_spec(state, root.id, proposal)
                root.specs.append(proposal)
                if state.schema_version == 3:
                    root.must_solve = [r.description for r in proposal.requirements if r.required]
                    root.root_contribution = state.user_request
                state.coverage.clear()
                root.coverage_ids.clear()
                from .models import CoverageItem

                for requirement in proposal.requirements:
                    item = CoverageItem(
                        id=f"requirement:{requirement.id}",
                        kind="REQUIREMENT",
                        owner_task_id=root.id,
                        root_requirement_ids=[requirement.id],
                        subject=requirement.description,
                        discovery_source_refs=[str(args.file.resolve())],
                    )
                    state.coverage[item.id] = item
                    root.coverage_ids.append(item.id)
                store.save(state, state.revision)
                print(json.dumps(status(state), ensure_ascii=False, indent=2))
            elif args.command == "approve":
                runtime = runtime_type(state)(store, None, None, None)
                runtime.approve_root(args.contract, args.confirmation)
                print(json.dumps(status(runtime.state), ensure_ascii=False, indent=2))
            elif args.command == "accept":
                runtime = runtime_type(state)(store, None, None, None)
                runtime.accept_user_criterion(args.task_id, args.criterion, args.confirmation)
                print(json.dumps(status(runtime.state), ensure_ascii=False, indent=2))
            elif args.command == "cancel":
                lease = store.acquire_workspace(state.workspace)
                refs = set(lease.unresolved_execution_refs)
                if state.scheduler.active_work:
                    refs.add(state.scheduler.active_work.id)
                refs.update(ref.id for ref in state.scheduler.active_siblings)
                for key in refs:
                    work = state.rounds.get(key) or state.verification_jobs.get(key)
                    if not work:
                        raise ValueError("unknown execution must be inspected before cancellation")
                    if work.execution_owner:
                        CodexAdapter.ensure_stopped(work.execution_owner)
                    elif key in lease.unresolved_execution_refs:
                        raise ValueError("unknown executor identity; workspace remains reserved")
                    if getattr(work, "decision_execution_owner", None):
                        CodexAdapter.ensure_stopped(work.decision_execution_owner)
                    work.phase = RoundPhase.INTERRUPTED if key in state.rounds else "CANCELLED"
                    work.error = "cancelled by user after executor confirmed stopped"
                state.scheduler.active_work = None
                state.scheduler.active_siblings = []
                store.execution_reservation([])
                state.status = RunStatus.CANCELLED
                for work in state.rounds.values():
                    if work.phase in {RoundPhase.QUEUED, RoundPhase.WAITING_CONTEXT}:
                        work.phase, work.error = RoundPhase.CANCELLED, "cancelled by user"
                for job in state.verification_jobs.values():
                    if job.phase in {"QUEUED", "WAITING_CONTEXT"}:
                        job.phase, job.error = "CANCELLED", "cancelled by user"
                for job in state.decision_jobs.values():
                    if job.status != "RESOLVED":
                        job.status = "SUPERSEDED"
                store.save(state, state.revision)
                store.release_workspace()
            elif args.command in {"run", "resume"}:
                if state.schema_version not in {2, 3}:
                    raise ValueError(
                        "legacy task-tree protocol is inspection-only; start a fresh run for model-gated decomposition"
                    )
                if args.command == "resume":
                    if state.status not in {RunStatus.BLOCKED, RunStatus.PAUSED}:
                        raise ValueError("resume expects a blocked or paused run")
                    state.background[f"resume:{state.revision}"] = args.reason
                    state.status, state.block_reason = RunStatus.ACTIVE, None
                    for pending in state.decision_jobs.values():
                        if pending.status == "WAITING_CONTEXT" and not any(
                            r.decision_job_id == pending.id and r.phase not in TERMINAL_ROUNDS
                            for r in state.rounds.values()
                        ):
                            pending.status = "QUEUED"
                            node = state.nodes[pending.owner_task_id]
                            node.status = (
                                NodeStatus.SUCCEEDED
                                if pending.prior_node_status == NodeStatus.SUCCEEDED
                                else NodeStatus.READY
                            )
                            node.block_reason = None
                    if not state.scheduler.active_work and not state.scheduler.active_siblings:
                        error_owners = {
                            state.verifications[key].task_id
                            for job in state.verification_jobs.values()
                            if job.phase == "ERROR"
                            for key in job.verification_ids
                            if state.verifications[key].status == "ERROR"
                            and (
                                current := latest_record(
                                    state,
                                    state.verifications[key].task_id,
                                    state.verifications[key].criterion_id,
                                )
                            )
                            is not None
                            and current.status == "ERROR"
                        }
                        for owner in error_owners:
                            node = state.nodes[owner]
                            if node.status != NodeStatus.SUPERSEDED and not any(
                                r.task_id == owner and r.phase not in TERMINAL_ROUNDS
                                for r in state.rounds.values()
                            ):
                                node.status = NodeStatus.READY
                                queue_round(
                                    state,
                                    owner,
                                    RoundKind.EXPLORE,
                                    f"verification-error-review:{state.revision}",
                                    purpose="inspect_verification_error_without_assuming_product_failure: "
                                    + args.reason,
                                )
                                parent = node.parent_id
                                while parent:
                                    state.nodes[parent].status = NodeStatus.WAITING_CHILDREN
                                    parent = state.nodes[parent].parent_id
                        for node in state.nodes.values():
                            if node.status == NodeStatus.BLOCKED and not any(
                                r.task_id == node.id and r.phase not in TERMINAL_ROUNDS
                                for r in state.rounds.values()
                            ):
                                node.status = NodeStatus.READY
                                kind = (
                                    RoundKind.RECONCILE
                                    if state.schema_version == 2 and state.children(node.id)
                                    and all(
                                        state.nodes[child].status == NodeStatus.SUCCEEDED
                                        for child in state.children(node.id)
                                    )
                                    else RoundKind.EXPLORE
                                )
                                if not state.root_approval:
                                    kind = (
                                        RoundKind.PREPARE
                                        if state.preparation_required
                                        and not state.current_preparation_id
                                        else RoundKind.SPECIFY
                                    )
                                queue_round(
                                    state,
                                    node.id,
                                    kind,
                                    f"user-resume:{state.revision}",
                                    purpose="inspect_new_context_and_correct_previous_blocker: "
                                    + args.reason,
                                )
                                if state.schema_version == 3:
                                    node.no_progress_rounds = 0
                    store.save(state, state.revision)
                run(store)
        return 0
    except KeyboardInterrupt:
        print("task-tree paused; recover with resume", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"task-tree: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
