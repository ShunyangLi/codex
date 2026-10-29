"""Opt-in, bounded real-Codex smoke for the Worker -> Luna -> execute -> Luna path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex_task_tree.action_runtime import ActionRuntime
from codex_task_tree.cli import export, run
from codex_task_tree.models import (
    AcceptanceCriterion,
    CheckSpec,
    ExecutionProfile,
    Requirement,
    RootApproval,
    RoundKind,
    RoundPhase,
    RunStatus,
    TaskSpec,
)
from codex_task_tree.runtime import queue_round
from codex_task_tree.store import RunStore, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--codex-bin", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    workspace = directory / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "numbers.txt").write_text("12\n30\n")
    binary = args.codex_bin.resolve()
    profile = ExecutionProfile(
        codex_bin=str(binary),
        codex_binary_sha256=file_hash(binary),
        code_mode_host_sha256=file_hash(binary.parent / "codex-code-mode-host"),
        max_worker_rounds=8,
        max_parallel_siblings=1,
        max_active_seconds=900,
        worker_timeout_seconds=300,
        decision_timeout_seconds=120,
    )
    store = RunStore(directory / "controller", "action-smoke")
    request = (
        "Local harness smoke fixture only. Sum the integer lines in numbers.txt, write answer.txt "
        "with the decimal sum and newline, and save a reproducible actual check in verification.txt. "
        "Do not use skills, web search or external inputs. This is one bounded arithmetic outcome; "
        "read, calculate, write and verify are supporting steps. Do not create a general calculator."
    )
    with store.locked():
        state = ActionRuntime.create(store, request, workspace, profile)
        # The fixture contract is preapproved by its caller; no synthetic user approval for a research task.
        state.preparation_required = False
        for work in state.rounds.values():
            work.phase = RoundPhase.CANCELLED
        root = state.nodes[state.root_task_id]
        root.specs = [
            TaskSpec(
                title="Sum fixture",
                goal=request,
                deliverables=["answer.txt", "verification.txt"],
                requirements=[
                    Requirement(
                        id="sum", description="Correct checked sum", acceptance_ids=["correct"]
                    )
                ],
                root_requirement_ids=["sum"],
                acceptance=[
                    AcceptanceCriterion(
                        id="correct",
                        description="answer.txt equals the decimal sum of numbers.txt plus newline; verification.txt contains an actual reproducible check and its result.",
                        check_kind="evidence",
                        root_requirement_ids=["sum"],
                        check_spec=CheckSpec(
                            artifact_paths=["numbers.txt", "answer.txt", "verification.txt"]
                        ),
                    )
                ],
            )
        ]
        root.must_solve, root.root_contribution = ["Deliver and verify the exact sum"], request
        state.root_approval = RootApproval(
            approved_spec_revision=1,
            contract_hash=root.spec.contract_hash(),
            user_confirmation_ref="explicit bounded smoke fixture contract",
        )
        queue_round(state, root.id, RoundKind.EXPLORE, "fixture-approved")
        store.save(state, state.revision)
        run(store)
    state = store.load()
    assert state.status == RunStatus.SUCCEEDED, state.block_reason
    assert (workspace / "answer.txt").read_text() == "42\n"
    rounds = [r for r in state.rounds.values() if r.phase == RoundPhase.COMMITTED]
    explore = [r for r in rounds if r.kind == RoundKind.EXPLORE]
    execute = [r for r in rounds if r.kind == RoundKind.EXECUTE]
    assert len(explore) == 1 and execute
    assert all(r.thread_id == explore[0].thread_id for r in explore + execute)
    assert state.nodes[state.root_task_id].worker_thread_id == explore[0].thread_id
    assert len(state.action_decisions) >= 2
    assert all(
        d.thread_id not in {r.thread_id for r in rounds} for d in state.action_decisions.values()
    )
    print(
        json.dumps(
            {
                "result": "PASS",
                "rounds": len(rounds),
                "luna_calls": len(state.action_decisions),
                "exports": str(export(store, state)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
