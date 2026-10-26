from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from .models import (
    NodeStatus,
    RoundKind,
    RoundPhase,
    RunState,
    RunStatus,
    WaveCursor,
    WorkRef,
)


@dataclass
class WorkSchedule:
    advance_ready: dict[int, deque[str]] = field(default_factory=dict)
    reconcile_ready: list[str] = field(default_factory=list)
    verification_ready: deque[str] = field(default_factory=deque)
    parent_waits: dict[str, set[str]] = field(default_factory=dict)
    not_ready: dict[WorkRef, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduleChoice:
    work_ref: WorkRef | None
    wave_after: WaveCursor | None


def build_work_schedule(state: RunState) -> WorkSchedule:
    from .runtime import certificate_valid

    view = WorkSchedule()
    advances: dict[int, list[str]] = defaultdict(list)
    for node in state.nodes.values():
        if node.status in {NodeStatus.SUPERSEDED, NodeStatus.BLOCKED, NodeStatus.NEEDS_CONTEXT}:
            continue
        children = state.children(node.id)
        if children:
            view.parent_waits[node.id] = {
                child for child in children if state.nodes[child].status != NodeStatus.SUCCEEDED
            }
    for work in sorted(state.rounds.values(), key=lambda item: item.created_seq):
        pending_retry = work.phase == RoundPhase.WAITING_CONTEXT and any(
            job.source_ref == WorkRef(kind="ROUND", id=work.id) and job.status == "QUEUED"
            for job in state.decision_jobs.values()
        )
        if work.phase != RoundPhase.QUEUED and not pending_retry:
            continue
        node = state.nodes[work.task_id]
        ref = WorkRef(kind="ROUND", id=work.id)
        reason = None
        if node.status in {NodeStatus.SUPERSEDED, NodeStatus.BLOCKED}:
            reason = "superseded"
        elif node.status == NodeStatus.NEEDS_CONTEXT and work.kind != RoundKind.CONTEXT_REVIEW:
            reason = "waiting for decision context"
        elif work.kind not in {RoundKind.PREPARE, RoundKind.SPECIFY} and not state.root_approval:
            reason = "root approval required"
        elif any(not certificate_valid(state, dep) for dep in node.depends_on):
            reason = "unmet dependency"
        elif work.kind == RoundKind.RECONCILE and view.parent_waits.get(node.id):
            reason = "waiting for child deliveries"
        if reason:
            view.not_ready[ref] = reason
        elif work.kind in {
            RoundKind.EXPLORE,
            RoundKind.SCOPE_REVIEW,
            RoundKind.DECOMPOSE,
            RoundKind.EXECUTE,
            RoundKind.REVIEW,
            RoundKind.CONTEXT_REVIEW,
        }:
            advances[node.depth].append(work.id)
        elif work.kind == RoundKind.RECONCILE:
            view.reconcile_ready.append(work.id)
    view.advance_ready = {depth: deque(ids) for depth, ids in sorted(advances.items())}
    view.reconcile_ready.sort(
        key=lambda key: (
            -state.nodes[state.rounds[key].task_id].depth,
            state.rounds[key].created_seq,
        )
    )
    for job in sorted(state.verification_jobs.values(), key=lambda item: item.created_seq):
        pending_retry = job.phase == "WAITING_CONTEXT" and any(
            item.source_ref == WorkRef(kind="VERIFICATION", id=job.id) and item.status == "QUEUED"
            for item in state.decision_jobs.values()
        )
        if (job.phase == "QUEUED" or pending_retry) and state.nodes[
            job.owner_task_id
        ].status not in {NodeStatus.SUPERSEDED, NodeStatus.BLOCKED, NodeStatus.NEEDS_CONTEXT}:
            view.verification_ready.append(job.id)
    return view


def select_next(state: RunState) -> ScheduleChoice:
    if state.scheduler.active_work:
        raise ValueError("recover active work before selecting another item")
    wave = state.scheduler.wave.model_copy(deep=True) if state.scheduler.wave else None
    if state.status not in (RunStatus.ACTIVE, RunStatus.AWAITING_APPROVAL):
        return ScheduleChoice(None, wave)
    if not state.root_approval:
        specifies = [
            item
            for item in state.rounds.values()
            if item.kind in {RoundKind.PREPARE, RoundKind.SPECIFY}
            and item.phase == RoundPhase.QUEUED
            and (
                item.kind == RoundKind.PREPARE
                or not state.preparation_required
                or state.current_preparation_id
            )
        ]
        if specifies:
            item = min(specifies, key=lambda item: item.created_seq)
            return ScheduleChoice(WorkRef(kind="ROUND", id=item.id), wave)
        return ScheduleChoice(None, wave)
    view = build_work_schedule(state)
    refs = [WorkRef(kind="VERIFICATION", id=key) for key in view.verification_ready]
    refs += [WorkRef(kind="ROUND", id=key) for key in view.reconcile_ready]
    refs += [WorkRef(kind="ROUND", id=key) for ids in view.advance_ready.values() for key in ids]
    ready = set(refs)
    if wave:
        while wave.next_index < len(wave.work_refs):
            item = wave.work_refs[wave.next_index]
            wave.next_index += 1
            if item in ready:
                return ScheduleChoice(item, wave)
    if refs:
        depth = min(
            state.nodes[state.rounds[ref.id].task_id].depth
            if ref.kind == "ROUND"
            else state.nodes[state.verification_jobs[ref.id].owner_task_id].depth
            for ref in refs
        )
        wave = WaveCursor(depth=depth, work_refs=refs, next_index=1)
        return ScheduleChoice(refs[0], wave)
    return ScheduleChoice(None, None)
