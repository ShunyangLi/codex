"""Fresh native decision sessions; the skill owns selection, the tool owns transport."""

from __future__ import annotations

import json

from .context import fingerprints, input_path, read_roots
from .decision_context import adverse_material, working_budget
from .decision_skill import instructions
from .jev import (
    PREDICATES,
    DecisionContextTooLarge,
    DecisionWorkerUnavailable,
    decision_policy_digest,
    prepare_request,
)
from .models import DecisionBrief, DecisionRecord, Model, digest


class DecisionSubmission(Model):
    decision_brief: DecisionBrief
    catalog_task_ids: list[str] | None = None


class DecisionWorkerResult(Model):
    decision_id: str


def question_catalog(state) -> list[dict]:
    return [
        {
            "id": node.id,
            "parent_id": node.parent_id,
            "status": node.status.value,
            "question": node.spec.goal,
            "scope": node.spec.scope,
        }
        for node in state.nodes.values()
        if node.specs and node.status.value != "SUPERSEDED"
    ]


def compact_spec(value):
    if not isinstance(value, dict):
        return value
    return {
        key: (
            [
                {k: v for k, v in c.items() if k not in {"check_spec", "adopted_check_revision_id"}}
                for c in value[key]
            ]
            if key == "acceptance"
            else value[key]
        )
        for key in (
            "title",
            "goal",
            "scope",
            "constraints",
            "deliverables",
            "acceptance",
            "required",
            "same_as_root",
        )
        if key in value
    }


def required_context(kind, payload):
    """Keep exact obligations and current adverse facts outside the worker's brief."""
    result = {}
    for key in (
        "root",
        "task",
        "target",
        "criterion",
        "candidate",
        "capability",
        "proposed_exclusion",
        "controller_facts",
        "behavioral_check",
        "independent_review",
    ):
        if key not in payload:
            continue
        value = payload[key]
        if key in {"root", "target"}:
            value = compact_spec(value)
        elif key in {"task", "candidate"} and isinstance(value, dict):
            value = {
                k: compact_spec(v) if k == "spec" else v
                for k, v in value.items()
                if k in {"id", "parent_id", "status", "spec", "open_gaps", "depends_on", "reason"}
            }
        result[key] = value
    current_spec = result.get("task", {}).get("spec", {})
    if current_spec.get("goal") and not current_spec.get("same_as_root") and "root" in result:
        root = result["root"]
        result["root"] = {key: root[key] for key in ("goal", "scope", "constraints") if key in root}
        result["scope_boundary"] = (
            "Judge ONLY task.spec and its assigned outcomes. Root goal is background; inherited constraints remain mandatory. Unassigned ancestor acceptance and integration belong to their owners, not this child."
        )
    if kind == "REUSE":
        source = payload.get("source", {})
        result["source"] = {"id": source.get("id"), "spec": compact_spec(source.get("spec"))}
        result["source_currently_verified"] = payload.get("source_currently_verified", False)
    if kind == "DUPLICATE":
        result["existing_task"] = payload["existing_task"]
    if kind == "CHECK_CHANGE":
        result["proposal"] = compact_spec(payload.get("proposal", {}).get("proposal", {}))
    # Do not ship large raw histories, passing logs or other tasks' results.
    if kind in {"COMPLETE", "EVIDENCE_SUPPORTED", "CHECK_CHANGE", "COVERAGE"}:
        result["current_adverse_facts"] = adverse_material(
            {
                key: payload[key]
                for key in ("task", "result_draft", "verification", "coverage", "behavioral_check")
                if key in payload
            },
            "current",
        )
    if "inherited_root_criterion_ids" in payload:
        result["inherited_root_criterion_ids"] = payload["inherited_root_criterion_ids"]
    if kind == "COMPLETE":
        result["verification"] = [
            {
                k: v[k]
                for k in ("task_id", "criterion_id", "status", "method", "exit_code")
                if k in v
            }
            for v in payload.get("verification", [])
        ]
        result["coverage"] = payload.get("coverage", [])
    return result


class CodexDecisionMaker:
    def __init__(self, adapter, transport, state_provider):
        self.adapter, self.transport, self.state_provider = adapter, transport, state_provider
        self.last_trace = None

    def __call__(self, kind, target, payload):
        state, store = self.state_provider(), self.transport.store
        if kind == "SHOULD_SPLIT":
            payload = dict(payload)
            for field in ("exploration", "exploration_draft"):
                explored = payload.get(field)
                if isinstance(explored, dict) and explored.get("split_opportunities") == []:
                    payload[field] = {
                        key: value
                        for key, value in explored.items()
                        if key != "split_opportunities"
                    }
            payload["controller_facts"] = {
                **payload.get("controller_facts", {}),
                "candidate_proposals_require_prior_split_authorization": True,
                "empty_legacy_exploration_candidates_are_expected": True,
            }
        protected = required_context(kind, payload)
        catalog = (
            question_catalog(state) if kind in {"SHOULD_SPLIT", "NECESSARY", "DUPLICATE"} else []
        )
        key = digest(
            [
                kind,
                target,
                payload,
                protected,
                catalog,
                decision_policy_digest(),
                DECISION_INSTRUCTIONS,
                self.transport.profile.worker_model,
                self.transport.profile.worker_effort,
                self.transport.profile.native_prompt_profile,
            ]
        )
        prefix = f"decision_workers/{key}"
        source_ref = store.write_json(f"{prefix}/source.json", payload)
        cached = store.directory / prefix / "answer.json"
        if cached.exists():
            saved = json.loads(cached.read_text())
            current = fingerprints(
                state.workspace, list(saved["fingerprints"]), [*read_roots(state), source_ref]
            )
            if current != saved["fingerprints"]:
                raise ValueError("saved decision evidence changed; regenerate the source context")
            return DecisionRecord.model_validate(saved["record"])
        allowed_criteria = set()

        def collect(value):
            if isinstance(value, dict):
                if "acceptance" in value:
                    allowed_criteria.update(c["id"] for c in value["acceptance"])
                if "description" in value and "check_kind" in value and "id" in value:
                    allowed_criteria.add(value["id"])
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(protected)
        paths_by_ref = {}
        for artifact in state.artifacts.values():
            paths_by_ref[artifact.id] = artifact.snapshot_ref or artifact.path
        for record in state.verifications.values():
            if record.output_ref:
                paths_by_ref[record.id] = record.output_ref
        roots = [*read_roots(state), source_ref]
        result = None
        calls = 0
        trace = {}
        last_error = None

        def submit(arguments):
            nonlocal result, calls, last_error
            if result is not None:
                return {"ok": True, "decision": result.model_dump(mode="json"), "cached": True}
            calls += 1
            if calls > 3:
                return {
                    "ok": False,
                    "code": "REPAIR_LIMIT",
                    "message": "Stop; three submissions exhausted. No verdict was recorded.",
                }
            store.write_json(f"{prefix}/submissions/{calls}.json", arguments)
            try:
                report = DecisionSubmission.model_validate(arguments)
                brief = report.decision_brief
                if not brief.claims:
                    raise ValueError("provide evidence-backed claims before requesting a verdict")
                ids = [claim.id for claim in brief.claims]
                if len(set(ids)) != len(ids):
                    raise ValueError("claim IDs must be unique")
                selected_ids = report.catalog_task_ids
                if selected_ids is not None and not set(selected_ids) <= {
                    item["id"] for item in catalog
                }:
                    raise ValueError("catalog_task_ids must contain actual supplied question IDs")
                selected_catalog = (
                    catalog
                    if selected_ids is None
                    else [item for item in catalog if item["id"] in selected_ids]
                )
                store.write_json(
                    f"{prefix}/catalog-selection/{calls}.json",
                    {
                        "selected_task_ids": [item["id"] for item in selected_catalog],
                        "omitted_task_ids": [
                            item["id"] for item in catalog if item not in selected_catalog
                        ],
                    },
                )
                used = {source_ref}
                for claim in brief.claims:
                    if not set(claim.criterion_ids) <= allowed_criteria:
                        raise ValueError(
                            f"unknown criterion IDs; allowed: {sorted(allowed_criteria)}"
                        )
                    for ref in claim.evidence_refs:
                        path = input_path(state.workspace, paths_by_ref.get(ref, ref), roots)
                        if not path.is_file():
                            raise ValueError(f"evidence is not an actual authorized file: {ref}")
                        used.add(str(path))
                before = fingerprints(state.workspace, sorted(used), roots)
                packet = {
                    **protected,
                    "task_catalog": selected_catalog,
                    "catalog_policy": "Question-only peers selected by the decision worker for relevance; omitted questions are not supplied and are not proven irrelevant. The exact current obligations and any bound duplicate pair remain supplied separately.",
                    "decision_brief": brief.model_dump(mode="json"),
                    "source_binding": key,
                    "source_policy": "Full source is archived and independently readable by this decision worker. Only the exact obligations, code facts and cited observations below are supplied; paths do not imply that Jev read files.",
                }
                prepared = prepare_request(
                    kind, self.transport.profile.jev_model, packet, str(store.directory)
                )
                if len(prepared.encoded) > working_budget(self.transport.profile):
                    raise DecisionContextTooLarge(
                        f"{len(prepared.encoded)} wire bytes exceed {working_budget(self.transport.profile)}; shorten observations and select relevant catalog_task_ids, preserving failures and limitations. Fixed obligations cannot be removed."
                    )
                result = self.transport(kind, target, packet)
                if fingerprints(state.workspace, sorted(used), roots) != before:
                    result = None
                    raise ValueError(
                        "evidence changed during the decision; inspect and resubmit current observations"
                    )
                result.worker_thread_id = trace.get("thread_id", "")
                result.worker_turn_id = trace.get("turn_id", "")
                store.write_json(
                    f"{prefix}/answer.json",
                    {"record": result.model_dump(mode="json"), "fingerprints": before},
                )
                return {
                    "ok": True,
                    "decision": result.model_dump(mode="json"),
                    "wire_bytes": len(prepared.encoded),
                }
            except (ValueError, OSError, RuntimeError) as error:
                code = (
                    "INPUT_TOO_LARGE"
                    if isinstance(error, DecisionContextTooLarge)
                    else "INVALID_INPUT"
                    if isinstance(error, ValueError)
                    else "PROVIDER_ERROR"
                )
                last_error = {
                    "ok": False,
                    "code": code,
                    "message": str(error),
                    "allowed_criterion_ids": sorted(allowed_criteria),
                    "working_budget_bytes": working_budget(self.transport.profile),
                }
                store.write_json(f"{prefix}/errors/{calls}.json", last_error)
                return last_error

        def started(thread_id, turn_id=None):
            trace["thread_id"] = thread_id
            if turn_id:
                trace["turn_id"] = turn_id
            self.last_trace = trace.copy()
            store.write_json(f"{prefix}/trace-{digest(trace)}.json", trace)

        context = {
            "kind": "DECISION",
            "predicate": PREDICATES[kind],
            "decision_kind": kind,
            "source_context_ref": source_ref,
            "protected_context": protected,
            "question_catalog": catalog,
            "allowed_criterion_ids": sorted(allowed_criteria),
            "working_budget_bytes": working_budget(self.transport.profile),
            "authorized_sources": state.input_refs,
            "background_index": str(state.workspace / "background/index.json"),
            "scope": "Read only current sources and this saved context. No other skills, no product work, no child proposals, no network calls except jev_decide.",
        }
        store.write_json(f"{prefix}/worker-context.json", context)
        try:
            self.adapter.run_decision(
                context,
                submit,
                started,
                min(self.transport.profile.worker_timeout_seconds, self.transport.time_left()),
            )
        except (OSError, ValueError, RuntimeError) as error:
            from .codex_adapter import UnresolvedWorkerExecution

            if isinstance(error, UnresolvedWorkerExecution):
                raise
            if result is None:
                raise DecisionWorkerUnavailable(
                    f"native decision worker failed: {error}"
                ) from error
        if result is None:
            error_type = (
                DecisionContextTooLarge
                if last_error and last_error["code"] == "INPUT_TOO_LARGE"
                else DecisionWorkerUnavailable
            )
            raise error_type(
                "decision worker did not obtain a valid verdict: "
                + json.dumps(last_error, ensure_ascii=False)
            )
        store.write_json(
            f"{prefix}/session-result.json",
            {
                "decision_id": result.id,
                "trace": trace,
                "usage": getattr(self.adapter, "last_usage", None),
                "submissions": calls,
            },
        )
        return result


DECISION_INSTRUCTIONS = """You are a fresh read-only decision worker. Your only job is to understand the ONE supplied predicate, select its relevant facts, and call jev_decide using the bundled Jev skill. Read source_context_ref and relevant background/evidence with native tools. Summarize the question and the established meanings; include a small relevant schema/interface excerpt or sample when useful, without assuming samples establish distributions. Preserve concrete conditions, counterevidence, failures, limitations and unknowns. Other tree tasks are provided only as questions with scope/status, never infer their results. For SHOULD_SPLIT/NECESSARY assess prospective independent outcomes, available inputs, workload, coupling and meaningful uncertainty, not delivery success. An unimplemented task, absent tests or unknown final answer is expected BEFORE execution, not adverse evidence against decomposition. A worker's ability to do everything is not evidence of one coherent responsibility. Several supporting steps/boundary checks may serve one outcome; distinct independently verifiable outcomes may deserve separate workers. For completion read the ACTUAL current delivery and check evidence in the source, not merely titles or paths. Do not solve the task, propose children, change scope, fabricate evidence, supply your own verdict or call a remote API directly. A tool error is not a no; correct the input and call the same tool within this conversation, at most three submissions. A valid verdict, including a no, ends this decision; never resample it. Return only {\"decision_id\": \"the actual tool-returned ID\"}. If no valid decision can be obtained, state the unresolved tool error; never invent an ID.
The supplied protected_context contains the bound declarations. source_context_ref is the full source payload with its own top-level keys, not a second protected_context wrapper; inspect its keys before selecting unfamiliar fields. Exploration is forbidden to propose children before SHOULD_SPLIT authorization. An absent or empty legacy split_opportunities field is expected protocol state, never an inconsistency or evidence against splitting. Assess independently deliverable responsibilities from actual findings instead. For planning predicates (SHOULD_SPLIT, CAPABILITY_NEEDED, NECESSARY, DUPLICATE and SUGGEST), select evidence rather than conduct another investigation. Start with the declared task contracts and saved preparation and exploration observations, plus a historical scope review when present. Read the relevant excerpts; do not repeat primary-source scans or reconstruct aggregates already established there. Only perform a bounded read-only check when a missing fact is critical to this one predicate. Leave execution questions and unresolved research to the responsible worker; do not answer them before deciding whether to delegate. For completion, inspect the current delivery and actual saved check evidence instead of repeating the whole execution.
""" + instructions()
