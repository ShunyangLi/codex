from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

PACKET_VERSION = 4

# Only independently reviewed background/history may be replaced by a brief.
# Goals, actual delivery, criteria, verification, coverage and gaps are not here.
REVIEWABLE_SECTIONS = frozenset(
    {
        "preparation.knowledge_documents",
        "background_inventory",
        "exploration",
        "exploration_draft",
        "scope_review",
        "last_decomposition",
        "task_catalog",
    }
)
ADVERSE_KEYS = frozenset(
    {
        "counter_evidence",
        "limitations",
        "uncertainties",
        "omissions",
        "required_followups",
        "execution_blockers",
        "gaps",
        "open_gaps",
        "unresolved",
    }
)


def adverse_material(value, path):
    """Keep explicit negative/unknown information even when a section is summarized."""
    if isinstance(value, dict):
        if {"kind", "target_id", "score", "threshold"} <= value.keys():
            score, threshold = value["score"], value["threshold"]
            if (
                type(score) in {int, float}
                and type(threshold) in {int, float}
                and score < threshold
            ):
                return [{"path": path, "value": value}]
        if isinstance(value.get("status"), str) and value["status"] in {
            "FAIL",
            "ERROR",
            "GAP",
            "UNASSESSED",
            "UNSUPPORTED",
            "NEEDS_PROBE",
            "BLOCKED",
            "NEEDS_CONTEXT",
        }:
            return [{"path": path, "value": value}]
        result = []
        for key, child in value.items():
            if key in ADVERSE_KEYS and child:
                result.append({"path": f"{path}.{key}", "value": child})
            else:
                result.extend(adverse_material(child, f"{path}.{key}"))
        return result
    if isinstance(value, list):
        return [
            item
            for i, child in enumerate(value)
            for item in adverse_material(child, f"{path}[{i}]")
        ]
    return []


def working_budget(profile) -> int:
    """A byte policy, not a claim to know the provider's tokenization."""
    return min(profile.max_jev_input_bytes, profile.jev_context_target_bytes)


def decision_packet(kind: str, original: dict) -> tuple[dict, list[dict]]:
    """Select evidence for one predicate; the caller archives the complete original."""
    packet = deepcopy(original)
    manifest = []
    preparation = packet.get("preparation")
    if isinstance(preparation, dict) and "knowledge_documents" in preparation:
        packet["preparation"] = {
            key: preparation[key]
            for key in (
                "round_id",
                "summary",
                "facts",
                "limitations",
                "investigation_inventory",
                "knowledge_documents",
                "detail_policy",
            )
            if key in preparation
        }
        manifest.append(
            {
                "path": "/preparation",
                "reason": "Shared knowledge and the full semantic inventory retained; duplicate index/statistics and artifact metadata remain archived.",
            }
        )
    override = packet.pop("decision_brief_override", None)
    if override:
        packet["independent_context_review"] = override
        adverse = []
        for section in override.get("reviewed_sections", []):
            if section not in REVIEWABLE_SECTIONS:
                raise ValueError("context reviewer may not replace protected decision material")
            keys = section.split(".")
            parent = packet
            for key in keys[:-1]:
                parent = parent.get(key, {}) if isinstance(parent, dict) else {}
            if not isinstance(parent, dict) or keys[-1] not in parent:
                continue
            adverse.extend(adverse_material(parent[keys[-1]], section))
            parent[keys[-1]] = {
                "content": "Original section is archived, not supplied here. Use the independent_context_review's supported observations, conditions and limits for this predicate only.",
                "review_ref": override.get("review_ref"),
            }
            manifest.append(
                {
                    "path": section,
                    "reason": "independently inspected section summarized in decision brief",
                }
            )
        if adverse:
            packet["retained_adverse_information"] = adverse
        if "preparation.knowledge_documents" in override.get("reviewed_sections", []):
            packet["preparation"]["detail_policy"] = (
                "Knowledge text was independently reviewed and is summarized in independent_context_review; original full text is archived, not supplied here. The explicit investigation inventory and limitations remain supplied."
            )
    if kind == "EVIDENCE_SUPPORTED" and isinstance(packet.get("review_material"), dict):
        criterion_id = packet.get("criterion", {}).get("id")
        scoped_refs = {
            ref
            for claim in (override or {}).get("decision_brief", {}).get("claims", [])
            if criterion_id and criterion_id in claim.get("criterion_ids", [])
            for ref in claim.get("evidence_refs", [])
        }
        for path, content in packet["review_material"].items():
            if (
                path not in scoped_refs
                or not isinstance(content, str)
                or len(content.encode()) <= 1000
            ):
                continue
            fingerprint = sha256(content.encode()).hexdigest()
            packet["review_material"][path] = {
                "original_text_sha256": fingerprint,
                "content": "Full original text is archived, not supplied. Use only this criterion's independently reviewed observations and actual check output; a reference/hash alone is not evidence of its meaning.",
            }
            manifest.append(
                {
                    "path": "review_material." + path,
                    "sha256": fingerprint,
                    "reason": "independent observations explicitly bound to this single criterion; original evidence archived without truncation",
                }
            )
    if kind == "COMPLETE" and isinstance(packet.get("artifact_contents"), list):
        owner = packet.get("task", {}).get("id")
        task_spec = packet.get("task", {}).get("spec", {})
        root = packet.get("root", {})
        # A child is judged against its inherited obligations, not every other
        # branch's future acceptance. Root goal, scope and constraints stay full.
        inherited = packet.get("inherited_root_criterion_ids")
        if inherited is None:
            roots = [
                item["id"]
                for item in original.get("task_catalog", [])
                if item.get("parent_id") is None
            ]
            root_id = packet.get("root_task_id") or (roots[0] if len(roots) == 1 else None)
            inherited = (
                task_spec.get("parent_criterion_ids", [])
                if root_id and packet.get("task", {}).get("parent_id") == root_id
                else []
            )
        parent_ids = set(inherited)
        known_ids = {item["id"] for item in root.get("acceptance", [])}
        if parent_ids and parent_ids <= known_ids:
            required = set(task_spec.get("root_requirement_ids", []))
            root["acceptance"] = [item for item in root["acceptance"] if item["id"] in parent_ids]
            root["requirements"] = [
                item
                for item in root.get("requirements", [])
                if item["id"] in required or set(item["acceptance_ids"]) & parent_ids
            ]
            manifest.append(
                {
                    "path": "root.acceptance",
                    "reason": "child receives mapped inherited clauses; complete root goal/scope/constraints retained, other branches' acceptance archived",
                }
            )
        represented = {item.get("artifact_id") for item in packet["artifact_contents"]}
        if isinstance(packet.get("artifacts"), list):
            before = packet["artifacts"]
            packet["artifacts"] = [item for item in before if item.get("id") not in represented]
            if len(before) != len(packet["artifacts"]):
                manifest.append(
                    {
                        "path": "artifacts",
                        "reason": "matching artifact paths/hashes/content are already represented in artifact_contents; original metadata archived",
                    }
                )
        for record in packet.get("verification", []):
            if record.get("status") == "PASS":
                for key in (
                    "input_fingerprints",
                    "input_paths",
                    "execution_profile_digest",
                    "check_bundle_hash",
                    "source_ref",
                ):
                    record.pop(key, None)
        if any(item.get("status") == "PASS" for item in packet.get("verification", [])):
            manifest.append(
                {
                    "path": "verification.PASS.provenance",
                    "reason": "exact fingerprint maps and execution metadata are archived; actual criterion, status, method, exit code and output remain supplied",
                }
            )
        child_proofs = packet.get("child_completion_proofs", [])
        certified = {item["task_id"] for item in child_proofs if item.get("currently_valid")}
        parents = {item["id"]: item.get("parent_id") for item in original.get("task_catalog", [])}

        def certified_branch(task_id):
            seen = set()
            while task_id and task_id != owner and task_id not in seen:
                if task_id in certified:
                    return True
                seen.add(task_id)
                task_id = parents.get(task_id)
            return False

        # The controller has validated each child certificate against current
        # bytes/checks. Do not re-send every descendant's passing check at each
        # ancestor. Failures and evidence without a current proof stay visible.
        for field in ("verification", "artifacts", "artifact_contents"):
            if not isinstance(packet.get(field), list):
                continue
            before = packet[field]
            packet[field] = [
                item
                for item in before
                if not certified_branch(item.get("task_id"))
                or field == "verification"
                and item.get("status") != "PASS"
            ]
            if len(before) != len(packet[field]):
                manifest.append(
                    {
                        "path": field,
                        "reason": "support is inherited only within the scopes of currently valid direct-child completion proofs",
                        "archived_items": len(before) - len(packet[field]),
                    }
                )
        for proof in child_proofs:
            certificate = proof.get("certificate")
            if proof.get("currently_valid") and isinstance(certificate, dict):
                verification_ids = certificate.pop("verification_ids", [])
                certificate["verified_check_count"] = len(verification_ids)
                proof["scope"] = (
                    "Controller validated this child's current scoped certificate; parent integration and parent goal fulfillment still require separate verification. Full check references remain archived."
                )
                manifest.append(
                    {
                        "path": "child_completion_proofs."
                        + proof["task_id"]
                        + ".certificate.verification_ids",
                        "reason": "exact check references archived; controller-validated count supplied without asking Jev to count",
                    }
                )
        packet["worker_results"] = [
            result
            for result in packet.get("worker_results", [])
            if owner is None or result.get("task_id") == owner or result.get("parent_id") == owner
        ]
        deduplicated = False
        for result in packet["worker_results"]:
            if (
                result.get("task_id") == owner
                and "spec" in result
                and "spec" in packet.get("task", {})
                and result["spec"] == packet["task"]["spec"]
            ):
                result["spec"] = {"same_as_current_task": True}
                manifest.append(
                    {
                        "path": "worker_results.current.spec",
                        "reason": "identical current task specification already supplied in task.spec",
                    }
                )
            for report_key in ("delivery_report", "latest_round_report"):
                report = result.get(report_key) or {}
                if "task_id" in report and report["task_id"] == result.get("task_id"):
                    report.pop("task_id")
                    deduplicated = True
            current = result.get("current_result") or {}
            latest = result.get("latest_round_report") or {}
            if "summary" in current and current["summary"] == latest.get("summary"):
                current.pop("summary")
                deduplicated = True
            delivery = result.get("delivery_report", {})
            if delivery.get("check_change_proposals"):
                delivery.pop("check_change_proposals")
                manifest.append(
                    {
                        "path": "worker_results.delivery_report.check_change_proposals",
                        "reason": "historical check implementation proposals archived; actual delivery, findings, gaps, current acceptance and check outcomes remain supplied",
                    }
                )
        if deduplicated:
            manifest.append(
                {
                    "path": "worker_results",
                    "reason": "repeated report task IDs and identical current/latest summary are supplied once in their enclosing actual result",
                }
            )
        reviewed_refs = {
            ref
            for claim in (override or {}).get("decision_brief", {}).get("claims", [])
            for ref in claim.get("evidence_refs", [])
        }
        records = {item["id"]: item for item in packet.get("verification", []) if "id" in item}
        criteria = {item["id"]: item for item in task_spec.get("acceptance", [])}
        for item in packet["artifact_contents"]:
            content = item.get("current_content", {})
            text = content.get("text")
            if not isinstance(text, str):
                continue
            path = item.get("path", "")
            inherited = item.get("task_id") in certified and item.get("task_id") != owner
            evidence_reviews = item.get("evidence_reviews", [])
            reviewed = item.get("artifact_id") in reviewed_refs or path in reviewed_refs
            reviewed_material = len(text.encode()) > 1000 and evidence_reviews
            if not (inherited or reviewed_material or reviewed and len(text.encode()) > 1000):
                continue
            content.pop("text")
            content["content"] = (
                "Full contents archived, not supplied. Use only the stated criterion evidence, "
                "independent observations or current child proof, each within its scope. "
                "Paths/hashes do not establish other properties."
            )
            manifest.append(
                {
                    "path": path,
                    "sha256": content.get("sha256"),
                    "reason": "valid child proof"
                    if inherited
                    else "criterion-scoped evidence already reviewed"
                    if evidence_reviews
                    else "independent context review",
                    "evidence_reviews": evidence_reviews,
                }
            )
            for review in evidence_reviews:
                record = records.get(review.get("verification_id"), {})
                for key in ("actual_output", "method"):
                    if key in record and review.get(key) == record[key]:
                        review.pop(key)
                criterion = criteria.get(review.get("criterion_id"), {})
                if criterion and review.get("description") == criterion.get("description"):
                    review.pop("description")
        packet["completion_material_policy"] = (
            "Judge this task's actual delivery and integration using supplied contents, actual check "
            "outputs and explicitly scoped evidence reviews. Necessary direct children include current "
            "proofs and actual results. Some full supporting contents are archived rather than repeated; "
            "their hashes are provenance, not a claim that you read them. Uncovered goals, contradictions, "
            "invalid checks and unresolved gaps still prevent completion."
        )
        if "context_selection" in packet:
            packet["context_selection"] = {
                "policy": "Original source context is archived. Current goals, actual results and check outcomes are supplied with explicit scoped deferrals; detailed selection provenance is archived separately."
            }
    # The detailed manifest (including proof provenance) is archived separately.
    # Repeating it inside the request would duplicate the same large evidence.
    packet["context_selection_manifest"] = {
        "affected_sections": list(dict.fromkeys(item["path"] for item in manifest)),
        "policy": "Detailed per-item selection/provenance is archived; deferred contents are explicitly marked, and their supporting evidence is supplied in the relevant packet fields.",
    }
    return packet, manifest
