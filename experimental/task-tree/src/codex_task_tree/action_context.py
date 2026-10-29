"""Bound prompt size without discarding the authoritative context on disk."""

import json
from copy import deepcopy

PROMPT_BYTES = 64_000  # UTF-8 bytes, not an estimate of the provider's tokens.


def bounded_context(context, source_ref, *, limit=PROMPT_BYTES):
    packet = deepcopy(context)
    # On the root visit these are the exact same node. Keep one authoritative copy.
    if packet.get("root", {}).get("id") == packet.get("task", {}).get("id"):
        packet["root"] = {"id": packet["task"]["id"], "same_as": "/task"}
    packet["full_context_ref"] = source_ref
    packet["context_policy"] = (
        "This is a bounded view. Original root/current responsibilities and acceptance are retained. "
        "Fields marked omitted_to remain in full_context_ref at the stated JSON pointer; open relevant "
        "evidence before deciding. Omission never means absent, passed, or irrelevant. Do not reload "
        "the whole tree. Catalogue references must be searched before proposing potentially duplicate work."
    )

    def size():
        return len(json.dumps(packet, ensure_ascii=False).encode())

    # Full context is already durable. Never clip text into an ambiguous partial assertion.
    for key in ("catalog", "preparation", "background", "last_drill_down", "ancestors", "last_visit"):
        if size() <= limit:
            return packet
        if packet.get(key):
            packet[key] = {
                "omitted_to": source_ref,
                "json_pointer": "/" + key,
                "detail_required_when_relevant": True,
            }
    # Preserve failures/blockers, result versions and evidence locations of direct inputs.
    for key in ("children", "dependencies"):
        if size() <= limit:
            return packet
        packet[key] = [
            {
                "id": n["id"],
                "status": n["status"],
                "block_reason": n.get("block_reason"),
                "must_solve": n["must_solve"],
                "root_contribution": n["root_contribution"],
                "result": {
                    key: n["result"][key]
                    for key in ("revision", "goal_answer", "must_solve_answers")
                    if key in n["result"]
                } if n["result"] else None,
                "result_report_ref": n["result_report_ref"],
                "omitted_to": source_ref,
                "json_pointer": f"/{key}/{index}",
            }
            for index, n in enumerate(packet.get(key, []))
        ]
    if size() > limit and packet.get("exploration") and packet.get("task", {}).get("latest_exploration"):
        previous = packet["task"]["latest_exploration"]
        packet["task"]["latest_exploration"] = {
            "round_id": previous["round_id"],
            "ref": previous["ref"],
            "summary": previous["brief"]["summary"],
            "remaining_work": previous["brief"]["remaining_work"],
            "omitted_to": source_ref,
            "json_pointer": "/task/latest_exploration",
            "detail_required_when_relevant": True,
        }
    if size() > limit:
        raise ValueError(
            "protected task/root responsibility, results, failures or exploration exceed context "
            "budget; provide a shorter evidence brief with file references or reduce the task scope. "
            "No criteria or adverse evidence were silently truncated."
        )
    return packet
