from __future__ import annotations

import json
from pathlib import Path

from codex_task_tree.luna_decision import LunaDecisionMaker
from codex_task_tree.models import NodeResult


def test_luna_boolean_verdict_is_independent_cached_and_question_only(setup):
    runtime, _, _ = setup
    node = runtime.state.nodes[runtime.state.root_task_id]
    node.result_history.append(NodeResult(revision=1, round_id="prior", summary="private result"))

    class Adapter:
        calls = []

        def run_boolean_decision(self, context, report_type, instructions, started, timeout):
            self.calls.append(context)
            started("luna-thread")
            started("luna-thread", "luna-turn")
            assert "SHOULD_SPLIT" not in context["question"]
            assert "private result" not in json.dumps(context["task_catalog"])
            return report_type(decision=True)

    adapter = Adapter()
    maker = LunaDecisionMaker(adapter, runtime.store, lambda: runtime.state, lambda: 600)
    payload = {"task": {"id": node.id, "spec": node.spec.model_dump(mode="json")}, "exploration_draft": {"findings": ["two independently verifiable outcomes"]}}
    first = maker("SHOULD_SPLIT", node.id, payload)
    second = maker("SHOULD_SPLIT", node.id, payload)
    assert first == second and first.accepted and first.score == 1
    assert first.model == "gpt-6-luna" and first.worker_thread_id == "luna-thread"
    assert len(adapter.calls) == 1
    saved = json.loads(Path(first.response_ref).read_text())
    assert saved["verdict"] == {"decision": True}


def test_luna_false_verdict_never_authorizes_split(setup):
    runtime, _, _ = setup

    class Adapter:
        def run_boolean_decision(self, context, report_type, instructions, started, timeout):
            started("luna-thread")
            return report_type(decision=False)

    maker = LunaDecisionMaker(Adapter(), runtime.store, lambda: runtime.state, lambda: 600)
    record = maker("SHOULD_SPLIT", runtime.state.root_task_id, {"task": {"id": runtime.state.root_task_id}})
    assert not record.accepted and record.score == 0
