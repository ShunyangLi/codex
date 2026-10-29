from __future__ import annotations

import json
from io import BytesIO

from codex_task_tree.cli import parser
from codex_task_tree.jev import JevDecisionMaker
from codex_task_tree.models import ExecutionProfile
from codex_task_tree.store import RunStore


def test_split_threshold_is_strict_and_does_not_change_other_verdicts(tmp_path):
    profile = ExecutionProfile(split_decision_threshold=0.4, decision_threshold=0.5)
    judge = JevDecisionMaker(profile, RunStore(tmp_path, "split_floor"), lambda: 10)

    class Opener:
        def open(self, request, timeout):
            return BytesIO(
                json.dumps(
                    {
                        "model": profile.jev_model,
                        "answers": {"decision": {"type": "noul", "noul": 0.48}},
                    }
                ).encode()
            )

    judge.opener = Opener()
    split = judge("SHOULD_SPLIT", "root", {"goal": "two independent findings"})
    complete = judge("COMPLETE", "root", {"goal": "two independent findings"})
    assert split.score == complete.score == 0.48
    assert split.threshold == 0.4 and split.exclusive and split.accepted
    assert complete.threshold == 0.5 and not complete.exclusive and not complete.accepted
    assert not split.model_copy(update={"score": 0.4}).accepted


def test_legacy_split_threshold_does_not_change_boolean_routing_basis(setup):
    runtime, _, _ = setup
    original = runtime.routing_basis(runtime.state.root_task_id)
    runtime.state.execution_profile.split_decision_threshold = 0.4
    assert runtime.routing_basis(runtime.state.root_task_id) == original
    runtime.state.execution_profile.split_decision_threshold = 0.5
    assert runtime.routing_basis(runtime.state.root_task_id) == original


def test_boolean_decision_model_is_configurable_at_start():
    assert parser().parse_args(["start", "task"]).decision_model is None
    assert parser().parse_args(["start", "task", "--decision-model", "gpt-6-luna"]).decision_model == "gpt-6-luna"
