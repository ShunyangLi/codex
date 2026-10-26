"""Opt-in live fixture using the fork's workers and the configured decision backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from codex_task_tree.cli import add_decision_arguments, decision_settings, export, status
from codex_task_tree.codex_adapter import CodexAdapter
from codex_task_tree.decision_worker import CodexDecisionMaker
from codex_task_tree.jev import JevDecisionMaker
from codex_task_tree.models import (
    AcceptanceCriterion,
    CheckSpec,
    DecisionRecord,
    ExecutionProfile,
    Requirement,
    RoundKind,
    RunStatus,
    SpecProposal,
    TaskSpec,
    digest,
)
from codex_task_tree.runtime import TaskRuntime, certificate_valid
from codex_task_tree.store import RunStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--codex-bin", required=True)
    parser.add_argument("--model", default="gpt-6-sol")
    add_decision_arguments(parser)
    parser.add_argument(
        "--scenario", choices=["addition", "storage", "employment"], default="addition"
    )
    parser.add_argument(
        "--data-path", type=Path, help="RevelioData directory for the employment fixture"
    )
    parser.add_argument(
        "--inject-parent-fault",
        action="store_true",
        help="controlled parent integration/failure fixture: preapprove the root split, use real decisions thereafter; storage only",
    )
    args = parser.parse_args()
    if args.scenario == "employment" and not args.data_path:
        parser.error("employment requires --data-path")
    workspace = args.directory.resolve() / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    store = RunStore(args.directory.resolve() / "controller", "live-" + args.scenario)
    profile = ExecutionProfile(
        codex_bin=str(Path(args.codex_bin).resolve()),
        worker_model=args.model,
        worker_effort="xhigh",
        native_prompt_profile="open-task-v1",
        max_worker_rounds=100 if args.scenario in {"storage", "employment"} else 12,
        max_active_seconds=7200 if args.scenario in {"storage", "employment"} else 1200,
        **decision_settings(args),
    )
    TaskRuntime.create(
        store,
        "Compute complete US position occupation counts and strict explicit AI-label counts from the authorized Parquet, with a reproducible combined result."
        if args.scenario == "employment"
        else "Implement a modular CSV gzip file store with exact string-row roundtrip and fresh-process recovery."
        if args.scenario == "storage"
        else "Implement calculator.py:add(a,b) for integers, with real executable checks.",
        workspace,
        profile,
        input_refs=[str(args.data_path.resolve())] if args.scenario == "employment" else None,
    )

    class FixtureAdapter(CodexAdapter):
        def __call__(self, context, report_type, on_started, timeout):
            report = super().__call__(context, report_type, on_started, timeout)
            marker = store.directory / "test-fixture/fault-injection.json"
            if (
                args.inject_parent_fault
                and context["kind"] == "RECONCILE"
                and context["task_id"] == store.load().root_task_id
                and not marker.exists()
            ):
                target = workspace / "storage.py"
                original = target.read_text()
                target.write_text(
                    original
                    + "\n\n# Controlled regression fixture: runtime must reject the composite.\ndef load_rows(path):\n    return []\n"
                )
                store.write_json(
                    "test-fixture/fault-injection.json",
                    {
                        "round_id": context["round_id"],
                        "path": str(target),
                        "fault": "load_rows returns an empty list after the worker reported its delivery; actual verification must fail",
                    },
                )
            return report

    adapter = FixtureAdapter(profile, workspace)
    runtime = TaskRuntime(store, adapter, None, adapter.command_exec)
    actual_judge = CodexDecisionMaker(
        adapter, JevDecisionMaker(profile, store, runtime.time_left), lambda: runtime.state
    )

    def fixture_judge(kind, target, context):
        # Isolate parent integration from the separately measured autonomous routing decision.
        if (
            args.inject_parent_fault
            and kind == "SHOULD_SPLIT"
            and target == runtime.state.root_task_id
            and not runtime.state.nodes[target].current_plan_revision_id
        ):
            ref = store.write_json(
                "test-fixture/preapproved-root-split.json",
                {
                    "purpose": "Controlled integration test only, NOT a Jev decision or autonomous split result. Root split is preapproved; all subsequent plan, child and completion decisions use the real backend.",
                    "context": context,
                },
            )
            return DecisionRecord(
                id="fixture_root_split_" + digest(context),
                kind=kind,
                target_id=target,
                input_digest=digest(context),
                score=1,
                threshold=profile.decision_threshold,
                model="fixture-preapproved-root-split",
                request_ref=ref,
                response_ref=ref,
            )
        return actual_judge(kind, target, context)

    runtime.judge = fixture_judge
    specification = TaskSpec(
        title="Integer addition",
        goal="calculator.py exports add(a,b) returning the sum of two integers.",
        scope=["Only integer addition; no input validation or other operations"],
        deliverables=["calculator.py"],
        root_requirement_ids=["r-add"],
        requirements=[
            Requirement(
                id="r-add",
                description="Importable integer addition works",
                acceptance_ids=["c-add"],
            )
        ],
        acceptance=[
            AcceptanceCriterion(
                id="c-add",
                description="add returns correct integer sums for positive, negative, mixed-sign and zero values",
                root_requirement_ids=["r-add"],
                check_spec=CheckSpec(
                    command=[
                        sys.executable,
                        "-B",
                        "-c",
                        "from calculator import add; assert [add(a,b) for a,b in [(2,3),(-4,-6),(-4,7),(0,9)]] == [5,-10,3,9]; print('addition verified')",
                    ],
                    expected_stdout="addition verified",
                    artifact_paths=["calculator.py"],
                ),
            )
        ],
    )
    if args.scenario == "storage":
        specification = storage_spec()
        if args.inject_parent_fault:
            specification.constraints["controlled_integration_fixture"] = (
                "This fixture specifically verifies actual parent integration and fault recovery. "
                "Delegate independent CSV and gzip codec implementations to child workers, "
                "then integrate storage.py in a fresh parent RECONCILE after their deliveries. "
                "Do not implement codec children in the root. This is a controlled integration "
                "test, not evidence of spontaneous decomposition."
            )
    elif args.scenario == "employment":
        specification = employment_spec(args.data_path.resolve())
    try:
        with store.locked():
            # This predeclared test contract is the fixture approval, never the product's CLI approval path.
            runtime.state.preparation_required = False
            ref = runtime.claim_next_work()
            work = runtime.state.rounds[ref.id]
            work.kind = RoundKind.SPECIFY
            proposal = SpecProposal(
                task_id=work.task_id, round_id=work.id, proposed_spec=specification
            )
            work.worker_report_ref = store.write_json(
                f"reports/{work.id}/worker.json", proposal.model_dump(mode="json")
            )
            runtime._commit_spec(work, proposal)
            runtime.approve_root(
                specification.contract_hash(),
                "user-authorized development integration fixture: " + args.scenario,
            )
            while runtime.state.status == RunStatus.ACTIVE:
                runtime.step()
                print(
                    json.dumps(
                        {
                            "status": runtime.state.status.value,
                            "rounds": runtime.state.progress.worker_attempts_started,
                            "nodes": len(runtime.state.nodes),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        adapter.close()
        export(store, runtime.state)
        print(json.dumps(status(runtime.state), indent=2, ensure_ascii=False))
    if runtime.state.status != RunStatus.SUCCEEDED:
        raise SystemExit(1)
    assert certificate_valid(runtime.state, runtime.state.root_task_id)
    threads = [r.thread_id for r in runtime.state.rounds.values() if r.thread_id]
    assert len(threads) == len(set(threads)) == runtime.state.progress.worker_attempts_started
    if args.inject_parent_fault:
        fixture_decisions = [
            d
            for d in runtime.state.decisions.values()
            if d.model == "fixture-preapproved-root-split"
        ]
        assert fixture_decisions and all(
            d.kind == "SHOULD_SPLIT" and d.target_id == runtime.state.root_task_id
            for d in fixture_decisions
        )
        fault = json.loads((store.directory / "test-fixture/fault-injection.json").read_text())
        failed = next(
            result
            for result in runtime.state.nodes[runtime.state.root_task_id].result_history
            if result.round_id == fault["round_id"]
        )
        assert any(
            runtime.state.verifications[key].status == "FAIL" for key in failed.verification_ids
        )
        assert not any(
            runtime.state.decisions[key].kind == "COMPLETE" for key in failed.decision_ids
        )


def storage_spec():
    codec_check = "from csv_codec import encode_rows,decode_rows; from gzip_codec import compress,decompress; rows=[{'a':'a,b','b':'你好'},{'a':'quote\"','b':'line\\nvalue'}]; raw=encode_rows(rows); assert isinstance(raw,bytes); assert decode_rows(raw)==rows; packed=compress(raw); assert packed[:2]==b'\\x1f\\x8b'; assert decompress(packed)==raw; print('codecs verified')"
    restart_check = "import json,sys,subprocess,tempfile; from pathlib import Path; from storage import save_rows,load_rows; rows=[{'a':'a,b','b':'你好'},{'a':'quote\"','b':'line\\nvalue'}]; tmp=tempfile.TemporaryDirectory(dir='.'); p=Path(tmp.name)/'rows.csv.gz'; save_rows(p,rows); assert p.read_bytes()[:2]==b'\\x1f\\x8b'; assert load_rows(p)==rows; output=subprocess.check_output([sys.executable,'-B','-c','import json,sys; from pathlib import Path; from storage import load_rows; print(json.dumps(load_rows(Path(sys.argv[1]))))',str(p)],text=True); assert json.loads(output)==rows; tmp.cleanup(); print('restart verified')"
    paths = ["csv_codec.py", "gzip_codec.py", "storage.py"]
    return TaskSpec(
        title="CSV gzip file storage",
        goal="Implement a CSV gzip file store that preserves string rows exactly and reopens correctly in a fresh Python process.",
        scope=[
            "csv_codec.encode_rows(list[dict[str,str]]) -> bytes and decode_rows(bytes) -> list[dict[str,str]]",
            "gzip_codec.compress(bytes) -> bytes and decompress(bytes) -> bytes using stdlib gzip",
            "storage.save_rows(path,rows) and storage.load_rows(path), using both codec modules; Unicode, commas, quotes, and line breaks must roundtrip",
            "No other database features, metadata service, CLI or dependencies",
        ],
        constraints={
            "dependencies": "Python standard library only; preserve the specified public APIs.",
            "user_restrictions": "Only the bundled Jev decision protocol skill is allowed. Do not use other skills, previous runs or external subagents. Use only the current workspace and supplied context.",
        },
        deliverables=paths,
        requirements=[
            Requirement(
                id="r-storage",
                description="Modular compressed CSV storage with exact roundtrip after process restart",
                acceptance_ids=["c-codecs", "c-restart"],
            )
        ],
        root_requirement_ids=["r-storage"],
        acceptance=[
            AcceptanceCriterion(
                id="c-codecs",
                description="Independent CSV and gzip codec modules expose the specified APIs and preserve string rows including Unicode and quoted fields",
                root_requirement_ids=["r-storage"],
                check_spec=CheckSpec(
                    command=[sys.executable, "-B", "-c", codec_check],
                    expected_stdout="codecs verified",
                    artifact_paths=paths,
                ),
            ),
            AcceptanceCriterion(
                id="c-restart",
                description="storage.save_rows writes actual gzip CSV; load_rows recovers identical rows locally and after reopening from a new Python process",
                root_requirement_ids=["r-storage"],
                check_spec=CheckSpec(
                    command=[sys.executable, "-B", "-c", restart_check],
                    expected_stdout="restart verified",
                    artifact_paths=paths,
                ),
            ),
        ],
    )


def employment_spec(dataset: Path):
    source = str(dataset / "position.parquet")
    prefix = (
        "import json,duckdb,re; from collections import Counter; from analysis import collect; "
        "actual=collect(); saved=json.load(open('employment_summary.json')); assert actual==saved; "
        f"rows=duckdb.connect().execute(\"SELECT id,role_k1500_v2,role_k17000_v3 FROM read_parquet(?) WHERE country='United States'\",[{source!r}]).fetchall(); "
    )
    occupation = prefix + (
        "expected=dict(Counter(str(r[1]) if r[1] else '__MISSING__' for r in rows)); "
        "assert actual['us_positions']==len(rows); assert actual['occupation_counts']==expected; "
        "print('all observed occupation categories verified')"
    )
    ai = prefix + (
        "pattern=re.compile(r'\\bartificial intelligence\\b|\\bmachine learning\\b',re.I); "
        "ids=sorted({str(r[0]) for r in rows if pattern.search(r[1] or '') or pattern.search(r[2] or '')}); "
        "assert actual['strict_ai_count']==len(ids); assert sorted(actual['strict_ai_ids'])==ids; "
        "print('strict explicit AI labels verified')"
    )
    paths = ["analysis.py", "employment_summary.json", "occupation.py", "ai_labels.py", source]
    return TaskSpec(
        title="US position occupation and explicit AI label evidence",
        goal="Produce reproducible full occupation-category counts and strict explicit AI-label counts for US-workplace positions in the supplied RevelioData, with a combined result and explicit measurement limits.",
        scope=[
            f"Read only {source}; use country == 'United States'.",
            "occupation.collect() returns us_positions and occupation_counts for ALL observed role_k1500_v2 categories; normalize null/empty to __MISSING__, never select only computer/mathematics categories.",
            "ai_labels.collect() returns strict_ai_count and sorted strict_ai_ids as strings; match whole phrases artificial intelligence or machine learning, case-insensitive, in role_k1500_v2 OR role_k17000_v3, then deduplicate position IDs.",
            "analysis.collect() integrates those two actual module results; analysis.py writes the combined employment_summary.json. Document that these are sample classification proxies, not an AI-caused national employment effect.",
            "No extra datasets, reports, classifiers or causal estimation. Multiple code steps may support one result; keep independently answerable outcomes focused.",
        ],
        constraints={
            "sources": "Original data is read-only. Write only in this workspace; no network, prior research outputs or external agents. Only the bundled Jev protocol skill is allowed.",
            "python": f"Use the installed interpreter {sys.executable}; duckdb is available for Parquet queries, do not install dependencies.",
        },
        deliverables=paths[:-1],
        requirements=[
            Requirement(
                id="occupation",
                description="Full observed US occupation distribution",
                acceptance_ids=["occupation"],
            ),
            Requirement(
                id="ai-labels",
                description="Reproducible strict AI classification signal",
                acceptance_ids=["ai-labels"],
            ),
        ],
        root_requirement_ids=["occupation", "ai-labels"],
        acceptance=[
            AcceptanceCriterion(
                id="occupation",
                description="The combined result and actual collect() reproduce every observed US occupation category and its denominator from the original Parquet.",
                root_requirement_ids=["occupation"],
                check_spec=CheckSpec(
                    command=[sys.executable, "-B", "-c", occupation],
                    expected_stdout="all observed occupation categories verified",
                    artifact_paths=paths,
                ),
            ),
            AcceptanceCriterion(
                id="ai-labels",
                description="The combined result and actual collect() reproduce the exact whole-phrase union of strict AI labels with deduplicated string IDs from original Parquet.",
                root_requirement_ids=["ai-labels"],
                check_spec=CheckSpec(
                    command=[sys.executable, "-B", "-c", ai],
                    expected_stdout="strict explicit AI labels verified",
                    artifact_paths=paths,
                ),
            ),
        ],
    )


if __name__ == "__main__":
    main()
