"""Explicit, bundled worker skill; preview never calls the API or mutates a run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .decision_context import REVIEWABLE_SECTIONS, working_budget
from .jev import PREDICATES, prepare_request
from .models import ContextReviewReport, ExecutionProfile

SKILL_DIRECTORY = Path(__file__).parent / "skills" / "jev-decision-input"
SKILL_PATH = SKILL_DIRECTORY / "SKILL.md"


def instructions() -> str:
    return (
        "\nThe harness explicitly supplies its bundled $jev-decision-input protocol skill below. "
        "This does not enable discovery or use of any other skill.\n"
        + SKILL_PATH.read_text().split("---", 2)[-1].strip()
    )


def support_context(profile: ExecutionProfile) -> dict:
    return {
        "skill": str(SKILL_PATH),
        "preview_script": str(SKILL_DIRECTORY / "scripts" / "preview.py"),
        "python": sys.executable,
        "working_budget_bytes": working_budget(profile),
        "hard_limit_bytes": profile.max_jev_input_bytes,
        "decision_model": profile.jev_model,
        "token_policy": "No exact provider tokenizer is configured; byte budgets are not token counts.",
        "reviewable_sections": sorted(REVIEWABLE_SECTIONS),
        "authority": "Fresh DECISION workers call the bound jev_decide tool using this skill. Other roles prepare evidence. The tool enforces transport/budget; the controller records decisions and alone changes tree state.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=sorted(PREDICATES))
    parser.add_argument("--model", default="decision-model-preview")
    parser.add_argument("--max-bytes", type=int, default=48000)
    parser.add_argument("--hard-limit-bytes", type=int, default=262144)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--review-report", help="ContextReviewReport JSON path, or - for stdin")
    parser.add_argument("--show-packet", action="store_true")
    args = parser.parse_args(argv)
    context = json.loads(args.context.read_text())
    if args.review_report:
        raw = (
            sys.stdin.read() if args.review_report == "-" else Path(args.review_report).read_text()
        )
        review = ContextReviewReport.model_validate_json(raw)
        context["decision_brief_override"] = {
            "decision_brief": review.decision_brief.model_dump(mode="json"),
            "reviewed_sections": review.reviewed_sections,
            "review_ref": "worker preview only; controller validation required",
        }
    prepared = prepare_request(args.kind, args.model, context, args.run_directory)
    profile = ExecutionProfile(
        max_jev_input_bytes=args.hard_limit_bytes, jev_context_target_bytes=args.max_bytes
    )
    fits = len(prepared.encoded) <= working_budget(profile)
    result = {
        "fits_working_budget": fits,
        "wire_bytes": len(prepared.encoded),
        "working_budget_bytes": working_budget(profile),
        "provider_token_count": None,
        "selections": prepared.selection,
        "authority": "Preview only: evidence is not certified, no API call or state update occurred.",
    }
    if args.show_packet:
        result["packet"] = prepared.selected
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fits else 2


if __name__ == "__main__":
    raise SystemExit(main())
