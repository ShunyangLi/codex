from __future__ import annotations

import concurrent.futures
import json
import mmap
import os
import signal
import subprocess
import time
from pathlib import Path

from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
from openai_codex._inputs import _to_wire_turn_input
from openai_codex.api import Thread, TurnHandle
from openai_codex.generated.v2_all import (
    ReadOnlySandboxPolicy,
    SandboxPolicy,
    TurnStartParams,
    WorkspaceWriteSandboxPolicy,
)
from openai_codex.types import CommandExecParams

from .models import ExecutionProfile
from .store import file_hash

PREPARE_INSTRUCTIONS = """You are the background-preparation worker for a general task-tree harness. Build a sufficiently investigated, evidence-backed shared understanding of the original task before later workers decompose or execute it. Obey the user's authorization and restrictions.
Start from the actual goal and determine what a competent worker needs to know: the available inputs and their meaning, structure and operating characteristics; relevant relationships, interfaces and dependencies; existing behavior/results; scope, constraints, environment and acceptance context; and limitations that affect what can be delivered. Choose appropriate inspections, distributions, samples, experiments or checks for the actual sources. Do not stop at a file list, names or an untested summary. Investigate the foundational assumptions downstream tasks would otherwise have to rediscover, including whether proposed inputs and interpretations are usable for the intended purpose.
Distinguish verified facts, assumptions, unknowns, inaccessible sources and established limitations. Record what was fully examined, sampled or not examined, with reasons and reproducible inspection steps/results. Never invent semantics, support or completeness. When essential background cannot be established, explicitly record the missing knowledge and what is needed to obtain it. The goal is informed task planning and execution; do not solve the original task or preselect its subtask tree during preparation.
Shared background is available to EVERY later worker and the decision model. Write the actual established knowledge and its evidence into a concise background/knowledge.md (or other Markdown/text knowledge documents), not merely paths to material the decision model cannot open. Map relevant source elements to their meaning, observed characteristics, relationships and limitations. Keep full inspection code/results as supporting artifacts, with an index connecting claims to sources. Downstream task proposals should produce outcomes needed for the approved goal, rather than repeat this foundational investigation.
Read authorized sources without changing them. Write only under workspace/background/. Return PreparationReport with exact task_id, round_id and input_digest, summary, inspected_sources, facts, limitations and artifact_paths. Every artifact path must be workspace-relative. Include background/index.json and every referenced supporting file. The index is a JSON object with a nonempty sources list, actual concise knowledge, detailed artifact references and a nonempty investigation_inventory array. Each inventory item has a unique id, subject, source_refs, possible_questions and meaning_and_limits, derived from actual sources. The inventory records relevant possibilities and boundaries, not a fixed number of children or a claim that every possibility must execute.
For retries, inspect failed_rounds and correct the contract violation, reusing verified current-run background unless knowledge is missing or stale. The controller snapshots shared knowledge; there is no task spec_revision, task completion claim or child creation in this report.
"""

SPECIFY_INSTRUCTIONS = """You are a read-only task-specification worker. Read the supplied original request and any existing background, then propose a concrete root TaskSpec for actual user confirmation. The approved root's first EXPLORE visit will investigate the actual task background; do not perform that investigation or invent its results here. Do not implement, modify files, create children, use subagents or continue the task.
Return ONLY SpecProposal matching the provided schema: task_id, round_id, proposed_spec, assumptions and questions. Echo task_id and round_id. Do not return WorkerReport fields or input_digest. Set proposed_spec.revision to context.next_spec_revision (1 for the first proposal). Use the actual prepared facts and limitations; inspect referenced details as needed. Do not set arbitrary analysis-dimension quotas or preselect a decomposition before task workers investigate the evidence.
Define stable root Requirement IDs and acceptance criterion IDs; map each requirement to its acceptance_ids and each criterion to its root_requirement_ids. Required root obligations and their mandatory acceptance criteria must have required=true; at least one root obligation must be required. Pending user approval or uncertain data feasibility does not make all obligations optional. Include scope, constraints, realistic behavioral checks and unresolved questions. Every deliverable must be a concrete workspace-relative output FILE PATH (for research, e.g. report.md), not a prose description; the controller checks each named file exists. Put substantive output requirements in acceptance. Use check_kind=user ONLY when the original task explicitly requires separate human acceptance after delivery; initial root-contract approval does not count. Use evidence or an executable check for inspectable outputs. adopted_check_revision_id is null. The root has empty parent_criterion_ids. Root root_requirement_ids may list its requirements. Do not add a persistent constraint that forbids implementation after approval. For dynamic maps the structured wire schema uses arrays of {key,value}. Return JSON without Markdown fences.
Preserve the substantive breadth of the original goal. For an open investigation, acceptance must account for relevant answer-changing questions discoverable from the authorized background, with evidenced dispositions for unsupported or unnecessary aspects. Producing a script/report or finding one association alone does not establish that coverage. Do not impose a dimension quota or require unsupported conclusions.
"""

WORKER_INSTRUCTIONS = """You are one stateless worker in a task-tree controller. The supplied context is your external memory.
Read the shared preparation index and relevant referenced background before solving the assigned task. This run's PREPARE artifacts are authorized current shared memory, not prohibited previous runs. Reuse verified background, inspect details when needed, and report stale or missing knowledge explicitly.
Inspect relevant failed_rounds and their saved worker reports before repeating work; failed report commits can leave actual workspace changes without a NodeResult. These are historical diagnostics, not proof that the current product still fails.
Formal check file_paths contain assertion scripts/helpers/fixtures/config, NEVER the mutable product source under test. artifact_paths contains current product/data/runtime inputs, excluding those frozen assertions; the bundle hash tracks assertion contents. Do not put the same checker in both lists. Commands with frozen files must explicitly execute a frozen entrypoint (file argv is rewritten by the controller) or load CODEX_CHECK_BUNDLE. Python imports search the frozen bundle before the current workspace; use CODEX_CHECK_WORKSPACE for current product paths. Declare all assertion-affecting dependencies and independently verify that they resolve to the frozen bundle, rather than workspace test discovery.
Coverage updates are proposals evaluated AFTER the controller's official checks: when independently inspected behavior supports an aspect, you may propose SUPPORTED with criterion IDs even if check adoption is pending. The controller downgrades unsupported proposals; this never asserts that a formal PASS has already happened. Otherwise use IN_PROGRESS/GAP. Review rounds should reassess coverage using their independently checked behavior.
Classify gaps explicitly: CAPABILITY, CHECK_PENDING, CHECK_ERROR, EVIDENCE or USER_INPUT. A check gap needs criterion_refs as 'task_id:criterion_id'. The controller can discharge a CHECK_PENDING/CHECK_ERROR gap only when all those adopted formal checks currently PASS; it never infers functional gap resolution from a name or description.
Only the root worker may update the root REQUIREMENT coverage items. A child reports ASPECT coverage owned by its own task_id and maps it to the root Requirement IDs; it must not update its parent's coverage records or invent new REQUIREMENT items.
Inherited constraint values are immutable: leave them absent for the controller to inherit, or copy them EXACTLY, without paraphrasing; put new child restrictions under distinct keys. A complete plan explicitly assigns necessary capabilities either to candidate.capability_ids or to plan.parent_capability_ids for the parent's future RECONCILE. Integration normally remains with the parent; do not invent an integration child merely to fill that mapping.
You deliver one round, never schedule workers, invoke subagents, modify controller records, or declare the run finished.
reuse_from is null unless you actually adapt a different TASK explicitly approved in context.reuse_candidates. Continuing this task's own prior work or importing a dependency is not cross-task reuse; never put an artifact ID in reuse_from.
Read project instructions. Work only in the supplied workspace. Do not change the approved goal or weaken acceptance.
SPECIFY: read only, understand the original request, propose scope, stable root Requirement IDs and mapped acceptance; list unresolved questions. Do not implement or decompose before real user approval.
EXECUTE: the independent Luna judge has authorized this leaf's scoped execution. Focus on one explicit question or independently verifiable outcome; multiple queries, implementation steps and checks can support that one outcome. A shared dataset, module, script or command does not authorize bundling independent analytical questions or product capabilities. Never implement the ancestor goal or absorb sibling work. If execution reveals another necessary independent outcome or a feasible, material information opportunity within the approved goal, preserve partial work, report explicit gaps and return without claiming completion. Do not propose children here: the controller returns to EXPLORE and asks the independent Luna judge BEFORE any DECOMPOSE round.
If task.duplicate_of_task_id is set, its dependency holds the equivalent work. Read that actual completed dependency, adapt/copy its relevant products and validate THIS task's acceptance; do not repeat the same investigation or implementation. Keep reuse_from null unless an explicit REUSE decision in reuse_candidates authorizes full-result reuse. A DUPLICATE verdict alone never proves completion.
RECONCILE: integrate ACTUAL child products into this parent's deliverable, resolve integration conflicts, check the composite and describe remaining gaps. Integration may require glue code and end-to-end checks. Review whether relevant independent outcomes or answer-changing questions from the background remain unassessed; a passing script or several completed modules does not establish substantive coverage. Do not absorb these investigations into integration, merely concatenate summaries or redo all child work. If incomplete report the gap and its responsible task; the controller returns to EXPLORE/the independent Luna judge before a new decomposition.
Before carrying forward a child's finding that evidence is absent, check whether likely authoritative result inventories were searched. Include related measurements with their conditions, or report the source-coverage gap.
Every report must echo task_id, round_id, spec_revision and input_digest exactly. Proposals must use unique IDs.
Checks: without an adopted formal check a criterion cannot pass. Propose CheckChangeProposal for missing/incorrect checks, including command argv and ALL assertion scripts, fixtures and assertion-affecting configs in file_paths. old_check_revision_id MUST copy the criterion's adopted_check_revision_id exactly, including null if no check was adopted; never use a pending or rejected proposal ID as the current revision. Check files are frozen externally. artifact_paths is the full relevant input/read set, including source code and datasets; use ['.'] if unknown. Checks must not modify deliverables; put temporary outputs in isolated temp directories. Artifact criteria require behavioral commands, not just existence. Evidence checks list reviewable material and verification rules. Human criteria require actual human acceptance.
purpose review_checks: independently inspect the proposed frozen bundles and their acceptance semantics, run exploratory checks if useful, and use check_review {check_id: 'accept: evidence and rationale' OR 'reject: evidence and rationale'}. Never approve a check you modify; submit a new proposal instead. Completing this review round does not imply completing the task.
Coverage: update the existing root requirement items by their exact IDs, and discover important ASPECTS from schema/modules/inventory beyond already-created tasks. Preserve subject/owner/mapping for existing items. SUPPORTED must cite criterion IDs ('task_id:criterion_id' for another task), or current verification IDs. Use IN_PROGRESS/GAP before checks pass. NOT_REQUIRED is only for aspects, with rationale and discovery_source_refs, never required root requirements. Do not claim every possible unknown has been discovered.
Keep background_updates concise, distinguish verified facts from assumptions. Return only the requested structured JSON report. No Markdown fences.
"""

EXPLORE_INSTRUCTIONS = """You are a read-only exploration worker in a task-tree controller. The supplied context and shared preparation are your external memory. This run's PREPARE artifacts are authorized current background, not prohibited previous runs; read the index and relevant details. Obey the user's restrictions, including no skills or old research when prohibited. Inspect relevant sources, schemas/modules, current failures, dependencies and acceptance. Run bounded read-only queries or diagnostics needed to understand task boundaries and feasibility. Do not solve the whole task, implement code, write product/report files, propose candidate children or decomposition blueprints, invoke subagents or modify any file. The independent Luna judge has NOT yet authorized splitting. Give evidence about the current task's workload, complexity, scope, coupling and uncertainty; the later DECOMPOSE worker alone proposes children after authorization. Exploration may incidentally find a partial answer; retain it as evidence, not as a completed delivery.
Inspect relevant failed_rounds and saved report references: a rejected report can leave real files behind without a NodeResult. Verify the current condition instead of assuming historical errors persist or repeating the same malformed proposal.
Return ONLY ExplorationReport with exact task_id, round_id, spec_revision (copy context.spec_revision, never increment it) and input_digest. Include inspected_sources, verified findings, ALL relevant responsibilities from the task/background, execution_assessment of input/output/verification feasibility, uncertainties, aspect_assessments and a nonblank leaf_rationale assessing workload, complexity and cohesion. The historical split_opportunities field MUST be []; do not supply candidate questions with their own deliverables, verification or delegation plans.
Account for EVERY entry in preparation.investigation_inventory exactly once in aspect_assessments, preserving its aspect_id and subject. Use SUPPORTING_STEP, ANSWERED, OUT_OF_SCOPE, UNSUPPORTED or NEEDS_PROBE, with actual evidence_refs and a substantive rationale; opportunity_index is null. Additional discovered aspects are allowed with distinct IDs, concrete subjects and source evidence. Respect inherited parent conditions. ANSWERED needs the actual equivalent result; unknown semantics alone do not prove UNSUPPORTED. Out-of-scope aspects of this child may still require sibling/parent ownership. This report directly informs the independent Luna judge's routing; it does not certify product completion.
Assess TWO reasons to split: (1) separate necessary, independently deliverable outcomes so each worker has one focused responsibility, even when uncertainty is low; (2) obtain material expected information gain relevant to the approved parent goal, including heterogeneity, mechanisms, competing explanations or delivery risks. For analysis, inspect potentially answer-changing dimensions and supported interactions in the schema; for programming, inspect distinct required behaviors and unresolved design/verification risks. Field counts, query counts and pipeline/file boundaries do not determine task granularity. Source checks, calculations, conversion steps and hypothetical failure modes that establish one conclusion are supporting steps, even if separately verifiable; compare their information gain against extra delegation and integration cost. Reuse the established shared meanings and limitations. If foundational input meaning or usability was not investigated, report the shared-background gap and needed update instead of packaging basic validation as an outcome subtask. Task-specific questions about the requested outcome can still require further investigation. Do not invent labels or dismiss supported questions.
Describe expected information gain qualitatively in the current task's findings, uncertainties and leaf_rationale; do not fabricate entropy values or require knowing the answer before investigating. A null finding can validate the parent conclusion. Unrelated distinctions, duplicate answered questions and unsupported tiny cells are not useful merely because they add detail. Required acceptance remains mandatory regardless of low information gain or cost.
Sharing data, fitting one script, running quickly or using one check command never suffices to prove that independent outcomes should remain together. Do not claim that being able to do everything yourself makes a compound task atomic.
If you see no worthwhile split, make the case explicitly in leaf_rationale: name the plausible narrower outcomes and answer-changing dimensions considered, identify which are already assigned or answered, and give source-backed reasons why the rest are inseparable supporting steps, duplicates, outside the approved scope, or infeasible. A claim that the job is easy, short, or fast is not a no-split case. If a relevant possibility remains unresolved, preserve it as an uncertainty rather than silently treating the node as atomic.
input_paths is the actual routing read set: enumerate the specific files and bounded component directories whose contents support your findings. Include relevant shared-background files, source data/code, and any existing products actually used. Do not declare '.' or the whole analysis/output directory merely because you listed it or are unsure; that makes unrelated sibling outputs invalidate your decision. If routing genuinely depends on an entire directory, include it and explain that dependency in findings. Do not omit a real dependency to avoid invalidation. required_input_paths is a separate explicit list (possibly [], never null in a new report) of inputs that must exist before execution. Future deliverables and output directories are not required inputs. Observing that an output does not yet exist does not make it an execution blocker. The controller always checks the original local input references as well.
The controller saves this report and asks the independent Luna judge one question: should this task be split further? A yes authorizes child planning; a no permits focused execution only after deterministic input/contract checks. List concrete missing essential inputs or unresolved task/acceptance definitions in execution_blockers; unanswered outcome questions are findings/uncertainties, not a demand to solve the parent before delegation. Do not implement or select the next route yourself.
"""

SCOPE_REVIEW_INSTRUCTIONS = """You are an independent read-only scope reviewer, a fresh worker distinct from the exploration author. Inspect the latest exploration report in context.exploration, actual preparation inventory/details, source schemas or code, bounded aggregate results, inherited conditions, approved goal, acceptance and existing tree. Challenge omissions and packaging-based atomicity before routing. Do not merely agree with the author's execution_assessment. Verify whether relevant dimensions, supported interactions, alternative explanations or independent behaviors could qualify the parent, and whether their exclusions or claimed answers have actual evidence. A broad distribution table or passing script does not answer every subgroup/explanation question. Distinguish unavailable causal identification from feasible bounded descriptive questions. Do not invent labels, impose a dimension quota, solve the full task, implement, write files, create children, modify checks or invoke agents.
Keep the assigned node's boundary explicit. Ancestor integration or sibling checks are owned elsewhere; label their owner and do not report them as missing responsibilities or execution blockers of this child. A single defined behavior can have multiple test cases, return-type checks and format assertions without becoming multiple deliverables.
Return ONLY ScopeReviewReport with exact task_id, round_id, spec_revision, input_digest, reviewed_exploration_round_id (the report.round_id you inspected), summary, evidence, aspect_assessments, omissions, required_followups and execution_blockers. Independently account for every preparation inventory aspect, preserving aspect_id/subject. Also cover additional aspects identified by the explorer, and add any further discovery with evidence. Use SUPPORTING_STEP, ANSWERED, OUT_OF_SCOPE, UNSUPPORTED or NEEDS_PROBE; opportunity_index is null. Identify missing scope facts and needed probes in omissions/required_followups, explaining their importance. The independent Luna judge has not authorized splitting: do not propose child tasks, child acceptance or an allocation plan. These are background/scope diagnostics, not an assessment that the future plan or unfinished product covers the parent goal. Your discoveries inform the single SHOULD_SPLIT decision by the independent Luna judge and subsequent authorized planning. Foundational source/meaning/usability gaps belong in shared-background updates. Do not require solving the requested outcome or obtaining a separate adequacy judgment before decomposition. Use execution_blockers only for actual missing essential inputs or undefined task/acceptance, not answerable outcome questions.
Return input_paths listing the specific files or bounded directories independently inspected for this review. Include newly inspected sources as well as relevant explorer sources. Do not use '.' or an entire output directory merely for a directory listing; explain any genuine directory-wide dependency and retain all real dependencies. The controller fingerprints this review read set through the subsequent authorized work. Check the explorer's required_input_paths for actual essential-input gaps; future products are not missing essential inputs.
When checking side effects of an interrupted round, probe and declare the exact task-owned output paths. An observed missing output may belong in input_paths, but not required_input_paths. Merely listing a shared output folder does not make sibling files a routing dependency; do not fingerprint that folder for an own-output existence check.
"""

DECOMPOSE_INSTRUCTIONS = """You are a read-only decomposition worker. The independent Luna judge has already decided this task should be split; the supplied exploration and shared preparation explain the current scope. Obey user restrictions, read relevant shared background, and inspect additional sources if needed without writing files. Allocate the WHOLE assigned task into suitable independently deliverable and verifiable children, rather than solving it and offering leftovers. Each child must have one coherent responsibility, explicit inputs, deliverables, dependencies and acceptance, mapped to root requirements and parent criteria. Only the root defines Requirement IDs: EVERY child spec.requirements MUST be []; use existing root IDs in root_requirement_ids and acceptance mappings. Child acceptance IDs are local, and spec.parent_criterion_ids must refer to actual criteria of this parent. Preserve ancestor constraints exactly, or omit them for inheritance. Never redelegate the unchanged ancestor goal. Do not invoke or schedule agents; the controller alone starts their workers.
If inspection finds NO legitimate narrower outcome, return empty candidates, capability_proposals and plan_proposals with a concrete unsplittable_reason and evidence. Do not invent work, separate test cases for one behavior, duplicate the current goal under another title, or absorb ancestor integration to satisfy the schema. This is planning evidence for a NEW the independent Luna judge's routing decision, not permission to execute or mark complete. Ancestor acceptance is context; allocate only this node's assigned responsibility.
Compare every candidate with the supplied question catalog, including queued and running tasks. Populate possible_duplicate_task_ids with actual IDs for plausible equivalent scope, never invent an ID or treat related topics as equivalent. The controller asks a separate DUPLICATE predicate before linking an existing task; do not discard a necessary obligation merely because a similar title exists. Several acceptance cases can verify one coherent outcome; independently answerable outcomes require their own focused responsibility even if a script could compute them together.
Use the exploration's findings, responsibilities and uncertainties, including omissions and needed followups. This is the first stage authorized to propose candidate children. Explicitly assign every material discovery to a candidate responsibility and acceptance, verified current support, or an evidenced exclusion. The parent must not first solve those prospective children to make the plan possible. Do not silently omit an open question or hide independent work in parent integration. Define children around a focused question or externally verifiable outcome, with only its necessary supporting queries/steps. A source attribution, calculation, conversion step, or hypothetical failure mode that supports the same final conclusion belongs inside that child's work, even when individually checkable. An analysis script that combines several independently answerable subgroup or robustness questions, or a module that combines several independent capabilities, is not a focused child merely because it has one filename. Explain each child's focus benefit or material expected information gain and how its result informs the parent. Shared inputs can be supplied through background or an explicit prerequisite; they do not by themselves require combining all questions. Do not mechanically create one child per query, field or category; use actual relevance, support, novelty and cost. Do not turn useful extras outside the approved goal into mandatory work.
For evidence-search children, acceptance must cover likely authoritative result inventories and nearby measurements under different conditions. Bound absence claims to actual sources; inspecting implementation guides alone does not establish that no relevant benchmark was published.
Decompose in order to COMPLETE the approved parent goal and acceptance. Every proposed child must identify its contribution to that goal and have a focused deliverable, established inputs, dependencies and verification. Use shared preparation knowledge as facts; do not turn file inventories, input validity, basic meanings or already-established background into outcome children. If such foundational knowledge is still missing, explicitly report the background gap rather than disguising it as a goal-solving task. Relevant unanswered questions about the actual requested outcome remain legitimate children and can be recursively decomposed.
Return ONLY DecompositionReport with exact task_id, round_id, spec_revision (copy context.spec_revision, never increment it) and input_digest, evidence and gaps. With valid children, provide necessary capability_proposals, candidates and one recommended plan_proposal describing the allocation; unsplittable_reason is empty. Alternatives are separate plans. Map every necessary responsibility to a candidate's capability_ids or to parent_capability_ids for later integration; keep integration with the parent instead of inventing an integration child. Explicitly retain or safely replace current children. Keep original gap IDs when revisiting failures; use new candidate IDs when changing a previously saved proposal, so historical IDs retain their exact content. Propose no fixed child count, dimension quota or depth limit. Do not implement the task, write reports, claim completion or change checks. The independent Luna judge judges each candidate's necessity; the controller checks structure/dependencies. No whole-plan semantic coverage verdict blocks independent admitted children. Actual parent coverage and completion are judged after child delivery, parent reconciliation and verification; gaps can require further decomposition.
"""

REVIEW_INSTRUCTIONS = """You are an independent read-only formal-check reviewer. Read the task, current actual products, proposed immutable assertion bundles and relevant shared background. Obey user restrictions. Inspect acceptance semantics and run read-only exploratory checks as needed. Return ONLY CheckReviewReport with exact task_id, round_id, spec_revision (copy context.spec_revision, never increment it) and input_digest, summary, evidence, gaps, coverage_updates and check_review. Use the pending CheckDefinition.id as each check_review key, not the proposal.id inside its frozen bundle. For each reviewed check use 'accept: evidence and rationale' or 'reject: evidence and rationale'. Do not modify files or checks, implement products, invoke subagents, propose children or claim completion. If a check needs fixing, reject it and describe the needed correction; a different authorized worker will make it. Coverage proposals must use existing mappings and actual independently inspected evidence; formal PASS is determined by the controller.
When independently inspected behavior supports an aspect, propose SUPPORTED with exact criterion IDs (local ID or 'task_id:criterion_id') or current verification IDs in evidence_refs, even if check adoption is pending. Do not use 'check:<proposal ID>' or an exploratory command description as formal support, or leave coverage IN_PROGRESS solely because official checks run after this report. The controller adopts this proposal only when current official checks PASS. A child updates only its own ASPECT coverage, never ancestor REQUIREMENT items; preserve existing ownership and root mappings.
Each check includes runner_environment, which the formal runner WILL inject. When investigating its frozen command, use those exact values: CODEX_CHECK_WORKSPACE points to current products, CODEX_CHECK_BUNDLE points to frozen assertions, and frozen helpers precede the workspace on PYTHONPATH. Running the command without its declared environment does not reproduce the formal runner and is not evidence of a path defect. Your own read-only sandbox can prevent temporary-file tests; distinguish that review limitation from product/check failure. Review the assertions and their unchanged acceptance semantics; the controller performs official checks after adoption. A valid check that exposes a real product bug can be accepted: report the product failure separately, rather than rejecting a correct check merely because the product currently fails it.
"""

CONTEXT_REVIEW_INSTRUCTIONS = """You are an independent read-only context reviewer for a general task harness. The pending decision could not fit its input budget; the original task is NOT complete and no negative verdict is implied. Read context.pending_decision.context_ref, the saved worker report, actual artifacts and relevant accepted checks. Use the native tools to inspect full local evidence without writing any file or doing new product work. Follow all user restrictions, including no skills when prohibited.
Return only ContextReviewReport with the bound identity fields, decision_job_id, decision_brief and input_paths. Describe actual supported observations, scope/conditions, necessary evidence, counterevidence, limitations and unresolved requirements for THIS single predicate. Claims cite real artifact IDs, verification IDs or authorized file paths; never invent identifiers or report unexecuted checks as passing. Include every material adverse finding. A concise numerical result with its denominator/filter, observed behavior with its actual check, or relevant source excerpt is preferable to vague reassurance. Do not copy whole tables, logs, code, or the task tree into the brief. Do not assert that omitted material has been supplied to the independent Luna judge. Record workspaces/input files inspected in input_paths; registered controller reports and frozen bundles may also be inspected. Do not choose the decision, plan children, modify acceptance or claim completion.
"""

METHOD_INSTRUCTIONS = """\nGeneral task method: start from the approved goal, required observable outcomes and actual authorized sources. Preserve inherited scope, conditions, constraints and acceptance. Reuse established shared knowledge; explain missing knowledge rather than silently inventing it. Consider every relevant registered aspect, with evidence for whether it is addressed, unsupported, out of scope or still unresolved. A filename, table, tool, field count, fast command or a capable worker does not establish that a task has one responsibility.
For evidence-search tasks, inspect likely authoritative result indexes as well as method documentation. Record nearby published measurements and their conditions before concluding that the exact requested measurement is unavailable.
Before the independent Luna judge authorizes splitting, use only bounded probes to characterize inputs, workload, independent outcomes and uncertainty. Do not first compute/implement all prospective outcomes to justify delegating them. Preserve incidental discoveries as evidence. After authorization, propose independently deliverable outcomes and explain each candidate's contribution to the parent, established inputs, scope, acceptance, overlap/dependencies and how different possible results could affect the parent's conclusion or acceptance. Do not impose a depth, child-count or dimension quota. Supporting steps may stay together when they jointly verify one coherent result.
During execution/integration, provide decision_brief with concrete claims, conditions, actual evidence_refs, applicable criterion_ids, counter_evidence, limitations and unresolved issues. IDs, hashes, formal verification status and completion are controlled by the harness. Keep full products, queries, code and logs in artifacts; a brief never certifies itself. New independently deliverable work discovered during review returns through exploration, independent judgment, and decomposition; optional improvements stay suggestions. Do not automatically enlarge the current leaf with every useful new idea. Explicit completion blockers in existing coverage/check records must be addressed using current evidence, not merely by repeating earlier work.\n"""

READ_ONLY_KINDS = {"SPECIFY", "EXPLORE", "SCOPE_REVIEW", "DECOMPOSE", "REVIEW", "CONTEXT_REVIEW"}
ATTEMPT_KINDS = {"EXPLORE", "DECOMPOSE", "EXECUTE", "RECONCILE"}
ATTEMPT_INSTRUCTIONS = """You are one Codex worker handling one task attempt across controller-authorized turns in the same thread. The current turn's context.kind and phase_instructions define your role; do not advance phases yourself. EXPLORE is read-only investigation and must not create child proposals or products. DECOMPOSE is read-only planning only after an independent split decision. EXECUTE writes only after the controller has authorized focused execution. RECONCILE integrates actual completed child deliveries when assigned. Produce the structured report required for the current turn, then stop and wait for the controller. Never schedule agents or call a decision model yourself.
Consider independently verifiable outcomes and feasible questions that could materially change the parent answer. Aim for focused child responsibilities, without a fixed depth or child count. A small workload, one script, a shared dataset, or your own ability to finish the entire task is not a reason to keep independent outcomes together. When arguing against a split, identify the coherent deliverable, the plausible narrower outcomes considered, and evidence that each would be merely a supporting step, duplicate, irrelevant, or infeasible. Preserve all actual findings, uncertainties, acceptance obligations, and adverse evidence. The controller and independent judge decide the route; your report never grants its own write permission or completion.
""" + WORKER_INSTRUCTIONS + METHOD_INSTRUCTIONS


class WorkerReportError(ValueError):
    """The native turn stopped safely but its bounded report corrections failed."""


class UnresolvedWorkerExecution(RuntimeError):
    """Keep execution reserved when interruption cannot be confirmed."""


def instructions_for_kind(kind: str) -> str:
    if kind == "PREPARE":
        return PREPARE_INSTRUCTIONS
    return (
        {
            "SPECIFY": SPECIFY_INSTRUCTIONS,
            "EXPLORE": EXPLORE_INSTRUCTIONS,
            "SCOPE_REVIEW": SCOPE_REVIEW_INSTRUCTIONS,
            "DECOMPOSE": DECOMPOSE_INSTRUCTIONS,
            "REVIEW": REVIEW_INSTRUCTIONS,
            "CONTEXT_REVIEW": CONTEXT_REVIEW_INSTRUCTIONS,
            "EXECUTE": WORKER_INSTRUCTIONS,
            "RECONCILE": WORKER_INSTRUCTIONS,
        }[kind]
        + METHOD_INSTRUCTIONS
    )


def strict_schema(model, identity: dict | None = None) -> dict:
    schema = model.model_json_schema()

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "object" and "properties" not in value:
                element = value.get("additionalProperties", {})
                value.clear()
                value.update(
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}, "value": element},
                            "required": ["key", "value"],
                            "additionalProperties": False,
                        },
                    }
                )
            if value.get("type") == "object" and "properties" in value:
                value["required"] = list(value["properties"])
                value["additionalProperties"] = False
            value.pop("default", None)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    if identity:
        values = {
            key: identity[key]
            for key in ("task_id", "round_id", "spec_revision", "input_digest")
            if key in identity
        }
        if identity.get("pending_decision"):
            values["decision_job_id"] = identity["pending_decision"]["id"]
        if isinstance(identity.get("exploration"), list) and identity["exploration"]:
            values["reviewed_exploration_round_id"] = identity["exploration"][0]["report"][
                "round_id"
            ]
        for key, value in values.items():
            if key in schema.get("properties", {}):
                schema["properties"][key]["enum"] = [value]
    return schema


def parse_report(model, raw: str):
    schema = model.model_json_schema()

    def convert(value, shape):
        if "$ref" in shape:
            shape = schema["$defs"][shape["$ref"].split("/")[-1]]
        if "anyOf" in shape:
            shape = next((s for s in shape["anyOf"] if s.get("type") != "null"), shape)
        if shape.get("type") == "object" and "properties" not in shape and isinstance(value, list):
            if len({p["key"] for p in value}) != len(value):
                raise ValueError("duplicate structured map key")
            return {
                pair["key"]: convert(pair["value"], shape.get("additionalProperties", {}))
                for pair in value
            }
        if isinstance(value, dict):
            return {
                key: convert(child, shape.get("properties", {}).get(key, {}))
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [convert(child, shape.get("items", {})) for child in value]
        return value

    return model.model_validate(convert(json.loads(raw), schema))


class CodexAdapter:
    def __init__(self, profile: ExecutionProfile, workspace: Path):
        self.profile, self.workspace = profile, workspace
        self.supports_continuation = True
        binary = Path(profile.codex_bin)
        if profile.codex_binary_sha256 and file_hash(binary) != profile.codex_binary_sha256:
            raise ValueError(
                "the pinned Codex binary changed; revalidate the execution profile before resuming"
            )
        host = binary.parent / "codex-code-mode-host"
        if profile.code_mode_host_sha256 and file_hash(host) != profile.code_mode_host_sha256:
            raise ValueError("the pinned code-mode host changed")
        if profile.native_prompt_profile == "open-task-v1":
            with (
                binary.open("rb") as handle,
                mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as compiled,
            ):
                if compiled.find(b"## Open task method (task-tree/open-task-v1)") < 0:
                    raise ValueError(
                        "native open-task-v1 instructions are not compiled into this Codex binary; rebuild codex-cli before selecting this profile"
                    )
        self.codex = Codex(
            CodexConfig(
                codex_bin=profile.codex_bin,
                cwd=str(workspace),
                start_new_session=True,
                rpc_timeout_seconds=30,
                env={
                    "CODEX_TASK_TREE_WORKER": "1",
                    "CODEX_TASK_TREE_PROMPT_PROFILE": profile.native_prompt_profile,
                },
                config_overrides=(
                    "features.multi_agent=false",
                    "features.multi_agent_v2=false",
                    "features.goals=false",
                    "memories.generate_memories=false",
                    "memories.use_memories=false",
                    "skills.include_instructions=false",
                    'sandbox_mode="workspace-write"',
                    'approval_policy="never"',
                    "sandbox_workspace_write.writable_roots=[]",
                    "sandbox_workspace_write.exclude_tmpdir_env_var=true",
                    "sandbox_workspace_write.exclude_slash_tmp=true",
                ),
            )
        )
        process = self.codex._client._proc
        self._owner = {
            "pid": str(process.pid),
            "start": process_start(process.pid),
            "codex_bin": self.profile.codex_bin,
        }
        self.last_usage = None
        if profile.worker_model is None:
            models = self.codex.models().data
            available = next(
                (model for model in models if model.is_default), models[0] if models else None
            )
            if available is None:
                self.close()
                raise RuntimeError("no worker model is available for this Codex account")
            profile.worker_model = available.model

    def close(self):
        if getattr(self, "_action_judge", None):
            self._action_judge.close()
            self._action_judge = None
        owner = self._owner if self.codex._client._proc else None
        self.codex.close()
        if owner:
            group = int(owner["pid"])
            if process_start(group) not in ("", owner["start"]):
                return
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    break

    def execution_owner(self) -> dict[str, str]:
        # The SDK owns the stdio child; its PID is needed to distinguish an
        # orphaned executor from a persisted but no-longer-running thread.
        return dict(self._owner)

    def recover(self, work, report_type):
        owner = work.execution_owner
        self.ensure_stopped(owner)
        if not work.thread_id:
            return None
        response = self.codex._client.thread_read(work.thread_id, include_turns=True)
        turns = response.thread.turns
        matching = [turn for turn in turns if not work.turn_id or turn.id == work.turn_id]
        if not matching:
            return None
        turn = matching[-1]
        text = []
        for wrapped in turn.items:
            item = getattr(wrapped, "root", wrapped)
            if getattr(item, "type", None) == "agentMessage" and getattr(item, "text", None):
                text.append(item.text)
        if str(getattr(turn.status, "value", turn.status)) == "completed" and text:
            try:
                return parse_report(report_type, text[-1])
            except ValueError:
                return None
        return None

    @staticmethod
    def ensure_stopped(owner):
        if owner:
            try:
                os.killpg(int(owner["pid"]), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(
                    "previous app-server process group is still alive; stop/recover that owner before replay"
                )
        if not owner:
            raise RuntimeError(
                "old executor identity is unknown; confirm termination before recovery"
            )

    def __call__(self, context, report_type, on_started, timeout):
        return self.run_with_feedback(context, report_type, on_started, timeout)

    def run_with_feedback(
        self, context, report_type, on_started, timeout, validator=None, on_feedback=None
    ):
        action_protocol = context.get("protocol_version") == 3
        if action_protocol:
            from .action_protocol import VISIT_INSTRUCTIONS
            context = {**context, "phase_instructions": VISIT_INSTRUCTIONS}
        elif context["kind"] in ATTEMPT_KINDS:
            context = {**context, "phase_instructions": instructions_for_kind(context["kind"])}
        deadline = time.monotonic() + timeout
        self.codex._client.config.rpc_timeout_seconds = min(30, max(0.01, timeout))
        read_only = context["kind"] in READ_ONLY_KINDS
        developer_instructions = (
            VISIT_INSTRUCTIONS if action_protocol else ATTEMPT_INSTRUCTIONS
            if context["kind"] in ATTEMPT_KINDS else instructions_for_kind(context["kind"])
        )
        if context.get("continuation_thread_id"):
            thread = self.codex.thread_resume(
                context["continuation_thread_id"],
                cwd=str(self.workspace),
                sandbox=Sandbox.read_only if read_only else Sandbox.workspace_write,
                approval_mode=ApprovalMode.deny_all,
            )
        else:
            thread = self.codex.thread_start(
                cwd=str(self.workspace),
                model=self.profile.worker_model,
                sandbox=Sandbox.read_only if read_only else Sandbox.workspace_write,
                approval_mode=ApprovalMode.deny_all,
                ephemeral=False,
                developer_instructions=developer_instructions,
            )
        on_started(thread.id)
        return self._collect_report(
            thread, context, report_type, on_started, deadline, validator, on_feedback
        )

    def run_action_visit(
        self, context, report_type, on_started, timeout, submit, validator=None, on_feedback=None
    ):
        from .action_protocol import VISIT_INSTRUCTIONS, ExploreBrief

        context = {**context, "phase_instructions": VISIT_INSTRUCTIONS}
        if not getattr(self, "_action_tool_ready", False):
            with (
                Path(self.profile.codex_bin).open("rb") as handle,
                mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as compiled,
            ):
                if compiled.find(b"task_tree_decide") < 0:
                    raise ValueError("rebuild this fork: the native task_tree_decide tool is unavailable")
            self._action_tool_ready = True
        client = self.codex._client
        previous = client._approval_handler
        thread_id = None

        def handler(method, params):
            if method != "item/tool/call":
                return {"decision": "decline"}
            if not params or params.get("tool") != "task_tree_decide" or params.get("threadId") != thread_id:
                answer = {"ok": False, "error": "unbound decision tool call"}
            else:
                try:
                    brief = parse_report(ExploreBrief, json.dumps(params.get("arguments")))
                    answer = submit(brief)
                except Exception as error:
                    answer = {"ok": False, "error": str(error)}
            return {
                "success": answer.get("ok", False),
                "contentItems": [{"type": "inputText", "text": json.dumps(answer, ensure_ascii=False)}],
            }

        client._approval_handler = handler
        try:
            if context.get("continuation_thread_id"):
                thread = self.codex.thread_resume(
                    context["continuation_thread_id"],
                    cwd=str(self.workspace),
                    sandbox=Sandbox.read_only,
                    approval_mode=ApprovalMode.deny_all,
                )
                thread_id = thread.id
            else:
                started = client.thread_start({
                    "cwd": str(self.workspace), "model": self.profile.worker_model,
                    "sandbox": "read-only", "approvalPolicy": "never",
                    "developerInstructions": VISIT_INSTRUCTIONS,
                    "dynamicTools": [{
                        "type": "function", "name": "task_tree_decide",
                        "description": "Submit this visit's explored evidence once to independent Luna; returns execute, drill_down or finish and a bound decision_id. Never call during execution or decomposition.",
                        "inputSchema": strict_schema(ExploreBrief),
                    }],
                })
                thread_id = started.thread.id
            on_started(thread_id)
            return self._collect_report(
                Thread(client, thread_id), context, report_type, on_started,
                time.monotonic() + timeout, validator, on_feedback,
            )
        finally:
            client._approval_handler = previous

    def run_action_decision(self, context, on_started, timeout, on_owner, validator=None):
        from .action_protocol import ACTION_JUDGE_INSTRUCTIONS
        from .models import ActionVerdict

        # Tool callbacks run on the SDK reader thread; a second transport avoids deadlock.
        if not getattr(self, "_action_judge", None):
            self._action_judge = CodexAdapter(self.profile, self.workspace)
        on_owner(self._action_judge.execution_owner())
        return self._action_judge.run_boolean_decision(
            context, ActionVerdict, ACTION_JUDGE_INSTRUCTIONS, on_started, timeout, validator=validator
        )

    def run_decision(self, context, submit, on_started, timeout):
        from .decision_worker import DECISION_INSTRUCTIONS, DecisionSubmission, DecisionWorkerResult

        with (
            Path(self.profile.codex_bin).open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as compiled,
        ):
            if compiled.find(b"jev_decide") < 0:
                raise ValueError(
                    "this fork does not expose the native Jev tool; rebuild codex-cli before running task-tree decisions"
                )
        client = self.codex._client
        prior_handler = client._approval_handler
        thread_id = None

        def handler(method, params):
            if method != "item/tool/call":
                # Decision workers have no authority to approve native commands or writes.
                return {"decision": "decline"}
            if (
                not params
                or params.get("tool") != "jev_decide"
                or params.get("threadId") != thread_id
            ):
                return {
                    "success": False,
                    "contentItems": [
                        {"type": "inputText", "text": "Unbound decision tool call rejected"}
                    ],
                }
            try:
                answer = submit(params.get("arguments"))
            except Exception as error:
                answer = {"ok": False, "code": "TOOL_ERROR", "message": str(error)}
            return {
                "success": answer.get("ok", False),
                "contentItems": [
                    {"type": "inputText", "text": json.dumps(answer, ensure_ascii=False)}
                ],
            }

        client._approval_handler = handler
        try:
            started = client.thread_start(
                {
                    "cwd": str(self.workspace),
                    "model": self.profile.worker_model,
                    "sandbox": "read-only",
                    "approvalPolicy": "never",
                    "developerInstructions": DECISION_INSTRUCTIONS,
                    "dynamicTools": [
                        {
                            "type": "function",
                            "name": "jev_decide",
                            "description": "Submit a focused evidence brief for this bound predicate. Returns the real Jev decision or a structured input/provider error. A valid no is final.",
                            "inputSchema": strict_schema(DecisionSubmission),
                        }
                    ],
                }
            )
            thread_id = started.thread.id
            on_started(thread_id)
            return self._collect_report(
                Thread(client, thread_id),
                context,
                DecisionWorkerResult,
                on_started,
                time.monotonic() + timeout,
            )
        finally:
            client._approval_handler = prior_handler

    def run_boolean_decision(
        self, context, report_type, developer_instructions, on_started, timeout, validator=None
    ):
        thread = self.codex.thread_start(
            cwd=str(self.workspace),
            model=self.profile.decision_model,
            config={"model_reasoning_summary": "none"},
            sandbox=Sandbox.read_only,
            approval_mode=ApprovalMode.deny_all,
            ephemeral=False,
            developer_instructions=developer_instructions,
        )
        on_started(thread.id)
        return self._collect_report(
            thread,
            context,
            report_type,
            on_started,
            time.monotonic() + timeout,
            validator=validator,
            model=self.profile.decision_model,
            effort="low",
        )

    def _collect_report(
        self, thread, context, report_type, on_started, deadline, validator=None, on_feedback=None,
        *, model=None, effort=None,
    ):
        prompt = context
        for attempt in range(3):
            if model is None and effort is None:
                result = self._run_turn(
                    thread, prompt, report_type, context, on_started, deadline
                )
            else:
                result = self._run_turn(
                    thread, prompt, report_type, context, on_started, deadline,
                    model=model, effort=effort,
                )
            try:
                report = parse_report(report_type, result.final_response)
                for key in ("task_id", "round_id", "spec_revision", "input_digest"):
                    if (
                        key in context
                        and hasattr(report, key)
                        and getattr(report, key) != context[key]
                    ):
                        raise ValueError(f"identity mismatch: {key}; expected {context[key]!r}")
                if validator:
                    validator(report)
                return report
            except ValueError as error:
                diagnostics = {
                    "code": "INVALID_REPORT",
                    "message": str(error),
                    "attempt": attempt + 1,
                    "instruction": "Correct only the report contract using the original task and actual evidence. Preserve product work, adverse findings and acceptance. Do not rerun completed implementation. Return the required structured report.",
                }
                if on_feedback:
                    on_feedback(diagnostics, result.final_response)
                if attempt == 2:
                    raise WorkerReportError(str(error)) from error
                prompt = {"validation_error": diagnostics}

    def sandbox_policy(self, kind):
        network = self.profile.permission_profile == "workspace-write-network"
        if kind not in {"PREPARE", "EXECUTE", "RECONCILE"}:
            return SandboxPolicy(root=ReadOnlySandboxPolicy(type="readOnly", network_access=network))
        return SandboxPolicy(
            root=WorkspaceWriteSandboxPolicy(
                type="workspaceWrite",
                writable_roots=[str(self.workspace)],
                exclude_tmpdir_env_var=True,
                exclude_slash_tmp=True,
                network_access=network,
            )
        )

    def _run_turn(
        self, thread, prompt, report_type, identity, on_started, deadline, *, model=None, effort=None
    ):
        text = json.dumps(prompt, ensure_ascii=False)
        model = model or self.profile.worker_model
        effort = effort or self.profile.worker_effort
        if self.profile.permission_profile == "workspace-write-network":
            wire_input, tool_output = _to_wire_turn_input(text)
            result, subscription = self.codex._client._start_turn(
                thread.id,
                wire_input,
                params=TurnStartParams(
                    thread_id=thread.id,
                    input=wire_input,
                    tool_output=tool_output,
                    model=model,
                    effort=effort,
                    output_schema=strict_schema(report_type, identity),
                    sandbox_policy=self.sandbox_policy(identity.get("kind")),
                ),
                for_handle=True,
            )
            turn = TurnHandle(
                self.codex._client, thread.id, result.turn.id, _subscription=subscription
            )
        else:
            turn = thread.turn(
                text,
                model=model,
                effort=effort,
                output_schema=strict_schema(report_type, identity),
                sandbox=Sandbox.read_only
                if identity.get("kind") in READ_ONLY_KINDS | {"JUDGE"}
                else Sandbox.workspace_write,
            )
        on_started(thread.id, turn.id)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(turn.run)
        try:
            result = future.result(timeout=max(0, deadline - time.monotonic()))
        except concurrent.futures.TimeoutError:
            self.codex._client.config.rpc_timeout_seconds = 5
            turn.interrupt()
            try:
                future.result(timeout=20)
            except concurrent.futures.TimeoutError:
                self.close()
                raise UnresolvedWorkerExecution(
                    "worker interruption unconfirmed; workspace remains reserved"
                ) from None
            raise TimeoutError("worker timeout; interruption acknowledged") from None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if (
            str(getattr(result.status, "value", result.status)) != "completed"
            or not result.final_response
        ):
            raise RuntimeError("worker did not produce a completed structured report")
        self.codex._client.config.rpc_timeout_seconds = min(
            30, max(0.01, deadline - time.monotonic())
        )
        saved = thread.read(include_turns=True)
        for saved_turn in saved.thread.turns:
            if saved_turn.id != turn.id:
                continue
            for wrapped in saved_turn.items:
                item = getattr(wrapped, "root", wrapped)
                status = getattr(item, "status", None)
                if (
                    getattr(item, "type", None) == "commandExecution"
                    and str(getattr(status, "value", status)) == "inProgress"
                ):
                    raise RuntimeError(
                        "worker left a command running; stop/recover it before another round"
                    )
        self.last_usage = result.usage.model_dump(mode="json") if result.usage else None
        return result

    def report_contract(self, kind, report_type):
        return {
            "developer_instructions": ATTEMPT_INSTRUCTIONS
            if kind in ATTEMPT_KINDS
            else instructions_for_kind(kind),
            "phase_instructions": instructions_for_kind(kind)
            if kind in ATTEMPT_KINDS
            else None,
            "output_schema": strict_schema(report_type),
            "identity_fields_bound_by_controller": True,
            "worker_model": self.profile.worker_model,
            "worker_effort": self.profile.worker_effort,
            "native_prompt_profile": self.profile.native_prompt_profile,
            "sandbox": "read-only" if kind in READ_ONLY_KINDS else "workspace-write",
        }

    def command_exec(
        self, command: list[str], cwd: Path, timeout: float, extra_env: dict | None = None
    ) -> dict:
        if timeout <= 0:
            raise TimeoutError("no command time budget remains")
        self.codex._client.config.rpc_timeout_seconds = timeout + 5
        policy = self.sandbox_policy("EXECUTE")
        result = self.codex.command_exec(
            CommandExecParams(
                command=command,
                cwd=str(cwd),
                timeout_ms=max(1, int(timeout * 1000)),
                sandbox_policy=policy,
                env={
                    "PYTHONPATH": str(self.workspace),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    **(extra_env or {}),
                },
            )
        )
        return result.model_dump()


def process_start(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, timeout=5
    )
    return result.stdout.strip() if result.returncode == 0 else ""
