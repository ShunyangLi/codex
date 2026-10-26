# Task Tree Harness

Experimental, serial controller for the [agreed design](../../../docs/task_tree_harness_architecture_v1.md). Python owns the tree and scheduling; this Codex fork owns model/tool execution; a System One decision model scores explicit decisions. New CLI runs default to Alibaba Cloud `decision-model-preview`; recorded local Ollama profiles remain supported.

## Run from this checkout

The current workstation has the Python package and local SDK installed in `../../../.venv` and the fork built in `../../codex-rs/target/debug`. Use the fork explicitly:

```sh
./codex/codex-rs/target/debug/codex task-tree doctor --probe-decision
./codex/codex-rs/target/debug/codex task-tree start \
  --workspace /absolute/path/to/project --model gpt-6-sol --effort xhigh \
  'Describe the task and desired deliverable'
```

`start` first runs a dedicated **PREPARE** agent. It inspects the task's sources, writes inspection code/results and knowledge under `background/`, and returns a structured report. The controller snapshots its files and saves a shared preparation record outside the child-task tree. A fresh **read-only SPECIFY** worker uses that background to propose the root contract, then stops for actual user confirmation. Review the contract and questions; edit an unapproved contract with `spec RUN --file task-spec.json` if necessary. Then:

```sh
./codex/codex-rs/target/debug/codex task-tree approve RUN \
  --contract DISPLAYED_HASH --confirmation 'User approved this exact specification'
./codex/codex-rs/target/debug/codex task-tree run RUN
./codex/codex-rs/target/debug/codex task-tree status RUN
./codex/codex-rs/target/debug/codex task-tree export RUN
```

Status can be read while another controller holds the run lock. `resume RUN --reason 'What changed'` resumes a blocker without clearing budgets or no-progress history. A current human criterion can be accepted with `accept RUN --task-id TASK --criterion ID --confirmation 'Actual review evidence'`; it cannot override command/evidence checks. `cancel`/`release` require no unresolved execution.

Default controller storage is `$CODEX_HOME/task-tree` (or `~/.codex/task-tree`), outside the target workspace. A different root goes **before** the subcommand: `task-tree --home /path/controller status RUN`. Use the same controller root for overlapping workspaces so its shared lease registry can enforce ownership. `CODEX_TASK_TREE_PYTHON` selects another Python interpreter. The Rust entry loads this checkout's controller/SDK and passes its own binary path.

## Shared preparation and decision backend

For task-authorized remote sources such as `ssh hk`, start with `--allow-worker-network`.
This records `workspace-write-network` in the execution profile and enables network access
for worker turns and formal command checks. Exploration, planning and decision roles retain
read-only filesystems; delivery roles retain workspace-only writes. The default stays offline.
Remote inputs should have a declared local source manifest and reproducible source-change
checks; a local descriptor alone does not establish remote data immutability.

```text
User task → PREPARE → shared background snapshot → SPECIFY → user approval
    → read-only EXPLORE → fresh Codex decision worker → Jev SHOULD_SPLIT: should this task be split further?
            yes → DECOMPOSE → per-child NECESSARY + structural checks → child EXPLORE
            no → deterministic input/contract checks → scoped leaf EXECUTE
                missing essentials → fresh EXPLORE or explicit blocker
    → fresh parent RECONCILE → checks and further decisions
```

Preparation is a general role: investigate the inputs, meaning, observed characteristics/behavior, relationships/interfaces/dependencies, existing results, constraints and limitations needed for the actual goal. Save evidence-backed shared knowledge, not merely names/paths; clearly distinguish unknown or inaccessible essential background. Outcome decomposition must reuse that knowledge and contribute to parent delivery, rather than repeat foundational validation.

Preparation is run-level knowledge, not a required child of an analysis/programming task. It does not publish a NodeResult or a completion claim and cannot create children. Each round uses a fresh agent; the preparation index, verified facts, limitations and snapshot references are included in subsequent worker and decision contexts. Detailed results remain in files rather than being silently truncated into the index. Inspection can be partial for large/inaccessible sources and must say so. Counts distinguish preparation rounds, task rounds and actual child nodes; all agent rounds consume the persisted global budget.

`background/index.json` must contain a nonempty, source-derived `investigation_inventory`. Each entry identifies a subject, source references, possible questions and meaning/limits. Analysis preparation maps relevant fields to constructs and dimensions; programming preparation maps required behaviors, interfaces and integration risks. Exploration inspects facts, task workload/complexity, unknowns and boundaries; it does not propose child blueprints before Jev authorizes splitting. New exploration reports keep the historical `split_opportunities` field empty. Discoveries inform the split decision and subsequent authorized planning. When Jev declines splitting, the controller checks explicit deliverables, declared essential-input blockers and actual input existence before authorizing execution. Exploration does not solve the whole task or publish a product result. There is no field, child-count or depth quota, and one focused outcome can need multiple queries.

New workers default to `gpt-6-sol` with `xhigh` effort, explicitly set on both thread and turn. Unrelated skill discovery and built-in memories are disabled for these workers; the bundled Jev input skill is explicitly supplied where needed. They retain Codex's native base instructions/tools; task-specific method and role instructions are supplied by this package, and external context is assembled afresh. The selected model must be supported by the current authentication; a rejected model fails explicitly without fallback. The initial development build reports `0.0.0` and returned HTTP 400 for `gpt-6.1-sol` and `gpt-6-sol`. Follow-up diagnostics isolated the version header: the installed `0.162.1` binary successfully called `gpt-6-sol / xhigh`, and this same fork succeeded when only the version header changed from `0.0.0` to `0.162.1` in a temporary diagnostic provider. Thus the original error does not establish lack of account access. The workspace now uses development build metadata `0.162.1+task-tree.1`. After rebuilding, the fork's built-in provider successfully called `gpt-6-sol / xhigh` without header overrides; the native preflight and cloud decision probe are retained under `.local/task-tree-validation/versioned_build_20261022_182302/` in the parent workspace. Only local package versions changed; external dependencies were unchanged. These one-line probes do not validate the full task workflow. Existing stored profiles are not silently rewritten.

The cloud endpoint is `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/systemone`, with the dedicated System One `state`/`questions` protocol. It is not Chat Completions. Authentication reads `QWEN_KEY` from the current process or, by default, this checkout's `codex/.env`; `--decision-env-file` selects a different file. Only the named variable is parsed, without executing the file. Keys are excluded from profiles, requests and logs; credentials are not forwarded on redirects. See the [official Token Plan decision-model integration](https://help.aliyun.com/zh/model-studio/token-plan-decision-model).

The cloud CLI byte guard defaults to 262,144 bytes; the provider's documented 65,536-token limit remains authoritative. Byte and token limits are different, and oversize requests still block explicitly. The threshold stays 0.5; scores from different models are not assumed calibrated identically. `doctor --probe-decision` makes a small real call and retains its response. A normal `doctor` checks configuration and credential availability without claiming that the model was contacted.

For an existing **unapproved, idle** run, prepare background and regenerate its specification while preserving old proposals and budgets:

```sh
./codex/codex-rs/target/debug/codex task-tree --home /path/controller prepare RUN \
  --decision-url https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --decision-model decision-model-preview --decision-key-env QWEN_KEY \
  --decision-env-file /absolute/path/to/.env --max-decision-input-bytes 262144
```

`prepare` preserves the recorded backend unless new flags are supplied. Approved or active runs cannot use this shortcut to change their contract. Newly generated root specifications require confirmation. To start explicitly with local Ollama, use `--decision-url http://127.0.0.1:11434 --decision-model nimble:9b-q4_K_M`; the legacy `--jev-url`/`--jev-model` spellings are aliases.

## Build / install

Python 3.10+, Pydantic 2.12+, packaging 26.2+, the local Python SDK, Rust 1.99.1, and a compatible `codex-code-mode-host` are required. The worker uses existing Codex authentication. The decision backend uses the explicitly recorded endpoint/model and credentials independently of the worker.

```sh
# From harness_analysis, using the existing virtualenv:
.venv/bin/uv pip install --python .venv/bin/python --no-deps \
  -e ./codex/sdk/python -e ./codex/experimental/task-tree

# From codex/codex-rs:
cargo build -p codex-cli --bin codex --locked
# When the upstream V8 artifacts are available:
cargo build -p codex-code-mode-host --bin codex-code-mode-host --locked
```

`--no-deps` intentionally avoids the SDK's packaged CLI: this controller pins the fork. Install Pydantic/packaging separately if not already present.

On this workstation, the code-mode host source build encounters an upstream V8 artifact 404. `target/debug/codex-code-mode-host` therefore links to the already installed official Codex 0.163.0 host. Actual worker file writes verified compatibility. The controller's native terminal/file tool ceiling is enforced in this fork's app-server when its private process has `CODEX_TASK_TREE_WORKER=1`; other Codex sessions are unchanged. Third-party MCP/UI tools are excluded from these workers because their writes bypass workspace sandboxing.

## Execution and persistence

Current lifecycle/input policy: [decision input v3](../../../docs/task_tree_harness_decision_input_v3.md), supplementing [task-tree v2](../../../docs/task_tree_harness_protocol_v2.md). The compiled fork's `codex task-tree` Rust command loads this Python package and starts this fork's app-server once per run/resume invocation, creating a fresh thread for each worker round. Ordinary `codex` still uses its original default workflow.

Lifecycle decisions now create their own fresh native Codex thread. Its bundled Jev skill calls a bound `jev_decide` dynamic tool, receives the actual score or structured error, and repairs input within that conversation (at most three submissions). Python remains the credential-bearing transport and tree authority. The worker selects relevant background/schema/examples and question-only peers; other tasks' result summaries are excluded. Valid negative answers are durable and never resampled. `doctor` remains a low-level connectivity probe.

`possible_duplicate_task_ids` lets a planner identify plausible existing work, including queued tasks; a separate DUPLICATE judgment links equivalent work as a dependency. That node adapts and verifies the existing delivery rather than repeating it. Full-result REUSE additionally requires conservative goal/scope/acceptance mappings in code. Related work can still be read as a dependency. Ordinary worker reports support bounded same-thread contract correction with archived diagnostics.

New runs use schema version 2. Version 1 remains readable/exportable/cancellable; execution and mutation commands reject it rather than silently changing its semantics. Start a fresh run to test the new policy. `ADVANCE` is retained only for reading historical records.

```text
PREPARE → SPECIFY → user approval → EXPLORE (read-only)
  → fresh decision Codex + Jev SHOULD_SPLIT
          yes → DECOMPOSE → per-child NECESSARY + structural checks → child EXPLORE
          no → input/contract checks → EXECUTE or further EXPLORE/blocker
child delivery → RECONCILE → actual checks → Jev COMPLETE
incomplete execution/integration → EXPLORE → Jev routing again
```

Routing uses ONE `SHOULD_SPLIT` decision. Focused ownership, task workload/complexity and expected information gain are evidence for that single question. `INFORMATION_GAIN`, `EXPLORATION_ADEQUATE` and `EXECUTABLE` remain readable in old records but are no longer requested. A negative split decision permits execution only when explicit deliverables, explorer `execution_blockers`, actual declared inputs and required child deliveries pass controller checks. Shared data, one script or fast execution do not prove one focused outcome; many queries can support one question.

Prompt version 14 assembles fixed common guidance, per-kind criteria and one final yes/no proposition in `questions.decision.instructions`; actual context remains in `state.context`. All guidance/questions and the version enter cache and authorization identity. The pre-execution whole-plan PLAN_ACCEPTABLE veto is retired; candidates are judged individually, with rejected prerequisites deferring dependents and replacements remaining atomic. `evaluate_node` unifies lifecycle routing without merging COMPLETE and SHOULD_SPLIT: failed acceptance returns to investigation, which may authorize leaf repair or a new split. COMPLETE runs only after actual delivery/integration and hard checks. EVIDENCE_SUPPORTED assesses one criterion. Current results, direct-child actual findings and current scoped proofs replace repeated descendant passing logs; failures, goals, acceptance, coverage and gaps stay visible.

The bundled [Jev input skill](src/codex_task_tree/skills/jev-decision-input/SKILL.md) is explicitly supplied to workers; unrelated skill discovery remains disabled. Workers prepare evidence briefs and can run its read-only, network-free preview helper. Only the controller submits formal API calls and updates the tree. Code handles arithmetic, dates, counts, dependencies and hard check states; Jev receives direct facts for a single semantic proposition. A score never grants permissions or overrides actual failures/human acceptance.

Evidence packet version 3 can replace explicitly inspected background/history sections with an independent review brief, while preserving structured adverse facts. Large artifacts require current criterion-scoped evidence or independent observations before full contents can be deferred. Child root-clause mappings are resolved in code, with global root goals/constraints retained. Repeated specs, check fingerprint maps and historical implementation proposals stay archived. Current frozen check code can use exact-byte independent CHECK_CHANGE evidence and an actual passing check. Context encoding version 3 removes only insignificant JSON whitespace and reversibly aliases opaque references and controller/workspace prefixes. Default working budget is 48,000 wire bytes (`--decision-context-bytes`); `--max-decision-input-bytes` retains the hard byte ceiling. Neither is an exact provider token count. An oversized request parks only its owner as NEEDS_CONTEXT, saves a DecisionJob and queues bounded read-only CONTEXT_REVIEW work; independent branches continue. Saved products are not re-executed just to replay a decision. Finite ready batches prevent fresh shallow retries from starving already-ready deeper work.

Routing read sets use specific files or bounded directories from EXPLORE. Real changes in that read set expire authorization; unrelated sibling outputs outside them do not. Workers must retain real dependencies and explain any directory-wide dependency instead of using `.` merely for a listing. EXPLORE's explicit `required_input_paths` lists inputs that must exist before execution; observed missing outputs are not automatically essential inputs. Original local input references are always checked and fingerprinted. Legacy reports without the field retain their previous existence checks.

An adopted evidence check with a command runs that frozen command before EVIDENCE_SUPPORTED. A wrong exit code/stdout fails, and execution errors remain errors; neither reaches the model. On command success, its actual exit code/stdout/stderr accompany the criterion and materials. Such records use `command+model`; older model-only records for checks with commands cannot certify completion. Evidence checks without commands keep their previous behavior.

Preparation includes verified Markdown/text knowledge documents inline as well as its index/facts/inventory. Profiles, scripts and binary contents remain explicitly referenced, not assumed read. The latest committed decomposition and all actual judgments are supplied to later workers/decisions, including declined or deferred candidates and necessary unassigned duties. Historical failures are retained, not resampled or overwritten.

EXPLORE, DECOMPOSE and REVIEW have separate output schemas and native read-only sandboxes. Historical SCOPE_REVIEW rounds remain readable and resumable for existing runs. Exploration records source observations and full responsibilities without implementing the task. Decomposition proposes the whole scope, never just leftovers. The controller verifies routing authorization, task/input versions and read-only workspace fingerprints. A no-split judgment alone never authorizes execution; transport errors cannot authorize either route. Reviewers reject erroneous checks without modifying them; an authorized worker repairs them.


1. Claim one persistent round/job and wave cursor in SQLite, then execute.
2. Start a fresh persistent Codex thread; never resume its conversation as the next worker's memory. Pass explicit context, background, actual child deliveries, coverage and frozen checks.
3. Save exploration and obtain Jev routing before any decomposition or execution. Save each role-specific report. For decomposition, validate declared assignments/dependencies and judge each candidate's necessity; admit safe independent positives without a whole-plan semantic veto. Preserve unassigned necessary duties as gaps. For execution/integration, verify the current effective acceptance set. `parent_capability_ids` assigns integration to the future parent worker; candidate capability IDs assign delegated work.
4. Commit the role-specific transition atomically. Only execution/integration and subsequent evidence updates publish product results; exploration and planning never issue completion certificates. An ended round is not a completed node.
5. Finish the current BFS wave, then reconcile the deepest ready parent. Missing behavior can produce another plan under the same parent. Reconciliation does not increase depth.
6. Require current checks, resolved coverage/gaps/children and one Jev COMPLETE judgment before issuing a completion certificate.

`TaskRound`/`VerificationJob` phases are authoritative. `WorkSchedule` is rebuilt from them; it is not a second persisted queue. The state snapshot uses revision compare-and-swap; referenced immutable files are written before their references are committed. Rejected candidates, failed rounds, raw Jev attempts and previous plans/results remain available.

Formal assertion bundles live outside the workspace. New/changed checks require a different worker's review and CHECK_CHANGE judgment. `file_paths` contains assertion scripts, helpers, fixtures and assertion-affecting configuration; it must not freeze the mutable product being tested. Commands must explicitly execute their frozen entrypoint or use `CODEX_CHECK_BUNDLE`; implicit discovery of mutable workspace tests is rejected when a frozen bundle is declared. The check environment puts frozen Python helpers before workspace imports and exposes the current project as `CODEX_CHECK_WORKSPACE`. Reviewers still need to verify that all assertion dependencies are declared and actually resolve to their frozen copies.

File changes first invalidate proof; a pure VerificationJob can refresh proof without inventing a new result or rerunning a child worker. Actual failure produces repair work. Verification errors remain distinct from implementation failure; after the blocker is resolved, `resume` arranges inspection of the current erroneous check definitions before parent reconciliation. It does not assume that the product needs changing.

Each declared file deliverable has a content-addressed immutable snapshot. A later metadata/review round retains earlier current artifact references; replacing a file preserves the older snapshot and publishes a new version. Output files also enter verification fingerprints, including when they were omitted from the original check's read set. Context contains concrete paths and snapshot references for inspecting actual child products.

Jev receives consistent aliases for opaque IDs, hashes and the controller directory prefix, reducing token use without dropping semantic content. Every decision retains its full original context, the exact reference map, the actual wire request and raw response. Cache identity binds the full context, so two different hashes cannot accidentally share a decision even if their symbolic packets look alike. Input that still exceeds the byte/token limit blocks explicitly; this prototype does not partition arbitrarily large trees automatically.

Recovery never assumes an unanswered call did nothing. It checks the saved executor process group and persisted thread/turn before retrying. Unknown or still-live execution reserves the workspace and blocks replacement. External side effects still require inspection; exactly-once execution is not promised.

## Tests

```sh
PYTHONPATH=codex/experimental/task-tree/src:codex/sdk/python/src \
  .venv/bin/python -m pytest codex/experimental/task-tree/tests -q -o addopts=''

# Opt-in live fixtures, from codex; each uses a new, empty directory:
../.venv/bin/python experimental/task-tree/tests/real_smoke.py \
  --directory /absolute/path/to/new/fixture --codex-bin codex-rs/target/debug/codex
../.venv/bin/python experimental/task-tree/tests/real_smoke.py --scenario storage \
  --inject-parent-fault \
  --directory /absolute/path/to/another/new/fixture --codex-bin codex-rs/target/debug/codex
../.venv/bin/python experimental/task-tree/tests/real_jev_cases.py \
  --directory /absolute/path/to/jev-probes
```

Live fixtures explicitly preapprove a fixed development test contract; the product CLI does not autoapprove user specifications. Without `--inject-parent-fault`, the storage fixture leaves routing to the actual decision model. With that flag, the controlled integration fixture preapproves **only the initial root split**, recording it as `fixture-preapproved-root-split`, never as a Jev result. Plan acceptance, child routing, execution and completion still use the real backend. It deliberately breaks the composite after the first parent report: actual checks must reject it, and a new worker must repair it. This isolates integration from autonomous routing, which is measured separately through the native CLI. Offline tests also exercise appended repair children and a second reconciliation under the same parent. Validation results and test scope are recorded in [the implementation record](../../../docs/task_tree_harness_implementation_v1.md).

This is a local macOS/Linux prototype. No concurrency, TUI, cross-project retrieval, distributed execution or benchmark-quality claim. Context that exceeds the recorded Jev input limit blocks visibly; it is not silently truncated or reported as a negative decision. Workers' decomposition and semantic judgments can still be wrong, so deterministic invariants and real checks remain mandatory.
