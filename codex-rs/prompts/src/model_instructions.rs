//! Returns literal model instructions from the captured model metadata.
//! Missing templates produce a warning; supplied empty templates remain valid input.

use crate::ResolvedModelMessages;
use codex_protocol::openai_models::ModelInfo;

const TASK_TREE_METHOD: &str = include_str!("task_tree_open_task.md");
const TASK_TREE_METHOD_MARKER: &str = "## Open task method (task-tree/open-task-v1)";

fn task_tree_profile_enabled(worker: Option<&str>, profile: Option<&str>) -> bool {
    worker == Some("1") && profile == Some("open-task-v1")
}

fn with_task_tree_method(base: String, enabled: bool) -> String {
    if !enabled || base.is_empty() || base.contains(TASK_TREE_METHOD_MARKER) {
        return base;
    }
    format!("{base}\n\n{TASK_TREE_METHOD}")
}

/// Renders base instructions from the caller's captured model, retaining missing-template warnings.
pub fn render_model_instructions(model_info: &ModelInfo) -> String {
    let template = ResolvedModelMessages::from_model(model_info).instructions_template();
    if template.is_none() {
        tracing::warn!(
            model = %model_info.slug,
            "Model has no instruction template; returning empty instructions."
        );
    }
    let worker = std::env::var("CODEX_TASK_TREE_WORKER").ok();
    let profile = std::env::var("CODEX_TASK_TREE_PROMPT_PROFILE").ok();
    with_task_tree_method(
        template.unwrap_or_default().to_owned(),
        task_tree_profile_enabled(worker.as_deref(), profile.as_deref()),
    )
}

#[cfg(test)]
#[path = "model_instructions_tests.rs"]
mod tests;
