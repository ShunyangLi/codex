//! Tool ceiling for the source-checkout task-tree controller's private stdio server.

use codex_extension_api::ToolPolicy;
use codex_tools::ToolName;

pub(crate) fn policy() -> ToolPolicy {
    ToolPolicy {
        allowed_tools: Some(
            [
                "exec",
                "wait",
                "exec_command",
                "write_stdin",
                "apply_patch",
                "view_image",
                "update_plan",
                "web",
                "jev_decide",
            ]
            .into_iter()
            .map(ToolName::plain)
            .collect(),
        ),
        expose_additional_permissions: false,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_workers_cannot_delegate_or_bypass_workspace_tools() {
        let policy = policy();
        for name in ["exec", "wait", "exec_command", "apply_patch", "jev_decide"] {
            assert!(policy.allows(&ToolName::plain(name)));
        }
        for name in ["spawn_agent", "create_goal", "update_goal"] {
            assert!(!policy.allows(&ToolName::plain(name)));
        }
        assert!(!policy.allows(&ToolName::namespaced("mcp__cua_repl", "js")));
        assert!(!policy.expose_additional_permissions);
    }
}
