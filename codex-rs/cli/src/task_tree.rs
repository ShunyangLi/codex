use clap::Args;
use std::ffi::OsString;
use std::path::PathBuf;

#[derive(Debug, Args)]
pub struct TaskTreeCommand {
    /// Arguments forwarded to the Python task-tree controller.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<OsString>,
}

fn controller_paths() -> (PathBuf, PathBuf) {
    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    (
        checkout.join("experimental/task-tree/src"),
        checkout.join("sdk/python/src"),
    )
}

pub async fn run(command: TaskTreeCommand) -> anyhow::Result<()> {
    let (controller, sdk) = controller_paths();
    anyhow::ensure!(
        controller.is_dir() && sdk.is_dir(),
        "task-tree requires this fork's source checkout (Python controller and SDK)"
    );
    let python = std::env::var_os("CODEX_TASK_TREE_PYTHON").unwrap_or_else(|| {
        let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.venv/bin/python");
        if local.is_file() {
            local.into_os_string()
        } else {
            OsString::from("python3")
        }
    });
    let mut paths = vec![controller, sdk];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    let exit = tokio::process::Command::new(python)
        .args(["-m", "codex_task_tree"])
        .args(command.args)
        .env("PYTHONPATH", std::env::join_paths(paths)?)
        .env("CODEX_TASK_TREE_CODEX_BIN", std::env::current_exe()?)
        .status()
        .await?;
    anyhow::ensure!(exit.success(), "task-tree controller exited with {exit}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        command: TaskTreeCommand,
    }

    #[test]
    fn forwards_controller_flags_without_reinterpreting_them() {
        let parsed = TestCli::parse_from([
            "test",
            "start",
            "build storage",
            "--workspace",
            "/tmp/project",
        ]);
        assert_eq!(
            parsed.command.args,
            ["start", "build storage", "--workspace", "/tmp/project"].map(OsString::from)
        );
        let (controller, sdk) = controller_paths();
        assert!(controller.join("codex_task_tree/cli.py").is_file());
        assert!(sdk.join("openai_codex/api.py").is_file());
    }
}
