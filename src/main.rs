use clap::Parser;
use color_eyre::eyre::Result;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

mod cli;
mod keyboard;
mod whisper_provider;

#[cfg(feature = "overlay")]
mod app;

#[cfg(feature = "overlay")]
mod hotkeys;

#[cfg(feature = "overlay")]
mod waybar;

pub fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("Setting up tokio runtime needs to succeed."))
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = cli::Cli::parse();

    match args.command {
        #[cfg(feature = "overlay")]
        cli::Command::WaybarStatus { connection_opts } => {
            runtime()
                .block_on(async move { waybar::main_waybar_status(&connection_opts).await })?;
        }
        #[cfg(feature = "overlay")]
        command @ cli::Command::Overlay { .. } => {
            app::launch_app(command)?;
        }
        cli::Command::Daemon { model, socket, .. } => {
            println!("LDS daemon starting...");
            println!("Loading model: {}", model);
            let _provider = whisper_provider::WhisperProvider::new(&model)?;
            println!("Model loaded with Vulkan GPU.");
            println!("IPC socket: {}", socket);
            println!("LDS daemon ready. (IPC + audio capture not yet wired)");
            std::thread::park();
        }
    }

    Ok(())
}
