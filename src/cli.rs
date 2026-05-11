use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about = "Linguistic Dispatch System")]
pub struct Cli {
    /// Config file path (default: $XDG_CONFIG_HOME/lds/config.toml)
    #[arg(short, long, global = true)]
    pub config: Option<String>,

    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand, Clone)]
pub enum Command {
    /// Run as a background daemon with IPC
    Daemon {
        /// Path to whisper ggml model file (overrides config)
        #[arg(short, long)]
        model: Option<String>,

        /// Unix domain socket path for IPC (overrides config)
        #[arg(short, long)]
        socket: Option<String>,

        /// Audio capture device name (empty = auto-detect PipeWire)
        #[arg(long)]
        device: Option<String>,
    },

    /// Generate a template config file
    InitConfig {
        /// Output path (default: $XDG_CONFIG_HOME/lds/config.toml)
        #[arg(short, long)]
        output: Option<String>,
    },
}
