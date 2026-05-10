use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(version, about = "Liam's Dictation Service")]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand, Clone)]
pub enum Command {
    /// Run as a background daemon with IPC
    Daemon {
        /// Path to whisper ggml model file
        #[arg(short, long)]
        model: String,

        /// Unix domain socket path for IPC
        #[arg(short, long, default_value = "/run/user/1000/ldsd.sock")]
        socket: String,

        /// Audio capture device name (empty = default)
        #[arg(long, default_value = "")]
        device: String,
    },

    #[cfg(feature = "overlay")]
    WaybarStatus {
        #[clap(flatten)]
        connection_opts: ConnectionOpts,
    },

    #[cfg(feature = "overlay")]
    Overlay {
        #[clap(flatten)]
        connection_opts: ConnectionOpts,

        /// An optional stylesheet for the overlay
        #[arg(short, long, default_value = None)]
        style: Option<std::path::PathBuf>,

        /// Hotkey to activate voice input
        #[arg(long, default_value = "KEY_RIGHTCTRL")]
        hotkey: String,
    },
}

#[derive(Debug, Args, Clone)]
pub struct ConnectionOpts {
    #[clap(short, long, default_value = "localhost:7007")]
    pub address: String,
}
