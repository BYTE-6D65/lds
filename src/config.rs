use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Path to whisper ggml model file
    pub model: String,

    /// Unix domain socket path for IPC
    #[serde(default = "default_socket")]
    pub socket: String,

    /// Audio capture device name (empty = auto-detect PipeWire)
    #[serde(default)]
    pub device: String,

    /// Write transcript to clipboard after transcription
    #[serde(default = "default_true")]
    pub clipboard: bool,

    /// Auto-type transcript after transcription (best-effort)
    #[serde(default = "default_true")]
    pub auto_type: bool,

    /// Log transcript text (set false for privacy)
    #[serde(default)]
    pub log_transcript: bool,

    /// Language hint for whisper (empty = auto-detect)
    #[serde(default)]
    pub language: String,

    /// Initial prompt for whisper (improves accuracy for domain-specific terms)
    #[serde(default)]
    pub initial_prompt: String,
}

fn default_socket() -> String {
    format!(
        "/run/user/{}/ldsd.sock",
        std::env::var("UID").unwrap_or_else(|_| "1000".into())
    )
}

fn default_true() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: String::new(),
            socket: default_socket(),
            device: String::new(),
            clipboard: true,
            auto_type: true,
            log_transcript: false,
            language: String::new(),
            initial_prompt: String::new(),
        }
    }
}

impl Config {
    /// Load config from file, falling back to defaults for missing fields.
    pub fn load(path: &Path) -> color_eyre::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&text)?;
        Ok(config)
    }

    /// Get the default config path: $XDG_CONFIG_HOME/lds/config.toml
    pub fn default_path() -> PathBuf {
        let config_dir = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(format!(
                "/home/{}/.config",
                std::env::var("USER").unwrap_or_else(|_| "byte".into())
            )));
        config_dir.join("lds").join("config.toml")
    }

    /// Save a template config to the given path.
    pub fn save_template(path: &Path) -> color_eyre::Result<()> {
        let config = Self {
            model: "/path/to/ggml-model.bin".into(),
            ..Self::default()
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(&config)?;
        std::fs::write(path, text)?;
        Ok(())
    }
}
