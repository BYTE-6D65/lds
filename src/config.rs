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

    /// Transcription mode: "batch" (record-stop-transcribe) or "streaming"
    #[serde(default = "default_mode")]
    pub mode: String,

    /// Streaming: VAD speech probability threshold (0.0 - 1.0)
    #[serde(default = "default_vad_threshold")]
    pub vad_threshold: f32,

    /// Streaming: minimum silence duration (ms) to trigger segment end
    #[serde(default = "default_vad_min_silence_ms")]
    pub vad_min_silence_ms: u32,

    /// Streaming: how often to process audio chunks (ms)
    #[serde(default = "default_chunk_interval_ms")]
    pub chunk_interval_ms: u64,

    /// Streaming: send partial_transcript IPC events
    #[serde(default = "default_true")]
    pub partial_results: bool,

    /// Streaming: minimum audio (ms) before first transcription pass
    #[serde(default = "default_min_audio_ms")]
    pub min_audio_ms: u64,
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

fn default_mode() -> String {
    "batch".into()
}

fn default_vad_threshold() -> f32 {
    0.5
}

fn default_vad_min_silence_ms() -> u32 {
    500
}

fn default_chunk_interval_ms() -> u64 {
    2000
}

fn default_min_audio_ms() -> u64 {
    1500
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: String::new(),
            socket: default_socket(),
            device: String::new(),
            auto_type: true,
            log_transcript: false,
            language: String::new(),
            initial_prompt: String::new(),
            mode: default_mode(),
            vad_threshold: default_vad_threshold(),
            vad_min_silence_ms: default_vad_min_silence_ms(),
            chunk_interval_ms: default_chunk_interval_ms(),
            partial_results: true,
            min_audio_ms: default_min_audio_ms(),
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
            .unwrap_or_else(|_| {
                PathBuf::from(format!(
                    "/home/{}/.config",
                    std::env::var("USER").unwrap_or_else(|_| "byte".into())
                ))
            });
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

    pub fn is_streaming(&self) -> bool {
        self.mode == "streaming"
    }
}
