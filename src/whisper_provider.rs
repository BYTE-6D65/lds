use color_eyre::eyre::{Context, Result};
use std::path::PathBuf;
use std::sync::Mutex;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

/// Pluggable STT provider. Currently wraps whisper-rs with Vulkan GPU.
pub struct WhisperProvider {
    ctx: WhisperContext,
    model_path: PathBuf,
}

impl WhisperProvider {
    /// Load a whisper model from disk with GPU acceleration.
    pub fn new(model_path: &str) -> Result<Self> {
        let mut params = WhisperContextParameters::default();
        params.use_gpu = true;

        let ctx = WhisperContext::new_with_params(model_path, params)
            .with_context(|| format!("failed to load whisper model from {}", model_path))?;

        Ok(Self {
            ctx,
            model_path: PathBuf::from(model_path),
        })
    }

    /// Transcribe audio samples (f32, mono, 16kHz) to text.
    /// Returns the full transcript as a single string.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut state = self
            .ctx
            .create_state()
            .with_context(|| "failed to create whisper state")?;

        let mut wparams = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        wparams.set_language(Some("en"));
        wparams.set_print_progress(false);
        wparams.set_print_timestamps(false);
        wparams.set_print_special(false);

        state
            .full(wparams, audio)
            .with_context(|| "whisper transcription failed")?;

        let n_segments = state.full_n_segments();
        let mut text = String::new();
        for i in 0..n_segments {
            if let Some(seg) = state.get_segment(i) {
                if let Ok(s) = seg.to_str() {
                    text.push_str(s);
                    text.push(' ');
                }
            }
        }

        Ok(text.trim().to_string())
    }

    pub fn model_path(&self) -> &PathBuf {
        &self.model_path
    }
}

/// Thread-safe wrapper for use across async tasks.
pub type SharedWhisperProvider = Mutex<WhisperProvider>;
