use color_eyre::eyre::{Context, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Pluggable STT provider wrapping whisper-rs with Vulkan GPU.
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

    /// Batch transcription: transcribe audio samples (f32, mono, 16kHz) to text.
    /// Returns the full transcript as a single string.
    pub fn transcribe(&self, audio: &[f32], language: &str, initial_prompt: &str) -> Result<String> {
        let mut state = self
            .ctx
            .create_state()
            .with_context(|| "failed to create whisper state")?;

        let mut wparams = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        if !language.is_empty() {
            wparams.set_language(Some(language));
        }
        if !initial_prompt.is_empty() {
            wparams.set_initial_prompt(initial_prompt);
        }
        wparams.set_print_progress(false);
        wparams.set_print_timestamps(false);
        wparams.set_print_special(false);

        state
            .full(wparams, audio)
            .with_context(|| "whisper transcription failed")?;

        Ok(assemble_transcript(&state))
    }

    /// Streaming transcription with per-segment callback.
    /// `on_segment` fires for each new segment as it's decoded.
    /// Returns the full assembled transcript.
    /// Set `abort_flag` to true to cancel mid-transcription.
    pub fn transcribe_streaming<F>(
        &self,
        audio: &[f32],
        language: &str,
        initial_prompt: &str,
        on_segment: F,
        abort_flag: Arc<AtomicBool>,
    ) -> Result<String>
    where
        F: FnMut(whisper_rs::SegmentCallbackData) + 'static,
    {
        let mut state = self
            .ctx
            .create_state()
            .with_context(|| "failed to create whisper state")?;

        let mut wparams = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        if !language.is_empty() {
            wparams.set_language(Some(language));
        }
        if !initial_prompt.is_empty() {
            wparams.set_initial_prompt(initial_prompt);
        }
        wparams.set_print_progress(false);
        wparams.set_print_timestamps(false);
        wparams.set_print_special(false);
        wparams.set_single_segment(true);
        wparams.set_split_on_word(true);

        // Safe segment callback — fires for each new segment with text included
        wparams.set_segment_callback_safe(on_segment);

        // Abort flag — checked each decoder step
        let flag = abort_flag.clone();
        wparams.set_abort_callback_safe(move || flag.load(Ordering::Relaxed));

        state
            .full(wparams, audio)
            .with_context(|| "whisper streaming transcription failed")?;

        Ok(assemble_transcript(&state))
    }

    pub fn model_path(&self) -> &PathBuf {
        &self.model_path
    }
}

/// Extract full transcript text from a completed whisper state.
fn assemble_transcript(state: &whisper_rs::WhisperState) -> String {
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
    text.trim().to_string()
}

/// Thread-safe wrapper for use across async tasks.
pub type SharedWhisperProvider = std::sync::Mutex<WhisperProvider>;
