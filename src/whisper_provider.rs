use color_eyre::eyre::{Context, Result};
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Common whisper hallucination patterns — discard if output matches.
const HALLUCINATIONS: &[&str] = &[
    "thank you",
    "thank you.",
    "thanks for watching",
    "thanks for watching.",
    "thank you for watching",
    "thank you for watching.",
    "subscribe",
    "subscribe.",
    "please subscribe",
    "like and subscribe",
    "thank you for your attention",
];

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

    /// Transcribe audio samples (f32, mono, 16kHz) to text.
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
        wparams.set_split_on_word(true);
        // Suppress hallucination — whisper sometimes loops on silence
        wparams.set_suppress_blank(true);

        state
            .full(wparams, audio)
            .with_context(|| "whisper transcription failed")?;

        let raw = assemble_transcript(&state);

        // Filter hallucinations
        Ok(filter_hallucinations(&raw))
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

/// Filter out known whisper hallucination patterns.
fn filter_hallucinations(text: &str) -> String {
    let trimmed = text.trim().to_lowercase();

    // If the ENTIRE output is a hallucination, discard it
    if HALLUCINATIONS.iter().any(|h| trimmed == *h) {
        return String::new();
    }

    // If it's very short (< 4 chars) and doesn't contain real words, discard
    if trimmed.len() < 4 {
        return String::new();
    }

    text.to_string()
}

/// Thread-safe wrapper for use across async tasks.
pub type SharedWhisperProvider = std::sync::Mutex<WhisperProvider>;
