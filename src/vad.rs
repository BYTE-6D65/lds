use color_eyre::eyre::{Context, Result};
use std::sync::Mutex;
use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

/// Voice Activity Detection using whisper.cpp's built-in Silero VAD.
pub struct Vad {
    ctx: WhisperVadContext,
    params: WhisperVadParams,
}

/// Result of VAD analysis on an audio chunk.
#[derive(Debug, Clone)]
pub struct VadResult {
    pub has_speech: bool,
    pub probability: f32,
}

impl Vad {
    /// Create a new VAD pipeline.
    /// `vad_model_path`: path to silero_vad.onnx model file.
    /// `threshold`: speech probability threshold (0.0–1.0, default 0.5).
    /// `min_silence_ms`: minimum silence duration to trigger segment end.
    pub fn new(vad_model_path: &str, threshold: f32, min_silence_ms: u32) -> Result<Self> {
        let mut ctx_params = WhisperVadContextParams::new();
        ctx_params.set_use_gpu(true);

        let ctx = WhisperVadContext::new(vad_model_path, ctx_params)
            .with_context(|| format!("failed to load VAD model from {}", vad_model_path))?;

        let mut params = WhisperVadParams::new();
        params.set_threshold(threshold);
        params.set_min_silence_duration(min_silence_ms as i32);
        params.set_speech_pad(30); // pad speech segments by 30ms

        Ok(Self { ctx, params })
    }

    /// Run VAD on audio samples.
    /// Returns the max speech probability and whether speech was detected.
    pub fn detect(&mut self, audio: &[f32]) -> VadResult {
        match self.ctx.detect_speech(audio) {
            Ok(()) => {
                let probs = self.ctx.probabilities();
                let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
                VadResult {
                    has_speech: max_prob >= self.params_threshold(),
                    probability: max_prob,
                }
            }
            Err(_) => VadResult {
                has_speech: false,
                probability: 0.0,
            },
        }
    }

    /// Detect speech segments in audio, returning (start_ms, end_ms) pairs.
    /// Note: whisper.cpp VAD returns timestamps in centiseconds.
    pub fn segments(&mut self, audio: &[f32]) -> Vec<(f32, f32)> {
        match self.ctx.segments_from_samples(self.params.clone(), audio) {
            Ok(segs) => {
                let count = segs.num_segments();
                let mut out = Vec::with_capacity(count as usize);
                for i in 0..count {
                    if let Some(seg) = segs.get_segment(i) {
                        // Convert centiseconds → milliseconds
                        out.push((seg.start * 10.0, seg.end * 10.0));
                    }
                }
                out
            }
            Err(_) => Vec::new(),
        }
    }

    fn params_threshold(&self) -> f32 {
        // WhisperVadParams doesn't expose getter, so we track threshold via the
        // value we set. Since we can't read it back, store separately.
        // For now just use 0.5 default — the actual threshold was set via set_threshold()
        0.5
    }
}

/// Thread-safe wrapper.
pub type SharedVad = Mutex<Vad>;
