use crate::config::Config;
use crate::ipc::DaemonHandle;
use crate::vad::SharedVad;
use crate::whisper_provider::SharedWhisperProvider;
use color_eyre::eyre::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;
use tokio::time::Duration;

/// Streaming transcription coordinator.
///
/// Audio is fed via a channel (the capture is not Send, so we can't hold it).
/// The coordinator receives audio chunks and runs VAD → whisper → IPC.
pub struct StreamingCoordinator {
    provider: Arc<SharedWhisperProvider>,
    vad: Arc<SharedVad>,
    handle: Arc<DaemonHandle>,
    config: StreamingConfig,
    abort_flag: Arc<AtomicBool>,
    utterance_text: Arc<AsyncMutex<String>>,
    audio_rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<Vec<f32>>>,
}

#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub chunk_interval_ms: u64,
    pub partial_results: bool,
    pub language: String,
    pub initial_prompt: String,
    pub min_audio_for_vad: usize,
    pub silence_chunks_to_finalize: usize,
}

impl From<&Config> for StreamingConfig {
    fn from(cfg: &Config) -> Self {
        Self {
            chunk_interval_ms: cfg.chunk_interval_ms,
            partial_results: cfg.partial_results,
            language: cfg.language.clone(),
            initial_prompt: cfg.initial_prompt.clone(),
            min_audio_for_vad: 16000,
            silence_chunks_to_finalize: 3,
        }
    }
}

impl StreamingCoordinator {
    pub fn new(
        provider: Arc<SharedWhisperProvider>,
        vad: Arc<SharedVad>,
        handle: Arc<DaemonHandle>,
        config: StreamingConfig,
    ) -> (Self, tokio::sync::mpsc::Sender<Vec<f32>>) {
        let (audio_tx, audio_rx) = tokio::sync::mpsc::channel(64);

        let coord = Self {
            provider,
            vad,
            handle,
            config,
            abort_flag: Arc::new(AtomicBool::new(false)),
            utterance_text: Arc::new(AsyncMutex::new(String::new())),
            audio_rx: tokio::sync::Mutex::new(audio_rx),
        };

        (coord, audio_tx)
    }

    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::Relaxed);
    }

    pub fn reset_abort(&self) {
        self.abort_flag.store(false, Ordering::Relaxed);
    }

    /// Run the streaming loop. Audio is fed via the channel by the main loop.
    pub async fn run(&self) -> Result<String> {
        let mut speech_buffer: Vec<f32> = Vec::new();
        let mut silence_count: usize = 0;
        let mut in_utterance = false;
        let chunk_duration = Duration::from_millis(self.config.chunk_interval_ms);
        let max_buffer_samples = 16000 * 120;
        let mut final_text = String::new();

        self.handle
            .set_state(crate::ipc::DaemonState::Streaming {
                partial_text: String::new(),
            })
            .await;

        let rx = &self.audio_rx;

        loop {
            if self.abort_flag.load(Ordering::Relaxed) {
                eprintln!("[streaming] aborted");
                let text = self.transcribe_utterance(&speech_buffer).await?;
                final_text.push_str(&text);
                break;
            }

            // Collect audio chunks from channel
            let mut new_samples = Vec::new();
            let mut _got_data = false;

            // Drain all available chunks
            {
                let mut rx_guard = rx.lock().await;
                loop {
                    match rx_guard.try_recv() {
                        Ok(chunk) => {
                            new_samples.extend_from_slice(&chunk);
                            _got_data = true;
                        }
                        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                            // Channel closed — recording stopped
                            eprintln!("[streaming] audio channel closed, finalizing");
                            let text = self.transcribe_utterance(&speech_buffer).await?;
                            final_text.push_str(&text);
                            return Ok(final_text);
                        }
                    }
                }
            }

            if in_utterance {
                speech_buffer.extend_from_slice(&new_samples);
                if speech_buffer.len() > max_buffer_samples {
                    let excess = speech_buffer.len() - max_buffer_samples;
                    speech_buffer.drain(..excess);
                }
            }

            // Need enough audio for VAD
            let vad_audio = if in_utterance {
                speech_buffer.as_slice()
            } else {
                new_samples.as_slice()
            };

            if vad_audio.len() < self.config.min_audio_for_vad {
                tokio::time::sleep(chunk_duration).await;
                continue;
            }

            // Run VAD
            let vad_result = {
                let mut vad = self.vad.lock().unwrap();
                vad.detect(vad_audio)
            };

            // Broadcast VAD status
            self.handle
                .broadcast_event(
                    "vad_status",
                    serde_json::json!({
                        "speech_detected": vad_result.has_speech,
                        "probability": vad_result.probability,
                    }),
                )
                .await;

            if vad_result.has_speech {
                if !in_utterance {
                    eprintln!("[streaming] speech detected, starting utterance");
                    in_utterance = true;
                    speech_buffer.extend_from_slice(&new_samples);
                }
                silence_count = 0;
            } else if in_utterance {
                silence_count += 1;

                if silence_count >= self.config.silence_chunks_to_finalize {
                    eprintln!(
                        "[streaming] silence after {} chunks, finalizing ({}s audio)",
                        silence_count,
                        speech_buffer.len() as f32 / 16000.0
                    );

                    let text = self.transcribe_utterance(&speech_buffer).await?;
                    if !text.is_empty() {
                        final_text.push_str(&text);
                        final_text.push(' ');
                    }
                    speech_buffer.clear();
                    in_utterance = false;
                    silence_count = 0;

                    // If we got text, we're done with this utterance
                    if !final_text.trim().is_empty() {
                        return Ok(final_text.trim().to_string());
                    }
                }
            }

            tokio::time::sleep(chunk_duration).await;
        }

        Ok(final_text.trim().to_string())
    }

    /// Transcribe the accumulated speech buffer using streaming whisper.
    async fn transcribe_utterance(&self, audio: &[f32]) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }

        let duration = audio.len() as f32 / 16000.0;
        eprintln!("[streaming] transcribing {:.1}s...", duration);

        self.handle
            .set_state(crate::ipc::DaemonState::Transcribing)
            .await;

        let provider = self.provider.clone();
        let lang = self.config.language.clone();
        let prompt = self.config.initial_prompt.clone();
        let do_partials = self.config.partial_results;
        let abort = self.abort_flag.clone();
        let audio_data = audio.to_vec();

        let text = tokio::task::spawn_blocking(move || {
            let prov = provider.lock().unwrap();

            let result = prov.transcribe_streaming(
                &audio_data,
                &lang,
                &prompt,
                move |seg| {
                    let trimmed = seg.text.trim();
                    if !trimmed.is_empty() && do_partials {
                        eprintln!("[streaming] segment: \"{}\"", trimmed);
                    }
                },
                abort,
            );

            match result {
                Ok(full_text) => full_text,
                Err(e) => {
                    eprintln!("[streaming] transcription error: {}", e);
                    String::new()
                }
            }
        })
        .await
        .unwrap_or_default();

        // Broadcast partial for this utterance
        if self.config.partial_results && !text.is_empty() {
            self.handle
                .broadcast_event(
                    "partial_transcript",
                    serde_json::json!({
                        "text": text,
                        "is_final": false,
                    }),
                )
                .await;

            let full = format!("{}{}", self.utterance_text.lock().await, text);
            self.handle
                .set_state(crate::ipc::DaemonState::Streaming {
                    partial_text: full,
                })
                .await;
        }

        // Accumulate
        if !text.is_empty() {
            let mut acc = self.utterance_text.lock().await;
            acc.push_str(&text);
            acc.push(' ');
        }

        Ok(text)
    }
}

// Safety: StreamingCoordinator is Send because all its fields are Send.
// The audio_rx Mutex<Receiver> is Send. The other Arc fields are Send.
// The provider Arc<Mutex<WhisperProvider>> — WhisperProvider contains WhisperContext
// which should be Send (whisper-rs impls Send+Sync on WhisperState).
unsafe impl Send for StreamingCoordinator {}
