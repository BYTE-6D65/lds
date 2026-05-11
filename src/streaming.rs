use crate::config::Config;
use crate::ipc::DaemonHandle;
use crate::vad::SharedVad;
use crate::whisper_provider::SharedWhisperProvider;
use color_eyre::eyre::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;
use tokio::time::Duration;

/// Rolling transcription coordinator.
///
/// Simple model: accumulate audio while speech is detected. Every N ms,
/// transcribe whatever's in the buffer, type the FULL output, then clear
/// the buffer. Next pass only sees fresh audio. No dedup needed.
pub struct StreamingCoordinator {
    provider: Arc<SharedWhisperProvider>,
    vad: Arc<SharedVad>,
    handle: Arc<DaemonHandle>,
    config: StreamingConfig,
    abort_flag: Arc<AtomicBool>,
    full_text: Arc<AsyncMutex<String>>,
    audio_rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<Vec<f32>>>,
}

#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// How often to run transcription while speech is active (ms)
    pub rolling_interval_ms: u64,
    /// Minimum audio to accumulate before transcription (ms of audio)
    pub min_audio_ms: u64,
    pub language: String,
    pub initial_prompt: String,
}

impl From<&Config> for StreamingConfig {
    fn from(cfg: &Config) -> Self {
        Self {
            rolling_interval_ms: cfg.chunk_interval_ms,
            min_audio_ms: 1500,
            language: cfg.language.clone(),
            initial_prompt: cfg.initial_prompt.clone(),
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
        let (audio_tx, audio_rx) = tokio::sync::mpsc::channel(128);

        let coord = Self {
            provider,
            vad,
            handle,
            config,
            abort_flag: Arc::new(AtomicBool::new(false)),
            full_text: Arc::new(AsyncMutex::new(String::new())),
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

    pub async fn run(&self) -> Result<String> {
        let mut buffer: Vec<f32> = Vec::new();
        let min_samples = (16000 * self.config.min_audio_ms / 1000) as usize;
        let rolling_interval = Duration::from_millis(self.config.rolling_interval_ms);

        // VAD sliding window
        let vad_window_samples = 16000 * 2;
        let mut vad_window: Vec<f32> = Vec::with_capacity(vad_window_samples);

        let mut in_speech = false;
        let mut silence_count: usize = 0;
        let mut last_transcribe = std::time::Instant::now();

        self.handle
            .set_state(crate::ipc::DaemonState::Streaming {
                partial_text: String::new(),
            })
            .await;

        let rx = &self.audio_rx;

        loop {
            if self.abort_flag.load(Ordering::Relaxed) {
                break;
            }

            // Drain all available audio
            let mut new_samples = Vec::new();
            {
                let mut rx_guard = rx.lock().await;
                loop {
                    match rx_guard.try_recv() {
                        Ok(chunk) => new_samples.extend_from_slice(&chunk),
                        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                            // Hotkey released — final transcription on remaining buffer
                            if buffer.len() >= min_samples {
                                eprintln!(
                                    "[streaming] final pass ({}s)",
                                    buffer.len() as f32 / 16000.0
                                );
                                self.transcribe_and_type(&buffer).await?;
                            }
                            let text = self.full_text.lock().await.clone();
                            return Ok(text.trim().to_string());
                        }
                    }
                }
            }

            if new_samples.is_empty() {
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }

            buffer.extend_from_slice(&new_samples);

            // Update VAD window
            vad_window.extend_from_slice(&new_samples);
            if vad_window.len() > vad_window_samples {
                let excess = vad_window.len() - vad_window_samples;
                vad_window.drain(..excess);
            }

            // VAD check
            let vad_result = {
                let mut vad = self.vad.lock().unwrap();
                vad.detect(&vad_window)
            };

            if vad_result.has_speech {
                if !in_speech {
                    eprintln!("[streaming] speech started");
                }
                in_speech = true;
                silence_count = 0;
            } else if in_speech {
                silence_count += 1;
            }

            // Rolling transcription: enough audio + enough time passed
            if in_speech
                && buffer.len() >= min_samples
                && last_transcribe.elapsed() >= rolling_interval
            {
                eprintln!(
                    "[streaming] rolling ({}s)",
                    buffer.len() as f32 / 16000.0
                );
                self.transcribe_and_type(&buffer).await?;
                // Clear buffer — next pass only gets fresh audio
                buffer.clear();
                last_transcribe = std::time::Instant::now();
            }

            // Prolonged silence while in speech — finalize this chunk
            if in_speech && silence_count >= 6 {
                eprintln!("[streaming] silence timeout");
                if !buffer.is_empty() {
                    self.transcribe_and_type(&buffer).await?;
                    buffer.clear();
                }
                in_speech = false;
                silence_count = 0;
                last_transcribe = std::time::Instant::now();
            }

            tokio::time::sleep(Duration::from_millis(80)).await;
        }

        let text = self.full_text.lock().await.clone();
        Ok(text.trim().to_string())
    }

    /// Transcribe audio buffer and type the full result.
    /// Each call covers only new audio since the last clear.
    async fn transcribe_and_type(&self, audio: &[f32]) -> Result<()> {
        if audio.is_empty() {
            return Ok(());
        }

        let provider = self.provider.clone();
        let lang = self.config.language.clone();
        let prompt = self.config.initial_prompt.clone();
        let audio_data = audio.to_vec();

        let text = tokio::task::spawn_blocking(move || {
            let prov = provider.lock().unwrap();
            match prov.transcribe(&audio_data, &lang, &prompt) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[streaming] error: {}", e);
                    String::new()
                }
            }
        })
        .await
        .unwrap_or_default();

        if text.trim().is_empty() {
            return Ok(());
        }

        eprintln!("[streaming] got: \"{}\"", text.trim());

        // Type it — prepend space if there's been previous output
        let text_for_type = {
            let full = self.full_text.lock().await;
            if full.is_empty() {
                text.clone()
            } else {
                format!(" {}", text.trim())
            }
        };
        tokio::task::spawn_blocking(move || {
            if let Err(e) = crate::auto_type(&text_for_type) {
                eprintln!("[streaming] type error: {}", e);
            }
        })
        .await
        .ok();

        // Accumulate full text and update clipboard
        let mut full = self.full_text.lock().await;
        if !full.is_empty() {
            full.push(' ');
        }
        full.push_str(text.trim());

        let clipboard_text = full.clone();
        tokio::task::spawn_blocking(move || {
            let _ = crate::write_clipboard(&clipboard_text);
        })
        .await
        .ok();

        // Broadcast
        self.handle
            .set_state(crate::ipc::DaemonState::Streaming {
                partial_text: full.clone(),
            })
            .await;

        Ok(())
    }
}

unsafe impl Send for StreamingCoordinator {}
