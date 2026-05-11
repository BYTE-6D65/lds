use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Smooth typist that emits characters at a controlled rate.
///
/// Instead of burst-typing entire segments via `wtype`, this queues text
/// and types it character-by-character at a calculated rate:
///   rate = chars_to_type / time_until_next_result
///
/// This creates a continuous typing feel that bridges the gap between
/// transcription passes.
pub struct SmoothTypist {
    tx: mpsc::UnboundedSender<TypistCommand>,
    /// Set when typist is actively emitting chars
    typing: Arc<AtomicBool>,
}

enum TypistCommand {
    /// Queue text for typing at the given rate (chars per second).
    Type {
        text: String,
        rate: f64,
    },
    /// Flush remaining text at max speed (for final delivery).
    Flush,
    /// Stop typing immediately and discard pending.
    Stop,
}

impl SmoothTypist {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<TypistCommand>();
        let typing = Arc::new(AtomicBool::new(false));
        let typing_clone = typing.clone();

        std::thread::spawn(move || {
            let mut pending: String = String::new();
            let mut rate: f64 = 60.0; // default: 60 chars/sec

            loop {
                // Drain all pending commands first
                loop {
                    match rx.try_recv() {
                        Ok(cmd) => match cmd {
                            TypistCommand::Type { text, rate: r } => {
                                pending.push_str(&text);
                                rate = r;
                            }
                            TypistCommand::Flush => {
                                if !pending.is_empty() {
                                    type_burst(&pending);
                                    pending.clear();
                                    typing_clone.store(false, Ordering::Relaxed);
                                }
                            }
                            TypistCommand::Stop => {
                                pending.clear();
                                typing_clone.store(false, Ordering::Relaxed);
                                return;
                            }
                        },
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            if !pending.is_empty() {
                                type_burst(&pending);
                            }
                            typing_clone.store(false, Ordering::Relaxed);
                            return;
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break,
                    }
                }

                // Type pending chars at rate
                if !pending.is_empty() {
                    typing_clone.store(true, Ordering::Relaxed);
                    // Calculate how many chars to type per tick (targeting ~60 ticks/sec)
                    let chars_per_tick = std::cmp::max(1, (rate / 60.0).ceil() as usize);
                    let tick_delay_us = std::cmp::max(1000, 1_000_000 / 60);

                    let end = std::cmp::min(chars_per_tick, pending.len());
                    let chunk: String = pending.drain(..end).collect();
                    type_burst(&chunk);

                    if !pending.is_empty() {
                        std::thread::sleep(std::time::Duration::from_micros(tick_delay_us));
                    } else {
                        typing_clone.store(false, Ordering::Relaxed);
                    }
                } else {
                    typing_clone.store(false, Ordering::Relaxed);
                    // Wait for new input
                    match rx.blocking_recv() {
                        Some(cmd) => match cmd {
                            TypistCommand::Type { text, rate: r } => {
                                pending.push_str(&text);
                                rate = r;
                            }
                            TypistCommand::Flush => {
                                if !pending.is_empty() {
                                    type_burst(&pending);
                                    pending.clear();
                                }
                            }
                            TypistCommand::Stop => return,
                        },
                        None => {
                            if !pending.is_empty() {
                                type_burst(&pending);
                            }
                            return;
                        }
                    }
                }
            }
        });

        Self { tx, typing }
    }

    /// Queue text for smooth typing.
    /// `next_result_interval` is the estimated seconds until the next transcription result.
    pub fn type_text(&self, text: &str, next_result_interval: f64) {
        if text.is_empty() {
            return;
        }
        let chars = text.len() as f64;
        // Rate: type all chars in ~75% of the interval, leaving headroom
        let rate = if next_result_interval > 0.1 {
            (chars / (next_result_interval * 0.75)).max(20.0).min(300.0)
        } else {
            100.0
        };
        let _ = self.tx.send(TypistCommand::Type {
            text: text.to_string(),
            rate,
        });
    }

    /// Flush remaining text at max speed.
    pub fn flush(&self) {
        let _ = self.tx.send(TypistCommand::Flush);
    }

    /// Stop typing and discard pending text.
    #[allow(dead_code)]
    pub fn stop(&self) {
        let _ = self.tx.send(TypistCommand::Stop);
    }

    /// Is the typist currently emitting characters?
    #[allow(dead_code)]
    pub fn is_typing(&self) -> bool {
        self.typing.load(Ordering::Relaxed)
    }
}

/// Type a string instantly via wtype (burst mode).
fn type_burst(text: &str) {
    use std::process::Command;
    if let Err(e) = Command::new("wtype").arg("--").arg(text).output() {
        eprintln!("[typist] wtype error: {}", e);
    }
}
