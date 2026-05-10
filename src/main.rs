use clap::Parser;
use color_eyre::eyre::Result;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::runtime::Runtime;

mod audio_capture;
mod cli;
mod keyboard;
mod whisper_provider;

#[cfg(feature = "overlay")]
mod app;

#[cfg(feature = "overlay")]
mod hotkeys;

#[cfg(feature = "overlay")]
mod waybar;

pub fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("Setting up tokio runtime needs to succeed."))
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = cli::Cli::parse();

    match args.command {
        #[cfg(feature = "overlay")]
        cli::Command::WaybarStatus { connection_opts } => {
            runtime()
                .block_on(async move { waybar::main_waybar_status(&connection_opts).await })?;
        }
        #[cfg(feature = "overlay")]
        command @ cli::Command::Overlay { .. } => {
            app::launch_app(command)?;
        }
        cli::Command::Daemon { model, socket, .. } => {
            run_daemon(&model, &socket)?;
        }
    }

    Ok(())
}

use std::sync::OnceLock;

fn run_daemon(model_path: &str, _socket_path: &str) -> Result<()> {
    println!("[lds] daemon starting...");

    // Load whisper model with GPU
    println!("[lds] loading model: {}", model_path);
    let provider = Arc::new(Mutex::new(whisper_provider::WhisperProvider::new(model_path)?));
    println!("[lds] model loaded with Vulkan GPU.");

    // Init audio capture
    let capture = Arc::new(audio_capture::AudioCapture::new()?);
    println!("[lds] audio capture ready.");

    // State machine
    let recording = Arc::new(Mutex::new(false));

    println!("[lds] daemon ready. Type 'start' / 'stop' / 'quit' and press Enter.");
    println!("[lds] (IPC not yet wired — using stdin for testing)");

    // Simple stdin loop for testing (will be replaced by IPC in Sprint 3)
    runtime().block_on(async {
        let stdin = BufReader::new(tokio::io::stdin());
        let mut lines = stdin.lines();

        loop {
            let Some(line) = lines.next_line().await.ok().flatten() else {
                break;
            };
            let cmd = line.trim();

            match cmd {
                "start" | "s" => {
                    if *recording.lock().unwrap() {
                        println!("[lds] already recording");
                        continue;
                    }
                    *recording.lock().unwrap() = true;
                    capture.start();
                    println!("[lds] ● recording...");
                }
                "stop" | "x" => {
                    if !*recording.lock().unwrap() {
                        println!("[lds] not recording");
                        continue;
                    }
                    *recording.lock().unwrap() = false;
                    let samples = capture.stop();

                    if samples.is_empty() {
                        println!("[lds] no audio captured");
                        continue;
                    }

                    let duration = samples.len() as f32 / 16000.0;
                    println!("[lds] transcribing {:.1}s of audio...", duration);

                    // Transcribe
                    let provider = provider.clone();
                    let text = tokio::task::spawn_blocking(move || {
                        provider.lock().unwrap().transcribe(&samples)
                    })
                    .await
                    .expect("transcription task panicked")
                    .expect("transcription failed");

                    if text.is_empty() {
                        println!("[lds] (no speech detected)");
                        continue;
                    }

                    println!("[lds] transcript: \"{}\"", text);

                    // Clipboard write (Sprint 2 — best-effort)
                    match write_clipboard(&text) {
                        Ok(()) => println!("[lds] ✓ copied to clipboard"),
                        Err(e) => println!("[lds] ✗ clipboard failed: {} (text printed above)", e),
                    }

                    // Auto-type (best-effort)
                    match auto_type(&text) {
                        Ok(()) => println!("[lds] ✓ auto-typed"),
                        Err(e) => println!("[lds] ✗ auto-type failed: {}", e),
                    }
                }
                "quit" | "q" => {
                    println!("[lds] shutting down");
                    break;
                }
                "" => continue,
                _ => println!("[lds] unknown command: {} (start/stop/quit)", cmd),
            }
        }
    });

    Ok(())
}

fn write_clipboard(text: &str) -> Result<()> {
    let mut clipboard = arboard::Clipboard::new()?;
    clipboard.set_text(text)?;
    Ok(())
}

fn auto_type(text: &str) -> Result<()> {
    use enigo::{Enigo, Keyboard, Settings};
    let mut enigo = Enigo::new(&Settings::default())?;
    enigo.text(text)?;
    Ok(())
}
