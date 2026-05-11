use clap::Parser;
use color_eyre::eyre::Result;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

mod audio_capture;
mod cli;
mod config;
mod ipc;
mod whisper_provider;

#[cfg(feature = "overlay")]
mod app;

#[cfg(feature = "overlay")]
mod hotkeys;

#[cfg(feature = "overlay")]
mod keyboard;

#[cfg(feature = "overlay")]
mod waybar;

#[cfg(feature = "overlay")]
pub fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Runtime::new().expect("Setting up tokio runtime needs to succeed.")
    })
}

/// Commands from IPC handler to daemon worker
enum DaemonCmd {
    Start,
    Stop,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = cli::Cli::parse();

    let rt = tokio::runtime::Runtime::new()?;

    match args.command {
        #[cfg(feature = "overlay")]
        cli::Command::WaybarStatus { connection_opts } => {
            rt.block_on(async { waybar::main_waybar_status(&connection_opts).await })?;
        }
        #[cfg(feature = "overlay")]
        command @ cli::Command::Overlay { .. } => {
            app::launch_app(command)?;
        }
        cli::Command::Daemon {
            model,
            socket,
            device,
        } => {
            // Load config file
            let config_path = args
                .config
                .map(std::path::PathBuf::from)
                .unwrap_or_else(config::Config::default_path);
            let mut cfg = config::Config::load(&config_path)?;

            // CLI args override config file
            if let Some(m) = model {
                cfg.model = m;
            }
            if let Some(s) = socket {
                cfg.socket = s;
            }
            if let Some(d) = device {
                cfg.device = d;
            }

            if cfg.model.is_empty() {
                color_eyre::eyre::bail!(
                    "No model path specified. Use --model or set 'model' in {}",
                    config_path.display()
                );
            }

            rt.block_on(run_daemon(cfg))?;
        }
        cli::Command::InitConfig { output } => {
            let path = output
                .map(std::path::PathBuf::from)
                .unwrap_or_else(config::Config::default_path);
            config::Config::save_template(&path)?;
            eprintln!("[lds] template config written to {}", path.display());
        }
    }

    Ok(())
}

async fn run_daemon(cfg: config::Config) -> Result<()> {
    eprintln!("[lds] daemon starting...");

    eprintln!("[lds] loading model: {}", cfg.model);
    let provider = Arc::new(Mutex::new(whisper_provider::WhisperProvider::new(
        &cfg.model,
    )?));
    eprintln!("[lds] model loaded with Vulkan GPU.");

    let capture = Arc::new(audio_capture::AudioCapture::new_with_device(&cfg.device)?);
    eprintln!("[lds] audio capture ready.");

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<DaemonCmd>();
    let handle = Arc::new(ipc::DaemonHandle::new());

    // Register IPC callbacks → send commands through channel
    {
        let cmd_tx = cmd_tx.clone();
        *handle.on_start.lock().await = Some(Box::new(move || {
            cmd_tx.send(DaemonCmd::Start).ok();
            Ok(())
        }));
    }
    {
        let cmd_tx = cmd_tx.clone();
        *handle.on_stop.lock().await = Some(Box::new(move || {
            cmd_tx.send(DaemonCmd::Stop).ok();
            Ok(String::new())
        }));
    }

    handle.set_state(ipc::DaemonState::Idle).await;
    eprintln!("[lds] daemon ready. IPC on {}", cfg.socket);

    // Spawn IPC server as a task on the SAME runtime
    let ipc_handle = handle.clone();
    let ipc_path = cfg.socket.clone();
    tokio::spawn(async move {
        if let Err(e) = ipc::serve(&ipc_path, ipc_handle).await {
            eprintln!("[ipc] server error: {}", e);
        }
    });

    // Daemon worker loop — processes commands from IPC
    let recording: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    loop {
        tokio::select! {
            Some(cmd) = cmd_rx.recv() => {
                match cmd {
                    DaemonCmd::Start => {
                        if *recording.lock().unwrap() {
                            eprintln!("[lds] already recording");
                            continue;
                        }
                        *recording.lock().unwrap() = true;
                        capture.start();
                        handle.set_state(ipc::DaemonState::Recording).await;
                        {
                            let mut dbg = std::fs::OpenOptions::new()
                                .create(true).append(true)
                                .open("/tmp/lds-debug.log")
                                .unwrap();
                            use std::io::Write;
                            writeln!(dbg, "[start] buffer_len={}", capture.buffer_len()).unwrap();
                        }
                    }
                    DaemonCmd::Stop => {
                        if !*recording.lock().unwrap() {
                            eprintln!("[lds] not recording");
                            continue;
                        }
                        *recording.lock().unwrap() = false;
                        let samples = capture.stop();

                        // Debug: dump to file
                        {
                            let mut dbg = std::fs::OpenOptions::new()
                                .create(true).append(true)
                                .open("/tmp/lds-debug.log")
                                .unwrap();
                            use std::io::Write;
                            writeln!(dbg, "[stop] samples={}", samples.len()).unwrap();
                        }

                        if samples.is_empty() {
                            handle.set_state(ipc::DaemonState::Error("no audio captured".into())).await;
                            continue;
                        }

                        let duration = samples.len() as f32 / 16000.0;
                        eprintln!("[lds] transcribing {:.1}s...", duration);
                        handle.set_state(ipc::DaemonState::Transcribing).await;

                        let prov = provider.lock().unwrap();
                        match prov.transcribe(&samples) {
                            Ok(text) if !text.is_empty() => {
                                {
                                    let mut dbg = std::fs::OpenOptions::new()
                                        .create(true).append(true)
                                        .open("/tmp/lds-debug.log")
                                        .unwrap();
                                    use std::io::Write;
                                    writeln!(dbg, "[transcribe] ok: \"{}\"", text).unwrap();
                                }
                                if cfg.log_transcript {
                                    eprintln!("[lds] transcript: \"{}\"", text);
                                } else {
                                    eprintln!("[lds] transcript: {} chars", text.len());
                                }
                                if cfg.clipboard {
                                    match write_clipboard(&text) {
                                        Ok(()) => {
                                            let mut dbg = std::fs::OpenOptions::new()
                                                .create(true).append(true)
                                                .open("/tmp/lds-debug.log")
                                                .unwrap();
                                            use std::io::Write;
                                            writeln!(dbg, "[clipboard] ok").unwrap();
                                        }
                                        Err(e) => {
                                            let mut dbg = std::fs::OpenOptions::new()
                                                .create(true).append(true)
                                                .open("/tmp/lds-debug.log")
                                                .unwrap();
                                            use std::io::Write;
                                            writeln!(dbg, "[clipboard] FAIL: {}", e).unwrap();
                                        }
                                    }
                                }
                                if cfg.auto_type {
                                    match auto_type(&text) {
                                        Ok(()) => eprintln!("[lds] ✓ auto-type"),
                                        Err(e) => eprintln!("[lds] ✗ auto-type: {}", e),
                                    }
                                }
                                handle.broadcast_event(
                                    "final_transcript",
                                    serde_json::json!({ "text": text }),
                                ).await;
                                handle.set_state(ipc::DaemonState::ClipboardWritten).await;
                            }
                            Ok(_) => {
                                {
                                    let mut dbg = std::fs::OpenOptions::new()
                                        .create(true).append(true)
                                        .open("/tmp/lds-debug.log")
                                        .unwrap();
                                    use std::io::Write;
                                    writeln!(dbg, "[transcribe] empty result").unwrap();
                                }
                                handle.set_state(ipc::DaemonState::Error("no speech detected".into())).await;
                            }
                            Err(e) => {
                                {
                                    let mut dbg = std::fs::OpenOptions::new()
                                        .create(true).append(true)
                                        .open("/tmp/lds-debug.log")
                                        .unwrap();
                                    use std::io::Write;
                                    writeln!(dbg, "[transcribe] error: {}", e).unwrap();
                                }
                                handle.set_state(ipc::DaemonState::Error(e.to_string())).await;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn write_clipboard(text: &str) -> Result<()> {
    // Prefer wl-copy on Wayland — it forks a background holder process
    // that keeps the clipboard offer alive. arboard's temporary Wayland
    // connection gets garbage-collected by the compositor.
    match std::process::Command::new("wl-copy")
        .arg("--trim-newline")
        .stdin(std::process::Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                stdin.write_all(text.as_bytes())?;
            }
            let status = child.wait()?;
            if status.success() {
                return Ok(());
            }
        }
        Err(_) => {} // wl-copy not found, fall through to arboard
    }
    // Fallback: arboard (works on X11, unreliable on Wayland for headless)
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
