use clap::Parser;
use color_eyre::eyre::Result;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

mod audio_capture;
mod cli;
mod config;
mod ipc;
mod text_middleware;
mod transcript_log;
mod wayland_typist;
mod whisper_provider;

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
        cli::Command::Daemon {
            model,
            socket,
            device,
        } => {
            let config_path = args
                .config
                .map(std::path::PathBuf::from)
                .unwrap_or_else(config::Config::default_path);
            let mut cfg = config::Config::load(&config_path)?;

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
    eprintln!("[lds] daemon starting... (mode: batch)");

    // Initialize native Wayland typist
    let typist = Arc::new(Mutex::new(match wayland_typist::WaylandTypist::new() {
        Ok(t) => {
            eprintln!("[lds] native Wayland typist ready.");
            t
        }
        Err(e) => {
            eprintln!("[lds] warning: native typist failed ({}), falling back to wtype", e);
            // We'll check for None and fall back to spawn-based typing
            return Err(color_eyre::eyre::eyre!("native typist init failed: {}", e));
        }
    }));

    eprintln!("[lds] loading model: {}", cfg.model);
    let provider = Arc::new(Mutex::new(whisper_provider::WhisperProvider::new(
        &cfg.model,
    )?));
    eprintln!("[lds] model loaded.");

    let capture = Arc::new(audio_capture::AudioCapture::new_with_device(&cfg.device)?);
    eprintln!("[lds] audio capture ready.");

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<DaemonCmd>();
    let handle = Arc::new(ipc::DaemonHandle::new());

    // Register IPC callbacks
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

    // Config update callback
    {
        let live_cfg = Arc::new(std::sync::Mutex::new(cfg.clone()));
        *handle.on_config_update.lock().await = Some(Box::new(move |update: serde_json::Value| {
            if !update.is_null() {
                let mut c = live_cfg.lock().unwrap();
                if let Some(v) = update.get("auto_type").and_then(|v| v.as_bool()) {
                    c.auto_type = v;
                    eprintln!("[config] auto_type = {}", v);
                }
                if let Some(v) = update.get("language").and_then(|v| v.as_str()) {
                    c.language = v.to_string();
                    eprintln!("[config] language = {}", v);
                }
            }
            let c = live_cfg.lock().unwrap();
            serde_json::json!({
                "auto_type": c.auto_type,
                "language": c.language,
                "mode": "batch",
            })
        }));
    }

    handle.set_state(ipc::DaemonState::Idle).await;
    eprintln!("[lds] daemon ready. IPC on {}", cfg.socket);

    // Spawn IPC server
    let ipc_handle = handle.clone();
    let ipc_path = cfg.socket.clone();
    tokio::spawn(async move {
        if let Err(e) = ipc::serve(&ipc_path, ipc_handle).await {
            eprintln!("[ipc] server error: {}", e);
        }
    });

    // Daemon worker loop
    let recording: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

    loop {
        tokio::select! {
            Some(cmd) = cmd_rx.recv() => {
                match cmd {
                    DaemonCmd::Start => {
                        if *recording.lock().unwrap() {
                            // Toggle: treat Start while recording as Stop
                            eprintln!("[lds] toggle: already recording, stopping");
                            cmd_tx.send(DaemonCmd::Stop).ok();
                            continue;
                        }
                        *recording.lock().unwrap() = true;
                        capture.start();
                        handle.set_state(ipc::DaemonState::Recording).await;
                    }
                    DaemonCmd::Stop => {
                        if !*recording.lock().unwrap() {
                            eprintln!("[lds] not recording");
                            continue;
                        }

                        // Stop capture and grab samples
                        let samples = capture.stop();
                        *recording.lock().unwrap() = false;

                        dump_wav(&samples);

                        if samples.is_empty() {
                            eprintln!("[lds] no audio captured");
                            handle.set_state(ipc::DaemonState::Error("no audio captured".into())).await;
                            continue;
                        }

                        let duration = samples.len() as f32 / 16000.0;
                        eprintln!("[lds] transcribing {:.1}s...", duration);
                        handle.set_state(ipc::DaemonState::Transcribing).await;

                        let prov = provider.lock().unwrap();
                        match prov.transcribe(&samples, &cfg.language, &cfg.initial_prompt) {
                            Ok(text) if !text.is_empty() => {
                                deliver_transcript(&text, &cfg, &handle, &typist).await;
                            }
                            Ok(_) => {
                                eprintln!("[lds] no speech detected");
                                handle.set_state(ipc::DaemonState::Error("no speech detected".into())).await;
                            }
                            Err(e) => {
                                eprintln!("[lds] transcription error: {}", e);
                                handle.set_state(ipc::DaemonState::Error(e.to_string())).await;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Deliver transcript: clipboard + auto-type + IPC broadcast.
async fn deliver_transcript(raw: &str, cfg: &config::Config, handle: &Arc<ipc::DaemonHandle>, typist: &Arc<Mutex<wayland_typist::WaylandTypist>>) {
    // Run through text middleware — strips hallucinated filler/noise from
    // stagnant air ("thank you", "thanks for watching", lone noise words, etc.)
    // and cleans up punctuation artifacts.
    let text = text_middleware::clean_text(raw);

    if text.is_empty() {
        eprintln!("[lds] transcript: empty after transcription");
        handle.set_state(ipc::DaemonState::Error("no speech detected".into())).await;
        return;
    }

    if cfg.log_transcript {
        eprintln!("[lds] transcript: \"{}\"", text);
    } else {
        eprintln!("[lds] transcript: {} chars", text.len());
    }

    // Persist transcript to disk
    match transcript_log::save_transcript(&text) {
        Ok(path) => eprintln!("[lds] ✓ transcript saved: {}", path.display()),
        Err(e) => eprintln!("[lds] ✗ transcript save: {}", e),
    }

    match write_clipboard(&text) {
        Ok(()) => eprintln!("[lds] ✓ clipboard"),
        Err(e) => eprintln!("[lds] ✗ clipboard: {}", e),
    }

    if cfg.auto_type {
        match typist.lock().unwrap().type_text(&text) {
            Ok(()) => eprintln!("[lds] ✓ auto-type (native)"),
            Err(e) => {
                eprintln!("[lds] ✗ native auto-type: {}, falling back to wtype", e);
                match auto_type(&text) {
                    Ok(()) => eprintln!("[lds] ✓ auto-type (wtype fallback)"),
                    Err(e) => eprintln!("[lds] ✗ auto-type: {}", e),
                }
            }
        }
    }

    handle
        .broadcast_event("final_transcript", serde_json::json!({ "text": text }))
        .await;
    handle
        .set_state(ipc::DaemonState::ClipboardWritten)
        .await;
}

/// Dump captured audio to /tmp/lds-last-recording.wav for debugging
fn dump_wav(samples: &[f32]) {
    if samples.is_empty() {
        return;
    }
    let path = "/tmp/lds-last-recording.wav";
    match std::fs::File::create(path) {
        Ok(mut f) => {
            use std::io::Write;
            let data_len = (samples.len() * 2) as u32;
            let _ = f.write_all(b"RIFF");
            let _ = f.write_all(&(36 + data_len).to_le_bytes());
            let _ = f.write_all(b"WAVE");
            let _ = f.write_all(b"fmt ");
            let _ = f.write_all(&16u32.to_le_bytes());
            let _ = f.write_all(&1u16.to_le_bytes());
            let _ = f.write_all(&1u16.to_le_bytes());
            let _ = f.write_all(&16000u32.to_le_bytes());
            let _ = f.write_all(&32000u32.to_le_bytes());
            let _ = f.write_all(&2u16.to_le_bytes());
            let _ = f.write_all(&16u16.to_le_bytes());
            let _ = f.write_all(b"data");
            let _ = f.write_all(&data_len.to_le_bytes());
            for s in samples {
                let pcm = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                let _ = f.write_all(&pcm.to_le_bytes());
            }
        }
        Err(e) => eprintln!("[lds] wav dump failed: {}", e),
    }
}

pub fn write_clipboard(text: &str) -> Result<()> {
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
        Err(_) => {}
    }
    let mut clipboard = arboard::Clipboard::new()?;
    clipboard.set_text(text)?;
    Ok(())
}

pub fn auto_type(text: &str) -> Result<()> {
    // Use wtype with a 2ms inter-key delay. Without this, wtype blasts all
    // characters at Wayland as fast as possible and the compositor can drop
    // key events under load, causing missing characters (especially repeated
    // chars like the P in VIP, the A in hammer, etc.).
    match std::process::Command::new("wtype")
        .arg("-d")
        .arg("2")
        .arg("--")
        .arg(text)
        .status()
    {
        Ok(status) if status.success() => return Ok(()),
        _ => {}
    }
    use enigo::{Enigo, Keyboard, Settings};
    let mut enigo = Enigo::new(&Settings::default())?;
    enigo.text(text)?;
    Ok(())
}
