use clap::Parser;
use color_eyre::eyre::Result;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

mod audio_capture;
mod cli;
mod config;
mod ipc;
mod smooth_typist;
mod streaming;
mod text_middleware;
mod transcript_log;
mod vad;
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
    let is_streaming = cfg.is_streaming();
    // Shared config for live updates from IPC
    let live_cfg = Arc::new(std::sync::Mutex::new(cfg.clone()));
    eprintln!(
        "[lds] daemon starting... (mode: {})",
        if is_streaming { "streaming" } else { "batch" }
    );

    eprintln!("[lds] loading model: {}", cfg.model);
    let provider = Arc::new(Mutex::new(whisper_provider::WhisperProvider::new(
        &cfg.model,
    )?));
    eprintln!("[lds] model loaded with Vulkan GPU.");

    let capture = Arc::new(audio_capture::AudioCapture::new_with_device(&cfg.device)?);
    eprintln!("[lds] audio capture ready.");

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<DaemonCmd>();
    let handle = Arc::new(ipc::DaemonHandle::new());

    // Always load VAD — needed for streaming mode, cheap at 865KB
    let vad: Option<Arc<Mutex<vad::Vad>>> = {
        let vad_model_path = find_vad_model();
        match vad_model_path {
            Some(path) => {
                eprintln!("[lds] loading VAD model: {}", path);
                match vad::Vad::new(&path, cfg.vad_threshold, cfg.vad_min_silence_ms) {
                    Ok(v) => {
                        eprintln!("[lds] VAD ready.");
                        Some(Arc::new(Mutex::new(v)))
                    }
                    Err(e) => {
                        eprintln!(
                            "[lds] warning: VAD init failed ({}), falling back to batch",
                            e
                        );
                        None
                    }
                }
            }
            None => {
                eprintln!(
                    "[lds] warning: no VAD model found, streaming disabled"
                );
                None
            }
        }
    };

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

    // Config update callback — returns current config, applies updates
    {
        let live_cfg = live_cfg.clone();
        *handle.on_config_update.lock().await = Some(Box::new(move |update: serde_json::Value| {
            if !update.is_null() {
                let mut cfg = live_cfg.lock().unwrap();
                if let Some(v) = update.get("min_audio_ms").and_then(|v| v.as_u64()) {
                    cfg.min_audio_ms = v;
                    eprintln!("[config] min_audio_ms = {}", v);
                }
                if let Some(v) = update.get("vad_threshold").and_then(|v| v.as_f64()) {
                    cfg.vad_threshold = v as f32;
                    eprintln!("[config] vad_threshold = {}", v);
                }
                if let Some(v) = update.get("chunk_interval_ms").and_then(|v| v.as_u64()) {
                    cfg.chunk_interval_ms = v;
                    eprintln!("[config] chunk_interval_ms = {}", v);
                }
                if let Some(v) = update.get("vad_min_silence_ms").and_then(|v| v.as_u64()) {
                    cfg.vad_min_silence_ms = v as u32;
                    eprintln!("[config] vad_min_silence_ms = {}", v);
                }
                if let Some(v) = update.get("mode").and_then(|v| v.as_str()) {
                    if v == "batch" || v == "streaming" {
                        cfg.mode = v.to_string();
                        eprintln!("[config] mode = {}", v);
                    }
                }
            }
            // Return current config
            let cfg = live_cfg.lock().unwrap();
            serde_json::json!({
                "min_audio_ms": cfg.min_audio_ms,
                "vad_threshold": cfg.vad_threshold,
                "chunk_interval_ms": cfg.chunk_interval_ms,
                "vad_min_silence_ms": cfg.vad_min_silence_ms,
                "mode": cfg.mode,
                "auto_type": cfg.auto_type,
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

    // Streaming state
    let stream_audio_tx: Arc<Mutex<Option<tokio::sync::mpsc::Sender<Vec<f32>>>>> =
        Arc::new(Mutex::new(None));

    loop {
        let chunk_sleep = tokio::time::sleep(Duration::from_millis(cfg.chunk_interval_ms));
        tokio::pin!(chunk_sleep);

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

                        if live_cfg.lock().unwrap().is_streaming() {
                            if let Some(ref vad_ctx) = vad {
                                let (coord, audio_tx) = streaming::StreamingCoordinator::new(
                                    provider.clone(),
                                    vad_ctx.clone(),
                                    handle.clone(),
                                    streaming::StreamingConfig::from(&*live_cfg.lock().unwrap()),
                                );
                                coord.reset_abort();

                                // Store the audio sender
                                *stream_audio_tx.lock().unwrap() = Some(audio_tx);

                                // Spawn the streaming coordinator
                                let handle_ref = handle.clone();
                                let cfg_ref = cfg.clone();
                                let recording_flag = recording.clone();
                                let stream_tx = stream_audio_tx.clone();

                                tokio::spawn(async move {
                                    let result = coord.run().await;

                                    // Streaming already typed + clipboard during rolling passes.
                                    // Just finalize state.
                                    match result {
                                        Ok(text) if !text.is_empty() => {
                                            eprintln!("[lds] transcript: \"{}\"", &text[..text.len().min(200)]);

                                            // Persist transcript to disk
                                            match transcript_log::save_transcript(&text) {
                                                Ok(path) => eprintln!("[lds] ✓ transcript saved: {}", path.display()),
                                                Err(e) => eprintln!("[lds] ✗ transcript save: {}", e),
                                            }

                                            // Clipboard already updated during rolling passes.
                                            // Only update if coordinator didn't (e.g. final pass had nothing new).
                                            if let Err(e) = crate::write_clipboard(&text) {
                                                eprintln!("[lds] ✗ clipboard: {}", e);
                                            } else {
                                                eprintln!("[lds] ✓ clipboard");
                                            }
                                            handle_ref.set_state(ipc::DaemonState::ClipboardWritten).await;
                                        }
                                        Ok(_) => {
                                            eprintln!("[streaming] no speech detected");
                                        }
                                        Err(e) => {
                                            eprintln!("[streaming] error: {}", e);
                                        }
                                    }

                                    *recording_flag.lock().unwrap() = false;
                                    *stream_tx.lock().unwrap() = None;
                                });
                            } else {
                                // No VAD — fall back to batch behavior
                                handle.set_state(ipc::DaemonState::Recording).await;
                            }
                        } else {
                            handle.set_state(ipc::DaemonState::Recording).await;
                        }
                    }
                    DaemonCmd::Stop => {
                        if !*recording.lock().unwrap() {
                            eprintln!("[lds] not recording");
                            continue;
                        }

                        // Stop capture and grab samples
                        let samples = capture.stop();
                        *recording.lock().unwrap() = false;

                        if live_cfg.lock().unwrap().is_streaming() {
                            // Streaming mode: close the audio channel — coordinator will finalize
                            drop(stream_audio_tx.lock().unwrap().take());
                        } else {
                            // Batch mode
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
                                    deliver_transcript(&text, &cfg, &handle).await;
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
            _ = &mut chunk_sleep, if *recording.lock().unwrap() && live_cfg.lock().unwrap().is_streaming() => {
                // Feed audio from capture to streaming coordinator
                let samples = capture.drain_buffer();
                if !samples.is_empty() {
                    let tx_guard = stream_audio_tx.lock().unwrap();
                    if let Some(ref sender) = *tx_guard {
                        let _ = sender.try_send(samples);
                    }
                }
            }
        }
    }
}

/// Deliver transcript: clipboard + auto-type + IPC broadcast.
async fn deliver_transcript(raw: &str, cfg: &config::Config, handle: &Arc<ipc::DaemonHandle>) {
    // Run text middleware on raw whisper output (period cleanup, filler removal, etc.)
    let text = text_middleware::clean_text(raw);

    if text.is_empty() {
        eprintln!("[lds] transcript: empty after middleware cleanup");
        handle.set_state(ipc::DaemonState::Error("no speech detected".into())).await;
        return;
    }

    if cfg.log_transcript {
        eprintln!("[lds] transcript: \"{}\"", text);
    } else {
        eprintln!("[lds] transcript: {} chars", text.len());
    }

    // Persist transcript to disk (belt-and-suspenders for clipboard failures)
    match transcript_log::save_transcript(&text) {
        Ok(path) => eprintln!("[lds] ✓ transcript saved: {}", path.display()),
        Err(e) => eprintln!("[lds] ✗ transcript save: {}", e),
    }

    match write_clipboard(&text) {
        Ok(()) => eprintln!("[lds] ✓ clipboard"),
        Err(e) => eprintln!("[lds] ✗ clipboard: {}", e),
    }

    if cfg.auto_type {
        match auto_type(&text) {
            Ok(()) => eprintln!("[lds] ✓ auto-type"),
            Err(e) => eprintln!("[lds] ✗ auto-type: {}", e),
        }
    }

    handle
        .broadcast_event("final_transcript", serde_json::json!({ "text": text }))
        .await;
    handle
        .set_state(ipc::DaemonState::ClipboardWritten)
        .await;
}

/// Find the Silero VAD model file.
fn find_vad_model() -> Option<String> {
    let user = std::env::var("USER").unwrap_or_default();
    let candidates = [
        // GGML Silero VAD models (whisper.cpp native format)
        format!("/home/{user}/.local/share/lds/ggml-silero-v6.2.0.bin"),
        format!("/home/{user}/.local/share/lds/ggml-silero-v5.1.2.bin"),
        // Relative to project
        "models/ggml-silero-v6.2.0.bin".into(),
        "models/ggml-silero-v5.1.2.bin".into(),
    ];

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.clone());
        }
    }

    None
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
    match std::process::Command::new("wtype")
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
