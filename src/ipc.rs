use color_eyre::eyre::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::net::UnixListener;
use tokio::sync::{broadcast, Mutex};
use tokio_tungstenite::tungstenite::Message;

/// IPC message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

/// Daemon state exposed via IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonState {
    Idle,
    Recording,
    Transcribing,
    Streaming {
        partial_text: String,
    },
    ClipboardWritten,
    Error(String),
}

/// Shared daemon state accessible by IPC handler
pub struct DaemonHandle {
    pub state: Mutex<DaemonState>,
    pub event_tx: broadcast::Sender<IpcMessage>,
    // Callbacks — set by main daemon loop
    pub on_start: Mutex<Option<Box<dyn Send + Sync + Fn() -> Result<()>>>>,
    pub on_stop: Mutex<Option<Box<dyn Send + Sync + Fn() -> Result<String>>>>,
    /// Streaming: abort in-flight transcription
    pub on_abort: Mutex<Option<Box<dyn Send + Sync + Fn()>>>,
}

impl DaemonHandle {
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(32);
        Self {
            state: Mutex::new(DaemonState::Idle),
            event_tx,
            on_start: Mutex::new(None),
            on_stop: Mutex::new(None),
            on_abort: Mutex::new(None),
        }
    }

    pub async fn set_state(&self, new_state: DaemonState) {
        let mut state = self.state.lock().await;
        *state = new_state.clone();
        drop(state);

        // Broadcast state change to all IPC clients
        let msg = IpcMessage {
            msg_type: "state".into(),
            id: None,
            ts: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            ),
            payload: Some(serde_json::to_value(&new_state).unwrap_or_default()),
        };
        let _ = self.event_tx.send(msg);
    }

    pub async fn broadcast_event(&self, msg_type: &str, payload: serde_json::Value) {
        let msg = IpcMessage {
            msg_type: msg_type.into(),
            id: None,
            ts: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            ),
            payload: Some(payload),
        };
        let _ = self.event_tx.send(msg);
    }
}

/// Start the UDS WebSocket IPC server.
/// Binds to `socket_path`, accepts connections, handles requests.
pub async fn serve(socket_path: &str, handle: Arc<DaemonHandle>) -> Result<()> {
    // Remove stale socket
    let path = Path::new(socket_path);
    if path.exists() {
        std::fs::remove_file(path).with_context(|| "failed to remove stale socket")?;
    }

    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("failed to bind IPC socket at {}", socket_path))?;
    println!("[ipc] listening on {}", socket_path);

    loop {
        let (stream, _) = listener
            .accept()
            .await
            .with_context(|| "failed to accept IPC connection")?;
        let ws_stream = tokio_tungstenite::accept_async(stream)
            .await
            .with_context(|| "WebSocket handshake failed")?;

        let (mut ws_sender, mut ws_receiver) = ws_stream.split();
        let handle = handle.clone();
        let mut event_rx = handle.event_tx.subscribe();

        println!("[ipc] client connected");

        // Spawn a task for this client
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Incoming request from client
                    msg = ws_receiver.next() => {
                        match msg {
                            Some(Ok(Message::Text(text))) => {
                                if let Ok(ipc_msg) = serde_json::from_str::<IpcMessage>(&text) {
                                    eprintln!("[ipc] received: {}", ipc_msg.msg_type);
                                    let response = handle_request(&handle, &ipc_msg).await;
                                    let resp_json = serde_json::to_string(&response).unwrap_or_default();
                                    if ws_sender.send(Message::Text(resp_json.into())).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            Some(Ok(Message::Ping(data))) => {
                                let _ = ws_sender.send(Message::Pong(data)).await;
                            }
                            Some(Ok(Message::Close(_))) | None => {
                                println!("[ipc] client disconnected");
                                break;
                            }
                            _ => {}
                        }
                    }
                    // Broadcast event to client
                    event = event_rx.recv() => {
                        if let Ok(event_msg) = event {
                            let json = serde_json::to_string(&event_msg).unwrap_or_default();
                            if ws_sender.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            }
        });
    }
}

async fn handle_request(handle: &DaemonHandle, msg: &IpcMessage) -> IpcMessage {
    match msg.msg_type.as_str() {
        "status" => {
            let state = handle.state.lock().await;
            IpcMessage {
                msg_type: "status".into(),
                id: msg.id.clone(),
                ts: None,
                payload: Some(serde_json::json!({
                    "state": serde_json::to_value(&*state).unwrap_or_default(),
                })),
            }
        }
        "start_session" => {
            let on_start = handle.on_start.lock().await;
            if let Some(ref callback) = *on_start {
                match callback() {
                    Ok(()) => {
                        handle.set_state(DaemonState::Recording).await;
                        IpcMessage {
                            msg_type: "session_started".into(),
                            id: msg.id.clone(),
                            ts: None,
                            payload: None,
                        }
                    }
                    Err(e) => IpcMessage {
                        msg_type: "error".into(),
                        id: msg.id.clone(),
                        ts: None,
                        payload: Some(serde_json::json!({ "error": e.to_string() })),
                    },
                }
            } else {
                IpcMessage {
                    msg_type: "error".into(),
                    id: msg.id.clone(),
                    ts: None,
                    payload: Some(serde_json::json!({ "error": "no start handler registered" })),
                }
            }
        }
        "stop_session" => {
            let on_stop = handle.on_stop.lock().await;
            if let Some(ref callback) = *on_stop {
                match callback() {
                    Ok(text) => {
                        handle.set_state(DaemonState::ClipboardWritten).await;
                        handle
                            .broadcast_event(
                                "final_transcript",
                                serde_json::json!({ "text": text }),
                            )
                            .await;
                        IpcMessage {
                            msg_type: "session_stopped".into(),
                            id: msg.id.clone(),
                            ts: None,
                            payload: Some(serde_json::json!({ "text": text })),
                        }
                    }
                    Err(e) => IpcMessage {
                        msg_type: "error".into(),
                        id: msg.id.clone(),
                        ts: None,
                        payload: Some(serde_json::json!({ "error": e.to_string() })),
                    },
                }
            } else {
                IpcMessage {
                    msg_type: "error".into(),
                    id: msg.id.clone(),
                    ts: None,
                    payload: Some(serde_json::json!({ "error": "no stop handler registered" })),
                }
            }
        }
        "abort" => {
            let on_abort = handle.on_abort.lock().await;
            if let Some(ref callback) = *on_abort {
                callback();
                IpcMessage {
                    msg_type: "aborted".into(),
                    id: msg.id.clone(),
                    ts: None,
                    payload: None,
                }
            } else {
                IpcMessage {
                    msg_type: "error".into(),
                    id: msg.id.clone(),
                    ts: None,
                    payload: Some(serde_json::json!({ "error": "no abort handler registered" })),
                }
            }
        }
        _ => IpcMessage {
            msg_type: "error".into(),
            id: msg.id.clone(),
            ts: None,
            payload: Some(serde_json::json!({
                "error": format!("unknown request type: {}", msg.msg_type)
            })),
        },
    }
}
