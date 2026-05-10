use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio_tungstenite::tungstenite::Message;

#[derive(Parser)]
#[command(name = "ldsctl", about = "LDS daemon control CLI")]
struct Cli {
    /// Socket path
    #[arg(long, default_value = "/run/user/1000/ldsd.sock")]
    socket: String,

    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Query daemon status
    Status,
    /// Start recording session
    Start,
    /// Stop recording session
    Stop,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Cli::parse();

    let stream = tokio::net::UnixStream::connect(&args.socket).await?;
    let (ws, _response) = tokio_tungstenite::client_async("ws://localhost", stream)
        .await
        .map_err(|e| color_eyre::eyre::eyre!("WebSocket handshake failed: {}", e))?;

    let (mut ws_sender, mut ws_receiver) = ws.split();

    match args.command {
        Command::Status => {
            let msg = json!({"type": "status", "id": "1"});
            ws_sender
                .send(Message::Text(msg.to_string().into()))
                .await?;
            if let Some(Ok(resp)) = ws_receiver.next().await {
                println!("{}", resp);
            }
        }
        Command::Start => {
            let msg = json!({"type": "start_session", "id": "2"});
            ws_sender
                .send(Message::Text(msg.to_string().into()))
                .await?;
            if let Some(Ok(resp)) = ws_receiver.next().await {
                println!("{}", resp);
            }
        }
        Command::Stop => {
            let msg = json!({"type": "stop_session", "id": "3"});
            ws_sender
                .send(Message::Text(msg.to_string().into()))
                .await?;
            while let Some(Ok(resp)) = ws_receiver.next().await {
                println!("{}", resp);
                let text = resp.to_string();
                if text.contains("session_stopped") || text.contains("\"error\"") {
                    break;
                }
            }
        }
    }

    ws_sender.close().await.ok();
    Ok(())
}
