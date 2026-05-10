# LDS — Liam's Dictation Service

Local speech-to-text daemon for Linux desktop. Vulkan GPU accelerated via whisper.cpp, shared mic via PipeWire, clipboard-first delivery, COSMIC panel applet.

## Architecture

```
Mic → PipeWire → cpal → whisper-rs (Vulkan GPU) → clipboard + auto-type
                   ↑                                   ↓
              UDS WebSocket IPC ←──────────── ldsctl / COSMIC applet
                                                   ↓
                                          AI-tubing adapter (Hermes)
                                                   ↓
                                         WebSocket → Mac Mini (TTS + Godot VRM)
```

## Features

- **Vulkan GPU** — AMD 9070 XT, 7.3x realtime in debug builds (faster in release)
- **Shared mic** — Uses PipeWire via cpal's ALSA plugin. OBS, Discord, browser all work simultaneously
- **Clipboard-first delivery** — Transcript goes to clipboard. Auto-type via enigo is best-effort bonus
- **UDS WebSocket IPC** — `ldsctl` CLI and COSMIC applet communicate over Unix domain socket
- **AI-tubing integration** — Hermes adapter consumes transcripts from ldsd, sends to Mac Mini for TTS + VRM
- **COSMIC panel applet** — Mic icon, click to toggle recording, status popup

## Build

```bash
# LDS daemon
cd lds
cargo build --release

# COSMIC applet (requires rustc 1.93+)
cd ../lds-cosmic-applet
cargo build --release
```

## Install

```bash
# Daemon
cp target/release/lds ~/.local/bin/
cp target/release/ldsctl ~/.local/bin/

# Config
mkdir -p ~/.config/lds
lds init-config            # writes template to ~/.config/lds/config.toml
# Edit config.toml — set model path

# Systemd (optional)
cp contrib/ldsd.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ldsd
```

## Usage

```bash
# CLI
ldsd daemon --model /path/to/ggml-large-v3-turbo-q5_0.bin
ldsctl status
ldsctl start
ldsctl stop

# Or use the COSMIC panel applet — click mic icon, hit Start/Stop
```

## Config

`~/.config/lds/config.toml`:

```toml
model = "/path/to/ggml-large-v3-turbo-q5_0.bin"
socket = "/run/user/1000/ldsd.sock"
device = ""                    # empty = auto-detect PipeWire
clipboard = true               # write transcript to clipboard
auto_type = true               # auto-type via enigo (best-effort)
log_transcript = false         # don't log transcript text (privacy)
language = ""                  # empty = auto-detect
initial_prompt = ""            # whisper prompt hint
```

CLI args override config file values.

## Model

Tested with `ggml-large-v3-turbo-q5_0.bin` (547 MiB) from whisper.cpp. Place in `~/Projects/AI-tubing/models/` or configure path.

## Related

- [lds-cosmic-applet](https://github.com/BYTE-6D65/lds-cosmic-applet) — COSMIC panel applet
- [whisper-rs](https://github.com/tazz4843/whisper-rs) — Rust bindings for whisper.cpp
- [whisper-overlay](https://github.com/oddlama/whisper-overlay) — Original fork base (uses Python server)

## License

MIT
