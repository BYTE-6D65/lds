# LDS — Linguistic Dispatch System

Local speech-to-text daemon for Linux desktop. Vulkan GPU accelerated via whisper.cpp, dual-mode (batch + streaming), COSMIC panel applet with live config tuning.

## Architecture

```
Mic → PipeWire → cpal → VAD (Silero) → whisper-rs (Vulkan GPU)
                                          ↓
                              ┌──── batch ────┐  ┌──── streaming ────┐
                              │ record-stop-  │  │ rolling transcribe│
                              │ transcribe    │  │ + smooth typist   │
                              └───────────────┘  │ + text middleware │
                                                 └───────────────────┘
                                          ↓
                              clipboard + auto-type (wtype)
                                          ↓
                              UDS WebSocket IPC ←── COSMIC applet / ldsctl
```

## Features

- **Dual mode** — Batch (record → transcribe) or streaming (rolling transcription with smooth typing). Swap live from the applet, no restart.
- **Vulkan GPU** — AMD 9070 XT, ~7.3x realtime with large-v3-turbo. NVIDIA would be 20-50x.
- **Smooth typist** — Rate-controlled character emission via wtype. Characters flow at calculated pace, flush on stop.
- **Text middleware** — 6-pass regex cleanup: filler removal, repeated phrase collapse, ellipsis normalization, sentence boundaries, orphan noise filter.
- **Silero VAD** — GGML voice activity detection on CPU. No GPU conflict with whisper model.
- **Live config** — Min audio ms, VAD threshold, chunk interval, mode toggle — all adjustable from the COSMIC applet popup, no restart needed.
- **COSMIC panel applet** — Mic icon, click to toggle recording, status popup with settings sliders.
- **Hallucination filter** — Blocklist + suppress_blank + min 4 chars. Catches whisper's greatest hits.
- **Shared mic** — PipeWire via cpal's ALSA plugin. OBS, Discord, browser all work simultaneously.

## Build

```bash
# Daemon
cd lds
cargo build --release

# COSMIC applet
cd lds-cosmic-applet
cargo build --release
```

## Install

```bash
# Daemon
sudo cp target/release/lds /usr/bin/lds
sudo cp target/release/ldsctl /usr/bin/ldsctl

# Applet
sudo cp target/release/lds-cosmic-applet /usr/bin/lds-cosmic-applet

# Config
mkdir -p ~/.config/lds
# See Config section below

# Systemd (optional)
cp contrib/ldsd.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ldsd
```

## Usage

```bash
# Start daemon
ldsd daemon

# CLI
ldsctl status
ldsctl start
ldsctl stop

# Or use the COSMIC panel applet — click mic icon to toggle
```

## Config

`~/.config/lds/config.toml`:

```toml
model = "/path/to/ggml-large-v3-turbo-q5_0.bin"
socket = "/run/user/1000/ldsd.sock"
device = ""                    # empty = auto-detect PipeWire
auto_type = true               # auto-type via wtype
log_transcript = true
language = "en"
initial_prompt = ""

# Mode: "batch" or "streaming"
mode = "streaming"
vad_threshold = 0.5            # speech probability gate (0.0-1.0)
vad_min_silence_ms = 300       # silence before segment end
chunk_interval_ms = 500        # rolling pass cadence
partial_results = true         # IPC partial events
min_audio_ms = 1500            # minimum audio before first transcription
```

All streaming parameters are live-tunable via the COSMIC applet — no restart needed.

## Models

- **Whisper**: `ggml-large-v3-turbo-q5_0.bin` (547 MiB) from whisper.cpp
- **VAD**: `ggml-silero-v6.2.0.bin` (865 KiB) from [ggml-org/whisper-vad](https://github.com/ggml-org/whisper-vad)

VAD model expected at `~/.local/share/lds/`.

## Pipeline (streaming mode)

1. Audio captured at 16kHz via PipeWire/cpal
2. Silero VAD detects speech (CPU, no GPU conflict)
3. Audio buffered until `min_audio_ms` reached
4. Whisper transcribes buffer (Vulkan GPU)
5. Text middleware cleans output (regex, ~0ms)
6. Smooth typist emits characters at calculated rate
7. Buffer cleared, next pass starts immediately
8. On stop: final clipboard write, no double-type

## Credits

- [whisper-rs](https://github.com/tazz4843/whisper-rs) — Rust bindings for whisper.cpp
- [whisper-overlay](https://github.com/oddlama/whisper-overlay) — Original inspiration (Python server)
- [libcosmic](https://github.com/pop-os/libcosmic) — COSMIC DE toolkit

## License

MIT
