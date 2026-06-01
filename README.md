# LDS — Linguistic Dispatch System

Local speech-to-text daemon for Linux desktop. Vulkan GPU accelerated via whisper.cpp, native Wayland virtual keyboard for reliable auto-typing, COSMIC panel applet with live config tuning.

## Architecture

```
Mic → PipeWire → cpal → whisper-rs (Vulkan GPU)
                              ↓
                    ┌──── batch mode ────┐
                    │ record-stop-       │
                    │ transcribe         │
                    └────────────────────┘
                              ↓
                    text middleware (hallucination filter)
                              ↓
                    native Wayland typist (zwp_virtual_keyboard_v1)
                    + clipboard (wl-copy / arboard)
                              ↓
                    UDS WebSocket IPC ←── COSMIC applet / ldsctl
```

## Features

- **Native Wayland typist** — Persistent `zwp_virtual_keyboard_v1` protocol connection. No process spawning, no per-character roundtrips. Characters are delivered at 200/sec via the Wayland virtual keyboard protocol — reliable across terminals and browsers.
- **Vulkan GPU** — AMD 9070 XT, ~7.3x realtime with large-v3-turbo.
- **Text middleware** — 6-pass regex cleanup: filler removal, repeated phrase collapse, ellipsis normalization, sentence boundaries, orphan noise filter. Catches Whisper's stagnant-air hallucinations ("thank you", "thanks for watching", lone noise words).
- **Batch mode** — Press to record, release to transcribe. Simple, reliable, correct.
- **Shared mic** — PipeWire via cpal's ALSA plugin. OBS, Discord, browser all work simultaneously.
- **Live config** — Language, auto-type toggle — adjustable from the COSMIC applet, no restart needed.
- **COSMIC panel applet** — Mic icon, click to toggle recording, status popup with settings.
- **Transcript log** — Timestamped transcripts saved to `~/.local/state/lds/transcripts/`, auto-pruned.

## Build

Requires: `cargo`, `cmake`, `clang`, `vulkan-headers`

```bash
cargo build --release --bin lds --bin ldsctl
```

For the COSMIC applet, see [`lds-cosmic-applet`](../lds-cosmic-applet/).

## Install

```bash
# Using the PKGBUILD (Arch Linux)
cd contrib
makepkg -sfC
sudo pacman -U lds-git-*.pkg.tar.zst

# Or manual
sudo cp target/release/lds /usr/bin/lds
sudo cp target/release/ldsctl /usr/bin/ldsctl

# Systemd
cp contrib/ldsd.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ldsd
```

## Usage

```bash
# Start daemon (typically via systemd)
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
auto_type = true               # auto-type via native Wayland typist
log_transcript = true
language = "en"
initial_prompt = ""
```

Run `lds init-config` to generate a template.

## Models

- **Whisper**: `ggml-large-v3-turbo-q5_0.bin` (547 MiB) from whisper.cpp

## Pipeline

1. Audio captured at 16kHz mono via PipeWire/cpal
2. Whisper transcribes full recording (Vulkan GPU)
3. Text middleware cleans output — hallucination suppression, filler removal, punctuation normalization
4. Transcript saved to disk, written to clipboard, auto-typed via native Wayland virtual keyboard
5. IPC broadcasts `final_transcript` event to connected clients

## Auto-typing

The daemon creates a persistent `zwp_virtual_keyboard_v1` at startup using the Wayland virtual keyboard protocol. Characters are converted to XKB keysyms via `xkbcommon`, a dynamic keymap is uploaded to the compositor, and key events are flushed per-character at ~200 chars/sec. Falls back to `wtype` if the native typist fails to initialize.

This replaced the previous approach of spawning a `wtype` process per transcript, which suffered from character drops under compositor load — especially in browsers — due to per-character Wayland roundtrips.

## Streaming (planned)

Streaming mode with rolling transcription, VAD-based segmentation, and live partial output is planned for a future release. The disabled modules (`streaming.rs`, `vad.rs`, `smooth_typist.rs`) are preserved in the source tree for reference.

## Credits

- [whisper-rs](https://github.com/tazz4843/whisper-rs) — Rust bindings for whisper.cpp
- [whisper-overlay](https://github.com/oddlama/whisper-overlay) — Original inspiration (Python server)
- [libcosmic](https://github.com/pop-os/libcosmic) — COSMIC DE toolkit
- [wtype](https://github.com/atx/wtype) — Reference implementation for the Wayland virtual keyboard approach

## License

MIT
