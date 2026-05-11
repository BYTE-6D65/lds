# LDS Architecture: Streaming STT & Multi-Model Support

**Branch:** `feat/streaming-and-models`
**Status:** Planning

---

## Part 1: Streaming Transcription

### Current Architecture (Batch Mode)

```
User presses Start
  → AudioCapture starts buffering f32 samples
User presses Stop
  → AudioCapture returns entire buffer
  → WhisperProvider::transcribe(&samples) — BLOCKS until complete
  → Final text → clipboard + auto-type + IPC event
```

**Problems:**
1. No feedback during recording — user sees "Recording" but has no idea what's being captured
2. Long recordings block the daemon — a 30s clip takes ~4s of GPU time where the daemon is stuck in `provider.lock().unwrap()`
3. No partial results — if the user stops too early or too late, they get nothing until the full transcription finishes
4. No VAD — silence is transcribed as garbage or empty segments

### Target Architecture (Streaming Mode)

```
User presses Start
  → AudioCapture runs continuously
  → VAD (whisper.cpp built-in) processes audio in chunks
  → On speech detected: feed chunk to whisper with new_segment_callback
  → Each partial segment → IPC "partial_transcript" event
  → Client sees live text appearing as user speaks
User presses Stop (or VAD detects end of speech)
  → Final assembled text → clipboard + auto-type + IPC "final_transcript" event
```

### Available whisper-rs APIs for Streaming

whisper-rs 0.16 exposes everything needed:

| API | Purpose |
|-----|---------|
| `WhisperVadContext` | Built-in VAD. Methods: `detect_speech`, `segments_from_samples`, `probabilities`. Feed raw audio, get speech/non-speech segments back. |
| `WhisperVadParams` | VAD tuning: threshold, min speech duration, min silence duration, window size, max speech duration. |
| `FullParams::set_new_segment_callback` | Callback fires every time whisper produces a new segment during `full()`. This IS the streaming hook. |
| `FullParams::set_new_segment_callback_safe` | Same but without raw pointers — safer Rust API. |
| `FullParams::set_segment_callback_safe_lossy` | Segment callback that may skip segments under load — useful for real-time UI where stale data is worse than missing a segment. |
| `FullParams::enable_vad` | Enable whisper.cpp's built-in VAD filtering on the transcription itself (separate from WhisperVadContext). |
| `FullParams::set_vad_model_path` | Path to VAD model (silero_vad.onnx typically). |
| `FullParams::set_vad_params` | Pass WhisperVadParams to transcription. |
| `FullParams::set_single_segment` | Force single-segment output — useful for dictation mode. |
| `FullParams::set_split_on_word` | Split segments on word boundaries instead of arbitrary token boundaries. Better for live display. |
| `WhisperState::full` | The main transcription call. With segment callbacks, this becomes a streaming source. |
| `FullParams::set_abort_callback_safe` | Allow cancelling in-flight transcription — critical for "user pressed stop during processing." |

### Architecture Change

```
src/
  audio_capture.rs     — unchanged (buffer management)
  whisper_provider.rs  → src/provider/
      mod.rs           — trait SttProvider
      whisper.rs       — existing WhisperProvider (batch mode, kept as fallback)
      streaming.rs     — NEW: StreamingWhisperProvider
  vad.rs              — NEW: VAD pipeline (silero via whisper.cpp built-in)
  streaming.rs        — NEW: streaming coordinator (audio → VAD → whisper → IPC)
  ipc.rs              — extended with partial_transcript events
  main.rs             — new daemon mode: "streaming" vs "batch"
```

### Streaming Coordinator Design

```rust
// src/streaming.rs

/// Runs the streaming transcription loop.
/// Spawns two tasks:
///   1. Audio chunk feeder — reads from AudioCapture in ~2s windows
///   2. VAD + transcriber — processes chunks, fires callbacks
///
/// Uses FullParams::set_new_segment_callback to emit partial results
/// via the existing IPC broadcast channel.

async fn run_streaming_loop(
    capture: Arc<AudioCapture>,
    provider: Arc<Mutex<WhisperProvider>>,
    handle: Arc<DaemonHandle>,
    config: StreamingConfig,
) {
    let mut ring_buffer: Vec<f32> = Vec::new();
    let chunk_size = 16000 * 2; // 2 seconds at 16kHz

    loop {
        // Collect samples from capture buffer
        let samples = capture.drain_buffer();
        ring_buffer.extend_from_slice(&samples);

        // VAD check: is there speech in the buffer?
        if ring_buffer.len() >= chunk_size {
            let vad_result = vad_detect(&ring_buffer);
            if vad_result.has_speech() {
                // Transcribe with streaming callbacks
                let partial = provider.lock().unwrap()
                    .transcribe_streaming(&ring_buffer, |segment_text| {
                        // Fire IPC event for each partial segment
                        handle.broadcast_event("partial_transcript", json!({
                            "text": segment_text,
                            "is_final": false,
                        }));
                    });
                // Reset buffer after transcription
                ring_buffer.clear();
            } else {
                // No speech — trim old audio to prevent unbounded growth
                let keep = 16000; // keep last 1s as context
                if ring_buffer.len() > keep {
                    ring_buffer.drain(..ring_buffer.len() - keep);
                }
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
```

### IPC Changes

New IPC events broadcast to all connected clients:

```jsonc
// Partial transcript (streaming)
{
  "type": "partial_transcript",
  "payload": {
    "text": "hello world",
    "segment_index": 3,
    "is_final": false,
    "confidence": 0.92
  }
}

// Final transcript (existing, unchanged)
{
  "type": "final_transcript",
  "payload": {
    "text": "Hello world, this is the complete transcript.",
    "is_final": true
  }
}

// VAD status (new)
{
  "type": "vad_status",
  "payload": {
    "speech_detected": true,
    "probability": 0.87
  }
}
```

### AudioCapture Extension

Current `AudioCapture` doesn't support draining the buffer without stopping. Need:

```rust
// New method on AudioCapture
/// Drain accumulated samples without stopping recording.
/// Returns what's accumulated since last drain, keeps recording.
pub fn drain_buffer(&self) -> Vec<f32> {
    std::mem::take(&mut *self.buffer.lock().unwrap())
}
```

### Config Additions

```toml
# Streaming mode
mode = "streaming"           # "batch" (current) or "streaming"
vad_threshold = 0.5          # VAD speech probability threshold
vad_min_silence_ms = 500     # minimum silence to trigger segment end
chunk_interval_ms = 2000     # how often to process audio chunks
partial_results = true       # send partial_transcript IPC events
```

### Implementation Phases

**Phase 1: AudioCapture drain + buffer management**
- Add `drain_buffer()` method
- Add `buffer_len()` (already exists)
- No behavioral change for batch mode

**Phase 2: VAD pipeline**
- Create `vad.rs` wrapping `WhisperVadContext`
- Test speech detection accuracy with existing model
- Tune threshold params for the 9070 XT setup

**Phase 3: StreamingWhisperProvider**
- New `transcribe_streaming()` method using `set_new_segment_callback`
- Abort callback support for "stop" during transcription
- Wire partial segments to IPC broadcast

**Phase 4: Streaming coordinator**
- Wire VAD → streaming whisper → IPC in a loop
- Ring buffer management with context window
- Handle edge cases: silence trimming, buffer overflow, very long speech

**Phase 5: Daemon mode switching**
- Config `mode = "streaming" | "batch"`
- Default to batch (backward compat)
- Streaming mode activates the coordinator loop

---

## Part 2: Model Support (Beyond Whisper)

### Current State

LDS is hardcoded to whisper.cpp via `whisper-rs`. Only supports whisper GGML models (`.bin` format). Currently using `ggml-large-v3-turbo-q5_0.bin` (547 MiB).

### The Question: Can We Support Other STT Models?

whisper.cpp only supports Whisper-family models. To support other architectures, we'd need different backends.

### Option Analysis

| Model Family | Backend | Rust Crate? | Vulkan? | Quality vs Whisper | Effort |
|-------------|---------|-------------|---------|-------------------|--------|
| **Whisper** (current) | whisper.cpp | whisper-rs ✅ | ✅ | Baseline | Done |
| **Whisper V3 Turbo** | whisper.cpp | whisper-rs ✅ | ✅ | Faster, same quality | Just swap model file |
| **Silero VAD** | ONNX Runtime | ort ✅ | Via CUDA/Vulkan | VAD only, not STT | Medium |
| **Moonshine** (usefulsense) | ONNX Runtime | ort ✅ | Via EP | Small, fast, English-only | High |
| **Paraformer** (FunASR) | ONNX Runtime | ort ✅ | Via EP | Chinese+English | High |
| **SeamlessM4T** | fairseq2 | ❌ No Rust | ❌ | Multilingual | Not viable |
| **faster-whisper** (CTranslate2) | CTranslate2 | ❌ No Rust | ❌ | 4x faster whisper | Not viable |
| **Whisper MLX** | Apple MLX | ❌ | ❌ | Apple only | Not viable |

### Recommendation: Stay on whisper.cpp + Add ONNX for Non-Whisper

**Keep whisper-rs as the primary backend.** It's already working, Vulkan GPU is fast, the API is stable. Whisper large-v3-turbo is good enough for English dictation.

**Add an ONNX Runtime backend** (`ort` crate) as an alternative provider. This opens up:
- **Moonshine** — tiny ONNX model (base: 200M params), designed for edge/real-time. Could be faster than whisper for streaming.
- **Silero VAD** — proper standalone VAD (the current `WhisperVadContext` uses silero internally, but direct ONNX gives more control)
- **Paraformer** — if you ever need Chinese/English mixed

### Provider Trait Design

```rust
// src/provider/mod.rs

/// Pluggable STT backend
pub trait SttProvider: Send + Sync {
    /// Load model from path
    fn new(model_path: &str) -> Result<Self> where Self: Sized;

    /// Batch transcription (current behavior)
    fn transcribe(&self, audio: &[f32]) -> Result<String>;

    /// Streaming transcription with partial callback
    fn transcribe_streaming(
        &self,
        audio: &[f32],
        on_segment: &dyn Fn(&str),
    ) -> Result<String>;

    /// Provider name for logging
    fn name(&self) -> &str;
}
```

Then:
- `WhisperProvider` implements `SttProvider` (existing code, add `transcribe_streaming`)
- `OnnxProvider` implements `SttProvider` (new, for Moonshine/Paraformer)
- `OrtProvider` or `MoonshineProvider` wraps the ONNX model

### ONNX Runtime Vulkan Status

The `ort` crate supports:
- **CUDA** execution provider
- **CPU** execution provider (always works)
- **Vulkan** — NOT directly supported as an EP. ONNX Runtime doesn't have a Vulkan EP.
- **ROCm** — supported as an EP for AMD GPUs

For your 9070 XT, the realistic paths are:
1. **ROCm EP** via `ort` — if ROCm is installed, ONNX Runtime can use it
2. **CPU** via `ort` — for small models like Moonshine base, CPU might be fast enough
3. **Stay on whisper.cpp Vulkan** — already working, already fast

### Practical Verdict

**For now, the best ROI is staying on whisper.cpp and optimizing the streaming pipeline.** The model quality is fine, Vulkan is working at 7.3x realtime, and the API supports everything needed for streaming (VAD, segment callbacks, abort).

Switching to ONNX models is a future option if:
- You need non-English languages (Paraformer)
- You want smaller/faster models for embedded use (Moonshine)
- You want to decouple from whisper.cpp's GGML format

The `SttProvider` trait should be extracted now (it's a clean refactor), but adding ONNX backends is Phase 2+ work.

---

## Summary of Changes

### What to Build

1. **Streaming transcription** — VAD + segment callbacks + partial IPC events
2. **SttProvider trait** — abstract the whisper dependency for future models
3. **AudioCapture drain** — non-destructive buffer reading for streaming
4. **Config `mode` switch** — batch (default) vs streaming

### What NOT to Build (Yet)

- ONNX Runtime integration
- Moonshine/Paraformer model support
- Multi-language model switching

### Model Compatibility (whisper.cpp GGML)

All these work with the current `whisper-rs` setup, just swap the file:

| Model | Size | Speed (9070 XT) | Quality |
|-------|------|-----------------|---------|
| `large-v3-turbo-q5_0` (current) | 547M | ~7.3x realtime | Excellent |
| `large-v3-turbo-q8_0` | ~900M | ~5x realtime | Slightly better |
| `medium-q5_0` | ~150M | ~15x realtime | Good |
| `small-q5_0` | ~50M | ~25x realtime | OK for English |
| `tiny-q5_0` | ~15M | ~50x realtime | Usable for quick commands |
| `base-q5_0` | ~30M | ~35x realtime | Decent for English |

For streaming, `medium` or `small` might be the sweet spot — fast enough for near-real-time with good enough quality for dictation.
