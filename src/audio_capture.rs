use color_eyre::eyre::{Context, ContextCompat, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

/// Captures audio from PipeWire via cpal's ALSA plugin.
/// Requests mono 16kHz f32 — PipeWire's ALSA plugin handles resampling.
pub struct AudioCapture {
    stream: cpal::Stream,
    buffer: Arc<Mutex<Vec<f32>>>,
    recording: Arc<Mutex<bool>>,
}

impl AudioCapture {
    /// Create with auto-detected device (prefers "pipewire" for shared mic).
    pub fn new() -> Result<Self> {
        Self::new_with_device("")
    }

    /// Create with a specific device name. Empty string = auto-detect.
    pub fn new_with_device(device_name: &str) -> Result<Self> {
        let host = cpal::default_host();

        let device = if device_name.is_empty() {
            // Prefer the "pipewire" device for shared mic access
            host.input_devices()
                .ok()
                .and_then(|mut devs| {
                    devs.find(|d| d.name().map(|n| n == "pipewire").unwrap_or(false))
                })
                .or_else(|| host.default_input_device())
                .with_context(|| "no audio input device available")?
        } else {
            host.input_devices()
                .ok()
                .and_then(|mut devs| {
                    devs.find(|d| d.name().map(|n| n == device_name).unwrap_or(false))
                })
                .or_else(|| host.default_input_device())
                .with_context(|| format!("device '{}' not found", device_name))?
        };

        println!(
            "[audio] input device: {}",
            device.name().unwrap_or_default()
        );

        // Find a supported config: 1ch, 16kHz, f32
        let supported_config = device
            .supported_input_configs()
            .with_context(|| "could not query supported configs")?
            .find(|c| {
                c.channels() == 1
                    && c.min_sample_rate().0 <= 16000
                    && c.max_sample_rate().0 >= 16000
                    && matches!(c.sample_format(), cpal::SampleFormat::F32)
            })
            .with_context(|| "no supported 1ch 16kHz f32 config")?
            .with_sample_rate(cpal::SampleRate(16000));

        let config = supported_config.config();
        println!(
            "[audio] config: {}ch @ {}Hz {:?}",
            config.channels, config.sample_rate.0, supported_config.sample_format()
        );

        let buffer: Arc<Mutex<Vec<f32>>> =
            Arc::new(Mutex::new(Vec::with_capacity(16000 * 60)));
        let recording = Arc::new(Mutex::new(false));
        let buf_clone = buffer.clone();
        let rec_clone = recording.clone();

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let is_recording = *rec_clone.lock().unwrap();
                    if !is_recording {
                        return;
                    }
                    buf_clone.lock().unwrap().extend_from_slice(data);
                },
                |err| eprintln!("[audio] capture error: {}", err),
                None,
            )
            .with_context(|| "failed to build audio input stream")?;

        stream
            .play()
            .with_context(|| "failed to start audio stream")?;

        Ok(Self {
            stream,
            buffer,
            recording,
        })
    }

    /// Start recording. Clears the buffer.
    pub fn start(&self) {
        self.buffer.lock().unwrap().clear();
        *self.recording.lock().unwrap() = true;
        println!("[audio] recording started");
    }

    /// Stop recording and return the captured samples.
    pub fn stop(&self) -> Vec<f32> {
        *self.recording.lock().unwrap() = false;
        let samples = std::mem::take(&mut *self.buffer.lock().unwrap());
        let duration = samples.len() as f32 / 16000.0;
        println!(
            "[audio] recording stopped: {:.1}s ({} samples)",
            duration,
            samples.len()
        );
        samples
    }

    /// Check if currently recording.
    pub fn is_recording(&self) -> bool {
        *self.recording.lock().unwrap()
    }

    /// Get current buffer length in samples without stopping.
    pub fn buffer_len(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }

    /// Drain accumulated samples without stopping recording.
    /// Returns what's accumulated since last drain, keeps recording.
    pub fn drain_buffer(&self) -> Vec<f32> {
        std::mem::take(&mut *self.buffer.lock().unwrap())
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.stream.pause().ok();
    }
}
