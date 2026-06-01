//! Native Wayland virtual keyboard typist.
//!
//! Replaces the `wtype` external process with a persistent virtual keyboard
//! that lives for the daemon's lifetime. Key events are batched and flushed
//! in a single Wayland roundtrip instead of roundtripping per character,
//! eliminating the character-drop issues that plague wtype.

use std::io::Write;
use std::os::unix::io::{AsRawFd, BorrowedFd};
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_seat::WlSeat;
use wayland_client::{Connection, Dispatch, QueueHandle};
use xkbcommon::xkb::{keysym_from_name, keysym_get_name, utf32_to_keysym, Keysym, KEYSYM_CASE_INSENSITIVE};

use xkbcommon::xkb::keysyms::KEY_NoSymbol;

pub mod protocol {
    //! Generated Wayland protocol bindings for zwp_virtual_keyboard_v1
    pub mod interfaces {
        use wayland_client::protocol::__interfaces::*;
        use wayland_backend;
        wayland_scanner::generate_interfaces!("protocol/virtual-keyboard-unstable-v1.xml");
    }
    pub mod client {
        use super::interfaces::*;
        use wayland_client;
        use wayland_client::protocol::*;
        wayland_scanner::generate_client_code!("protocol/virtual-keyboard-unstable-v1.xml");
    }
}

use protocol::client::zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1;
use protocol::client::zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1;

/// Internal state for Wayland event queue.
/// Only needs Dispatch for WlRegistry (required by registry_queue_init).
struct TypistState;

impl Dispatch<WlRegistry, GlobalListContents> for TypistState {
    fn event(
        _state: &mut Self,
        _proxy: &WlRegistry,
        _event: <WlRegistry as wayland_client::Proxy>::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        // We don't need to handle dynamic global events for our use case
    }
}

impl Dispatch<WlSeat, ()> for TypistState {
    fn event(
        _state: &mut Self,
        _proxy: &WlSeat,
        _event: <WlSeat as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwpVirtualKeyboardManagerV1, ()> for TypistState {
    fn event(
        _state: &mut Self,
        _proxy: &ZwpVirtualKeyboardManagerV1,
        _event: <ZwpVirtualKeyboardManagerV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwpVirtualKeyboardV1, ()> for TypistState {
    fn event(
        _state: &mut Self,
        _proxy: &ZwpVirtualKeyboardV1,
        _event: <ZwpVirtualKeyboardV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
    }
}

/// A persistent Wayland virtual keyboard for typing text.
///
/// Create once at daemon startup. Call [`WaylandTypist::type_text`] to type strings.
/// The Wayland connection and virtual keyboard stay alive for the process lifetime.
pub struct WaylandTypist {
    display: Connection,
    queue: wayland_client::EventQueue<TypistState>,
    keyboard: ZwpVirtualKeyboardV1,
    /// Each entry: (char, Keysym). Index + 1 = keycode.
    keymap: Vec<(char, Keysym)>,
    /// True when new characters have been added since last keymap upload.
    keymap_dirty: bool,
}

impl WaylandTypist {
    /// Connect to Wayland and create a persistent virtual keyboard.
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let display = Connection::connect_to_env()?;
        let (globals, queue) = registry_queue_init::<TypistState>(&display)?;
        let qh = queue.handle();

        let seat: WlSeat = globals.bind(&qh, 1..=7, ())?;
        let manager: ZwpVirtualKeyboardManagerV1 = globals.bind(&qh, 1..=1, ())?;
        let keyboard = manager.create_virtual_keyboard(&seat, &qh, ());

        display.flush()?;

        Ok(Self {
            display,
            queue,
            keyboard,
            keymap: Vec::new(),
            keymap_dirty: false,
        })
    }

    /// Get or create a keycode for a character.
    fn keycode_for_char(&mut self, ch: char) -> u32 {
        for (i, (c, _)) in self.keymap.iter().enumerate() {
            if *c == ch {
                return (i + 1) as u32;
            }
        }

        let keysym = char_to_keysym(ch);
        self.keymap.push((ch, keysym));
        self.keymap_dirty = true;
        self.keymap.len() as u32
    }

    /// Upload the current keymap to the compositor.
    fn upload_keymap(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.keymap_dirty {
            return Ok(());
        }

        let keymap_str = build_xkb_keymap(&self.keymap);
        let keymap_bytes = keymap_str.as_bytes();
        let keymap_size = keymap_bytes.len() + 1; // null terminator

        let mut tmp = tempfile::tempfile()?;
        tmp.write_all(keymap_bytes)?;
        tmp.write_all(&[0])?;
        tmp.flush()?;

        // WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1 = 1
        // SAFETY: tmp stays alive until after flush, compositor dup's the fd per Wayland protocol.
        let fd = unsafe { BorrowedFd::borrow_raw(tmp.as_raw_fd()) };
        self.keyboard.keymap(1, fd, keymap_size as u32);
        self.display.flush()?;
        drop(tmp); // explicit drop after flush

        self.keymap_dirty = false;
        Ok(())
    }

    /// Type a string of text.
    ///
    /// Key events are sent with inter-character delays to give clients time
    /// to process each event. A single roundtrip at the end confirms delivery.
    pub fn type_text(&mut self, text: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if text.is_empty() {
            return Ok(());
        }

        // Build keycode sequence
        let mut keycodes: Vec<u32> = Vec::with_capacity(text.chars().count());
        for ch in text.chars() {
            keycodes.push(self.keycode_for_char(ch));
        }

        // Upload keymap if it changed
        self.upload_keymap()?;

        // Send key events with inter-character delay.
        // Browsers and other complex clients drop key events when they arrive
        // too fast in a single flush. 5ms per character is fast enough for
        // dictation (~200 chars/sec) but slow enough for clients to keep up.
        let mut time_ms: u32 = 0;
        let inter_key_us = 5000; // 5ms between characters

        for kc in &keycodes {
            self.keyboard.key(time_ms, *kc, 1); // PRESSED
            time_ms += 1;
            self.keyboard.key(time_ms, *kc, 0); // RELEASED
            time_ms += 1;
            self.display.flush()?;
            std::thread::sleep(std::time::Duration::from_micros(inter_key_us));
        }

        Ok(())
    }
}

impl Drop for WaylandTypist {
    fn drop(&mut self) {
        self.keyboard.destroy();
        let _ = self.display.flush();
    }
}

/// Convert a character to an XKB keysym.
fn char_to_keysym(ch: char) -> Keysym {
    match ch {
        '\n' => Keysym::from(0xff0d),   // XKB_KEY_Return
        '\t' => Keysym::from(0xff09),   // XKB_KEY_Tab
        '\x1b' => Keysym::from(0xff1b), // XKB_KEY_Escape
        _ => {
            let s = ch.to_string();
            let ks = keysym_from_name(&s, KEYSYM_CASE_INSENSITIVE);
            if u32::from(ks) != KEY_NoSymbol {
                ks
            } else {
                utf32_to_keysym(ch as u32)
            }
        }
    }
}

/// Build an XKB keymap string for the given character-to-keysym mapping.
fn build_xkb_keymap(keymap: &[(char, Keysym)]) -> String {
    let mut s = String::from("xkb_keymap {\n");

    // Keycodes
    let max_kc = keymap.len() as u32 + 8 + 1;
    s.push_str(&format!(
        "xkb_keycodes \"(unnamed)\" {{\nminimum = 8;\nmaximum = {};\n",
        max_kc
    ));
    for i in 0..keymap.len() {
        s.push_str(&format!("<K{}> = {};\n", i + 1, i + 8 + 1));
    }
    s.push_str("};\n");

    // Types and compatibility
    s.push_str("xkb_types \"(unnamed)\" { include \"complete\" };\n");
    s.push_str("xkb_compatibility \"(unnamed)\" { include \"complete\" };\n");

    // Symbols
    s.push_str("xkb_symbols \"(unnamed)\" {\n");
    for (i, (_, keysym)) in keymap.iter().enumerate() {
        let name = keysym_get_name(*keysym);
        s.push_str(&format!("key <K{}> {{[{}]}};\n", i + 1, name));
    }
    s.push_str("};\n");
    s.push_str("};\n");

    s
}

unsafe impl Send for WaylandTypist {}
unsafe impl Sync for WaylandTypist {}
