# LDS Middleware: Future Improvements

## Fuzzy hallucination matching with `strsim`

**Crate**: [`strsim`](https://crates.io/crates/strsim) (814M+ downloads, zero deps, pure Rust)

**Why**: The current middleware uses exact regex matching for hallucinated phrases
("thank you", "thanks for watching", etc.). Whisper misspells these frequently —
"thank yu", "thanks for wathing", "subcribe" — and exact matches miss them.

**API we'd use**:
- `strsim::jaro_winkler(a, b) -> f64` — similarity 0.0–1.0, rewards prefix matches
- `strsim::normalized_levenshtein(a, b) -> f64` — edit distance normalized 0.0–1.0
- `strsim::sorensen_dice(a, b) -> f64` — bigram overlap, good for short phrases

**Proposed approach**: Build a hallucination blocklist and fuzzy-match incoming
transcripts against it with a tunable threshold (~0.88 Jaro-Winkler). Short-circuit
with exact match first for performance.

```rust
const HALLUCINATIONS: &[&str] = &[
    "thank you", "thanks for watching", "thanks for listening",
    "thank you for watching", "subscribe", "like and subscribe",
    "see you next time", "bye bye", "goodbye", "the end",
];

fn is_hallucination(text: &str) -> bool {
    let lower = text.trim().to_lowercase();
    if HALLUCINATIONS.contains(&lower.as_str()) {
        return true;
    }
    for h in HALLUCINATIONS {
        if strsim::jaro_winkler(&lower, h) > 0.88 {
            return true;
        }
    }
    false
}
```

This would replace or augment Pass 5 (single-word noise filter) and could also
catch multi-word hallucinations that the current regex-only system misses.

**Status**: Not yet implemented. Waiting for streaming system redesign.
