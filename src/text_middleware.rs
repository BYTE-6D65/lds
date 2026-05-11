use fancy_regex::Regex;

/// Post-processing middleware between whisper output and typist.
///
/// Rule-based text cleanup — regex only, zero latency, zero memory.
/// Runs on CPU in microseconds. No LLM needed.
pub fn clean_text(raw: &str) -> String {
    let mut text = raw.to_string();

    // --- Pass 1: Remove filler words ---
    let fillers = Regex::new(
        r"(?i)(?:^|\s)(?:um+|uh+|ah+|hmm+|huh|like,?\s|you know,?\s|so,?\s|I mean,?\s)(?:\s|[.,!?]|$)"
    ).unwrap();
    text = fillers.replace_all(&text, " ").to_string();

    // --- Pass 2: Collapse repeated phrases ---
    // Repeated word: "word word" → single
    let repeated_word = Regex::new(r"(?i)\b(\w+[,.]?)\s+\1\b").unwrap();
    // fancy-regex supports backrefs but replace_all needs a different approach
    // Use a loop to avoid issues
    for _ in 0..3 {
        let before = text.clone();
        text = repeated_word.replace_all(&text, "$1").to_string();
        if text == before { break; }
    }

    // Repeated short phrases (2-4 words)
    let repeated_phrase = Regex::new(r"(?i)\b((?:\w+\s+){1,3}\w+[,.]?)\s+\1\b").unwrap();
    for _ in 0..3 {
        let before = text.clone();
        text = repeated_phrase.replace_all(&text, "$1").to_string();
        if text == before { break; }
    }

    // --- Pass 3: Normalize ellipses (3+ dots only) ---
    let trailing_ellipsis = regex::Regex::new(r"\.{3,}").unwrap();
    text = trailing_ellipsis.replace_all(&text, ",").to_string();

    // "---" or " - " → em dash
    let dash_noise = regex::Regex::new(r"\s*[-—]{2,}\s*").unwrap();
    text = dash_noise.replace_all(&text, " — ").to_string();

    // Missing space after period/exclamation/question (any letter, not just caps)
    let missing_space = regex::Regex::new(r"([.!?])([a-zA-Z])").unwrap();
    text = missing_space.replace_all(&text, "$1 $2").to_string();

    // Double punctuation ".." or "!!" → single
    let double_punct = regex::Regex::new(r"([.!?]){2,}").unwrap();
    text = double_punct.replace_all(&text, "$1").to_string();

    // Space before punctuation
    let space_before_punct = regex::Regex::new(r"\s+([.,!?;:])").unwrap();
    text = space_before_punct.replace_all(&text, "$1").to_string();

    // Capitalize first character
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        if first.is_ascii_lowercase() {
            text = format!("{}{}", first.to_uppercase(), chars.as_str());
        }
    }

    // Capitalize after ". " "? " "! "
    let capitalize_after = regex::Regex::new(r"([.!?]\s+)([a-z])").unwrap();
    text = capitalize_after
        .replace_all(&text, |caps: &regex::Captures| {
            format!("{}{}", &caps[1], caps[2].to_uppercase())
        })
        .to_string();

    // --- Pass 5: Strip orphan noise words ---
    let trimmed = text.trim();
    if trimmed.split_whitespace().count() == 1 {
        let noise_words = regex::Regex::new(
            r"(?i)^(?:okay|yeah|yes|no|um|uh|hmm|huh|oh|ah|so|and|but|or|the|a|an)$"
        ).unwrap();
        if noise_words.is_match(trimmed) {
            return String::new();
        }
    }

    // --- Pass 6: Final whitespace cleanup ---
    let multi_space = regex::Regex::new(r" {2,}").unwrap();
    text = multi_space.replace_all(&text, " ").to_string();
    text = text.trim().to_string();

    // Strip leading comma (artifact from filler removal)
    text = text.trim_start_matches(',').trim().to_string();

    // Ensure trailing period for complete sentences (4+ words)
    if !text.is_empty()
        && !text.ends_with(|c: char| c == '.' || c == '!' || c == '?' || c == ',' || c == ':')
    {
        let word_count = text.split_whitespace().count();
        if word_count >= 3 {
            text.push('.');
        }
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filler_removal() {
        let result = clean_text("Um, this is a test");
        assert!(result.contains("this is a test") || result.contains("This is a test"));
    }

    #[test]
    fn test_repeated_words() {
        assert_eq!(clean_text("Hello hello world"), "Hello world");
    }

    #[test]
    fn test_ellipsis_normalization() {
        assert_eq!(clean_text("I was thinking..."), "I was thinking,");
        assert_eq!(
            clean_text("what the fuck... seriously..."),
            "What the fuck, seriously,"
        );
    }

    #[test]
    fn test_single_word_noise() {
        assert_eq!(clean_text("okay"), "");
        assert_eq!(clean_text("Um"), "");
        assert_eq!(clean_text("Hello"), "Hello");
    }

    #[test]
    fn test_sentence_cleanup() {
        assert_eq!(
            clean_text("hello world.this is a test"),
            "Hello world. This is a test."
        );
    }

    #[test]
    fn test_double_punct() {
        assert_eq!(clean_text("Hello world.."), "Hello world.");
        assert_eq!(clean_text("What??"), "What?");
    }
}
