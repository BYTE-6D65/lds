use std::fs;
use std::path::PathBuf;

use chrono::Local;

const RETENTION_DAYS: u64 = 7;

fn transcript_dir() -> PathBuf {
    let base = std::env::var("XDG_STATE_HOME")
        .unwrap_or_else(|_| {
            let user = std::env::var("USER").unwrap_or_else(|_| "byte".into());
            format!("/home/{user}/.local/state")
        });
    PathBuf::from(format!("{base}/lds/transcripts"))
}

/// Save a transcript to a timestamped file. Returns the path written.
pub fn save_transcript(text: &str) -> Result<PathBuf, std::io::Error> {
    let dir = transcript_dir();
    fs::create_dir_all(&dir)?;

    let now = Local::now();
    let secs = now.timestamp().max(0) as u64;

    // Filename: YYYY-MM-DD_HH-MM-SS_unixepoch
    let datetime = now.format("%Y-%m-%d_%H-%M-%S");
    let filename = format!("{datetime}_{secs}.txt");
    let path = dir.join(&filename);

    fs::write(&path, text)?;

    // Prune old entries each time we write
    let _ = prune_old_transcripts();

    Ok(path)
}

/// Remove transcript files older than RETENTION_DAYS.
fn prune_old_transcripts() -> Result<usize, std::io::Error> {
    let dir = transcript_dir();
    if !dir.exists() {
        return Ok(0);
    }

    let cutoff = (Local::now()
        .timestamp()
        .max(0) as u64)
        .saturating_sub(RETENTION_DAYS * 86400);

    let mut removed = 0;
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map_or(true, |e| e != "txt") {
            continue;
        }

        // Parse the unix timestamp from the filename suffix (before .txt)
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        // stem format: "YYYY-MM-DD_HH-MM-SS_<unixepoch>"
        if let Some(epoch_str) = stem.rsplit('_').next() {
            if let Ok(epoch) = epoch_str.parse::<u64>() {
                if epoch < cutoff {
                    fs::remove_file(&path)?;
                    removed += 1;
                }
            }
        }
    }

    if removed > 0 {
        eprintln!("[lds] pruned {removed} old transcript(s)");
    }

    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_prune() {
        let dir = std::env::temp_dir().join("lds_test_transcripts");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.txt");
        fs::write(&path, "hello").unwrap();
        assert!(path.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
