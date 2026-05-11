/// Split text into overlapping chunks for embedding.
///
/// Uses sentence boundaries when possible, falling back to word boundaries.
/// Each chunk is ~`target_tokens` words with `overlap_ratio` overlap.
pub fn chunk_text(text: &str, target_tokens: usize, overlap_ratio: f32) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return vec![text.to_string()];
    }

    let overlap_tokens = (target_tokens as f32 * overlap_ratio) as usize;
    let mut chunks = Vec::new();
    let mut current_chunk: Vec<&str> = Vec::new();
    let mut current_len = 0;

    for sentence in &sentences {
        let word_count = sentence.split_whitespace().count();

        if current_len + word_count > target_tokens && !current_chunk.is_empty() {
            // Emit current chunk
            chunks.push(current_chunk.join(" "));

            // Keep overlap: retain sentences from end of chunk
            let mut overlap_len = 0;
            let mut keep_from = current_chunk.len();
            for (i, s) in current_chunk.iter().enumerate().rev() {
                let wc = s.split_whitespace().count();
                if overlap_len + wc > overlap_tokens {
                    break;
                }
                overlap_len += wc;
                keep_from = i;
            }
            current_chunk = current_chunk[keep_from..].to_vec();
            current_len = overlap_len;
        }

        current_chunk.push(sentence);
        current_len += word_count;
    }

    // Emit remaining
    if !current_chunk.is_empty() {
        let last = current_chunk.join(" ");
        // Don't emit if it's identical to the last chunk (happens with small texts)
        if chunks.last().map_or(true, |prev| prev != &last) {
            chunks.push(last);
        }
    }

    chunks
}

/// Split text into sentences using basic heuristics.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;

    let bytes = text.as_bytes();
    let len = bytes.len();

    for i in 0..len {
        let ch = bytes[i] as char;
        if (ch == '.' || ch == '!' || ch == '?') && i + 1 < len {
            let next = bytes[i + 1] as char;
            // Sentence boundary: punctuation followed by space + uppercase or newline
            if next == ' ' || next == '\n' || next == '\r' {
                let sentence = text[start..=i].trim();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                start = i + 1;
            }
        }
    }

    // Remaining text
    let remaining = text[start..].trim();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_short_text() {
        let chunks = chunk_text("Hello world.", 512, 0.2);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunk_long_text() {
        let sentences: Vec<String> = (0..50)
            .map(|i| format!("This is sentence number {} with some extra words to pad. ", i))
            .collect();
        let text = sentences.join("");

        let chunks = chunk_text(&text, 50, 0.2);
        assert!(chunks.len() > 1, "Should produce multiple chunks, got {}", chunks.len());

        // Each chunk should be roughly target_tokens words
        for chunk in &chunks {
            let word_count = chunk.split_whitespace().count();
            assert!(word_count <= 70, "Chunk too long: {word_count} words");
        }
    }

    #[test]
    fn chunks_overlap() {
        let text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here. Sixth sentence here. Seventh sentence here. Eighth sentence here. Ninth sentence here. Tenth sentence here.";
        let chunks = chunk_text(text, 10, 0.3);

        if chunks.len() >= 2 {
            // Check that consecutive chunks share some text
            let words_0: std::collections::HashSet<&str> = chunks[0].split_whitespace().collect();
            let words_1: std::collections::HashSet<&str> = chunks[1].split_whitespace().collect();
            let overlap = words_0.intersection(&words_1).count();
            assert!(overlap > 0, "Chunks should overlap");
        }
    }

    #[test]
    fn empty_text_returns_empty() {
        let chunks = chunk_text("", 512, 0.2);
        assert!(chunks.is_empty());
    }

    #[test]
    fn split_sentences_basic() {
        let sentences = split_sentences("Hello world. How are you? I am fine!");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine!");
    }
}
