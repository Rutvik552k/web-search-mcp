use siphasher::sip::SipHasher;
use std::hash::{Hash, Hasher};

/// Generate a 64-bit SimHash fingerprint for near-duplicate detection.
///
/// Near-duplicates have hamming distance ≤ 3.
pub fn simhash(text: &str) -> u64 {
    let mut v = [0i32; 64];
    let words: Vec<&str> = text.split_whitespace().collect();

    // Generate shingles (3-grams of words)
    for window in words.windows(3) {
        let shingle = window.join(" ");
        let hash = hash_token(&shingle);

        for i in 0..64 {
            if (hash >> i) & 1 == 1 {
                v[i] += 1;
            } else {
                v[i] -= 1;
            }
        }
    }

    // Collapse to 64-bit fingerprint
    let mut fingerprint: u64 = 0;
    for (i, &count) in v.iter().enumerate() {
        if count > 0 {
            fingerprint |= 1 << i;
        }
    }
    fingerprint
}

/// Compute hamming distance between two SimHash fingerprints.
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Check if two documents are near-duplicates (hamming distance ≤ threshold).
pub fn is_near_duplicate(a: u64, b: u64, threshold: u32) -> bool {
    hamming_distance(a, b) <= threshold
}

fn hash_token(token: &str) -> u64 {
    let mut hasher = SipHasher::new();
    token.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_texts_have_zero_distance() {
        let text = "the quick brown fox jumps over the lazy dog";
        let a = simhash(text);
        let b = simhash(text);
        assert_eq!(hamming_distance(a, b), 0);
    }

    #[test]
    fn similar_texts_have_smaller_distance_than_different() {
        let a = simhash("the quick brown fox jumps over the lazy dog in the park near the river on a sunny day");
        let b = simhash("the quick brown fox leaps over the lazy dog in the park near the river on a sunny day");
        let c = simhash("quantum computing enables parallel processing of information using qubits and entanglement theory");
        // Similar texts should be closer to each other than to unrelated text
        let dist_similar = hamming_distance(a, b);
        let dist_different = hamming_distance(a, c);
        assert!(dist_similar < dist_different, "similar={dist_similar}, different={dist_different}");
    }

    #[test]
    fn different_texts_have_large_distance() {
        let a = simhash("the quick brown fox jumps over the lazy dog");
        let b = simhash("quantum computing enables parallel processing of information");
        assert!(hamming_distance(a, b) > 10);
    }
}
