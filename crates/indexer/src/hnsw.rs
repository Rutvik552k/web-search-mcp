use dashmap::DashMap;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use web_search_common::{Error, Result};

/// In-memory HNSW-like vector index for approximate nearest neighbor search.
///
/// Simplified implementation using a flat index with partitioned search.
/// For production scale (>1M vectors), replace with `hnsw_rs` crate.
pub struct HnswIndex {
    /// All stored vectors: doc_id → embedding
    vectors: DashMap<String, Vec<f32>>,
    dim: usize,
}

#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub doc_id: String,
    pub score: f32, // cosine similarity
}

impl HnswIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: DashMap::new(),
            dim,
        }
    }

    /// Insert a vector for a document.
    pub fn insert(&self, doc_id: &str, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::Index(format!(
                "vector dim mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        self.vectors.insert(doc_id.to_string(), vector);
        Ok(())
    }

    /// Remove a vector.
    pub fn remove(&self, doc_id: &str) -> bool {
        self.vectors.remove(doc_id).is_some()
    }

    /// Search for top-K nearest neighbors by cosine similarity.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorSearchResult>> {
        if query.len() != self.dim {
            return Err(Error::Index(format!(
                "query dim mismatch: expected {}, got {}",
                self.dim,
                query.len()
            )));
        }

        // Min-heap: keeps top-K highest scores
        let mut heap: BinaryHeap<MinScored> = BinaryHeap::new();

        for entry in self.vectors.iter() {
            let score = cosine_similarity(query, entry.value());

            if heap.len() < top_k {
                heap.push(MinScored {
                    score,
                    doc_id: entry.key().clone(),
                });
            } else if let Some(min) = heap.peek() {
                if score > min.score {
                    heap.pop();
                    heap.push(MinScored {
                        score,
                        doc_id: entry.key().clone(),
                    });
                }
            }
        }

        // Extract sorted by score descending
        let mut results: Vec<VectorSearchResult> = heap
            .into_iter()
            .map(|ms| VectorSearchResult {
                doc_id: ms.doc_id,
                score: ms.score,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get a vector by doc_id.
    pub fn get(&self, doc_id: &str) -> Option<Vec<f32>> {
        self.vectors.get(doc_id).map(|v| v.value().clone())
    }

    /// Save the vector index to a file (bincode serialization).
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let entries: Vec<(String, Vec<f32>)> = self
            .vectors
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();

        let serialized = serde_json::to_vec(&serde_json::json!({
            "dim": self.dim,
            "vectors": entries,
        }))
        .map_err(|e| Error::Index(format!("serialize vectors: {e}")))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(path)?;
        file.write_all(&serialized)?;

        tracing::info!(vectors = entries.len(), path = %path.display(), "Vector index saved");
        Ok(())
    }

    /// Load a vector index from a file.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        if !path.exists() {
            return Err(Error::Index(format!("vector index not found: {}", path.display())));
        }

        let data = std::fs::read(path)?;
        let parsed: serde_json::Value = serde_json::from_slice(&data)
            .map_err(|e| Error::Index(format!("deserialize vectors: {e}")))?;

        let dim = parsed["dim"].as_u64().unwrap_or(384) as usize;
        let index = HnswIndex::new(dim);

        if let Some(vectors) = parsed["vectors"].as_array() {
            for entry in vectors {
                if let (Some(key), Some(vec_arr)) = (entry[0].as_str(), entry[1].as_array()) {
                    let vec: Vec<f32> = vec_arr
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    if vec.len() == dim {
                        index.vectors.insert(key.to_string(), vec);
                    }
                }
            }
        }

        tracing::info!(vectors = index.len(), path = %path.display(), "Vector index loaded");
        Ok(index)
    }

    /// Load from file if exists, otherwise create new empty index.
    pub fn open_or_create(path: &std::path::Path, dim: usize) -> Self {
        match Self::load(path) {
            Ok(idx) => idx,
            Err(_) => {
                tracing::info!("Creating new vector index");
                Self::new(dim)
            }
        }
    }
}

/// Helper for min-heap (smallest score at top).
struct MinScored {
    score: f32,
    doc_id: String,
}

impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for MinScored {}

impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order: smallest score first (min-heap)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn insert_and_search() {
        let idx = HnswIndex::new(3);
        idx.insert("a", normalized(&[1.0, 0.0, 0.0])).unwrap();
        idx.insert("b", normalized(&[0.0, 1.0, 0.0])).unwrap();
        idx.insert("c", normalized(&[0.9, 0.1, 0.0])).unwrap();

        let results = idx.search(&normalized(&[1.0, 0.0, 0.0]), 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "a"); // exact match
        assert_eq!(results[1].doc_id, "c"); // close match
    }

    #[test]
    fn search_empty_index() {
        let idx = HnswIndex::new(3);
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn top_k_limits_results() {
        let idx = HnswIndex::new(2);
        for i in 0..100 {
            let angle = (i as f32) * 0.01;
            idx.insert(
                &format!("doc_{i}"),
                normalized(&[angle.cos(), angle.sin()]),
            )
            .unwrap();
        }

        let results = idx.search(&normalized(&[1.0, 0.0]), 5).unwrap();
        assert_eq!(results.len(), 5);
        // Scores should be descending
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn remove_works() {
        let idx = HnswIndex::new(2);
        idx.insert("a", vec![1.0, 0.0]).unwrap();
        idx.insert("b", vec![0.0, 1.0]).unwrap();
        assert_eq!(idx.len(), 2);

        assert!(idx.remove("a"));
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[1.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "b");
    }

    #[test]
    fn dim_mismatch_errors() {
        let idx = HnswIndex::new(3);
        assert!(idx.insert("a", vec![1.0, 0.0]).is_err());
        assert!(idx.search(&[1.0, 0.0], 5).is_err());
    }
}
