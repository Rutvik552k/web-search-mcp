
use crate::engine::SearchEngine;

/// Dispatch an MCP tool call to the appropriate engine method.
///
/// Returns JSON string result.
pub async fn dispatch_tool(
    engine: &SearchEngine,
    tool_name: &str,
    args: &serde_json::Map<String, serde_json::Value>,
) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
    match tool_name {
        // ── Smart Tools ──────────────────────────────────────────
        "deep_research" => {
            let query = get_str(args, "query")?;
            let max_pages = get_u64(args, "max_pages").unwrap_or(500) as usize;
            let depth = get_u64(args, "depth").unwrap_or(3) as u8;
            let time_limit = get_u64(args, "time_limit_secs").unwrap_or(120);

            let response = engine.deep_research(&query, max_pages, depth, time_limit).await?;
            Ok(serde_json::to_string_pretty(&response)?)
        }

        "quick_search" => {
            let query = get_str(args, "query")?;
            let max_results = get_u64(args, "max_results").unwrap_or(10) as usize;

            let response = engine.quick_search(&query, max_results).await?;
            Ok(serde_json::to_string_pretty(&response)?)
        }

        "explore_topic" => {
            let topic = get_str(args, "topic")?;
            // explore_topic uses deep_research with wide breadth
            let response = engine.deep_research(&topic, 200, 3, 60).await?;
            Ok(serde_json::to_string_pretty(&response)?)
        }

        "verify_claim" => {
            let claim = get_str(args, "claim")?;
            let min_sources = get_u64(args, "min_sources").unwrap_or(5) as usize;

            let response = engine.verify_claim(&claim, min_sources).await?;
            Ok(serde_json::to_string_pretty(&response)?)
        }

        "compare_sources" => {
            let urls = get_str_array(args, "urls")?;
            let aspect = get_str(args, "aspect")?;

            let mut extractions = Vec::new();
            for url in &urls {
                match engine.extract(url).await {
                    Ok(json) => extractions.push(serde_json::json!({
                        "url": url,
                        "extraction": serde_json::from_str::<serde_json::Value>(&json).unwrap_or_default(),
                    })),
                    Err(e) => extractions.push(serde_json::json!({
                        "url": url,
                        "error": e.to_string(),
                    })),
                }
            }

            Ok(serde_json::to_string_pretty(&serde_json::json!({
                "aspect": aspect,
                "sources": extractions,
            }))?)
        }

        // ── Atomic Tools ─────────────────────────────────────────
        "fetch_page" => {
            let url = get_str(args, "url")?;
            engine.fetch_page(&url).await.map_err(Into::into)
        }

        "extract" => {
            let url = get_str(args, "url")?;
            engine.extract(&url).await.map_err(Into::into)
        }

        "follow_links" => {
            let url = get_str(args, "url")?;
            let pattern = args.get("pattern").and_then(|v| v.as_str()).map(|s| s.to_string());
            let depth = get_u64(args, "depth").unwrap_or(1) as u8;

            engine.follow_links(&url, pattern.as_deref(), depth).await.map_err(Into::into)
        }

        "paginate" => {
            let url = get_str(args, "url")?;
            let max_pages = get_u64(args, "max_pages").unwrap_or(10) as u32;

            engine.paginate(&url, max_pages).await.map_err(Into::into)
        }

        "search_index" => {
            let query = get_str(args, "query")?;
            let max_results = get_u64(args, "max_results").unwrap_or(20) as usize;

            engine.search_index(&query, max_results).await.map_err(Into::into)
        }

        "find_similar" => {
            let text = get_str(args, "text")?;
            let top_k = get_u64(args, "top_k").unwrap_or(10) as usize;

            engine.find_similar(&text, top_k).await.map_err(Into::into)
        }

        "get_entities" => {
            let text = get_str(args, "text")?;
            // Basic NER placeholder: extract capitalized multi-word phrases
            let entities = extract_basic_entities(&text);
            Ok(serde_json::to_string_pretty(&serde_json::json!({
                "entities": entities,
            }))?)
        }

        "get_link_graph" => {
            let url = get_str(args, "url")?;
            let depth = get_u64(args, "depth").unwrap_or(1) as u8;

            engine.get_link_graph(&url, depth).await.map_err(Into::into)
        }

        _ => Err(format!("Unknown tool: {tool_name}").into()),
    }
}

// ── Argument helpers ─────────────────────────────────────────────

fn get_str(
    args: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("Missing required parameter: {key}").into())
}

fn get_u64(
    args: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64())
}

fn get_str_array(
    args: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> std::result::Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    args.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .ok_or_else(|| format!("Missing required array parameter: {key}").into())
}

/// Basic entity extraction (placeholder for full NER).
///
/// Finds capitalized multi-word phrases that likely represent
/// named entities (persons, organizations, locations).
fn extract_basic_entities(text: &str) -> Vec<serde_json::Value> {
    let mut entities = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    let mut i = 0;
    while i < words.len() {
        // Look for sequences of capitalized words
        if words[i].chars().next().map_or(false, |c| c.is_uppercase()) {
            let mut end = i + 1;
            while end < words.len() {
                let word = words[end];
                let first_char = word.chars().next().unwrap_or(' ');
                // Continue if capitalized or connecting word (of, the, and)
                if first_char.is_uppercase()
                    || matches!(word, "of" | "the" | "and" | "for" | "in" | "de" | "van" | "von")
                {
                    end += 1;
                } else {
                    break;
                }
            }

            if end > i + 1 || words[i].len() > 1 {
                let entity_text = words[i..end].join(" ");
                // Filter out common sentence starters
                let skip = matches!(
                    entity_text.as_str(),
                    "The" | "This" | "That" | "These" | "Those"
                    | "It" | "He" | "She" | "They" | "We"
                    | "However" | "Therefore" | "Moreover" | "Furthermore"
                    | "In" | "On" | "At" | "For" | "But" | "And" | "Or"
                );

                if !skip && entity_text.len() > 1 {
                    let entity_type = guess_entity_type(&entity_text);
                    entities.push(serde_json::json!({
                        "name": entity_text,
                        "type": entity_type,
                    }));
                }
            }

            i = end;
        } else {
            i += 1;
        }
    }

    // Deduplicate
    entities.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    entities.dedup_by(|a, b| a["name"] == b["name"]);

    entities
}

fn guess_entity_type(entity: &str) -> &'static str {
    let lower = entity.to_lowercase();

    // Location signals
    if lower.contains("city") || lower.contains("state") || lower.contains("country")
        || lower.contains("river") || lower.contains("mountain")
        || lower.ends_with("land") || lower.ends_with("burg") || lower.ends_with("stan")
    {
        return "location";
    }

    // Organization signals
    if lower.contains("university") || lower.contains("institute")
        || lower.contains("corporation") || lower.contains("company")
        || lower.contains("inc") || lower.contains("ltd") || lower.contains("llc")
        || lower.contains("association") || lower.contains("foundation")
    {
        return "organization";
    }

    // Date signals
    if lower.contains("january") || lower.contains("february") || lower.contains("march")
        || lower.contains("april") || lower.contains("may") || lower.contains("june")
        || lower.contains("july") || lower.contains("august") || lower.contains("september")
        || lower.contains("october") || lower.contains("november") || lower.contains("december")
    {
        return "date";
    }

    // Default: treat multi-word capitalized as person or unknown
    if entity.contains(' ') {
        "person"
    } else {
        "other"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_entities_finds_names() {
        let entities = extract_basic_entities(
            "Albert Einstein developed the theory of relativity at Princeton University."
        );
        let names: Vec<&str> = entities.iter()
            .filter_map(|e| e["name"].as_str())
            .collect();
        assert!(names.iter().any(|n| n.contains("Einstein")));
        assert!(names.iter().any(|n| n.contains("Princeton")));
    }

    #[test]
    fn extract_entities_skips_sentence_starters() {
        let entities = extract_basic_entities(
            "The quick brown fox. However this is different. They went home."
        );
        let names: Vec<&str> = entities.iter()
            .filter_map(|e| e["name"].as_str())
            .collect();
        assert!(!names.contains(&"The"));
        assert!(!names.contains(&"However"));
        assert!(!names.contains(&"They"));
    }

    #[test]
    fn guess_entity_types() {
        assert_eq!(guess_entity_type("Princeton University"), "organization");
        assert_eq!(guess_entity_type("New York City"), "location");
        assert_eq!(guess_entity_type("January 2025"), "date");
        assert_eq!(guess_entity_type("Albert Einstein"), "person");
        assert_eq!(guess_entity_type("Rust"), "other");
    }

    #[test]
    fn get_str_works() {
        let mut args = serde_json::Map::new();
        args.insert("query".into(), serde_json::Value::String("test".into()));
        assert_eq!(get_str(&args, "query").unwrap(), "test");
        assert!(get_str(&args, "missing").is_err());
    }

    #[test]
    fn get_str_array_works() {
        let mut args = serde_json::Map::new();
        args.insert("urls".into(), serde_json::json!(["https://a.com", "https://b.com"]));
        let urls = get_str_array(&args, "urls").unwrap();
        assert_eq!(urls.len(), 2);
    }
}
