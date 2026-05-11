use tantivy::schema::*;

/// Build the tantivy schema for the search index.
pub fn build_schema() -> Schema {
    let mut builder = SchemaBuilder::new();

    // Stored fields (returned in results)
    builder.add_text_field("url", STRING | STORED);
    builder.add_text_field("title", TEXT | STORED);
    builder.add_text_field("domain", STRING | FAST | STORED);

    // Full-text searchable body (not stored to save space; original in page cache)
    builder.add_text_field("body", TEXT);

    // Date field for freshness scoring
    builder.add_date_field("published_date", INDEXED | FAST | STORED);

    // Content hash for exact dedup
    builder.add_bytes_field("content_hash", FAST | STORED);

    // Source tier for authority scoring
    builder.add_u64_field("source_tier", FAST | STORED);

    // Extraction confidence
    builder.add_f64_field("extraction_confidence", FAST | STORED);

    builder.build()
}
