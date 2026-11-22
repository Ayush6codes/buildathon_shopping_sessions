# Shopping Sessions — Rust pipeline

Single static Rust binary implementing `rules.md`: mmap'd parallel CSV parsing, union-find for linked IDs,
hash-partitioned dedup, per-person sessionization + click attribution, Parquet output.

Build/run: `docker build -t ss . && docker run -v $DATA:/data:ro -v $OUT:/output ss`
Local: `cargo build --release && ./target/release/shopping DATA_DIR OUTPUT_DIR`
