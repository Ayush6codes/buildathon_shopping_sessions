FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN cargo build --release && strip target/release/shopping target/release/diag

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /src/target/release/shopping /app/shopping
COPY --from=build /src/target/release/diag /app/diag
COPY run_diag.sh /app/run_diag.sh
ENV MIMALLOC_ALLOW_LARGE_OS_PAGES=1 MIMALLOC_PURGE_DELAY=-1 MIMALLOC_ARENA_EAGER_COMMIT=1 SS_QUIET=1
USER 1000
ENTRYPOINT ["/app/run_diag.sh"]
