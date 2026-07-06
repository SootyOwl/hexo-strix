//! `hexo-infer-server` — pure-Rust HX04 inference server binary.
//!
//! A drop-in replacement for `python -m hexo_a0.inference_server`: speaks the
//! exact same binary stdin/stdout protocol (HX04 v2) but runs the pure-Rust
//! `hexo_infer::InferModel` forward pass, no libtorch. The self-play binary
//! spawns it via `--inference-bin`.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    hexo_infer::server::run_cli()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
