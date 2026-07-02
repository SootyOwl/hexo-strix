//! Queue-based evaluation batcher for parallel self-play.
//!
//! Game threads send [`EvalRequest`]s via an MPSC channel. The batcher
//! accumulates requests up to `max_batch_size` or a short timeout, calls
//! the real eval function once, and dispatches results back via per-request
//! oneshot channels.

use rustc_hash::FxHashMap as HashMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use hexo_engine::types::Coord;
use hexo_engine::GameState;

/// A single evaluation request from a game thread.
pub struct EvalRequest {
    /// States to evaluate (typically 1 for root eval, or m_actions for a halving batch).
    pub states: Vec<GameState>,
    /// Channel to send the response back on.
    pub response_tx: mpsc::SyncSender<EvalResponse>,
}

/// Response sent back to a game thread after evaluation.
pub struct EvalResponse {
    pub logits: Vec<HashMap<Coord, f64>>,
    pub values: Vec<f64>,
}

/// Configuration for the batcher.
pub struct BatcherConfig {
    /// Maximum number of states to accumulate before dispatching a batch.
    pub max_batch_size: usize,
    /// Maximum time to wait for additional requests before dispatching.
    pub timeout: Duration,
}

/// Run the batcher loop with a generic eval function.
///
/// Collects [`EvalRequest`]s from `request_rx`, batches them, calls `eval_fn`,
/// and dispatches results. Returns when all senders are dropped (all game
/// threads finished), or when `eval_fn` returns `Err` (e.g. KeyboardInterrupt).
pub fn batcher_loop<F>(
    request_rx: &mpsc::Receiver<EvalRequest>,
    config: &BatcherConfig,
    eval_fn: &mut F,
) -> Result<(), String>
where
    F: FnMut(&[GameState]) -> Result<(Vec<HashMap<Coord, f64>>, Vec<f64>), String>,
{
    loop {
        // Block until the first request arrives (or all senders drop)
        let first = match request_rx.recv() {
            Ok(req) => req,
            Err(mpsc::RecvError) => break,
        };

        let mut pending: Vec<EvalRequest> = vec![first];
        let mut total_states: usize = pending[0].states.len();
        let deadline = Instant::now() + config.timeout;

        // Drain additional requests up to max_batch_size or timeout
        while total_states < config.max_batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match request_rx.recv_timeout(remaining) {
                Ok(req) => {
                    total_states += req.states.len();
                    pending.push(req);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        // Flatten all states into one batch
        let mut all_states: Vec<GameState> = Vec::with_capacity(total_states);
        let mut slices: Vec<(usize, usize)> = Vec::with_capacity(pending.len());
        for req in &pending {
            let start = all_states.len();
            all_states.extend(req.states.iter().cloned());
            slices.push((start, all_states.len()));
        }

        // Call the real eval function once for the entire batch
        let (all_logits, all_values) = match eval_fn(&all_states) {
            Ok(result) => result,
            Err(e) => {
                // Send dummy responses so game threads unblock and can exit
                for req in pending {
                    let n = req.states.len();
                    let _ = req.response_tx.send(EvalResponse {
                        logits: vec![HashMap::default(); n],
                        values: vec![0.0; n],
                    });
                }
                return Err(e);
            }
        };

        // Dispatch results back to each requester
        for (i, req) in pending.into_iter().enumerate() {
            let (start, end) = slices[i];
            let response = EvalResponse {
                logits: all_logits[start..end].to_vec(),
                values: all_values[start..end].to_vec(),
            };
            let _ = req.response_tx.send(response);
        }
    }
    Ok(())
}
