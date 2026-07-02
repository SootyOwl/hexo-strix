//! Acting-time weight construction for opening-exploration moves,
//! including the Q-margin truncation gate
//! (see docs/research/2026-06-11-q-margin-truncated-exploration.md).

/// Result of building exploration acting weights.
pub struct ActingWeights {
    /// Sampling weights aligned with the legal-move ordering (unnormalised).
    pub weights: Vec<f64>,
    /// Fraction of the pre-gate weight total zeroed by the Q-margin gate.
    /// 0.0 when the gate is off or the pre-gate total is 0.
    ///
    /// **Denominator note**: the denominator is the FULL pre-gate acting
    /// distribution total (the sum of all legal-move weights including
    /// non-candidate mass present in `improved_policy` mode). `mass_removed`
    /// therefore measures truncation of the *actual sampling distribution*,
    /// not just candidate-set aggression. In `visit_counts` mode non-candidate
    /// weights are already zero, so numerator and denominator agree with
    /// candidate-set intuition. In `improved_policy` mode, non-candidate prior
    /// mass is never gated and lowers the reported ratio accordingly.
    pub mass_removed: f64,
    /// Max completed-Q over the candidate set (NEG_INFINITY if empty).
    /// Callers use `q_max - per_child_q[chosen]` as the acted-move deficit.
    ///
    /// **Caller contract**: callers MUST guard `q_max.is_finite()` before
    /// computing deficits. When the candidate set is empty, `q_max` is
    /// `NEG_INFINITY`, and naive subtraction with `.max(0.0)` would silently
    /// produce a 0.0 deficit rather than being skipped altogether.
    pub q_max: f64,
}

/// Build sampling weights for an opening-exploration move, with an optional
/// Q-margin truncation gate.
///
/// `use_visit_counts == false` (default) returns the improved policy π';
/// `true` returns root visit counts cast to f64 (paper-style acting).
/// Both are unnormalised; the caller normalises by the sum.
///
/// When `truncate_delta = Some(δ)`, every **candidate** index whose
/// completed-Q is more than δ below `q_max` has its weight zeroed.
/// Non-candidate indices are never gated. Ties at exactly δ are kept
/// (`q_max - q[i] > δ` is the gate predicate). If the gate zeroes every
/// candidate, the returned weights may sum to 0; the caller handles fallback.
///
/// NaN Q entries are never gated and never poison `q_max` (fail-safe: a
/// corrupt Q keeps its weight rather than silently discarding the move).
///
/// δ < 0 gates every candidate, including the argmax (deficit 0 > δ);
/// callers must pass δ > 0 — upstream config treats δ ≤ 0 as gate-off.
///
/// **Unvisited candidates**: when `n_simulations < m_actions` or under a
/// fast-cap search, some candidates may carry zero visits. Their `per_child_q`
/// entry will be an optimistic v_mix estimate rather than a search-backed
/// completed-Q. The gate's soundness guarantee — that zeroed candidates are
/// genuinely refuted — is therefore strongest when all candidates have
/// real visits (production regime: `n ≫ m`). Callers logging deficit
/// telemetry should note this when `n_simulations` is close to `m_actions`.
///
/// # Panics
/// Panics if any index in `candidate_indices` is out of bounds for
/// `per_child_q`, `improved_policy`, or `visit_counts`.
pub fn exploration_weights(
    improved_policy: &[f64],
    visit_counts: &[u32],
    per_child_q: &[f64],
    candidate_indices: &[usize],
    use_visit_counts: bool,
    truncate_delta: Option<f64>,
) -> ActingWeights {
    let mut weights: Vec<f64> = if use_visit_counts {
        visit_counts.iter().map(|&n| n as f64).collect()
    } else {
        improved_policy.to_vec()
    };

    let q_max = candidate_indices
        .iter()
        .map(|&i| per_child_q[i])
        .fold(f64::NEG_INFINITY, f64::max);

    let mut mass_removed = 0.0;
    if let Some(delta) = truncate_delta
        && !candidate_indices.is_empty()
    {
        let total: f64 = weights.iter().sum();
        let mut removed = 0.0;
        for &i in candidate_indices {
            if q_max - per_child_q[i] > delta && weights[i] > 0.0 {
                removed += weights[i];
                weights[i] = 0.0;
            }
        }
        if total > 0.0 {
            mass_removed = removed / total;
        }
    }

    ActingWeights { weights, mass_removed, q_max }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 5 legal moves; candidates are 0..4 (index 4 unsearched).
    fn fixture() -> (Vec<f64>, Vec<u32>, Vec<f64>, Vec<usize>) {
        let improved = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        let visits = vec![22, 14, 6, 2, 0];
        let q = vec![0.40, 0.35, 0.30, -0.10, 0.00]; // idx 4 = v_mix fallback
        let cands = vec![0, 1, 2, 3];
        (improved, visits, q, cands)
    }

    #[test]
    fn gate_off_visit_counts_matches_legacy() {
        let (imp, vis, q, c) = fixture();
        let out = exploration_weights(&imp, &vis, &q, &c, true, None);
        assert_eq!(out.weights, vec![22.0, 14.0, 6.0, 2.0, 0.0]);
        assert_eq!(out.mass_removed, 0.0);
    }

    #[test]
    fn gate_off_improved_policy_matches_legacy() {
        let (imp, vis, q, c) = fixture();
        let out = exploration_weights(&imp, &vis, &q, &c, false, None);
        assert_eq!(out.weights, imp);
        assert_eq!(out.mass_removed, 0.0);
    }

    #[test]
    fn gate_zeroes_refuted_candidates_and_reports_mass() {
        let (imp, vis, q, c) = fixture();
        // q_max = 0.40; δ = 0.25 → only idx 3 (deficit 0.50) is gated.
        let out = exploration_weights(&imp, &vis, &q, &c, true, Some(0.25));
        assert_eq!(out.weights, vec![22.0, 14.0, 6.0, 0.0, 0.0]);
        assert!((out.mass_removed - 2.0 / 44.0).abs() < 1e-12);
        assert_eq!(out.q_max, 0.40);
    }

    #[test]
    fn tie_at_exactly_delta_is_kept() {
        // Use exact IEEE 754 fractions to avoid floating-point rounding errors
        // in the deficit computation. q_max=0.5, q[2]=0.25 → deficit=0.25
        // exactly. δ=0.25 → idx 2 is a tie (kept); idx 3 deficit=0.75 → gated.
        let imp = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        let vis = vec![22u32, 14, 6, 2, 0];
        let q = vec![0.5, 0.375, 0.25, -0.25, 0.0];
        let c = vec![0, 1, 2, 3];
        let out = exploration_weights(&imp, &vis, &q, &c, true, Some(0.25));
        assert_eq!(out.weights, vec![22.0, 14.0, 6.0, 0.0, 0.0]);
        // Confirm the predicate: deficit at idx 2 is exactly δ, so it is kept.
        assert_eq!(q[0] - q[2], 0.25_f64);
    }

    #[test]
    fn gate_ignores_non_candidates_in_improved_policy_mode() {
        let (imp, vis, q, c) = fixture();
        // idx 4 is not a candidate: weight survives even though its
        // v_mix q (0.0) is more than δ below q_max.
        // fixture: improved = [0.5, 0.3, 0.1, 0.05, 0.05], total = 1.0.
        // Only idx 3 is gated (deficit 0.50 > δ=0.25); its weight is 0.05.
        // mass_removed = 0.05 / 1.0 = 0.05 (denominator is FULL pre-gate
        // distribution total, including non-candidate idx 4).
        let out = exploration_weights(&imp, &vis, &q, &c, false, Some(0.25));
        assert_eq!(out.weights[4], 0.05);
        assert_eq!(out.weights[3], 0.0); // candidate, gated
        assert!((out.mass_removed - 0.05).abs() < 1e-12,
            "mass_removed should be gated_weight(0.05)/total(1.0)=0.05, got {}", out.mass_removed);
    }

    #[test]
    fn empty_candidates_gate_is_noop() {
        let (imp, vis, q, _) = fixture();
        let out = exploration_weights(&imp, &vis, &q, &[], true, Some(0.25));
        assert_eq!(out.weights, vec![22.0, 14.0, 6.0, 2.0, 0.0]);
        assert_eq!(out.mass_removed, 0.0);
        assert_eq!(out.q_max, f64::NEG_INFINITY);
    }

    #[test]
    fn all_candidates_gated_weights_sum_to_zero() {
        let (imp, vis, q, c) = fixture();
        // With δ > 0 the argmax candidate (deficit 0) can never be gated, so
        // the only all-gated case is δ < 0 — the documented hazard: every
        // candidate's deficit (≥ 0) exceeds a negative δ. In visit-counts
        // mode the non-candidate idx 4 already has weight 0, so the result
        // sums to 0 and the caller's fallback path must engage.
        let out = exploration_weights(&imp, &vis, &q, &c, true, Some(-0.1));
        assert_eq!(out.weights, vec![0.0; 5]);
        assert_eq!(out.weights.iter().sum::<f64>(), 0.0);
        assert_eq!(out.mass_removed, 1.0); // 44/44 of pre-gate mass zeroed
        assert_eq!(out.q_max, 0.40);
    }

    #[test]
    fn all_candidates_gated_non_candidate_mass_survives() {
        let (imp, vis, q, c) = fixture();
        // Improved-policy mode: same δ < 0 gates all four candidates, but the
        // non-candidate idx 4 keeps its prior mass → weights still sum > 0.
        let out = exploration_weights(&imp, &vis, &q, &c, false, Some(-0.1));
        assert_eq!(out.weights, vec![0.0, 0.0, 0.0, 0.0, 0.05]);
        assert!(out.weights.iter().sum::<f64>() > 0.0);
        assert!((out.mass_removed - 0.95).abs() < 1e-12); // 0.95 of total 1.0
    }

    #[test]
    fn nan_q_entry_is_kept_and_does_not_poison_q_max() {
        let (imp, vis, _, c) = fixture();
        // idx 1 has a NaN completed-Q. Fail-safe direction: q_max stays
        // finite (NaN ignored by the max fold), the NaN candidate is KEPT
        // (gate predicate NaN > δ is false), and mass_removed stays finite.
        let q = vec![0.40, f64::NAN, 0.30, -0.10, 0.00];
        let out = exploration_weights(&imp, &vis, &q, &c, true, Some(0.25));
        assert_eq!(out.q_max, 0.40);
        assert_eq!(out.weights, vec![22.0, 14.0, 6.0, 0.0, 0.0]); // idx 1 kept
        assert!(out.mass_removed.is_finite());
        assert!((out.mass_removed - 2.0 / 44.0).abs() < 1e-12);
    }

    #[test]
    fn plus_inf_q_forces_that_candidate_and_gates_all_finite() {
        // One candidate has per_child_q = +INFINITY. In visit_counts mode:
        //   q_max = +inf (the fold uses f64::max, which propagates +inf).
        //   +inf candidate: deficit = inf - inf = NaN; gate predicate NaN > δ
        //     is false → kept (fail-safe: never gate a move with NaN deficit).
        //   finite candidates: deficit = inf - finite = +inf > δ → gated.
        // Net result: only the +inf candidate survives. This degenerates the
        // gate to forcing the +inf move — a **sick network** precondition.
        // Bounded Q values are the caller's contract.
        //
        // NOTE: this test pins CURRENT behavior. If acting.rs changes the
        // +inf handling, update this test deliberately.
        let imp = vec![0.2, 0.3, 0.3, 0.2];
        let vis = vec![10u32, 20, 20, 10];
        let q = vec![f64::INFINITY, 0.35, 0.30, -0.10];
        let c = vec![0, 1, 2, 3];
        let out = exploration_weights(&imp, &vis, &q, &c, true, Some(0.25));
        // q_max = +inf
        assert_eq!(out.q_max, f64::INFINITY);
        // +inf candidate kept; all finite candidates gated
        assert_eq!(out.weights[0], 10.0); // +inf candidate: NaN deficit → kept
        assert_eq!(out.weights[1], 0.0);  // deficit inf > δ → gated
        assert_eq!(out.weights[2], 0.0);  // deficit inf > δ → gated
        assert_eq!(out.weights[3], 0.0);  // deficit inf > δ → gated
        // mass_removed = (20+20+10)/60 ≈ 0.8333…; +inf candidate weight
        // (10) is not gated so numerator = 60 − 10 = 50.
        assert!((out.mass_removed - 50.0 / 60.0).abs() < 1e-12,
            "mass_removed should be 50/60, got {}", out.mass_removed);
    }

    #[test]
    fn one_hot_shortcut_visits_pass_through() {
        // Depth-1/2 shortcut returns one-hot visits; per-doc per_child_q is
        // uniform there. Gate must not disturb the forced move.
        let imp = vec![0.0, 1.0, 0.0];
        let vis = vec![0u32, 1, 0];
        let q = vec![0.2, 0.2, 0.2];
        let out = exploration_weights(&imp, &vis, &q, &[0, 1, 2], true, Some(0.25));
        assert_eq!(out.weights, vec![0.0, 1.0, 0.0]);
        assert_eq!(out.mass_removed, 0.0);
    }
}
