# 2026-07-12 — Hayes vs strix@237000 (512 sims): a proven win missed by the live solver budget

Analysis of a live loss (strix as P1, jkcat checkpoint 237000, 512 sims, solver-assisted
serving path). Player report: "it could've blocked my single threat with a double threat
(giving it full tempo which would've won), and it seems like it partially understands that
in the analysis but it still didn't play it in game."

The player's read is exactly right — but the moment is turn 16, not the endgame.

## Game summary

- htttx log: 27 turns, P2 (human) wins with (9,-1),(10,-2),(11,-3),(12,-4),(13,-5),(14,-6)
  on the (1,-1) axis (internal coords).
- From turn 19 (P2's broken five (6,-1) _ (8,-3),(9,-4),(10,-5),(11,-6)) every strix turn
  was forced defense: (7,-2)+(12,-7); q=9 column four → (9,0); q=11 column four; double
  four (diagonal + r=-4 row, 3 blockers needed vs 2 stones) → resign-equivalent.
- `solve_defense` from after-42 onward: **killers=[], pairs=[]** — no defense existed.
  Every late strix move matches the solver-defense outputs exactly (surviving pair
  (6,-3),(8,-2) at t16; best_delay (9,0)@42, (8,-1)@43, (11,-8)@46, (11,-2)@47,
  (7,-4)@50, (7,1)@51) → the solver-defense layer was active and stalling a lost game.

## The missed win (turn 16, after 30 placements)

P2 threatened (6,-3),(7,-3) completing r=-3 (stones (4,-3),(5,-3),(8,-3),(9,-3)).
P1 had a **proven forced win**: **(6,-5)+(6,-3)** — (6,-3) blocks the threat AND completes
a q=6-column four (6,-5),(6,-4),(6,-3),(6,-2); the cascade continues into P1's far-left
cluster. Deep prover (race): **WIN, depth 7 attacker-turns**, idtt 0.15 s + pdspn agree;
26-cell PV.

Winning line (analysis URL, PV appended after move 30):
https://hexo.tyto.cc/analysis#c=AQIAAgEBAwIDAQgCAgMEAgQFCAUIAQYDCgMKBQYFCgEGAQgDDAcHBAUCEgMEAAwDBAQQBw0IDwgSBRAFDAkMBQwNDAELCAkIEwgHCAcGBQQNDAEACwQJBA8EAwQNBgkCEwwHAAsGCQYPBgkACQoJDA

## Why live play missed it — two independent failures

1. **Own-win solve budget.** The win is found by the TIGHT generator (wide irrelevant)
   at **35k nodes / 116 ms**, at any depth cap 8–20. `LIVE_NODE_BUDGET = 20_000` → none.
   Missed by <2×. The 20k tuning sweeps (2026-07-04/05) showed zero coverage gain beyond
   20k on sampled positions; this game is a live counterexample.
   Bisect at after-30 (tight, d16): 20k none · 35k WIN · 50k–250k WIN, all ≤150 ms.

2. **Defense killer ordering.** At placement 31 (after the block (6,-3)), the single
   stone (6,-5) still won — again invisible at 20k. `_defensive_move` (live budget) saw
   P2's residual threat with killers **[(8,-2), (8,-1), (8,-6), (8,0), (6,0), (7,-1)]**
   and played killers[0] = (8,-2). But **(6,-1) — also in the killers list — is itself a
   proven win** (post-move `solve_defense` from P2's side: killers=[], pairs=[]).
   (8,-2) is the one choice that lets P2 escape (surviving defenses must also poison the
   left side: pairs like ((6,-1),(-5,4)); Hayes found (6,-1)+(9,-4) which held).

The MCTS layer never got a say (tactical layers preempt it), but notably 512-sim MCTS's
argmax at placement 31 was **(6,-1)** (q=+0.47) — pure MCTS would have kept the win that
the defense layer discarded. NN detail: the quiet builder (6,-5) has policy prior ~8e-5
(rank 25); root values +0.12/+0.37 in a proven-won position.

Analysis-screen note: the analysis pipeline (wide, d12/250k) finds the win at both
after-30 and after-31 (~260 ms) — hence "it partially understands that in the analysis
but didn't play it in game."

## Shipped fixes (same day)

1. **`LIVE_NODE_BUDGET` 20k → 50k** (own-win solve only; `DEF_NODE_BUDGET` stays at its
   measured knee — the defensive sweep stacks budget-bound re-solves, the own-win solve
   runs once per placement). Re-measured over every prefix of the 5 line fixtures
   (748 solves per config, both sides, dev box):

   | depth/budget | p95 | p99 | max | wins sighted |
   |---|---|---|---|---|
   | 8 / 20k  | 133 ms | 179 ms | 253 ms | 323 |
   | 8 / 50k  | 288 ms | 418 ms | 470 ms | 335 |
   | 16 / 20k | 129 ms | 189 ms | 243 ms | 323 |
   | 16 / 50k | 265 ms | 415 ms | 463 ms | 335 |

   Worst case stays under 0.5 s (≈1.4 s on the 2–3× slower production host) against
   3–10 s defense deadlines, and 50k sights 12 additional wins. The 2026-07-05 sweep's
   "p99 0.9–2.7 s at 50k" applied to the full depth×budget grid tail; on the live tier
   depth caps it does not materialize.

2. **Killer ranking in `_defensive_move`**: among verified killers, prefer one whose
   placement keeps/creates the bot's own forcing win — probe per candidate at the live
   budget (`solve_forcing` on the hypothetical board when another placement remains this
   turn, flipped-perspective `solve_threat` on the turn's last placement), deadline-
   bounded. The literal "block the single threat with a double threat" rule.

3. **Fixtures**: `hexo-a0/tests/fixtures/hayes_20260712.htttx` (pytest regression: the
   after-30/after-31 wins must be visible at `LIVE_NODE_BUDGET`, the bot must execute
   the win end-to-end, and the killer ranking must beat the policy tie-break) and
   `scripts/fixtures/forcing_puzzles/hayes_20260712_line.json` (added to
   `forcing_depth_sweep.py`'s line-fixture corpus).

## Residual observations

- Hayes's endgame suspicion ("my win wasn't real") is disproven: from after-42 no strix
  defense existed; his turn-17 (6,-1) was the exact killer of strix's last resource.
- The NN policy prior of the winning quiet builder (6,-5) was ~8e-5 (rank 25) and root
  value only +0.12 in a proven-won position — policy/value distillation target material
  (see the solver-eval distillation backlog).
