# PCZero path-consistency over a random replay buffer — correctness pitfalls

**Status:** the trainer's PCZero grouping was unsound; fixed 2026-07-07.
`pc_loss_weight` defaults to 0 in every config, so the bugs never affected a
run. This note records the analysis so future trajectory-based regularizers
don't repeat it.

## The two grouping bugs

PCZero (Zhao et al., 2022) penalises value predictions along a game
trajectory for (a) pairwise inconsistency under alternating movers and
(b) a terminal anchor to the outcome. `path_consistency_loss`
(`hexo-a0/src/hexo_a0/loss.py`) assumes its `values` are ordered, consecutive
game positions. The trainer's grouping violated that:

1. **No temporal ordering.** `traj_groups` collected each trajectory's
   indices in *batch-sample order* — i.e. random, because the replay buffer
   samples uniformly. `values[indices]` was a shuffled sequence, so the
   consecutive term `values[1:] + values[:-1]` paired arbitrary same-game
   positions.
2. **Wrong terminal sign.** The terminal term paired `values[-1]` (latest
   position) with `v_targets[indices[0]]` (an *earlier* position's label).
   `value_target` is side-to-move-relative, so the earliest and latest
   positions' labels differ in sign whenever their movers differ in parity
   — ~half the time it penalised a correct prediction.

## Why the consistency term needs move-parity, not just order

Even after sorting, a random replay minibatch contains a *subset* of a
trajectory's positions, with gaps. The consistency term `v[t] + v[t'] ≈ 0`
holds only for an **odd** game-move gap (opponent perspective); an even gap
shares the mover, so `v[t] + v[t'] ≈ 2·v`, and penalising that sum punishes a
correct prediction. So a sound term must pair only odd-gap positions, which
requires knowing each position's within-game move parity.

## The insertion id is a within-game move-order proxy

The replay buffer assigns contiguous ascending ids in `add_many`
(`replay_buffer.py`), and games are added in temporal order, so for a single
trajectory the insertion id `sid` is a proxy for the within-game move index:
consecutive sids = consecutive game positions, and **`sid` parity tracks the
side to move**. The buffer now attaches `ex._sid` at sample time and the
trainer sorts by it and passes it to `path_consistency_loss`, which pairs
every even-sid × odd-sid position (all odd-gap pairs in the sampled subset)
and ignores even-gap pairs.

This is the same "order is not persisted downstream" constraint that shaped
the horizon-head design (see `2026-07-06-distributional-value-heads-design.md`:
horizon targets must be computed at ingest). PCZero's consistency term faces
it too — except here the insertion id happens to recover enough (order +
parity) at train time, no data-format change required.

## Residual limitation

The terminal anchor pairs the *latest-in-batch* position's prediction with
its own `value_target`. That is correct (it is the value loss at that
position) but redundant with the main value loss — the true PCZero terminal
anchor (the game's *actual* final position = ±1) needs the game length,
which the buffer does not store. Identifying the true terminal would require
persisting the within-game move index / game length (a future data-format
change, e.g. riding along in HX08). Until then the terminal term adds only a
small extra weight on the latest sampled position, and the consistency term
is the real signal.

## Takeaway for similar regularizers

Any trajectory-based loss over a random replay minibatch needs (a) recovered
within-game order and (b) move-parity to pair only opposite-mover positions.
Both are available from the buffer insertion id *as long as whole games are
added in one `add_many` call in temporal order* — worth preserving as an
invariant if more such regularizers are added.