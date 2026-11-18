# Exp 09: A syllabus with review, and the block size where belief flips (pre-registration)

This is a follow-up to Exp 06/07, using the same `worlds.py`, data, model, hyperparameters
and seeds 1-3. The new options are `lp_mix` and `blocks`.

## Part 1: lp_mix (curriculum with review)

Every batch mixes all four depths. The shares are a softmax over each depth's
moving-average compression progress, with a floor of 10% per depth (β = 3, EMA 0.9).
Exp 06's random and lp runs are the comparison; they share the code and settings.

- **H1. The syllabus is still rediscovered.** Kendall τ between the step of each depth's
  peak share and the depth is >= 0.67 in each seed.
- **H2. Review restores generalisation.** Final accuracy on held-out depth-3/4
  compositions (mean) is >= random's (0.660 in Exp 06).
- **H3. No cost.** Mean accuracy over seen depths 1-4 is at least random's (0.997) minus
  0.02.

## Part 2: blocks (when does belief flip?)

The Exp 07 data is learnt once, in K alternating pairs of blocks: a shortcut chunk, then a
rule-only chunk, repeated K times. K = 1 is Exp 07's rule_last. At K = 312 each rule-only
block is a single batch, which is close to interleaved. The sweep is
K ∈ {1, 2, 4, 8, 16, 32, 64, 128, 312}, 27 runs in all.

- **H4. Belief changes with interleaving.** Final rule belief at K = 312 exceeds that at K = 1
  by >= 0.5.
- **H5. A phase transition, not a slope.** Some step between consecutive K values carries at
  least half of the total change in rule belief from K = 1 to K = 312.

Every run is reported.
