# Exp 08: A model that remembers its path, and one that can refuse (pre-registration)

Follow-up to Exp 05. That experiment found three things:

- Order changes the prequential code length of the same dataset by up to 1.4 bits per
  token.
- Code length predicts new-fact accuracy (Spearman ρ = -0.77) but not held-out story loss
  (ρ = 0.21).
- Orders that defer the noise end the epoch on noise and wreck the final model (story loss
  7-8).

The diagnosis: code length measures the whole learning path, while the final model is
dominated by the last data. Exp 08 tests two remedies. It uses the same data, prior, seeds
and hyperparameters as Exp 05, through new flags in `order.py`.

1. **Path ensemble (`--path_ensemble`).** The prequential code is the code of the sequence
   of predictors, not of the last one. So also evaluate the mixture of the predictive
   distributions of the 10 checkpoints taken every 150 steps along the path.
2. **Right to refuse (`--allow_skip`).** Kolmogorov-random data teaches nothing. An item the
   learner codes no better than chance (loss >= 97% of ln 4096 nats) is refused: it still
   pays its code length, but it is not trained on. The rule is the same for every order.

## Runs

The 6 orders x 3 seeds, in two conditions:

- `ens`: no refusal, with the path ensemble;
- `skip`: refusal, with the path ensemble.

That is 36 runs.

## Hypotheses

- **H1. The path ensemble restores MDL.** In `ens`, the path ensemble's held-out story loss
  correlates with learnable code length at Spearman ρ >= 0.5. The final models gave
  ρ = 0.21 in Exp 05.
- **H2. The path ensemble undoes the recency damage.** In `ens`, easy_first and lp have a
  path-ensemble story loss <= 3.0. Their final models were at 8.36 and 7.47 in Exp 05.
- **H3. Refusal removes the noise and the damage.**
  - In `skip`, under 5% of noise items are trained on, for every order.
  - Every order's final story loss is within 0.15 of random-with-skip's.
- **H4. With refusal, compression predicts generalisation.** In `skip`, final story loss
  correlates with learnable code length at Spearman ρ >= 0.5, and final fact accuracy at
  ρ <= -0.5.

Each hypothesis is reported as holding or failing.
