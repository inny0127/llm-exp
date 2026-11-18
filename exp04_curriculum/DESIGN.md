# Exp 04: Choosing what to learn next (pre-registration)

A small GPT is pretrained on TinyStories (20M tokens) to get a prior. It then trains for
3,000 steps of 32 windows each. A selection policy picks every batch from a pool of four
equally likely sources:

| Source | What it is | Category (Mindermann et al., 2022) |
|---|---|---|
| stories | held-out TinyStories | learnable, worth learning |
| facts | bios of 300 new invented people, templates 1-6 | learnable, worth learning, novel |
| noise | uniformly random tokens | not learnable |
| dupes | 64 fixed story windows, repeated | learnable, then already learnt |

Policies:

- **random:** a uniform draw from the pool.
- **maxloss:** the top 32 of 128 candidates by current loss ("curiosity").
- **rho:** the top 32 of 128 candidates by reducible loss: current loss minus the loss of a
  reference model. The reference is the prior trained 1,500 steps on holdout stories and
  holdout bios of the same people.
- **lp:** an Exp3 bandit over the four sources. Its reward is the relative loss drop on the
  batch just learnt (learning progress, Graves et al., 2017). Exploration 0.1.

Metrics, measured at the end of the 3,000 steps, means over seeds 1-3:

- fact accuracy on probe template 0;
- fact accuracy on held-out probe template 7, which never appears in training (guard);
- validation loss on TinyStories;
- the share of picks per source.

Chance on the fact probes is about 7%.

Hypotheses, fixed before any full run:

- **H1:** maxloss spends more than 40% of its picks on noise, and ends with worse validation
  loss than random (the "noisy TV" trap).
- **H2:** rho beats random by at least 0.10 in fact accuracy (template 0), with validation
  loss no more than 0.02 worse. The held-out-template accuracy must also improve.
- **H3:** lp beats random by the same margins as H2.
- **H4:** rho and lp each put under 10% of their picks on noise.

Hyperparameters were set once and not tuned: AdamW at lr 5e-4, batch 32, ctx 128,
candidate factor 4. A 20-step CPU smoke run checked only that the code runs. Every run is
reported.
