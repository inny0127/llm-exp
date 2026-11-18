# Exp 05: Order matters for compression (pre-registration)

**Question.** Take one fixed dataset D, learnt in one epoch with every item seen exactly once.
Does the order change how short the learner compresses D? And does the order that
compresses D shortest also generalise best?

## Background

- **Prequential code length** (Blier & Ollivier, 2018): the code length of D under a learner
  is the sum of its loss on each batch just before it learns that batch.
- **Ilya Sutskever (2023): compression ↔ generalisation.**
- **Kolmogorov complexity** belongs to D as a set and does not depend on order. The code
  length a particular learner actually reaches can.
- **Schmidhuber: compression progress as curiosity.** This is the `lp` order below.

## Setup

- **Data.** D has 48,000 windows of 128 tokens, 12,000 from each source: stories, bios of 300
  new people, uniform random noise, and 64 story windows repeated. D is fixed per seed and
  shared by every order with that seed.
- **Prior.** Exp 04's prior (a 3.8M-parameter GPT pretrained on 20M tokens of TinyStories).
- **Training.** 1,500 steps of 32 windows. The model, optimiser and number of steps are the
  same for every order.

## Orders (each with seeds 1, 2, 3; 18 runs)

| Order | Rule |
|---|---|
| random | a random permutation |
| easy_first | ascending loss under the prior, fixed in advance |
| hard_first | descending loss under the prior, fixed in advance |
| maxloss | greedy: the highest current loss among 128 candidates still unused |
| rho | greedy: the highest reducible loss (current minus Exp 04's reference model) |
| lp | Exp3 bandit over the four sources; reward = compression progress on the batch just learnt |

## Measures (per run)

- **Code length of D**, in bits per token: total, the learnable part (stories, facts, dupes),
  and noise.
- **Final generalisation:**
  - held-out story loss;
  - accuracy on the new facts with probe template 0;
  - accuracy on the new facts with held-out template 7, which never appears in training.

## Hypotheses (fixed before any full run)

- **H1. Order changes compression.** Across the six orders, the spread (max minus min) of
  mean learnable code length is larger than 3x the largest across-seed SD.
- **H2. Shorter code, better generalisation (MDL).** Across all 18 runs, learnable code length
  correlates with final held-out story loss at Spearman ρ >= 0.5, and with final
  fact accuracy at ρ <= -0.5.
- **H3. Compression progress compresses.** lp has lower mean learnable code length than
  random, and maxloss has higher.
- **H4. Noise is incompressible in any order.** No order codes the noise below its entropy,
  log2(4096) = 12 bits per token. The excess above 12 bits is reported per order: it is the
  cost of the prior's confident English predictions before the model learns the noise is
  uniform. This replaces a first version ("within 2% of 12 bits"), which the smoke run showed
  cannot hold: the prior pays about 17 bits per noise token early on. The change was made
  before any full run.

Each hypothesis is reported as holding or failing. An outcome where H1 holds and H2 fails
(code length and generalisation come apart) counts as a result, not a failure.

## Implementation notes

The code is `order.py`. A 64-items-per-source CPU smoke run checked only that every order
runs. Exp 04's prior, reference model and data are reused unchanged; `data_md5.txt` checks
that the data is identical when rebuilt on another machine.
