# Exp 01 — Two-speed weights: sparse fast writes + sleep consolidation

> Added after the dev stream: the protocol changes made there, before any
> test run, are listed with their evidence in `RESULTS.md` (Deviations). The
> text below is the pre-registration as committed.

Pre-registered design. Written before any stream run; hyperparameters marked
*dev-selected* are chosen on the dev stream (seed 0) and frozen before the test
streams (seeds 1+) are run.

## Origin

From the continual-learning ideation thread ("idea 2: two-speed weights"):
write new experience into sparse memory slots, then periodically consolidate
into the dense weights offline and free the slots. Meta's sparse memory
finetuning (Lin et al. 2025, arXiv:2510.15103) covers the fast half: on a 1.3B
model, NaturalQuestions F1 fell 89% after full finetuning on new facts, 71%
with LoRA, and 11% with TF-IDF-selected memory-slot updates. The slow half
(consolidation) is the open part.

## Decision this informs

Is two-speed weights worth scaling to the RTX 5090 (tier 1 to tier 2)? It is
only worth it if (Q1) the published sparse-write advantage reproduces in this
testbed and (Q2) consolidation adds something sparse writes alone do not.

## Questions, in gated order

**Q1 — replication gate.** At matched learning, does TF-IDF sparse memory
finetuning forget less than full finetuning and LoRA?
If no, stop: the testbed does not reproduce the published effect.

**Q2 — the new part.** Does a periodic sleep (self-quiz, distil into dense
weights with replay, reset the slots) keep learning and retention up late in
the stream without adding forgetting? Q2 is only meaningful if sparse-only
writes visibly saturate. If they do not, the honest answer is "sleep has
nothing to fix at this stream length."

## What changed from the thread's plan, and why

| Thread plan | Here | Why |
|---|---|---|
| 30M-param GPT on one GPU | 3.8M dense params + 16.8M-param memory table, 4 CPU cores | Only CPUs available; tests the mechanism, not scale |
| 5,000 facts, sleep every 500 | 1,000 facts, sleep every 100 | Same 10:1 ratio at CPU cost |
| Arms A, B, C, D | 2x2 factorial {full FT, sparse} x {no sleep, sleep} + LoRA | D also gets pretraining replay and extra rehearsal. Without an "A + sleep" arm, a D win could be replay alone |
| One learning rate per arm | LR sweep per arm, compared on the learning-forgetting frontier | A tiny LR forgets nothing and learns nothing; single-LR comparisons can be rigged by the LR choice |
| Success = acquisition stays flat | Success = retention plus forgetting, at matched learning | Sparse writes can always fit the current fact; saturation should show up as old stream facts being overwritten, not as failure to acquire |
| Free-text answers | Single-token answers scored over the relation's candidates | 20-30x cheaper probes, no length bias, chance is known |
| Forgetting = TinyStories loss | TinyStories loss **and** old-fact accuracy | Meta's headline is knowledge forgetting (NQ F1); old facts are this testbed's NQ |

Deleted outright: LoRA rank/alpha sweeps (rank 16, alpha 32 fixed), TF-only and
random-slot ablations, and scale-up runs. None changes the go/no-go call.

## Testbed

- **Pretraining corpus** (24M tokens, one epoch): TinyStories (77%) plus bios
  of 500 *old* invented entities (23%). Each old entity has six attributes
  (home, job, pet, color, food, toy) and appears in 100 bios built from eight
  paraphrase templates per attribute.
- **Model**: GPT, d=256, 4 layers, 4 heads, context 128, 4,096-token BPE
  trained on TinyStories only. Block 2's FFN is a product-key memory: 65,536
  slots (256^2 product keys), 4 heads x top-16, value dim 256, gated output
  `(y * silu(x W1)) W2` as in Meta's memory layers.
- **Stream**: 1,000 *new* invented entities, one fact each (relation and value
  uniform at random), each shown once as a batch of 16 paraphrases drawn from
  seven templates, then K gradient steps (K *dev-selected*, same for all arms).
- **Probes**: cloze prompt, answer = argmax over that relation's candidate
  tokens (12-20 candidates; chance about 7%). Two probe templates: one seen in
  stream batches (in-distribution) and one held out of every stream batch but
  present in pretraining bios (tests whether the fact was stored as knowledge,
  not as a string).

## Arms

Wake phase (one write per fact):

- **A — full FT**: AdamW on all parameters.
- **B — LoRA**: rank 16, alpha 32 on every attention and FFN matrix (memory
  gate/out included, lookup excluded), AdamW.
- **C — sparse memory FT**: per fact batch, rank slots by TF-IDF
  `tf(i) * log((|B|+1)/(df(i)+1))`, with df from 1,000 background batches of
  pretraining data; SGD on the values of the top-t slots only (t
  *dev-selected*). Keys and everything else frozen.

Sleep factor (every 100 facts, not after the last one):

- **Self-quiz**: the model answers the in-distribution probe for every fact it
  has been told about so far (it keeps the cues, never the answers). Its own
  answers become the labels: no ground truth enters sleep.
- **Reset** (sparse arm only): slots written since the last sleep return to
  their pretrained values *before* distillation, so the dense weights are
  trained in the configuration they will run in.
- **Distil**: AdamW on the dense parameters only (everything outside the memory
  lookup) over self-labelled paraphrases of the cued facts, each step mixed
  with pretraining replay windows. Budget *dev-selected*, identical for both
  sleep arms.

Cells: **A**, **A+sleep**, **C**, **D = C+sleep**, plus **B**.

## Metrics

Learning:
- **Acquisition**: probe accuracy on a fact right after its write.
- **Retention**: probe accuracy at end of stream, overall and on the first 100
  facts ("early").
- **Saturation diagnostics**: acquisition and retention by 100-fact bin;
  for sparse arms, the fraction of each fact's top-t slots already written by
  earlier facts.

Forgetting (relative to the pretrained model):
- **delta-NLL**: change in held-out TinyStories loss (nats/token, 128 fixed
  validation windows).
- **Old-fact drop**: change in probe accuracy on 600 old facts (100 old
  entities x 6 attributes), in-distribution template.

## Protocol

1. Dev stream (seed 0): coarse LR grids for each arm to find the range where
   acquisition goes from low to high; choose K, t and the sleep budget. Then
   freeze grids and settings.
2. Test streams (seeds 1+): every arm x every LR in its frozen grid.
3. Report test-stream results; dev results are reported separately and never
   pooled with test.

## Pre-registered decision rules

An arm's **operating point** is its LR with the highest end-of-stream
retention (in-distribution) whose forgetting is within budget:
delta-NLL <= 0.05 nats and old-fact drop <= 5 points (mean over test seeds).

- **Q1 passes** if, at matched acquisition (interpolated on the LR frontier at
  the highest acquisition level both A and C reach, capped at 0.9), C's
  old-fact drop and delta-NLL are each at most half of A's. Also report B.
- **Saturation present** if C's operating point shows early-fact retention at
  least 15 points below its mean acquisition, or last-bin acquisition at least
  10 points below first-bin acquisition.
- **Q2 passes** if saturation is present and D's operating point beats C's
  operating point on overall retention by at least 10 points, and beats
  A+sleep's operating point by at least 5 points.
- **Informative failure** if sleep raises forgetting beyond budget, or
  post-sleep retention of the facts just consolidated falls more than 20
  points below their pre-sleep retention (the dense update is the bottleneck,
  which points to the thread's idea 4: a curvature shield on dense updates).

## Known limits

One pretrained model, so no variance across pretraining seeds. Tiny model:
mechanism evidence only, not evidence about behaviour at scale. Facts are
independent single-hop triples; no compositional or procedural learning.
