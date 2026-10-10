# Exp 02 — Fixing the two bottlenecks of two-speed weights

Pre-registration for the test phase (stream seeds 1-3). The plan came from a
design panel (four independent designers running their own dev screens,
three judges, one synthesizer) and an adversarial code review; every choice
below is either fixed outright or made by a mechanical rule on the dev
stream (seed 0). This file is committed before any test run, with the frozen
choices filled in at the bottom.

Same testbed, pretrained model, fact streams and metrics as Exp 01. Code
changes are new flags in `../exp01_two_speed_weights/stream.py`; with every
new flag at its default the code is bit-identical to Exp 01's committed
`stream.py` (checked on dev for 50 facts, with and without Exp 01's sleep:
identical write steps, slot overlaps, acquisitions, log-probs, checkpoints
and sleep records).

## Where Exp 01 left off

Sparse memory writes beat dense fine-tuning but still lost 24 points of
old-fact accuracy over 500 facts; sleep added 19 points of retention but cost
19 points of old facts and +0.06 nats.

## Hypotheses

- **H1 (wake).** Wake damage reaches old facts through slots that new and old
  facts share: syllable-piece name slots, hit by the gradient that memorises
  the new name's spelling, and template-format slots the writer's own
  paraphrases share with old facts. Training only the answer token removes
  the first; down-weighting slots that known (non-eval) entities read in the
  writer's own templates removes the second.
- **H2 (sleep).** Sleep's old-fact cost is entity-level interference in the
  dense name-to-attribute map. Training the dense weights only on answer
  tokens, rehearsing answers about other old entities the frozen pretrained
  model knows, and freezing the tied embeddings removes it.
- **H3.** The two fixes compose.

## Arms

Writers (wake phase):

| Name | Flags | Role |
|---|---|---|
| `C` | Exp 01 writer (TF-IDF top-128, loss on every token) | baseline and FU2 partner; rerun on test and checked bit for bit against Exp 01's files |
| `p06ans` | `--write_loss answer --protect_gamma 1 --protect_templates stream` | **FU1 primary**. Slot score divided by 1 + P, P = slot reads by the frozen pretrained model on cloze prompts of entities 100-499 in templates 0-6 (the writer's own) |
| `pnpans` | as `p06ans` with P from templates 1-6 | dev control: does protection only work because P contains a probe template? |
| `ans` | `--write_loss answer` | loss-mask-only reference (test extra E2 at lr 0.5) |

Sleeps (every 100 facts, not after the last; Exp 01 schedule: 3 epochs,
lr 1e-4, 32 fact sentences and 16 clean replay windows per step):

| Name | Flags |
|---|---|
| `gentle` | `--sleep --sleep_fact_loss answer --sleep_rehearse 16 --rehearse_labels teacher --rehearse_templates nonprobe --sleep_freeze embed --clean_replay` |
| `gentle32` | `gentle` with `--sleep_rehearse 32` (anchor-free reducer of dense damage) |
| `sleepclean` | Exp 01 sleep with `--clean_replay` (test extra E3) |

Gentle sleep in words: self-quiz on every fact told so far, reset the written
slots, then train the dense weights (embeddings frozen) on the answer tokens
of the model's own answers, plus answers about other old entities labelled by
the frozen pretrained model (templates 1-6 only, own RNG), plus replay of
pretraining windows that never touch an eval entity's document.

Dropped, with reasons: EWC anchored at the pretrained weights (on dev it
pulled back facts consolidated in earlier sleeps: all-fact recall -5 points
by sleep 4); KL-to-pretrained replay (no gain in three screens); freezing
lower blocks, 2-epoch sleep (no gain, more damage); full FT with answer-only
loss (bears on neither bar).

## Leakage rules

1. Eval old entities are 0-99. Protection counts, rehearsal pool and teacher
   labels use entities 100-499 only (asserted).
2. Template 7 appears in no Exp 02 statistic and no training sentence; it is
   the clean held-out probe and a binding guard.
3. Every sleep arm uses clean replay: windows overlapping any pretraining
   document that names an eval entity are rejected (a token-level document
   mask; 0 eval-name fragments in 3,000 sampled windows). Exp 01's sleep
   replayed raw text, ~9% of whose windows touch eval bios, which flattered
   it; on dev the clean version is worse (old-fact drop 44.2 vs 39.2).
4. The IDF background is unchanged from Exp 01 (raw pretraining text, the
   system's own data) and identical across arms.
5. The in-sleep old-fact probes are logging only; they never feed training.

## Dev rules (mechanical; test seeds untouched)

**FU1 primary LR.** Candidates: `p06ans` at lr 4 and 8 (2 only if 4 is
ineligible). Eligible on dev only if old-fact drop <= 6.0, retention >= 0.50,
held-out retention >= 0.40, held-out old-fact drop <= 8.0, and <= 5% of facts
hit the 20-step cap. L* = eligible LR with the highest retention (within 0.02:
lower drop wins). If none is eligible: lowest-drop `p06ans` LR with retention
>= 0.50 and held-out retention >= 0.40, flagged "thin margin"; failing that,
`ans` lr 0.5.

**FU2 recipe R.** Gate for each of `gentle`, `gentle32` on `C` lr 16 against
`C` lr 16 without sleep: retention +0.15 or more, held-out retention +0.15 or
more, old-fact drop and held-out old-fact drop not worse, delta NLL <= +0.010,
mean (pre_all - post_all) per sleep <= 0.01. Choose `gentle32` if both pass
and its summed dense damage is >= 2.0 points lower than `gentle`'s with
retention no more than 0.02 lower and delta NLL no more than 0.003 higher;
else `gentle` if it passes; else `gentle32` if only it passes; else `gentle`
goes to test anyway and the dev failure is reported.

## Test plan (seeds 1, 2, 3; 500 facts)

Core (12 runs): `p06ans` at L*; `C` lr 16 (rerun, bit-identity check against
Exp 01); `C+R` lr 16; `p06ans+R` at L*.

Extras, in order, budget permitting (every launched run is reported):
E1 `p06ans` at L*/2 and 2L*; E2 `ans` lr 0.5; E3 `C+sleepclean` lr 16.

## Decision rules (means over seeds 1-3, final checkpoint)

- **FU1 passes** if `p06ans` at L* has old-fact drop <= 10 and retention >=
  0.45, with binding guards held-out old-fact drop <= 10 and held-out
  retention >= 0.40. Bar met but a guard failed = "template-bound pass",
  counted as a fail.
- **FU2 passes** if `C+R` minus `C` (same seeds) has delta NLL <= +0.02,
  old-fact drop change <= +3, retention change >= +0.10, with binding guards
  held-out old-fact drop change <= +3 and held-out retention change >= +0.10,
  and consolidation intact (mean pre_all - post_all <= 0.01). Reported with
  the summed reset gain and dense damage: "gentle consolidation" if mean
  summed dense damage <= 3 points, otherwise "passes through slot-reset
  repair; the dense update still costs X points".
- **Combined passes** if `p06ans+R` at L* meets every FU1 criterion in
  absolute terms and every FU2 criterion against `p06ans` at L*.
- **Go to the 5090** if and only if FU1 and Combined pass. A mean within one
  across-seed SD of its threshold is flagged "marginal"; the verdict still
  follows the mean.

## Frozen choices

Filled in from the dev rules before any test run:

- L* = 4 (freeze.py on runs/exp02_dev4: lr 4 meets all five eligibility checks: retention 0.79, held-out 0.63, old-fact drop 5.7, held-out drop 6.5, 0.6% at the cap; lr 8 fails with drop 11.7). Committed before any test run that uses it.
- R = gentle32 (freeze.py: both recipes pass the dev gate; gentle32 paired +0.31 retention, -12.0 old-fact drop, -0.004 NLL, summed dense damage 4.7 vs 7.3 for gentle, i.e. 2.6 points lower, with retention 0.752 vs 0.734 and NLL not higher). Committed before any sleep test run.
