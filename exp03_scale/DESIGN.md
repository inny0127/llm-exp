# Exp 03: Does the Exp 02 recipe hold at 8x scale and 4x longer streams?

This is the pre-registration. It was committed before any Colab run. Every choice below is
either fixed outright or made by a mechanical rule on the dev stream (seed 0). The driver is
`run_exp03.py`, and it applies these rules itself.

## Question

Exp 02 fixed both bottlenecks of two-speed weights on the 3.8M-parameter testbed with
500-fact streams:

- Wake: the protected answer-only writer `p06ans` (FU1).
- Sleep: gentle sleep with 32 rehearsal sentences, `gentle32` (FU2).

Does the same recipe, with its hyperparameters frozen except the write LR, still pass the same
bars when two things change?

1. **Scale.** The model has 26M dense parameters and 134M memory parameters, i.e. 262K slots
   with 32 read per head.
2. **Stream length.** The stream holds 2,000 facts with 19 sleeps, so old knowledge is
   exposed to four times as many writes.

## Testbed changes from Exp 02 (nothing else changes)

| | Exp 02 (`tiny`) | Exp 03 (`small`) |
|---|---|---|
| Model | 4 layers, d 256, ctx 128 | 8 layers, d 512, ctx 256, memory in block 4 |
| Memory | 256² slots, 4 heads x top-16 | 512² slots, 4 heads x top-32 |
| Pretraining | 30M tokens (200 MB of TinyStories V2) | 300M tokens (about half of TinyStories V2 train), bf16 with fp32 slot routing |
| Old entities | 500 (100 bios each) | 1,000 (100 bios each: same exposures per entity) |
| Stream | 500 facts, 4 sleeps | 2,000 facts, 19 sleeps |
| IDF background batch | 32 x 128 tokens | 16 x 256 tokens (the same 4,096 tokens) |
| Checkpoints | every 62-100 facts | every 250 facts |

Everything else is unchanged from Exp 02:

- Tokenizer: Exp 01's file. Names split into the same syllable pieces.
- Templates, relations and probes. Template 7 stays held out.
- The leakage rules:
  - eval old entities are 0-99;
  - protection, rehearsal and teacher labels use the other old entities;
  - clean replay uses the token-level eval-document mask.
- The writer, with its train-to-criterion cap of 20 steps.
- The sleep schedule and every gentle-sleep setting.

Pretraining runs one epoch with AdamW (0.9, 0.95), weight decay 0.1 on the dense matrices,
batch 64 x 256, LR 8e-4, 500 warmup steps, and a cosine decay to 10%.

**Validity condition:** the pretrained model must answer at least 80% of the eval old-fact
probes (template 0). Otherwise there is nothing to forget, and the experiment stops and reports.

## Arms

- **`C`**: the Exp 01 writer, TF-IDF top-128 slots with loss on every token. FU2 partner.
- **`p06ans`**: the frozen FU1 writer. It is answer-only and soft-protects slots that
  non-eval old entities read in templates 0-6.
- **`C+gentle32`, `p06ans+gentle32`**: the same writers with R = `gentle32`. R is fixed by
  Exp 02 and not re-selected.
- **`A`**: full fine-tuning with Adam (beta1 = 0). It is a dense reference and enters no
  decision.

## Dev rules (seed 0; the test streams stay untouched)

The scale change can move every LR, so each writer runs a grid around its Exp 02 value:
`C` at 16, 32 and 64; `p06ans` at 2, 4, 8 and 16; `A` at 3e-6, 1e-5 and 3e-5.

- **L\*** for `p06ans`: Exp 02's `freeze.fu1`, unchanged.
  - Eligibility at dev: old-fact drop <= 6, retention >= 0.50, held-out retention >= 0.40,
    held-out drop <= 8, and <= 5% of facts at the step cap.
  - Among eligible LRs, pick the highest retention; within 0.02 of it, the lower drop wins.
  - If no LR is eligible, apply the thin-margin rule.
  - If even that fails, the LR with the highest retention goes to test, flagged "no
    eligible LR at this scale", and the failure is reported.
  - The eligibility thresholds are Exp 02's. A 2,000-fact stream makes them stricter, and
    that is intended.
- **C LR, A LR**: the highest retention on dev; within 0.02 of it, the lower old-fact drop
  wins.

## Test (stream seeds 1, 2, 3; 15 runs)

`p06ans+gentle32` at L*, `C+gentle32` at C LR, `A` at A LR, `p06ans` at L*, and `C` at C LR.

## Decision rules (means over seeds 1-3, final checkpoint)

These are Exp 02's bars, unchanged and measured over 2,000 facts (`analyze02.py`):

- **FU1 holds** if `p06ans` at L* has:
  - old-fact drop <= 10 points;
  - retention >= 0.45;
  - binding guards: held-out drop <= 10 and held-out retention >= 0.40.
- **FU2 holds** if `C+gentle32` minus `C` (paired by seed) has:
  - delta NLL <= +0.02;
  - old-fact drop change <= +3;
  - retention change >= +0.10;
  - the same two held-out guards;
  - consolidation intact: mean pre_all - post_all <= 0.01.
  - The sleep is labelled "gentle consolidation" if its dense damage is <= 0.75 points per
    sleep. That is 3.0 points over Exp 02's 4 sleeps, the same per-sleep rate.
- **Combined holds** if `p06ans+gentle32` meets every FU1 criterion in absolute terms and every
  FU2 criterion against `p06ans`.
- **The recipe scales** if and only if FU1 and Combined hold. `analyze02.py` still writes
  this field as `go_to_5090`.
  - A mean within one across-seed SD of its threshold is flagged "marginal".
  - The verdict still follows the mean.

## Budget rules (Colab A100, compute units)

`--budget_cu 60` is about 30% of a month's AI Pro allotment at about 13 CU/h. The driver
stops itself in three cases:

1. **Smoke stage:** if the timed pretraining steps project pretraining to take more than
   half the remaining budget.
2. **Probe stage:** if the timed 200-fact runs project dev + test above the remaining
   budget. The projection assumes no speed-up beyond the concurrency it measured.
3. **Any stage:** no new run starts once the budget is spent.

Every launched run is reported, including the probe runs. If the budget stops the test
early, the verdict is "incomplete" and no partial-seed verdict is drawn.
