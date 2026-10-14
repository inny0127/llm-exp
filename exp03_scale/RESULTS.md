# Exp 03 results (interim): the Exp 02 recipe at 8x scale and 4x longer streams

> **Status: interim.** 2 of 15 test runs are done; the other 13 are queued on a free Colab
> T4. The free tier grants about one 2-hour runtime a day, so they will take a few more days.
> This page states what the finished runs already settle and what they do not. It is updated
> as runs land.
>
> Under DESIGN.md, the formal verdict needs all 15 test runs. The finished runs bound the
> combined arm firmly enough that no outcome of the remaining runs can make it pass.

**Bottom line so far.**

- **The wake fix scales very well.** On dev (seed 0), the protected answer-only writer stores
  2,000 facts at 0.96 retention while losing 0.5 points of old facts. Without protection,
  the Exp 01 writer loses 60-79 points; full fine-tuning loses 88.
- **The sleep fix does not scale.** On two test seeds, gentle sleep raises retention to 0.99
  but costs 32 points of old facts, against a 10-point bar.
  - Every point comes from the dense update. It takes about 1.7 points per sleep, roughly
    flat over the 19 sleeps.
  - Resetting the written slots gives back nothing, because the protected writer barely
    damages old facts to begin with.
  - Exp 02 predicted this accumulation.
- **The recipe as a whole does not scale.** The combined arm cannot meet the 10-point bar:
  its two finished seeds lost 32.0 and 32.7 points. Even a third seed that lost nothing
  would leave the mean above 21.
- **The next lever is the sleep.** Either change the dense consolidation step or drop it.
  At this scale the writer alone already does most of what the recipe was meant to do.

## Setup (attempt 2; DESIGN.md has the full pre-registration and amendments)

| | |
|---|---|
| Model | 8 layers, d 512, ctx 256 |
| Parameters | 26M dense + 134M memory (512² slots, 4 heads x top-32) |
| Pretraining | 299M tokens, fp16 on a T4, resumed once across runtimes |
| Pretrained model | held-out loss 1.268; old-fact accuracy 1.00 on template 0 and 1.00 on the held-out template (validity bar: 0.80) |
| Streams | 2,000 new facts; sleep every 100 facts (19 sleeps) |
| Frozen on dev | L* = 4 for `p06ans` (eligible, no flag); C LR = 16; A LR = 3e-5; R = gentle32 (fixed by Exp 02) |

Attempt 1 is in `results_attempt1/`: same data, a different pretrained model that was lost
before its test finished. It is reported, not counted. Its dev results and 2 test runs agree
with attempt 2 to within a few points.

## Dev (seed 0, 2,000 facts; selects LRs only)

Old-fact drop is in points. The curve column gives the drop at every 250 facts.

| Writer | LR | Retention | Held-out | First 100 | Old-fact drop | Delta NLL | Old-fact drop curve |
|---|---|---|---|---|---|---|---|
| `p06ans` | 2 | 0.937 | 0.764 | 0.90 | 0.2 | +0.001 | 0.2 → 0.2 |
| **`p06ans`** | **4** | **0.964** | **0.854** | **0.92** | **0.5** | **+0.001** | 0.2 → 0.5 |
| `p06ans` | 8 | 0.964 | 0.903 | 0.91 | 1.8 | +0.003 | 0.2 → 1.8 |
| `p06ans` | 16 | 0.970 | 0.929 | 0.95 | 5.0 | +0.011 | 0.5 → 5.0 |
| `C` (Exp 01 writer) | 16 | 0.554 | 0.530 | 0.41 | 60.0 | +0.001 | 36.8 at 250 → 60.0 |
| `C` | 32 | 0.562 | 0.533 | 0.47 | 71.2 | +0.002 | 65.0 at 250 → 71.2 |
| `C` | 64 | 0.546 | 0.509 | 0.51 | 79.0 | +0.005 | 77.5 at 250 → 79.0 |
| `A` (full FT) | 3e-6 | 0.081 | 0.093 | 0.07 | 88.2 | +0.136 | 49.8 at 250 → 88.2 |
| `A` | 1e-5 | 0.097 | 0.093 | 0.07 | 88.3 | +0.168 | 58.5 at 250 → 88.3 |
| `A` | 3e-5 | 0.114 | 0.115 | 0.07 | 87.5 | +0.300 | 76.2 at 250 → 87.5 |

**Scale made the Exp 01 writer much worse and the protected writer much better.**

- At 500 facts on the 3.8M model, the two writers lost 23.9 and 7.9 points (Exp 02, test).
  Here the Exp 01 writer loses 37-78 points within the first 250 facts.
  - It writes the top-128 TF-IDF slots of each fact's sentences, name pieces and template
    words included.
  - In a 262K-slot memory those slots are concentrated, and heavily shared with the old
    entities.
- Training only the answer token and down-weighting slots that known entities read
  (`p06ans`) stops nearly all of it.
- Exp 02 predicted that the writer alone would cross the 10-point bar at 2,000 facts. That
  prediction was wrong: the larger memory made protection more effective, not less.

## Test (seeds 1-3): 2 of 15 runs done

| Run | Retention | Held-out | First 100 | Old-fact drop | Held-out drop | Delta NLL |
|---|---|---|---|---|---|---|
| `p06ans+gentle32` lr 4, seed 1 | 0.986 | 0.984 | 0.97 | 32.0 | 31.7 | +0.016 |
| `p06ans+gentle32` lr 4, seed 2 | 0.988 | 0.985 | 1.00 | 32.7 | 31.3 | +0.018 |
| *attempt 1, seed 1 (other model)* | *0.994* | *0.989* | | *29.0* | *28.3* | *+0.016* |
| *attempt 1, seed 2 (other model)* | *0.987* | *0.984* | | *27.8* | *26.5* | *+0.014* |

Old-fact drop over the stream, at every 250 facts:

- Seed 1: 5.0, 8.0, 12.5, 14.7, 18.7, 28.7, 25.3, 32.0.
- Seed 2: 3.0, 12.3, 10.3, 12.7, 19.2, 25.7, 27.7, 32.7.

The loss grows steadily with the number of sleeps, while the writer alone (dev) stays flat.

Each sleep logs old-fact accuracy before the sleep, after the slot reset, and after the
dense update:

| | Seed 1 | Seed 2 |
|---|---|---|
| Slot reset gives back (sum over 19 sleeps) | -3.5 | -0.7 |
| Dense update takes (sum) | 32.0 | 32.8 |
| Dense update per sleep (mean) | 1.7 | 1.7 |
| Self-labelling accuracy | 0.987 | 0.990 |
| Consolidation loss (pre_all - post_all) | 0.0003 | 0.0001 |

The sleep does its own job almost perfectly:

- it labels the stream facts correctly 99% of the time;
- it moves them into the dense weights without loss;
- it lifts retention from about 0.96 (writer alone, dev) to about 0.99, with the held-out
  template at 0.98.

But each sleep's dense update still interferes with the probed old entities. Those
entities are deliberately excluded from rehearsal, so they measure interference on
knowledge the sleep never sees. At about 1.7 points per sleep, this is the same rate Exp 02
measured on the 3.8M model (1.5-1.7 per sleep). Scaling the model 8x did not reduce it,
and a 4x longer stream multiplies it.

## Verdict against the pre-registered rules (provisional)

| Rule | Status |
|---|---|
| FU1: `p06ans` at L* = 4 (bar: drop <= 10, retention >= 0.45, held-out guards) | **Pending** (0 of 3 test runs). Dev: drop 0.5, retention 0.96. |
| FU2: `C+gentle32` minus `C` at lr 16 | **Pending** (0 of 6 runs). |
| Combined: `p06ans+gentle32` at L* = 4 (absolute FU1 bar) | **Cannot pass.** 2 of 3 seeds lost 32.0 and 32.7 points; the mean stays above 21 whatever the third seed does. |
| The recipe scales (FU1 and Combined) | **No** (follows from Combined). |

## What to do next (Exp 04 candidates; not pre-registered yet)

1. **Writer only, no sleep.** It already holds 0.96 of 2,000 facts at 0.5 points of old-fact
   loss. Sleep buys 3 points of retention for 32 points of old knowledge, which is not worth
   it at this scale.
2. **Sleep without dense interference.**
   - Train only the blocks above the memory layer.
   - Consolidate into a separate fresh memory region instead of the shared dense weights.
   - Or sleep less often: the damage is per sleep, not per fact.
3. **Rehearsal that covers what it protects.** Today the dense update never sees the eval
   entities, by design. A rehearsal set drawn from the model's own knowledge across all
   entities (self-generated, so not a leak) would test whether coverage, not volume, is
   what is missing.

## Compute and logistics

- **Hardware:** a free Colab T4 driven through the Colab CLI by `colab_cli_runner.py`. The
  account has no compute units and A100 is not available to it.
- **Session pattern:** runtimes lasted about 2-5 hours, and new ones were granted roughly
  once a day.
- **Fixes this forced, all checked:**
  - **fp16 pretraining:** same loss as bf16, 2.9x faster on T4.
  - **Streaming data reads:** the 12 GB VM ran out of memory otherwise.
  - **Copies over SSH, at about 20 MB/s:** the runtime's file API returned errors. Attempt
    1's model was lost before this fix.
  - **Resumable stream runs:** a resumed run is identical to an uninterrupted one on CPU.
  - **Detecting a CPU runtime** silently auto-created after a lost GPU one.
