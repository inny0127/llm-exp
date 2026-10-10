# Exp 03, attempt 1 (free Colab T4, 2026-10-10): incomplete

What ran, all on one pretrained model:

- data and smoke checks;
- pretraining: 18,239 steps on 299M tokens, fp16;
  - val loss 1.269;
  - old-fact accuracy 1.00 on template 0 and 1.00 on the held-out template;
- the cost probe and all 10 dev runs;
- the freeze (L* = 4, C LR = 32, A LR = 3e-5; no flag);
- 2 of the 15 test runs: `p06ans+gentle32` at lr 4, seeds 1 and 2.

**Why it stopped.** Colab reclaimed the runtime after about 5 hours (20:33 UTC), and new T4
requests then returned "Service Unavailable". The pretrained weights had not finished copying
off the VM (the runtime's file API returned HTTP 500, and the slower fallback copy reached
200 of 640 MB). Pretraining on CUDA is not bit-reproducible: `embedding_bag`'s backward pass
uses atomic adds. So the test cannot be resumed on the same model.

**What counts.** Per DESIGN.md, every launched run is reported. These runs are reported here
and do **not** enter the Exp 03 verdict. The verdict needs dev, freeze and all 15 test runs
on one pretrained model. A rerun pretrains a new model and repeats dev and freeze before any
test run.

**Preliminary readings (one model, dev seed and 2 test seeds):**

| Arm (dev, seed 0) | LR | Retention | Old-fact drop (pts) | Delta NLL |
|---|---|---|---|---|
| `p06ans` | 4 | 0.957 | 0.3 | +0.001 |
| `C` | 32 | 0.544 | 70.0 | +0.002 |
| `A` | 3e-5 | 0.126 | 89.0 | +0.311 |

| Test run | Retention | Held-out retention | Old-fact drop (pts) | Delta NLL |
|---|---|---|---|---|
| `p06ans+gentle32` lr 4, seed 1 | 0.994 | 0.989 | 29.0 | +0.016 |
| `p06ans+gentle32` lr 4, seed 2 | 0.987 | 0.984 | 27.8 | +0.014 |

Over the 19 sleeps in the two test runs, slot resets returned 1.7 and 4.3 points of old
facts, and the dense update took 29.5 and 28.5 points (about 1.5 per sleep). Consolidation
itself lost nothing: pre_all - post_all averaged 0.0002 and 0.0004.

These match Exp 02's prediction that the sleep's per-sleep dense damage would add up over a
long stream. They contradict its prediction that the writer alone would exceed the 10-point
bar at 2,000 facts: on dev it lost 0.3 points.
