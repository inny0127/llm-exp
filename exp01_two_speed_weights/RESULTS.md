# Exp 01 results — Two-speed weights

<!-- VERDICT -->

## Deviations from the pre-registered design

All of these were decided on the dev stream (seed 0) before any test-stream run.
The pre-registered thresholds and decision rules are unchanged; one rule (Q1) had
to be restated because the new protocol makes acquisition about 1 by
construction.

| # | Pre-registered | Changed to | Why (dev evidence below) |
|---|---|---|---|
| 1 | K fixed gradient steps per fact | Train each fact until its in-distribution probe is answered, max 20 steps | With one step per fact no arm learned anything: acquisition 0.09–0.30 against chance 0.07 (table D1) |
| 2 | AdamW, beta1 = 0.9 (A, B) | beta1 = 0 | With one step per fact, momentum makes each write mostly a replay of earlier facts' gradients. Full FT at lr 1e-5: old-fact drop 59 points with momentum vs 23.5 without, at half the steps (table D2) |
| 3 | IDF background: 1,000 batches (size unspecified; first run used 384 tokens) | 1,000 batches of 4,096 tokens (32 windows), Meta's 64x64 batch shape | Sparse memory at lr 64: old-fact drop 63 points with 384-token batches vs 8.8 with 4,096 (table D2, slot probe) |
| 4 | 1,000-fact stream | 500 facts, sleep every 100 (4 sleeps) | Training to criterion costs up to 7x more steps per fact for full FT |
| 5 | LR grids of 4 | 3 per arm, spanning the range where the arm learns | Compute |
| 6 | Q1 at matched acquisition, interpolated | Least old-fact drop among LRs that reach acquisition >= 0.9 | Acquisition is ~1 at every LR that stays under the step cap |
| 7 | Sleep budget dev-selected | Sleep lr 1e-4, 3 epochs over the self-labelled set, 16 replay windows (2,048 tokens) per step | Cheapest setting that kept consolidated facts intact without blowing the NLL budget (table D3) |

### D1 — pre-registered protocol fails to learn (dev, 200 facts, 1 step/fact)

| Arm | LR | Acquisition | Old-fact drop (pts) | Delta NLL |
|---|---|---|---|---|
| Full FT | 1e-5 / 3e-5 / 1e-4 / 3e-4 / 1e-3 | 0.09 / 0.12 / 0.11 / 0.13 / 0.12 | 7.5 / 56.0 / 78.2 / 88.3 / 90.7 | +0.009 / +0.049 / +0.170 / +0.731 / +6.96 |
| LoRA | 1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2 | 0.10 / 0.10 / 0.12 / 0.12 / 0.14 | 67.8 / 62.7 / 81.0 / 89.3 / 90.2 | +0.082 / +0.105 / +0.170 / +1.07 / +7.46 |
| Sparse memory | 1 / 4 / 16 / 64 / 256 / 1024 | 0.12 / 0.17 / 0.22 / 0.22 / 0.23 / 0.30 | 8.8 / 25.3 / 49.7 / 69.0 / 77.0 / 84.2 | +0.000 / +0.001 / +0.002 / +0.006 / +0.023 / +0.078 |

Forgetting is easy and learning is hard: a 4M-parameter model needs many
exposures per new fact (it took about 50 bio exposures per old entity during
pretraining before old-fact accuracy passed 60%).

### D2 — protocol diagnostics (dev, 50 facts, train to criterion, cap 30)

| Setting | Retention | Held-out template | Old-fact drop (pts) | Delta NLL | Steps/fact |
|---|---|---|---|---|---|
| Full FT lr 1e-5, beta1 0.9 | 0.48 | 0.44 | 59.2 | +0.065 | 14.9 |
| Full FT lr 1e-5, beta1 0 | 0.48 | 0.50 | 23.5 | +0.022 | 7.2 |
| Full FT lr 3e-5, beta1 0 | 0.48 | 0.52 | 29.7 | +0.028 | 2.9 |
| Full FT lr 3e-6, beta1 0 | 0.28 | 0.34 | 17.7 | +0.016 | 19.3 (32% hit cap) |
| LoRA lr 3e-4, beta1 0 | 0.18 | 0.16 | 70.0 | +0.123 | 2.5 |
| Sparse lr 64, IDF batch 384 tok | 0.50 | 0.50 | 63.2 | +0.005 | 2.5 |
| Sparse lr 64, IDF batch 4,096 tok | 0.92 | 0.78 | 8.8 | +0.002 | 2.0 |
| Sparse lr 256, IDF batch 4,096 tok | 0.82 | 0.72 | 32.8 | +0.013 | 1.6 |

### Why sparse writes still hit old facts (slot probe, dev, 40 facts, t = 128)

| IDF batch | Selected-slot reads at entity-name tokens | at template tokens | at the value-predicting position | Selected slots also read in old-fact prompts (base rate 17.9%) |
|---|---|---|---|---|
| 384 tokens | 95% | 5% | 0.4% | 88% |
| 4,096 tokens | 88% | 10% | 1.4% | 71% |

TF-IDF picks slots that encode the new entity's *name*. Invented names share
syllable pieces, so those slots are also part of how old entities' names are
read. Rewriting them garbles old names; old answers are then lost even though
the slots read at the answer position are almost never touched (1.4–2.5%).
Larger IDF batches push the commonly shared name-piece slots out of the
selection but do not remove the collision.

### D3 — sleep budget (dev, 300 facts, sparse lr 64; sleeps after 100 and 200)

| Sleep lr | Epochs | Replay windows/step | Last 100 facts before → after each sleep | Retention @300 | Old-fact drop | Delta NLL |
|---|---|---|---|---|---|---|
| (no sleep) | | | | 0.62 | 31.8 | +0.005 |
| 1e-5 | 3 | 4 | 0.82→0.15, 0.67→0.15 | 0.27 | 45.3 | +0.016 |
| 3e-5 | 3 | 4 | 0.82→0.38, 0.70→0.39 | 0.39 | 43.2 | +0.026 |
| 3e-5 | 3 | 16 | 0.82→0.39, 0.73→0.43 | 0.40 | 40.7 | +0.018 |
| **1e-4** | **3** | **16** | **0.82→0.80, 0.85→0.85** | **0.73** | **38.8** | **+0.044** |
| 3e-4 | 1 | 4 | 0.82→0.64, 0.74→0.64 | 0.59 | 47.0 | +0.162 |
| 3e-4 | 3 | 4 | 0.82→0.82, 0.86→0.86 | 0.79 | 45.5 | +0.258 |
| 1e-3 | 1 | 4 | 0.82→0.74, 0.64→0.62 | 0.58 | 71.9 | +0.689 |
| 1e-3 | 3 | 4 | 0.82→0.82, 0.61→0.61 | 0.67 | 71.2 | +0.789 |

Below lr 1e-4 the dense weights do not absorb the facts before the slots are
reset, so the reset deletes them. At and above 3e-4 the facts survive but the
dense update damages everything else.
