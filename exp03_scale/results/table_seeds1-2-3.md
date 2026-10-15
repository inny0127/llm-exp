| Arm | LR | Seeds | Retention | Held-out template | First 100 | Old-fact drop (pts) | Old drop, held-out template | Delta NLL | Steps/fact | Over all sleeps: reset gives back / dense update takes (pts) |
|---|---|---|---|---|---|---|---|---|---|---|
| Full fine-tuning | 3e-05 | 3 | 0.13 ± 0.00 | 0.12 | 0.06 | 89.0 ± 0.5 | 87.9 | +0.319 | 1.7 |  |
| Exp 01 writer | 16 | 3 | 0.57 ± 0.00 | 0.54 | 0.43 | 61.1 ± 1.3 | 62.3 | +0.001 | 1.8 |  |
| Exp 01 writer + gentle sleep (32 rehearsal) | 16 | 3 | 0.45 ± 0.01 | 0.45 | 0.26 | 57.6 ± 2.1 | 57.0 | +0.020 | 1.1 | +458.3 / +36.2 |
| Protected answer-only writer | 4 | 3 | 0.95 ± 0.00 | 0.85 | 0.93 | 0.7 ± 0.5 | 0.8 | +0.001 | 1.6 |  |
| Protected answer-only writer + gentle sleep (32 rehearsal) | 4 | 3 | 0.99 ± 0.00 | 0.98 | 0.99 | 30.7 ± 2.3 | 30.3 | +0.018 | 1.5 | -1.2 / +30.8 |
