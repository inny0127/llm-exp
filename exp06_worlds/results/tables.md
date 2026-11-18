## Exp 06: syllabus (final accuracy, means over seeds)

| Policy | Seen d1 | d2 | d3 | d4 | Mean seen | Held-out d3 | Held-out d4 | Peak step of each depth's share (seed means) |
|---|---|---|---|---|---|---|---|---|
| random | 1.00 | 1.00 | 1.00 | 0.99 | 0.997 | 0.68 | 0.64 | 3567, 2533, 5017, 4467 |
| lp | 0.99 | 0.98 | 0.96 | 0.98 | 0.976 | 0.24 | 0.21 | 583, 1267, 3600, 7050 |
| easy_first | 0.03 | 0.08 | 0.03 | 0.99 | 0.282 | 0.01 | 0.32 | 250, 2250, 4250, 6250 |
| hard_first | 1.00 | 0.05 | 0.07 | 0.05 | 0.292 | 0.07 | 0.05 | 6250, 4250, 2250, 250 |

## Exp 07: beliefs (conflict probes, means over seeds)

| Order | Code (bits/item) | Rule belief | Shortcut belief |
|---|---|---|---|
| random | 0.861 ± 0.149 | 0.848 ± 0.115 | 0.040 |
| rule_first | 1.074 ± 0.009 | 0.000 ± 0.000 | 1.000 |
| rule_last | 1.021 ± 0.004 | 0.040 ± 0.007 | 0.043 |
| lp | 1.174 ± 0.032 | 0.605 ± 0.209 | 0.259 |
