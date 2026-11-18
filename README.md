# llm-exp

Small, pre-registered experiments on how neural networks could keep learning
after deployment.

| # | Experiment | Status |
|---|---|---|
| 01 | [Two-speed weights](exp01_two_speed_weights/): sparse memory writes + sleep consolidation | Done; [RESULTS](exp01_two_speed_weights/RESULTS.md) |
| 02 | [Fixing the two bottlenecks](exp02_fix_bottlenecks/) | Done; [RESULTS](exp02_fix_bottlenecks/RESULTS.md) |
| 03 | [The recipe at 8x scale](exp03_scale/) | Done; [RESULTS](exp03_scale/RESULTS.md) |
| 04 | [Choosing what to learn next](exp04_curriculum/) | Done; `results_gpu/` |
| 05 | [Order matters for compression](exp05_order_compression/) | Done; `results/` |
| 06-07 | [Rediscovering the syllabus; order decides beliefs](exp06_worlds/) | Done; `results/` |
| 08 | [Path ensemble and refusal](exp05_order_compression/DESIGN_exp08.md) | Done; `exp05_order_compression/results08/` |
| 09 | [Syllabus with review; belief phase transition](exp06_worlds/DESIGN_exp09.md) | Done; `exp06_worlds/results09/` |
| 10 | [Value-function teacher](exp10_value/) | Done; `results/` |

**Report:** [REPORT.md](REPORT.md) ([PDF](REPORT.pdf)) — Exp 01-10 in one write-up: what to learn and in
which order, seen through compression. Key result: prequential code length predicts the
generalisation of the learning *path* (checkpoint ensemble, Spearman ρ = 0.96), not of the final
model (ρ = 0.21).
