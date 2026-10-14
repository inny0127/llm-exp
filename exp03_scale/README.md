# Exp 03: The Exp 02 recipe at scale, on a Colab A100

This experiment asks whether the recipe that fixed both bottlenecks in
[Exp 02](../exp02_fix_bottlenecks/) still works under two changes:

- a model about 8x larger (26M dense + 134M memory parameters);
- streams 4x longer (2,000 facts, 19 sleeps).

The recipe is the protected answer-only writer plus gentle sleep.

## Files

- `DESIGN.md`: the pre-registration. It covers the testbed changes, dev rules, the bars
  (unchanged from Exp 02) and the budget rules.
- `exp03_colab.ipynb`: open it in Colab, choose an A100 runtime, then *Run all*. About 2 h,
  roughly 25-30 CU, capped at 60.
- `run_exp03.py`: the driver behind the notebook. Its stages are data, smoke, pretrain,
  prep, probe, dev, freeze, test, analyze and package. It is resumable through a Drive mirror
  and keeps track of the budget. `--dry --device cpu` runs the whole pipeline in miniature.
- `tokenizer.json`: Exp 01's tokenizer, reused so that names split the same way.
- `RESULTS.md`: results (interim until all 15 test runs land); `results_attempt2/` holds the runs, `results_attempt1/` the lost first attempt.

## Engine

The engine is `../exp01_two_speed_weights/`, shared with Exp 01 and 02, and it is now
device-aware:

- `--device cuda` and `--size small`;
- bf16 pretraining with resume.

On CPU it is bit-identical to the code that produced the Exp 01 and Exp 02 results.
