# Exp 02 — Fixing the two bottlenecks of two-speed weights

Follow-up to [Exp 01](../exp01_two_speed_weights/), which found that sparse
memory writes beat fine-tuning but still damaged old facts, and that sleep
consolidation added retention at the cost of old knowledge. Exp 02 tests one
fix for each half, and the two together.

- `DESIGN.md`: the pre-registration (design panel, dev rules, decision rules),
  committed before any test run.
- `RESULTS.md`: what happened and the go/no-go.
- `freeze.py`: applies the dev rules mechanically and prints the frozen choices.
- `run_test.sh`: the frozen test runs (`LSTAR=... R=... ./run_test.sh core|extras`).
- `analyze02.py`: decision rules, paired sleep effects, the reset-versus-dense
  split of every sleep, figures. Each run's flags are checked against its name.

The engine is `../exp01_two_speed_weights/stream.py`; Exp 02 only adds flags
whose defaults reproduce Exp 01 bit for bit.
