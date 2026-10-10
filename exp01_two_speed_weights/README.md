# Exp 01 — Two-speed weights

Can a network keep learning new facts after deployment without forgetting what
it knew? This tests the "two-speed weights" idea: write each new fact into a
few memory slots (fast), then periodically consolidate into the dense weights
during a "sleep" and free the slots (slow).

- `DESIGN.md`: the pre-registered design and decision rules (committed before
  any results).
- `RESULTS.md`: what happened and the verdict.
- `results/`: figures, tables and the decision JSON for each seed set.

## Files

| File | What it does |
|---|---|
| `facts.py` | Invented entities, six relations, eight paraphrase templates each |
| `data.py` | 4K BPE tokenizer, pretraining token stream (TinyStories + old-entity bios), fact streams |
| `model.py` | GPT with a product-key memory layer in block 2; LoRA wrappers |
| `pretrain.py` | One epoch over the 24M-token corpus |
| `stream.py` | One continual-learning run: arm (A full FT, B LoRA, C sparse memory), LR, seed, optional sleep |
| `sweep.py` | Runs the frozen LR grid for every arm in parallel, one thread per run |
| `analyze.py` | Aggregates runs, applies the decision rules, draws the figures |
| `run_all.sh` | Everything above, end to end |

## Reproduce

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
PY=venv/bin/python ./run_all.sh
```

On 4 CPU cores pretraining takes about 75 minutes and each stream run 4-12
minutes single-threaded. The code is CPU-only as written.
