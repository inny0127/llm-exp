#!/usr/bin/env bash
# Reproduce Exp 01 end to end on CPU (about 5 hours on 4 cores).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

mkdir -p data/raw
HF=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
[ -f data/raw/TinyStoriesV2-GPT4-valid.txt ] || curl -sSL -o data/raw/TinyStoriesV2-GPT4-valid.txt "$HF/TinyStoriesV2-GPT4-valid.txt"
[ -f data/raw/TinyStoriesV2-GPT4-train-head200MB.txt ] || curl -sSL -r 0-209715199 -o data/raw/TinyStoriesV2-GPT4-train-head200MB.txt "$HF/TinyStoriesV2-GPT4-train.txt"

[ -f data/train.bin ] || $PY data.py --data_dir data
[ -f runs/pretrain/model.pt ] || $PY pretrain.py --out runs/pretrain

# IDF background (shared by every sparse run), then the frozen test grid.
$PY -c "import stream, torch, os; m = stream.load_model('runs/pretrain/model.pt'); m.eval(); stream.background_df(m, 'data/train.bin', 1000, 32, 'runs/pretrain/bg_df_w32.pt')"
$PY sweep.py --seeds 1 2 3 --workers 4
$PY analyze.py --runs runs/stream --seeds 1 2 3 --out results
