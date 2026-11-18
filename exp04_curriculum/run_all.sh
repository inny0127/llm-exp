#!/usr/bin/env bash
# Full Exp 04 on CPU: data, prior + reference, then 4 policies x 3 seeds, 4 at a time.
set -e
cd "$(dirname "$0")"
P=${PY:-/home/user/venv/bin/python}
[ -f data/people.json ] || $P curriculum.py prep
[ -f runs/reference.pt ] || $P curriculum.py prior --threads 4
for s in 1 2 3; do for p in random maxloss rho lp; do echo "$p $s"; done; done | \
  xargs -P 4 -n 2 sh -c '[ -f runs/sel/$0_s$1.json ] || '"$P"' curriculum.py run --policy $0 --seed $1 --threads 1 > runs/log_$0_s$1.txt 2>&1'
echo ALL DONE
