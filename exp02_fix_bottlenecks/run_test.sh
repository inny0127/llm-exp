#!/usr/bin/env bash
# Exp 02 test phase (seeds 1-3), frozen per DESIGN.md. Usage: LSTAR=4 R=gentle ./run_test.sh [core|extras]
set -euo pipefail
: "${LSTAR:?set LSTAR to the frozen FU1 LR}" "${R:?set R to gentle or gentle32}"
cd "$(dirname "$0")/../exp01_two_speed_weights"
PY=${PY:-python}
OUT=runs/exp02_test
mkdir -p $OUT/logs
PART=${1:-core}

GENTLE="--sleep --sleep_fact_loss answer --rehearse_labels teacher --rehearse_templates nonprobe --sleep_freeze embed --clean_replay"
case $R in
  gentle) SLEEP="$GENTLE --sleep_rehearse 16" ;;
  gentle32) SLEEP="$GENTLE --sleep_rehearse 32" ;;
  *) echo "unknown R=$R"; exit 1 ;;
esac
P06="--write_loss answer --protect_gamma 1 --protect_templates stream"
half=$(python3 -c "print(f'{$LSTAR/2:g}')"); double=$(python3 -c "print(f'{$LSTAR*2:g}')")

jobs=()
if [ "$PART" = core ]; then
  for s in 1 2 3; do jobs+=("--arm C --lr 16 $SLEEP --seed $s --name C+${R}_lr16_s$s"); done
  for s in 1 2 3; do jobs+=("--arm C --lr $LSTAR $P06 $SLEEP --seed $s --name p06ans+${R}_lr${LSTAR}_s$s"); done
  for s in 1 2 3; do jobs+=("--arm C --lr $LSTAR $P06 --seed $s --name p06ans_lr${LSTAR}_s$s"); done
  for s in 1 2 3; do jobs+=("--arm C --lr 16 --seed $s --name C_lr16_s$s"); done
else
  for s in 1 2 3; do
    jobs+=("--arm C --lr $half $P06 --seed $s --name p06ans_lr${half}_s$s")
    jobs+=("--arm C --lr $double $P06 --seed $s --name p06ans_lr${double}_s$s")
  done
  for s in 1 2 3; do jobs+=("--arm C --lr 0.5 --write_loss answer --seed $s --name ans_lr0.5_s$s"); done
  for s in 1 2 3; do jobs+=("--arm C --lr 16 --sleep --clean_replay --seed $s --name C+sleepclean_lr16_s$s"); done
fi

printf '%s\n' "${jobs[@]}" | xargs -P 4 -I{} sh -c '
  name=$(echo "{}" | sed "s/.*--name //")
  [ -f '"$OUT"'/$name.json ] && exit 0
  '"$PY"' stream.py {} --out '"$OUT"' > '"$OUT"'/logs/$name.log 2>&1 && echo "done $name" || echo "FAILED $name"'
