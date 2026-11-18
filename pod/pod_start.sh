#!/usr/bin/env bash
# Runs on a RunPod GPU pod: rebuild Exp 04 data, check it matches, run Exp 04 (12 runs) and
# Exp 05 (18 runs) on the GPU, and serve results from /workspace/out on port 8000.
cd /workspace
OUT=/workspace/out; mkdir -p $OUT
(cd $OUT && nohup python -m http.server 8000 > /dev/null 2>&1 &)
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $OUT/STATUS; }
log "pod started"
[ -d llm-exp ] || git clone -q -b claude/experiment-design-run-sjkj74 https://github.com/inny0127/llm-exp.git
cd llm-exp && git pull -q; log "code $(git log --oneline -1)"
pip install -q tokenizers matplotlib > /dev/null 2>&1
RAW=/workspace/raw; mkdir -p $RAW
HF=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
[ -f $RAW/valid.txt ] || curl -sSL -o $RAW/valid.txt $HF/TinyStoriesV2-GPT4-valid.txt
[ -f $RAW/train200.txt ] || curl -sSL -r 0-209715199 -o $RAW/train200.txt $HF/TinyStoriesV2-GPT4-train.txt
cd exp04_curriculum
[ -f data/people.json ] || python curriculum.py prep --raw $RAW > $OUT/prep.log 2>&1
if (cd data && md5sum -c ../data_md5.txt) > $OUT/md5.txt 2>&1; then log "data identical"; else log "DATA MISMATCH"; fi
mkdir -p runs && cp models/*.pt runs/
cd ..
( while true; do
    mkdir -p $OUT/exp04 $OUT/exp05
    cp exp04_curriculum/runs/sel/*.json exp04_curriculum/runs/log_* $OUT/exp04/ 2>/dev/null
    cp exp05_order_compression/runs/order/*.json exp05_order_compression/runs/log_* $OUT/exp05/ 2>/dev/null
    sleep 60
  done ) &
W=${WORKERS:-8}
{
  for s in 1 2 3; do for p in random maxloss rho lp; do echo "4 $p $s"; done; done
  for s in 1 2 3; do for o in random easy_first hard_first maxloss rho lp; do echo "5 $o $s"; done; done
} | xargs -P $W -n 3 sh -c '
  if [ $0 = 4 ]; then cd exp04_curriculum && [ -f runs/sel/$1_s$2.json ] || python curriculum.py run --policy $1 --seed $2 --device cuda --threads 1 > runs/log_$1_s$2.txt 2>&1
  else cd exp05_order_compression && mkdir -p runs && [ -f runs/order/$1_s$2.json ] || python order.py --order $1 --seed $2 --device cuda --threads 1 > runs/log_$1_s$2.txt 2>&1; fi'
sleep 70
log "ALL DONE"
sleep infinity
