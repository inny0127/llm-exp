"""Run a grid of stream runs in parallel (one thread per run); finished runs are skipped.

Usage: python sweep.py --seeds 1 2 --workers 4 [--arms A A+sleep B C C+sleep] [-- extra stream.py args]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# Frozen after the dev-stream calibration (see DESIGN.md / RESULTS.md).
GRIDS = {
    "A": [1e-5, 3e-5, 1e-4],
    "B": [3e-5, 1e-4, 3e-4],
    "C": [16.0, 64.0, 256.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["A", "A+sleep", "B", "C", "C+sleep"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/stream")
    ap.add_argument("--lrs", type=float, nargs="*", help="override the frozen grid")
    ap.add_argument("--grid", type=json.loads, help='per-arm override, e.g. {"A": [1e-4]}')
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)

    jobs = []
    for arm in args.arms:
        base, sleep = arm.split("+")[0], arm.endswith("+sleep")
        for lr in args.lrs or (args.grid or GRIDS)[base]:
            for seed in args.seeds:
                name = f"{arm}_lr{lr:g}_s{seed}"
                if os.path.exists(os.path.join(args.out, name + ".json")):
                    continue
                cmd = [sys.executable, "stream.py", "--arm", base, "--lr", str(lr),
                       "--seed", str(seed), "--out", args.out, "--name", name] + extra
                if sleep:
                    cmd.append("--sleep")
                jobs.append((name, cmd))
    # Longest-running arms first so the pool drains evenly.
    order = {"A+sleep": 0, "A": 1, "B": 2, "C+sleep": 3, "C": 4}
    jobs.sort(key=lambda j: order.get(j[0].split("_")[0], 9))
    print(f"{len(jobs)} runs to do", flush=True)

    def run(job):
        name, cmd = job
        with open(os.path.join(args.out, "logs", name + ".log"), "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if rc == 0 else f'FAILED({rc})'} {name}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
