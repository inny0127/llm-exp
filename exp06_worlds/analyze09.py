"""Exp 09 analysis: lp_mix vs Exp 06 random/lp; rule belief vs block count K.

Usage: python analyze09.py [--runs results09/runs] [--runs06 results/runs] [--out results09]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze06 import kendall


def peaks(r):
    sh = np.array([c["window_share"] for c in r["curve"]])
    sm = np.stack([np.convolve(sh[:, d], np.ones(9) / 9, mode="same") for d in range(4)], 1)
    return [int(r["curve"][int(np.argmax(sm[:, d]))]["step"]) for d in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results09/runs")
    ap.add_argument("--runs06", default="results/runs")
    ap.add_argument("--out", default="results09")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out, lines = {}, ["## Part 1: syllabus with review", "",
                      "| Policy | Mean seen acc | Held-out d3/d4 | Kendall tau per seed |", "|---|---|---|---|"]
    rows = {}
    for pol, d in (("random", a.runs06), ("lp", a.runs06), ("lp_mix", a.runs)):
        rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "syllabus", f"{pol}_s*.json")))]
        if not rs:
            continue
        fin = [r["curve"][-1] for r in rs]
        seen = float(np.mean([np.mean([f[f"acc_seen_d{k}"] for k in range(1, 5)]) for f in fin]))
        held = float(np.mean([(f["acc_held_d3"] + f["acc_held_d4"]) / 2 for f in fin]))
        taus = [kendall(np.array(peaks(r), float), np.arange(4.0)) for r in rs]
        rows[pol] = {"seen": seen, "held": held, "taus": taus}
        lines.append(f"| {pol} | {seen:.3f} | {held:.3f} | {', '.join(f'{t:.2f}' for t in taus)} |")
    if "lp_mix" in rows and "random" in rows:
        m, r0 = rows["lp_mix"], rows["random"]
        out["H1 lp_mix tau >= 0.67 each seed"] = {"taus": m["taus"], "holds": all(t >= 0.67 for t in m["taus"])}
        out["H2 lp_mix held-out >= random"] = {"lp_mix": m["held"], "random": r0["held"], "holds": m["held"] >= r0["held"]}
        out["H3 lp_mix seen >= random - 0.02"] = {"lp_mix": m["seen"], "random": r0["seen"], "holds": m["seen"] >= r0["seen"] - 0.02}
    lines += ["", "## Part 2: rule belief vs number of block pairs K", "", "| K | Rule belief | Shortcut belief | Code (bits/item) |", "|---|---|---|---|"]
    ks, rule = [], []
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 312):
        rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.runs, "beliefs", f"blocks_k{k}_s*.json")))]
        if not rs:
            continue
        rb = [r["curve"][-1]["rule"] for r in rs]
        ks.append(k)
        rule.append(rb)
        lines.append(f"| {k} | {np.mean(rb):.3f} ± {np.std(rb):.3f} | {np.mean([r['curve'][-1]['shortcut'] for r in rs]):.3f} | "
                     f"{np.mean([sum(r['bits']) / r['n_items'] for r in rs]):.3f} |")
    if len(ks) >= 2:
        mu = [float(np.mean(x)) for x in rule]
        total = mu[-1] - mu[0]
        jumps = np.diff(mu)
        out["H4 rule(K=312) - rule(K=1) >= 0.5"] = {"diff": total, "holds": total >= 0.5}
        out["H5 one step carries >= half the change"] = {"largest_step": float(jumps.max()) if len(jumps) else None,
                                                         "between": [ks[int(np.argmax(jumps))], ks[int(np.argmax(jumps)) + 1]] if len(jumps) else None,
                                                         "holds": bool(total > 0 and jumps.max() >= 0.5 * total)}
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for i, k in enumerate(ks):
            ax.scatter([k] * len(rule[i]), rule[i], color="#2a78d6", alpha=0.5, s=18)
        ax.plot(ks, mu, color="#2a78d6", lw=2)
        ax.set_xscale("log")
        ax.set(xlabel="number of alternating block pairs K (log)", ylabel="rule belief on conflict probes",
               title="Same data: how finely it is interleaved decides the belief")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "belief_vs_blocks.png"), dpi=150)
    md = "\n".join(lines)
    open(os.path.join(a.out, "tables.md"), "w").write(md + "\n")
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(md)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
