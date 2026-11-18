"""Exp 06 (syllabus) and Exp 07 (beliefs) analysis: tables, hypothesis checks, figures.

Usage: python analyze06.py [--runs results/runs] [--out results]
"""
import argparse
import glob
import json
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def kendall(a, b):
    s = sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in combinations(range(len(a)), 2))
    return s / (len(a) * (len(a) - 1) / 2)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        out.setdefault(r["args"]["policy"], []).append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    S, B = load(os.path.join(a.runs, "syllabus")), load(os.path.join(a.runs, "beliefs"))
    res, lines = {}, []

    # ---- Exp 06
    lines += ["## Exp 06: syllabus (final accuracy, means over seeds)", "",
              "| Policy | Seen d1 | d2 | d3 | d4 | Mean seen | Held-out d3 | Held-out d4 | Peak step of each depth's share (seed means) |",
              "|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for pol in ("random", "lp", "easy_first", "hard_first"):
        rs = S.get(pol, [])
        if not rs:
            continue
        fin = [r["curve"][-1] for r in rs]
        m = {k: float(np.mean([f[k] for f in fin])) for k in fin[0] if k.startswith("acc_")}
        m["mean_seen"] = float(np.mean([m[f"acc_seen_d{d}"] for d in range(1, 5)]))
        m["held"] = float(np.mean([m["acc_held_d3"], m["acc_held_d4"]]))
        peaks = []
        for r in rs:
            sh = np.array([c["window_share"] for c in r["curve"]])
            k = 9  # smooth over ~450 steps
            sm = np.stack([np.convolve(sh[:, d], np.ones(k) / k, mode="same") for d in range(4)], 1)
            peaks.append([int(r["curve"][int(np.argmax(sm[:, d]))]["step"]) for d in range(4)])
        m["peaks"] = np.mean(peaks, 0).tolist()
        m["taus"] = [kendall(np.array(p, float), np.arange(4.0)) for p in peaks]
        rows[pol] = m
        lines.append(f"| {pol} | " + " | ".join(f"{m[f'acc_seen_d{d}']:.2f}" for d in range(1, 5))
                     + f" | {m['mean_seen']:.3f} | {m['acc_held_d3']:.2f} | {m['acc_held_d4']:.2f} | "
                     + ", ".join(f"{p:.0f}" for p in m["peaks"]) + " |")
    if "lp" in rows and "random" in rows:
        res["exp06"] = {
            "H1 lp Kendall tau >= 0.67 in each seed": {"taus": rows["lp"]["taus"], "holds": all(t >= 0.67 for t in rows["lp"]["taus"])},
            "H2 lp held-out >= random + 0.10": {"lp": rows["lp"]["held"], "random": rows["random"]["held"],
                                                "holds": rows["lp"]["held"] >= rows["random"]["held"] + 0.10},
            "H3 lp mean seen >= random - 0.02": {"lp": rows["lp"]["mean_seen"], "random": rows["random"]["mean_seen"],
                                                 "holds": rows["lp"]["mean_seen"] >= rows["random"]["mean_seen"] - 0.02}}

    # ---- Exp 07
    lines += ["", "## Exp 07: beliefs (conflict probes, means over seeds)", "",
              "| Order | Code (bits/item) | Rule belief | Shortcut belief |", "|---|---|---|---|"]
    allr, rb = [], {}
    for pol in ("random", "rule_first", "rule_last", "lp"):
        rs = B.get(pol, [])
        if not rs:
            continue
        code = [sum(r["bits"]) / r["n_items"] for r in rs]
        rule = [r["curve"][-1]["rule"] for r in rs]
        sc = [r["curve"][-1]["shortcut"] for r in rs]
        allr += list(zip(code, rule))
        rb[pol] = float(np.mean(rule))
        lines.append(f"| {pol} | {np.mean(code):.3f} ± {np.std(code):.3f} | {np.mean(rule):.3f} ± {np.std(rule):.3f} | {np.mean(sc):.3f} |")
    if rb:
        rho = spearman([c for c, _ in allr], [r for _, r in allr]) if len(allr) > 3 else None
        res["exp07"] = {
            "H1 |rule_first - rule_last| >= 0.20": {"diff": rb.get("rule_first", 0) - rb.get("rule_last", 0),
                                                     "holds": abs(rb.get("rule_first", 0) - rb.get("rule_last", 0)) >= 0.20},
            "H2 rho(code, rule belief) <= -0.5": {"rho": rho, "holds": rho is not None and rho <= -0.5},
            "H3 lp rule belief >= random": {"lp": rb.get("lp"), "random": rb.get("random"),
                                            "holds": rb.get("lp", -1) >= rb.get("random", 2)}}
    md = "\n".join(lines)
    open(os.path.join(a.out, "tables.md"), "w").write(md + "\n")
    json.dump({"exp06_rows": rows, "checks": res}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(md)
    print(json.dumps(res, indent=1, default=float))

    # ---- figures
    if "lp" in S:
        fig, ax = plt.subplots(1, len(S["lp"]), figsize=(5 * len(S["lp"]), 3.6), sharey=True)
        cols = ["#2a78d6", "#1baf7a", "#eda100", "#d9483b"]
        for i, r in enumerate(S["lp"]):
            st = [c["step"] for c in r["curve"]]
            sh = np.array([c["window_share"] for c in r["curve"]])
            for d in range(4):
                ax[i].plot(st, np.convolve(sh[:, d], np.ones(9) / 9, mode="same"), color=cols[d], lw=1.8, label=f"depth {d + 1}")
            ax[i].set(title=f"What the learner chose to study (seed {r['args']['seed']})", xlabel="training step")
            ax[i].spines[["top", "right"]].set_visible(False)
        ax[0].set_ylabel("share of batches")
        ax[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "syllabus.png"), dpi=150)
    if B:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        pols = [p for p in ("random", "rule_first", "rule_last", "lp") if p in B]
        ax.bar(range(len(pols)), [np.mean([r["curve"][-1]["rule"] for r in B[p]]) for p in pols], color="#2a78d6", label="answers by the rule")
        ax.bar(range(len(pols)), [np.mean([r["curve"][-1]["shortcut"] for r in B[p]]) for p in pols],
               bottom=[np.mean([r["curve"][-1]["rule"] for r in B[p]]) for p in pols], color="#d9483b", label="answers by the shortcut")
        ax.set_xticks(range(len(pols)), pols)
        ax.set(ylabel="share of conflict probes", title="Same data, different order: what the model believes")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "beliefs.png"), dpi=150)


if __name__ == "__main__":
    main()
