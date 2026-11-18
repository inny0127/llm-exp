"""Exp 05 analysis: code length per order, MDL correlations, hypothesis checks, figure.

Usage: python analyze05.py [--runs runs/order] [--out results]
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDERS = ["random", "easy_first", "hard_first", "maxloss", "rho", "lp"]
COLOR = {"random": "#7a7a7a", "easy_first": "#eda100", "hard_first": "#8a63d2", "maxloss": "#d9483b",
         "rho": "#2a78d6", "lp": "#1baf7a"}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/order")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    per = []
    for p in sorted(glob.glob(os.path.join(a.runs, "*.json"))):
        r = json.load(open(p))
        g, b, tpg = r["groups"], np.array(r["bits_per_group"]), r["tokens_per_group"]
        learn = [g.index(k) for k in ("stories", "facts", "dupes")]
        fin = r["curve"][-1]
        per.append({"order": r["args"]["order"], "seed": r["args"]["seed"],
                    "code_total": b.sum() / (tpg * len(g)), "code_learnable": b[learn].sum() / (tpg * len(learn)),
                    "code_noise": b[g.index("noise")] / tpg, "val_loss": fin["val_loss"],
                    "fact_acc": fin["fact_acc"], "fact_acc_heldout": fin["fact_acc_heldout"],
                    "per_group": dict(zip(g, (b / tpg).round(3).tolist()))})
    by = defaultdict(list)
    for x in per:
        by[x["order"]].append(x)
    keys = ["code_total", "code_learnable", "code_noise", "val_loss", "fact_acc", "fact_acc_heldout"]
    rows = {o: {k: (float(np.mean([x[k] for x in by[o]])), float(np.std([x[k] for x in by[o]]))) for k in keys}
            | {"n": len(by[o])} for o in ORDERS if o in by}
    lines = ["| Order | Seeds | Code, learnable (bits/token) | Code, noise | Held-out story loss | Fact accuracy | Held-out template |",
             "|---|---|---|---|---|---|---|"]
    for o, r in rows.items():
        lines.append(f"| {o} | {r['n']} | {r['code_learnable'][0]:.3f} ± {r['code_learnable'][1]:.3f} | "
                     f"{r['code_noise'][0]:.3f} | {r['val_loss'][0]:.3f} ± {r['val_loss'][1]:.3f} | "
                     f"{r['fact_acc'][0]:.3f} ± {r['fact_acc'][1]:.3f} | {r['fact_acc_heldout'][0]:.3f} |")
    md = "\n".join(lines)
    cl = [x["code_learnable"] for x in per]
    checks = {}
    if len(rows) >= 2:
        means = [r["code_learnable"][0] for r in rows.values()]
        sd = max(r["code_learnable"][1] for r in rows.values())
        checks["H1 spread > 3x SD"] = {"spread": max(means) - min(means), "max_sd": sd,
                                       "holds": bool(max(means) - min(means) > 3 * sd)}
    if len(per) >= 4:
        rv, rf = spearman(cl, [x["val_loss"] for x in per]), spearman(cl, [x["fact_acc"] for x in per])
        checks["H2 MDL correlation"] = {"rho_val_loss": rv, "rho_fact_acc": rf, "holds": bool(rv >= 0.5 and rf <= -0.5)}
    if all(o in rows for o in ("random", "lp", "maxloss")):
        r0 = rows["random"]["code_learnable"][0]
        checks["H3 lp < random < maxloss"] = {"holds": bool(rows["lp"]["code_learnable"][0] < r0 < rows["maxloss"]["code_learnable"][0])}
    if per:
        lo = min(x["code_noise"] for x in per)
        checks["H4 noise never below 12 bits"] = {"min_noise_bits": lo, "holds": bool(lo >= math.log2(4096))}
    open(os.path.join(a.out, "table.md"), "w").write(md + "\n")
    json.dump({"rows": rows, "runs": per, "checks": checks}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(md)
    print(json.dumps(checks, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for x in per:
        ax[0].scatter(x["code_learnable"], x["val_loss"], color=COLOR[x["order"]], s=40)
        ax[1].scatter(x["code_learnable"], x["fact_acc"], color=COLOR[x["order"]], s=40)
    for o in ORDERS:
        ax[0].scatter([], [], color=COLOR[o], label=o)
    ax[0].set(xlabel="code length of the learnable data (bits/token)", ylabel="held-out story loss",
              title="Shorter code vs generalisation (stories)")
    ax[1].set(xlabel="code length of the learnable data (bits/token)", ylabel="new-fact accuracy",
              title="Shorter code vs generalisation (facts)")
    ax[0].legend(frameon=False, fontsize=8)
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "mdl.png"), dpi=150)


if __name__ == "__main__":
    main()
