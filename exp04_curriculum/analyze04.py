"""Exp 04 analysis: final metrics per policy (mean ± SD over seeds), hypothesis checks, figures.

Usage: python analyze04.py [--runs runs/sel] [--out results]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

POLICIES = ["random", "maxloss", "rho", "lp"]
LABEL = {"random": "Random", "maxloss": "Max loss (curiosity)", "rho": "Reducible loss (RHO)",
         "lp": "Learning-progress bandit"}
COLOR = {"random": "#7a7a7a", "maxloss": "#d9483b", "rho": "#2a78d6", "lp": "#1baf7a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/sel")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(a.runs, "*.json"))):
        r = json.load(open(p))
        runs[r["args"]["policy"]].append(r)
    groups = next(iter(runs.values()))[0]["groups"]

    rows = {}
    for pol in POLICIES:
        fin = [r["curve"][-1] for r in runs.get(pol, [])]
        if not fin:
            continue
        share = np.array([np.array(f["picks"]) / np.sum(f["picks"]) for f in fin])
        row = {"n": len(fin), "seeds": sorted(r["args"]["seed"] for r in runs[pol])}
        for k in ("fact_acc", "fact_acc_heldout", "val_loss"):
            v = [f[k] for f in fin]
            row[k], row[k + "_sd"] = float(np.mean(v)), float(np.std(v))
        row["share"] = dict(zip(groups, share.mean(0).round(3).tolist()))
        rows[pol] = row

    base = rows.get("random")
    checks = {}
    if base and "maxloss" in rows:
        m = rows["maxloss"]
        checks["H1 maxloss >40% noise and worse val loss"] = bool(m["share"]["noise"] > 0.40 and m["val_loss"] > base["val_loss"])
    for h, pol in (("H2", "rho"), ("H3", "lp")):
        if base and pol in rows:
            r = rows[pol]
            checks[f"{h} {pol} facts +0.10, val <= +0.02, held-out up"] = bool(
                r["fact_acc"] - base["fact_acc"] >= 0.10 and r["val_loss"] - base["val_loss"] <= 0.02
                and r["fact_acc_heldout"] > base["fact_acc_heldout"])
    if "rho" in rows and "lp" in rows:
        checks["H4 rho and lp <10% noise"] = bool(rows["rho"]["share"]["noise"] < 0.10 and rows["lp"]["share"]["noise"] < 0.10)

    lines = ["| Policy | Seeds | Fact accuracy (template 0) | Held-out template | Val loss | Picks: stories / facts / noise / dupes |",
             "|---|---|---|---|---|---|"]
    for pol, r in rows.items():
        s = r["share"]
        lines.append(f"| {LABEL[pol]} | {r['n']} | {r['fact_acc']:.3f} ± {r['fact_acc_sd']:.3f} | "
                     f"{r['fact_acc_heldout']:.3f} | {r['val_loss']:.3f} ± {r['val_loss_sd']:.3f} | "
                     f"{s['stories']:.2f} / {s['facts']:.2f} / {s['noise']:.2f} / {s['dupes']:.2f} |")
    md = "\n".join(lines)
    open(os.path.join(a.out, "table.md"), "w").write(md + "\n")
    json.dump({"rows": rows, "checks": checks}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(md)
    print(json.dumps(checks, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for pol in POLICIES:
        for i, r in enumerate(runs.get(pol, [])):
            st = [c["step"] for c in r["curve"]]
            kw = dict(color=COLOR[pol], alpha=0.85, lw=1.6, label=LABEL[pol] if i == 0 else None)
            ax[0].plot(st, [c["fact_acc"] for c in r["curve"]], **kw)
            ax[1].plot(st, [c["val_loss"] for c in r["curve"]], **kw)
            pk = np.array([c["picks"] for c in r["curve"]], float)
            tot = pk.sum(1)
            ax[2].plot(st[1:], (pk[1:, groups.index("noise")] / tot[1:]), **kw)
    ax[0].set(title="New facts learnt (probe accuracy)", xlabel="training step", ylabel="accuracy")
    ax[1].set(title="Held-out story loss (lower is better)", xlabel="training step", ylabel="nats/token")
    ax[2].set(title="Share of training picks that were noise", xlabel="training step", ylabel="share")
    ax[0].axhline(1 / 14, color="k", ls=":", lw=0.8)
    ax[0].legend(frameon=False, fontsize=9)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "curves.png"), dpi=150)
    print("saved", os.path.join(a.out, "curves.png"))


if __name__ == "__main__":
    main()
