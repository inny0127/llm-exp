"""Aggregate stream runs, apply the pre-registered decision rules, draw the figures.

Usage: python analyze.py --runs runs/stream --seeds 1 2 --out results
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

ARMS = ["A", "A+sleep", "B", "C", "C+sleep"]
LABEL = {"A": "Full FT", "A+sleep": "Full FT + sleep", "B": "LoRA",
         "C": "Sparse memory", "C+sleep": "Sparse memory + sleep"}
# Reference categorical slots 1-3 (validated all-pairs); sleep = dashed + hollow marker.
COLOR = {"A": "#2a78d6", "A+sleep": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "C+sleep": "#1baf7a"}
DASH = {a: ("--" if a.endswith("+sleep") else "-") for a in ARMS}
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BUDGET_NLL, BUDGET_OLD = 0.05, 5.0


def binned(a, n_bins):
    a = np.asarray(a, float)
    n_bins = max(1, min(n_bins, len(a)))
    return a[:len(a) // n_bins * n_bins].reshape(n_bins, -1).mean(1)


def run_metrics(r):
    c0, cN = r["checkpoints"][0], r["checkpoints"][-1]
    acq = np.asarray(r["acq_in"], float)
    ret, ret_out = np.asarray(cN["ret_in"], float), np.asarray(cN["ret_out"], float)
    nb = len(acq) // 100
    sl = r["sleeps"]
    return {
        "acq_in": acq.mean(), "acq_out": float(np.mean(r["acq_out"])),
        "ret_in": ret.mean(), "ret_out": ret_out.mean(), "ret_early": ret[:100].mean(),
        "d_nll": cN["val_loss"] - c0["val_loss"],
        "old_drop": 100 * (c0["old_in"] - cN["old_in"]),
        "old_drop_out": 100 * (c0["old_out"] - cN["old_out"]),
        "acq_bins": binned(acq, nb), "ret_bins": binned(ret, nb),
        "overlap_bins": binned(r["slot_overlap"], nb) if r["slot_overlap"] else None,
        "consol_loss": float(np.mean([s["pre_recent"] - s["post_recent"] for s in sl])) if sl else None,
        "self_label_acc": float(np.mean([s["self_label_acc"] for s in sl])) if sl else None,
        "sleep_pre": [s["pre_recent"] for s in sl], "sleep_post": [s["post_recent"] for s in sl],
        "seconds": r["seconds"], "old0": c0["old_in"], "nll0": c0["val_loss"],
    }


def load(runs, seeds):
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(runs, "*.json"))):
        r = json.load(open(path))
        if r["args"]["seed"] not in seeds:
            continue
        arm = r["name"].split("_lr")[0]
        groups[(arm, r["args"]["lr"])].append(run_metrics(r))
    agg = {}
    for key, ms in groups.items():
        a = {"n": len(ms)}
        for k in ms[0]:
            vals = [m[k] for m in ms]
            if vals[0] is None:
                a[k] = None
            elif isinstance(vals[0], (list, np.ndarray)):
                a[k] = np.mean([np.asarray(v, float) for v in vals], 0)
            else:
                a[k], a[k + "_sd"] = float(np.mean(vals)), float(np.std(vals))
        agg[key] = a
    return agg


def frontier(agg, arm):
    return sorted([(lr, m) for (a, lr), m in agg.items() if a == arm], key=lambda t: t[0])


def operating_point(agg, arm):
    ok = [(lr, m) for lr, m in frontier(agg, arm)
          if m["d_nll"] <= BUDGET_NLL and m["old_drop"] <= BUDGET_OLD]
    return max(ok, key=lambda t: t[1]["ret_in"]) if ok else None


def at_acquisition(agg, arm, level):
    pts = sorted((m["acq_in"], m["old_drop"], m["d_nll"]) for _, m in frontier(agg, arm))
    acq = np.array([p[0] for p in pts])
    if level < acq.min() - 1e-9 or level > acq.max() + 1e-9:
        return None
    return float(np.interp(level, acq, [p[1] for p in pts])), float(np.interp(level, acq, [p[2] for p in pts]))


def decide(agg):
    d = {"budget": {"d_nll": BUDGET_NLL, "old_drop_pp": BUDGET_OLD}}
    ops = {a: operating_point(agg, a) for a in ARMS if frontier(agg, a)}
    d["operating_points"] = {a: (None if op is None else {"lr": op[0], **{k: op[1][k] for k in (
        "acq_in", "acq_out", "ret_in", "ret_out", "ret_early", "d_nll", "old_drop")}})
        for a, op in ops.items()}
    if frontier(agg, "A") and frontier(agg, "C"):
        level = min(max(m["acq_in"] for _, m in frontier(agg, "A")),
                    max(m["acq_in"] for _, m in frontier(agg, "C")), 0.9)
        q1 = {"matched_acquisition": level}
        for arm in ("A", "B", "C"):
            if frontier(agg, arm):
                v = at_acquisition(agg, arm, level)
                q1[arm] = None if v is None else {"old_drop": v[0], "d_nll": v[1]}
        if q1["A"] and q1["C"]:
            q1["pass"] = (q1["C"]["old_drop"] <= 0.5 * q1["A"]["old_drop"]
                          and q1["C"]["d_nll"] <= 0.5 * q1["A"]["d_nll"])
        d["Q1"] = q1
    if ops.get("C"):
        m = ops["C"][1]
        d["saturation"] = {"early_gap": m["acq_in"] - m["ret_early"],
                           "acq_first_minus_last_bin": float(m["acq_bins"][0] - m["acq_bins"][-1])}
        d["saturation"]["present"] = bool(d["saturation"]["early_gap"] >= 0.15
                                          or d["saturation"]["acq_first_minus_last_bin"] >= 0.10)
    if ops.get("C") and ops.get("C+sleep") and ops.get("A+sleep"):
        c, dd, asl = ops["C"][1], ops["C+sleep"][1], ops["A+sleep"][1]
        d["Q2"] = {"D_minus_C_ret": dd["ret_in"] - c["ret_in"],
                   "D_minus_Asleep_ret": dd["ret_in"] - asl["ret_in"],
                   "pass": bool(d.get("saturation", {}).get("present")
                                and dd["ret_in"] - c["ret_in"] >= 0.10
                                and dd["ret_in"] - asl["ret_in"] >= 0.05)}
    fail = {}
    for arm in ("A+sleep", "C+sleep"):
        if frontier(agg, arm):
            fail[arm] = {"within_budget_lr_exists": ops.get(arm) is not None,
                         "mean_consolidation_loss_by_lr": {lr: m["consol_loss"] for lr, m in frontier(agg, arm)}}
    d["sleep_failure_checks"] = fail
    return d, ops


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_frontier(agg, ops, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURF)
    for ax, key, xlabel, budget in [(axes[0], "old_drop", "Old-fact accuracy drop (points)", BUDGET_OLD),
                                    (axes[1], "d_nll", "Held-out TinyStories loss increase (nats/token)", BUDGET_NLL)]:
        style(ax)
        ax.axvspan(-1e9, budget, color="#f0efec", zorder=0)
        for arm in ARMS:
            pts = frontier(agg, arm)
            if not pts:
                continue
            xs, ys = [m[key] for _, m in pts], [m["ret_in"] for _, m in pts]
            ax.plot(xs, ys, DASH[arm], color=COLOR[arm], linewidth=2, label=LABEL[arm],
                    solid_capstyle="round", zorder=2)
            hollow = arm.endswith("+sleep")
            ax.scatter(xs, ys, s=64, zorder=3, facecolor=SURF if hollow else COLOR[arm],
                       edgecolor=COLOR[arm] if hollow else SURF, linewidth=2)
            if ops.get(arm):
                lr, m = ops[arm]
                ax.annotate(LABEL[arm], (m[key], m["ret_in"]), xytext=(6, 6),
                            textcoords="offset points", fontsize=8.5, color=INK2)
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
        lo = min(0, min(m[key] for (_, _), m in agg.items()))
        hi = max(m[key] for (_, _), m in agg.items())
        ax.set_xlim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.08 * (hi - lo + 1e-9))
    axes[0].set_ylabel("End-of-stream retention, all new facts", color=INK2, fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.suptitle("Learning vs forgetting: one point per learning rate (shaded = within forgetting budget)",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def plot_position(ops, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURF)
    for ax, key, title in [(axes[0], "acq_bins", "Acquisition: right after each write"),
                           (axes[1], "ret_bins", "Retention: at end of stream")]:
        style(ax)
        for arm in ARMS:
            if not ops.get(arm):
                continue
            lr, m = ops[arm]
            y = m[key]
            x = (np.arange(len(y)) + 0.5) * 100
            ax.plot(x, y, DASH[arm], color=COLOR[arm], linewidth=2, label=f"{LABEL[arm]} (lr {lr:g})")
            ax.scatter(x[-1:], y[-1:], s=64, zorder=3, facecolor=COLOR[arm], edgecolor=SURF, linewidth=2)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("Position of fact in stream", color=INK2, fontsize=10)
    axes[0].set_ylabel("Probe accuracy (chance about 0.07)", color=INK2, fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.suptitle("Each arm at its operating point (best retention within the forgetting budget)",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def plot_sleep(agg, ops, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    ax = axes[0]
    style(ax)
    for arm in ("A+sleep", "C+sleep"):
        if not ops.get(arm):
            continue
        lr, m = ops[arm]
        x = (np.arange(len(m["sleep_pre"])) + 1) * 100
        ax.plot(x, m["sleep_pre"], "-", color=COLOR[arm], linewidth=2, label=f"{LABEL[arm]}: before sleep")
        ax.plot(x, m["sleep_post"], "--", color=COLOR[arm], linewidth=2, label=f"{LABEL[arm]}: after sleep")
    ax.set_ylim(0, 1)
    ax.set_title("Last 100 facts, recalled before vs after each sleep", color=INK, fontsize=10.5, loc="left")
    ax.set_xlabel("Facts seen when sleep ran", color=INK2, fontsize=10)
    ax.set_ylabel("Probe accuracy", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    ax = axes[1]
    style(ax)
    for arm in ("C", "C+sleep"):
        if not ops.get(arm) or ops[arm][1]["overlap_bins"] is None:
            continue
        lr, m = ops[arm]
        y = m["overlap_bins"]
        ax.plot((np.arange(len(y)) + 0.5) * 100, y, DASH[arm], color=COLOR[arm], linewidth=2,
                label=LABEL[arm])
    ax.set_ylim(0, 1)
    ax.set_title("Slots a new fact must share with earlier facts", color=INK, fontsize=10.5, loc="left")
    ax.set_xlabel("Position of fact in stream", color=INK2, fontsize=10)
    ax.set_ylabel("Fraction of top-t slots already written", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def table(agg, ops):
    rows = ["| Arm | LR | Acquisition | Retention (all) | Retention (first 100) | Retention, held-out template"
            " | Old-fact drop (pts) | Delta NLL | Op. point |",
            "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for lr, m in frontier(agg, arm):
            op = "yes" if ops.get(arm) and ops[arm][0] == lr else ""
            rows.append(f"| {LABEL[arm]} | {lr:g} | {m['acq_in']:.2f} | {m['ret_in']:.2f} ± {m['ret_in_sd']:.2f}"
                        f" | {m['ret_early']:.2f} | {m['ret_out']:.2f} | {m['old_drop']:.1f} ± {m['old_drop_sd']:.1f}"
                        f" | {m['d_nll']:+.3f} | {op} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/stream")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    agg = load(args.runs, set(args.seeds))
    dec, ops = decide(agg)
    tag = "seeds" + "-".join(map(str, args.seeds))
    plot_frontier(agg, ops, os.path.join(args.out, f"frontier_{tag}.png"))
    plot_position(ops, os.path.join(args.out, f"position_{tag}.png"))
    plot_sleep(agg, ops, os.path.join(args.out, f"sleep_{tag}.png"))
    md = table(agg, ops)
    with open(os.path.join(args.out, f"table_{tag}.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(args.out, f"decision_{tag}.json"), "w") as f:
        json.dump(dec, f, indent=2, default=float)
    print(md)
    print(json.dumps(dec, indent=2, default=float))


if __name__ == "__main__":
    main()
