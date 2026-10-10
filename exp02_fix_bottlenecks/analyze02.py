"""Exp 02 analysis: follow-up bars, paired sleep effects, sleep damage decomposition, figures.

Runs are named "{writer}[+{sleep}]_lr{lr}_s{seed}". Each name is checked against the flags the
run actually used, so a mislabelled run fails loudly instead of being paired with the wrong
partner. Exp 01 runs (same code path, verified bit-identical) join via --extra_runs.

Usage: python analyze02.py --runs ../exp01_two_speed_weights/runs/exp02_test \
           --extra_runs ../exp01_two_speed_weights/runs/stream --seeds 1 2 3 \
           --primary p06ans:4 --out results
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp01_two_speed_weights"))
from analyze import INK, INK2, SURF, run_metrics, style  # noqa: E402

# Expected stream.py flags per writer and per sleep. Missing keys in Exp 01 runs = defaults.
DEFAULTS = {"arm": "C", "write_loss": "all", "protect_gamma": 0.0, "protect_templates": "all", "sleep": False,
            "sleep_fact_loss": "all", "sleep_rehearse": 0, "sleep_freeze": "none",
            "clean_replay": False, "sleep_ewc": 0.0, "rehearse_templates": "all"}
WRITERS = {
    "C": ("Exp 01 writer", {}),
    "ans": ("Answer-only writer", {"write_loss": "answer"}),
    "p06ans": ("Protected answer-only writer", {"write_loss": "answer", "protect_gamma": 1.0,
                                                "protect_templates": "stream"}),
    "pnpans": ("Protected answer-only writer, P from templates 1-6",
               {"write_loss": "answer", "protect_gamma": 1.0, "protect_templates": "nonprobe"}),
}
GENTLE = {"sleep": True, "sleep_fact_loss": "answer", "sleep_rehearse": 16, "sleep_freeze": "embed",
          "clean_replay": True, "rehearse_templates": "nonprobe"}
SLEEPS = {
    "sleep": ("Exp 01 sleep", {"sleep": True}),
    "sleepclean": ("Exp 01 sleep, clean replay", {"sleep": True, "clean_replay": True}),
    "gentle": ("gentle sleep", GENTLE),
    "gentle32": ("gentle sleep (32 rehearsal)", {**GENTLE, "sleep_rehearse": 32}),
}
COLOURS = {"C": "#2a78d6", "ans": "#1baf7a", "p06ans": "#eb6834", "pnpans": "#eda100"}
FU1_BAR = {"old_drop": 10.0, "ret_in": 0.45, "old_drop_out": 10.0, "ret_out": 0.40}  # last two: guards
FU2_BAR = {"d_nll": 0.02, "old_drop": 3.0, "ret_in": 0.10, "old_drop_out": 3.0, "ret_out": 0.10,
           "consol": 0.01}  # paired; old_drop_out / ret_out are guards; consol = mean(pre_all - post_all)
GENTLE_DAMAGE = 3.0  # summed dense damage (pts) over the 4 sleeps that still counts as gentle


def parse_arm(arm):
    writer, _, sleep = arm.partition("+")
    if writer not in WRITERS or (sleep and sleep not in SLEEPS):
        raise ValueError(f"unknown arm {arm!r}")
    expected = {**DEFAULTS, **WRITERS[writer][1], **(SLEEPS[sleep][1] if sleep else {})}
    label = WRITERS[writer][0] + (f" + {SLEEPS[sleep][0]}" if sleep else "")
    return writer, sleep, expected, label


def check_flags(r, expected):
    a = r["args"]
    for k, v in expected.items():
        got = a.get(k, DEFAULTS[k])
        if (abs(float(got) - float(v)) > 1e-9) if isinstance(v, (int, float)) and not isinstance(v, bool) else got != v:
            raise ValueError(f"{r['name']}: flag {k}={got!r}, arm name implies {v!r}")


def load(dirs, seeds, arms_wanted=None):
    groups, seen = defaultdict(list), set()
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(path))
            arm = r["name"].split("_lr")[0]
            if r["args"]["seed"] not in seeds or (arms_wanted and arm not in arms_wanted):
                continue
            _, _, expected, _ = parse_arm(arm)
            check_flags(r, expected)
            key = (arm, float(r["args"]["lr"]), r["args"]["seed"])
            if key in seen:
                raise ValueError(f"duplicate run {key}")
            seen.add(key)
            m = run_metrics(r)
            sl = r["sleeps"]
            if sl:
                m["consol"] = float(np.mean([s["pre_all"] - s["post_all"] for s in sl]))
            if sl and "old_after_reset" in sl[0]:  # summed over the run's sleeps
                m["reset_gain"] = 100 * float(np.sum([s["old_after_reset"] - s["old_before"] for s in sl]))
                m["dense_damage"] = 100 * float(np.sum([s["old_after_reset"] - s["old_after_sleep"] for s in sl]))
            m["seed"] = r["args"]["seed"]
            groups[(arm, float(r["args"]["lr"]))].append(m)
    return groups


def summarize(groups):
    rows = {}
    for (arm, lr), ms in sorted(groups.items()):
        row = {"arm": arm, "label": parse_arm(arm)[3], "lr": lr, "n": len(ms), "seeds": sorted(m["seed"] for m in ms)}
        row["per_seed"] = {m["seed"]: {k: m.get(k) for k in ("ret_in", "ret_out", "old_drop", "old_drop_out", "d_nll")}
                           for m in ms}
        for k in ("acq_in", "ret_in", "ret_out", "ret_early", "old_drop", "old_drop_out", "d_nll", "steps",
                  "cap_frac", "reset_gain", "dense_damage", "consol"):
            vals = [m[k] for m in ms if m.get(k) is not None]
            row[k], row[k + "_sd"] = (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
        rows[(arm, lr)] = row
    return rows


def paired(groups, rows, n_seeds):
    """Sleep arm minus its own writer without sleep, matched on LR and seed."""
    out = []
    for (arm, lr), ms in sorted(groups.items()):
        writer, sleep, _, label = parse_arm(arm)
        if not sleep or (writer, lr) not in groups:
            continue
        base = {m["seed"]: m for m in groups[(writer, lr)]}
        keys = ("ret_in", "ret_out", "ret_early", "old_drop", "old_drop_out", "d_nll")
        diffs = [{k: m[k] - base[m["seed"]][k] for k in keys} for m in ms if m["seed"] in base]
        if not diffs:
            continue
        eff = {k: float(np.mean([d[k] for d in diffs])) for k in keys}
        complete = len(diffs) == n_seeds
        sd = {k + "_sd": float(np.std([d[k] for d in diffs])) for k in keys}
        consol = rows[(arm, lr)]["consol"]
        checks = {"d_nll": eff["d_nll"] <= FU2_BAR["d_nll"], "old_drop": eff["old_drop"] <= FU2_BAR["old_drop"],
                  "ret_in": eff["ret_in"] >= FU2_BAR["ret_in"],
                  "guard_old_drop_out": eff["old_drop_out"] <= FU2_BAR["old_drop_out"],
                  "guard_ret_out": eff["ret_out"] >= FU2_BAR["ret_out"],
                  "consolidation_intact": consol is not None and consol <= FU2_BAR["consol"]}
        marginal = [k for k, thr in (("d_nll", FU2_BAR["d_nll"]), ("old_drop", FU2_BAR["old_drop"]),
                                     ("ret_in", FU2_BAR["ret_in"])) if abs(eff[k] - thr) <= sd[k + "_sd"]]
        dmg = rows[(arm, lr)]["dense_damage"]
        out.append({"sleep_arm": arm, "label": label, "partner": writer, "lr": lr, "n_pairs": len(diffs),
                    **eff, **sd, "per_seed": diffs, "checks": checks, "marginal": marginal,
                    "reset_gain_sum": rows[(arm, lr)]["reset_gain"], "dense_damage_sum": dmg,
                    "consol": consol,
                    "kind": (None if dmg is None else "gentle consolidation" if dmg <= GENTLE_DAMAGE else
                             f"passes through slot-reset repair; the dense update still costs {dmg:.1f} pts"),
                    "passes_fu2_bar": bool(all(checks.values())) if complete else "incomplete"})
    return out


def fu1_check(r):
    bar = r["old_drop"] <= FU1_BAR["old_drop"] and r["ret_in"] >= FU1_BAR["ret_in"]
    guards = r["old_drop_out"] <= FU1_BAR["old_drop_out"] and r["ret_out"] >= FU1_BAR["ret_out"]
    return "pass" if bar and guards else "template-bound pass (fail)" if bar else "fail"


def decide(rows, pairs, primary, n_seeds):
    d = {"bars": {"fu1": FU1_BAR, "fu2": FU2_BAR}, "primary_writer": primary}
    pw, plr = primary
    r = rows.get((pw, plr))
    if r is None or r["n"] < n_seeds:
        d["fu1"] = "incomplete"
    else:
        v = fu1_check(r)
        d["fu1"] = {k: r[k] for k in ("ret_in", "ret_out", "old_drop", "old_drop_out", "d_nll", "cap_frac",
                                      "ret_in_sd", "old_drop_sd")}
        d["fu1"].update({"verdict": v, "pass": v == "pass", "per_seed": r["per_seed"]})
    d["fu2"] = [p for p in pairs if p["partner"] == "C"]
    combo = []
    for p in pairs:
        if p["partner"] != pw or p["lr"] != plr:
            continue
        a = rows[(p["sleep_arm"], plr)]
        complete = a["n"] == n_seeds and p["n_pairs"] == n_seeds
        absolute = fu1_check(a)
        ok = absolute == "pass" and p["passes_fu2_bar"] is True
        combo.append({"arm": p["sleep_arm"], "abs_ret_in": a["ret_in"], "abs_ret_out": a["ret_out"],
                      "abs_old_drop": a["old_drop"], "abs_old_drop_out": a["old_drop_out"],
                      "abs_d_nll": a["d_nll"], "absolute_fu1": absolute, "paired_fu2": p["passes_fu2_bar"],
                      "paired_checks": p["checks"], "pass": bool(ok) if complete else "incomplete"})
    d["combined"] = combo
    fu1_ok = isinstance(d["fu1"], dict) and d["fu1"]["pass"]
    d["go_to_5090"] = bool(fu1_ok and any(c["pass"] is True for c in combo))
    return d


def plot_frontier(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURF)
    arms = sorted({a for a, _ in rows})
    for ax, key, xlabel in [(axes[0], "old_drop", "Old-fact accuracy drop (points)"),
                            (axes[1], "d_nll", "Held-out TinyStories loss increase (nats/token)")]:
        style(ax)
        if key == "old_drop":
            ax.axvspan(-1e9, FU1_BAR["old_drop"], color="#f0efec", zorder=0)
        for arm in arms:
            writer, sleep, _, label = parse_arm(arm)
            colour, is_sleep = COLOURS[writer], bool(sleep)
            dash = {"": "-", "sleep": ":", "sleepclean": ":", "gentle": "--", "gentle32": "-."}[sleep]
            pts = sorted([(lr, r) for (a, lr), r in rows.items() if a == arm], key=lambda t: t[0])
            xs, ys = [r[key] for _, r in pts], [r["ret_in"] for _, r in pts]
            ax.plot(xs, ys, dash, color=colour, linewidth=2, label=label, zorder=2)
            ax.scatter(xs, ys, s=64, zorder=3, facecolor=SURF if is_sleep else colour,
                       edgecolor=colour if is_sleep else SURF, linewidth=2)
        xs_all = [r[key] for r in rows.values()]
        lo, hi = min(0, min(xs_all)), max(xs_all)
        ax.set_xlim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.08 * (hi - lo + 1e-9))
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    axes[0].set_ylabel("End-of-stream retention, all new facts", color=INK2, fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.suptitle("Exp 02: learning vs forgetting, one point per learning rate (shaded = follow-up 1 bar)",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def table(rows):
    lines = ["| Arm | LR | Seeds | Retention | Held-out template | First 100 | Old-fact drop (pts)"
             " | Old drop, held-out template | Delta NLL | Steps/fact | Over all sleeps: reset gives back / dense update takes (pts) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (arm, lr), r in rows.items():
        sl = f"{r['reset_gain']:+.1f} / {r['dense_damage']:+.1f}" if r["dense_damage"] is not None else ""  # summed
        lines.append(f"| {r['label']} | {lr:g} | {r['n']} | {r['ret_in']:.2f} ± {r['ret_in_sd']:.2f}"
                     f" | {r['ret_out']:.2f} | {r['ret_early']:.2f} | {r['old_drop']:.1f} ± {r['old_drop_sd']:.1f}"
                     f" | {r['old_drop_out']:.1f} | {r['d_nll']:+.3f} | {r['steps']:.1f} | {sl} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--extra_runs", nargs="*", default=[])
    ap.add_argument("--extra_arms", nargs="*", default=["C", "C+sleep"],
                    help="arms taken from --extra_runs (Exp 01 controls)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--primary", required=True, help="frozen follow-up-1 writer and LR, e.g. p06ans:4")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = set(args.seeds)
    groups = load([args.runs], seeds)
    for k, v in load(args.extra_runs, seeds, set(args.extra_arms)).items():
        if k in groups:
            raise ValueError(f"{k} present in both --runs and --extra_runs")
        groups[k] = v
    rows = summarize(groups)
    pairs = paired(groups, rows, len(seeds))
    pw, plr = args.primary.split(":")
    dec = decide(rows, pairs, (pw, float(plr)), len(seeds))
    dec["paired"] = pairs
    dec["rows"] = list(rows.values())
    tag = "seeds" + "-".join(map(str, args.seeds))
    plot_frontier(rows, os.path.join(args.out, f"frontier_{tag}.png"))
    md = table(rows)
    open(os.path.join(args.out, f"table_{tag}.md"), "w").write(md + "\n")
    json.dump(dec, open(os.path.join(args.out, f"decision_{tag}.json"), "w"), indent=2)
    print(md)
    print(json.dumps({k: dec[k] for k in ("fu1", "fu2", "combined", "go_to_5090")}, indent=2))


if __name__ == "__main__":
    main()
