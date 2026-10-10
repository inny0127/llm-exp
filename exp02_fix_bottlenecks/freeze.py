"""Apply the pre-registered dev rules (DESIGN.md) to the dev runs and print the frozen L* and R.

Usage: python freeze.py --dev ../exp01_two_speed_weights/runs/exp02_dev4 \
           --base ../exp01_two_speed_weights/runs/exp02_dev/base_C_lr16_s0.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp01_two_speed_weights"))
from analyze import run_metrics  # noqa: E402


def metrics(path):
    r = json.load(open(path))
    m = run_metrics(r)
    m["consol"] = (sum(s["pre_all"] - s["post_all"] for s in r["sleeps"]) / len(r["sleeps"])
                   if r["sleeps"] else None)
    m["dense_damage_sum"] = (100 * sum(s["old_after_reset"] - s["old_after_sleep"] for s in r["sleeps"])
                             if r["sleeps"] else None)
    return m


def fu1(dev):
    runs = {float(os.path.basename(p).split("_lr")[1].split("_s")[0]): metrics(p)
            for p in glob.glob(os.path.join(dev, "p06ans_lr*_s0.json"))}
    report, eligible = {}, []
    for lr, m in sorted(runs.items()):
        checks = {"old_drop<=6": m["old_drop"] <= 6.0, "ret_in>=0.50": m["ret_in"] >= 0.50,
                  "ret_out>=0.40": m["ret_out"] >= 0.40, "old_drop_out<=8": m["old_drop_out"] <= 8.0,
                  "cap<=0.05": m["cap_frac"] <= 0.05}
        report[lr] = {"ret_in": m["ret_in"], "ret_out": m["ret_out"], "old_drop": m["old_drop"],
                      "old_drop_out": m["old_drop_out"], "cap_frac": m["cap_frac"], "checks": checks}
        if all(checks.values()):
            eligible.append((lr, m))
    if 4.0 in runs and not all(report[4.0]["checks"].values()) and 2.0 not in runs:
        return {"report": report, "decision": "RUN p06ans lr 2 on dev, then re-apply the rule"}
    if eligible:
        best = max(m["ret_in"] for _, m in eligible)
        near = [(lr, m) for lr, m in eligible if best - m["ret_in"] <= 0.02]
        lr, _ = min(near, key=lambda t: t[1]["old_drop"])
        return {"report": report, "L*": lr, "flag": None}
    ok = [(lr, m) for lr, m in runs.items() if m["ret_in"] >= 0.50 and m["ret_out"] >= 0.40]
    if ok:
        lr, _ = min(ok, key=lambda t: t[1]["old_drop"])
        return {"report": report, "L*": lr, "flag": "thin margin"}
    return {"report": report, "L*": "fallback: ans lr 0.5"}


def fu2(dev, base_path):
    base = metrics(base_path)
    report = {}
    for R in ("gentle", "gentle32"):
        p = os.path.join(dev, f"C+{R}_lr16_s0.json")
        if not os.path.exists(p):
            return {"decision": f"missing {p}"}
        m = metrics(p)
        d = {k: m[k] - base[k] for k in ("ret_in", "ret_out", "old_drop", "old_drop_out", "d_nll")}
        checks = {"d_ret_in>=0.15": d["ret_in"] >= 0.15, "d_ret_out>=0.15": d["ret_out"] >= 0.15,
                  "d_old_drop<=0": d["old_drop"] <= 0.0, "d_old_drop_out<=0": d["old_drop_out"] <= 0.0,
                  "d_nll<=0.010": d["d_nll"] <= 0.010, "consol<=0.01": m["consol"] <= 0.01}
        report[R] = {"paired": d, "consol": m["consol"], "dense_damage_sum": m["dense_damage_sum"],
                     "ret_in": m["ret_in"], "d_nll_abs": m["d_nll"], "checks": checks,
                     "gate": all(checks.values())}
    g, g32 = report["gentle"], report["gentle32"]
    if (g["gate"] and g32["gate"] and g32["dense_damage_sum"] <= g["dense_damage_sum"] - 2.0
            and g32["ret_in"] >= g["ret_in"] - 0.02 and g32["d_nll_abs"] <= g["d_nll_abs"] + 0.003):
        R = "gentle32"
    elif g["gate"]:
        R = "gentle"
    elif g32["gate"]:
        R = "gentle32"
    else:
        R = "gentle (dev gate failed; reported)"
    return {"report": report, "R": R}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    out = {"fu1": fu1(args.dev), "fu2": fu2(args.dev, args.base)}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
