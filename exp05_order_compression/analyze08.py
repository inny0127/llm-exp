"""Exp 08 analysis: final model vs path ensemble, with and without refusal.

Usage: python analyze08.py [--runs results08/runs] [--out results08]
"""
import argparse
import glob
import json
import os

import numpy as np

from analyze05 import ORDERS, spearman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results08/runs")
    ap.add_argument("--out", default="results08")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = []
    for p in sorted(glob.glob(os.path.join(a.runs, "*.json"))):
        r = json.load(open(p))
        g, b, tpg = r["groups"], np.array(r["bits_per_group"]), r["tokens_per_group"]
        learn = [g.index(k) for k in ("stories", "facts", "dupes")]
        tr = np.array(r["trained_per_group"])
        runs.append({"cond": "skip" if r["args"]["allow_skip"] else "ens", "order": r["args"]["order"],
                     "seed": r["args"]["seed"], "code": b[learn].sum() / (tpg * len(learn)),
                     "final_val": r["curve"][-1]["val_loss"], "final_fact": r["curve"][-1]["fact_acc"],
                     "ens_val": r["path_ensemble"]["val_loss"], "ens_fact": r["path_ensemble"]["fact_acc"],
                     "noise_trained": float(tr[g.index("noise")] / (tpg / 128))})
    lines = ["| Condition | Order | Code, learnable | Final story loss | Path-ensemble story loss | Final fact acc | Ensemble fact acc | Noise trained |",
             "|---|---|---|---|---|---|---|---|"]
    rows = {}
    for c in ("ens", "skip"):
        for o in ORDERS:
            rs = [x for x in runs if x["cond"] == c and x["order"] == o]
            if not rs:
                continue
            m = {k: float(np.mean([x[k] for x in rs])) for k in ("code", "final_val", "ens_val", "final_fact", "ens_fact", "noise_trained")}
            rows[(c, o)] = m
            lines.append(f"| {c} | {o} | {m['code']:.3f} | {m['final_val']:.3f} | {m['ens_val']:.3f} | {m['final_fact']:.3f} | "
                         f"{m['ens_fact']:.3f} | {m['noise_trained']:.2f} |")
    E = [x for x in runs if x["cond"] == "ens"]
    S = [x for x in runs if x["cond"] == "skip"]
    ck = {}
    if len(E) >= 4:
        ck["H1 ens: rho(code, ensemble story loss) >= 0.5"] = {"rho": spearman([x["code"] for x in E], [x["ens_val"] for x in E])}
        ck["H1 ens: rho(code, ensemble story loss) >= 0.5"]["holds"] = ck["H1 ens: rho(code, ensemble story loss) >= 0.5"]["rho"] >= 0.5
        ck["H1 reference: rho(code, final story loss)"] = {"rho": spearman([x["code"] for x in E], [x["final_val"] for x in E])}
    if ("ens", "easy_first") in rows and ("ens", "lp") in rows:
        ck["H2 ensemble story loss <= 3.0 for easy_first and lp"] = {
            "easy_first": rows[("ens", "easy_first")]["ens_val"], "lp": rows[("ens", "lp")]["ens_val"],
            "holds": rows[("ens", "easy_first")]["ens_val"] <= 3.0 and rows[("ens", "lp")]["ens_val"] <= 3.0}
    if ("skip", "random") in rows:
        base = rows[("skip", "random")]["final_val"]
        sk = [rows[k] for k in rows if k[0] == "skip"]
        ck["H3 skip: noise trained < 5% and final story loss within 0.15 of random"] = {
            "max_noise_trained": max(m["noise_trained"] for m in sk),
            "max_gap": max(abs(m["final_val"] - base) for m in sk),
            "holds": max(m["noise_trained"] for m in sk) < 0.05 and max(abs(m["final_val"] - base) for m in sk) <= 0.15}
    if len(S) >= 4:
        rv = spearman([x["code"] for x in S], [x["final_val"] for x in S])
        rf = spearman([x["code"] for x in S], [x["final_fact"] for x in S])
        ck["H4 skip: rho(code, story loss) >= 0.5 and rho(code, facts) <= -0.5"] = {"rho_val": rv, "rho_fact": rf, "holds": rv >= 0.5 and rf <= -0.5}
    md = "\n".join(lines)
    open(os.path.join(a.out, "table.md"), "w").write(md + "\n")
    json.dump({"runs": runs, "checks": ck}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(md)
    print(json.dumps(ck, indent=1, default=float))


if __name__ == "__main__":
    main()
