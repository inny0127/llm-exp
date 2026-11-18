"""Exp 10 analysis: value-function teacher vs random / greedy / lp on held-out datasets."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "runs")
rows = {}
for p in sorted(glob.glob(os.path.join(R, "*.json"))):
    r = json.load(open(p))
    name = "teacher" if "train_log" in r else r["args"]["policy"]
    rows.setdefault(name, []).extend(r["eval"])
lines = ["| Policy | Evals | Code, learnable | Code, noise | Story loss | Fact acc | Noise mean position |", "|---|---|---|---|---|---|---|"]
m = {}
for k in ("random", "greedy", "lp", "teacher"):
    e = rows.get(k, [])
    if not e:
        continue
    m[k] = {f: float(np.mean([x[f] for x in e])) for f in ("code_learnable", "code_noise", "val_loss", "fact_acc")}
    m[k]["noise_pos"] = float(np.mean([x["mean_position"]["noise"] for x in e]))
    lines.append(f"| {k} | {len(e)} | {m[k]['code_learnable']:.3f} | {m[k]['code_noise']:.3f} | {m[k]['val_loss']:.3f} | "
                 f"{m[k]['fact_acc']:.3f} | {m[k]['noise_pos']:.2f} |")
ck = {}
if {"teacher", "random", "lp"} <= m.keys():
    t = m["teacher"]
    ck["H1 teacher code < random and lp"] = bool(t["code_learnable"] < m["random"]["code_learnable"] and t["code_learnable"] < m["lp"]["code_learnable"])
    ck["H2 teacher noise position < 0.5"] = {"pos": t["noise_pos"], "holds": t["noise_pos"] < 0.5}
    ck["H3 teacher story loss <= random + 0.10"] = {"teacher": t["val_loss"], "random": m["random"]["val_loss"], "holds": t["val_loss"] <= m["random"]["val_loss"] + 0.10}
logs = {os.path.basename(p): [(x["episode"], round(x["code_learnable"], 3), round(x["mean_position"]["noise"], 2)) for x in json.load(open(p))["train_log"][::5]]
        for p in sorted(glob.glob(os.path.join(R, "teacher_s*.json")))}
md = "\n".join(lines)
open(os.path.join(HERE, "results", "table.md"), "w").write(md + "\n")
json.dump({"rows": m, "checks": ck, "learning_curves": logs}, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)
print(md)
print(json.dumps(ck, indent=1))
print(json.dumps(logs, indent=0)[:1500])
