"""Exp 05: Order matters for compression.

One fixed, finite dataset D (stories, new-person bios, random-token noise, repeated windows)
is learnt in ONE epoch, every item exactly once, in an order chosen by a policy. The
prequential code length of D under that order is the sum, over batches, of the model's loss
on the batch just before it trains on it (Blier & Ollivier, 2018), in bits. Same data, same
model, same steps: only the order changes.

Orders:
  random      - a random permutation
  easy_first  - ascending loss under the prior (static)
  hard_first  - descending loss under the prior (static)
  maxloss     - greedy: highest current loss among 4x candidates from what is left
  rho         - greedy: highest reducible loss (current minus reference) among 4x candidates
  lp          - Exp3 bandit over sources, reward = compression progress on the batch just learnt

Reuses the prior, reference model, data and helpers of ../exp04_curriculum.
Usage: python order.py --order rho --seed 1 [--device cuda]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp04_curriculum"))
import curriculum as cu  # noqa: E402
import evals  # noqa: E402
import facts  # noqa: E402
from model import GPT  # noqa: E402

ORDERS = ["random", "easy_first", "hard_first", "maxloss", "rho", "lp"]
LN2 = math.log(2)


def build_dataset(data, n_per_group, seed):
    """Fixed D: n windows per source, the same for every order with this seed."""
    pool = cu.Pool(data, seed=10_000 + seed)
    w, lab = [], []
    for k, g in enumerate(cu.GROUPS):
        if g == "dupes":  # the 64 dupe windows, each repeated to fill the group
            reps = int(math.ceil(n_per_group / len(pool.dupes)))
            w.append(np.tile(pool.dupes, (reps, 1))[:n_per_group])
        else:
            w.append(pool.draw(g, n_per_group))
        lab.append(np.full(n_per_group, k))
    return np.concatenate(w), np.concatenate(lab)


def run(a):
    from tokenizers import Tokenizer
    torch.manual_seed(a.seed)
    dev = a.device
    model = GPT(cu.cfg()).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.runs4, "prior.pt"), map_location=dev))
    ref = None
    if a.order == "rho":
        ref = GPT(cu.cfg()).to(dev)
        ref.load_state_dict(torch.load(os.path.join(a.runs4, "reference.pt"), map_location=dev))
        ref.eval()
    tok = Tokenizer.from_file(os.path.join(HERE, "..", "exp03_scale", "tokenizer.json"))
    people = json.load(open(os.path.join(a.data, "people.json")))
    items = [(e["name"], r, e["attrs"][r]) for e in people for r in facts.REL_NAMES]
    probes = {"fact_acc": evals.Probe(tok, items, 0), "fact_acc_heldout": evals.Probe(tok, items, 7)}
    v = np.load(os.path.join(a.data, "val.npy"))[300_000:]
    vs = np.random.default_rng(0).integers(0, len(v) - cu.CTX - 1, 256)
    val = np.stack([v[i:i + cu.CTX + 1] for i in vs])

    D, lab = build_dataset(a.data, a.n_per_group, a.seed)
    N = len(D)
    left = np.ones(N, bool)
    rng, nrng = random.Random(a.seed), np.random.default_rng(a.seed)
    if a.order in ("easy_first", "hard_first"):  # static order by the prior's loss
        x, y = cu.to_xy(D, dev)
        prior_loss = cu.item_loss(model, x, y, bs=512).cpu().numpy()
        static = np.argsort(prior_loss if a.order == "easy_first" else -prior_loss, kind="stable")
    elif a.order == "random":
        static = nrng.permutation(N)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    logw = np.zeros(len(cu.GROUPS))
    bits = np.zeros(len(cu.GROUPS))  # code length per source (bits)
    steps = N // a.batch
    curve, t0, start, spent = [], time.time(), 0, 0.0
    os.makedirs(os.path.join(a.runs, "order"), exist_ok=True)
    stem = os.path.join(a.runs, "order", f"{a.order}_s{a.seed}")
    if os.path.exists(stem + ".state.pt"):
        st = torch.load(stem + ".state.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        rng.setstate(st["rng"]); nrng.bit_generator.state = st["nrng"]
        left, logw, bits, curve, start, spent = st["left"], st["logw"], st["bits"], st["curve"], st["step"], st["seconds"]
        print(f"resumed at step {start}", flush=True)
    for s in range(start, steps):
        if a.order in ("random", "easy_first", "hard_first"):
            idx = static[s * a.batch:(s + 1) * a.batch]
        elif a.order == "lp":
            avail = np.array([left[lab == k].any() for k in range(len(cu.GROUPS))])
            p = np.exp(logw - logw.max()) * avail
            p = (1 - a.gamma) * p / p.sum() + a.gamma * avail / avail.sum()
            k = rng.choices(range(len(cu.GROUPS)), weights=p)[0]
            pool_k = np.flatnonzero(left & (lab == k))
            idx = nrng.choice(pool_k, min(a.batch, len(pool_k)), replace=False)
            if len(idx) < a.batch:  # source nearly exhausted: top up at random
                rest = np.flatnonzero(left & ~np.isin(np.arange(N), idx))
                idx = np.concatenate([idx, nrng.choice(rest, a.batch - len(idx), replace=False)])
        else:  # maxloss / rho: greedy among candidates drawn from what is left
            rest = np.flatnonzero(left)
            cand = nrng.choice(rest, min(len(rest), a.batch * a.cand), replace=False)
            cx, cy = cu.to_xy(D[cand], dev)
            score = cu.item_loss(model, cx, cy)
            if ref is not None:
                score = score - cu.item_loss(ref, cx, cy)
            idx = cand[torch.topk(score, a.batch).indices.cpu().numpy()]
        left[idx] = False
        x, y = cu.to_xy(D[idx], dev)
        before = cu.item_loss(model, x, y)  # prequential cost: loss BEFORE learning the batch
        b = before.cpu().numpy() * cu.CTX / LN2
        np.add.at(bits, lab[idx], b)
        cu.train_step(model, opt, x, y)
        if a.order == "lp":  # compression progress on the batch just learnt
            after = cu.item_loss(model, x, y).mean().item()
            r = float(np.clip((before.mean().item() - after) / before.mean().item() / a.lp_scale, 0, 1))
            logw[k] += a.gamma * r / (len(cu.GROUPS) * p[k])
        if (s + 1) % a.eval_every == 0 or s + 1 == steps:
            rec = {"step": s + 1, "bits": bits.tolist(), **cu.evaluate(model, val, probes, dev)}
            curve.append(rec)
            print(f"[{a.order} s{a.seed}] {s + 1}/{steps} code {bits.sum() / ((s + 1) * a.batch * cu.CTX):.3f} bits/tok"
                  f" val {rec['val_loss']:.3f} facts {rec['fact_acc']:.3f}/{rec['fact_acc_heldout']:.3f}"
                  f" ({spent + time.time() - t0:.0f}s)", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.getstate(),
                        "nrng": nrng.bit_generator.state, "left": left, "logw": logw, "bits": bits,
                        "curve": curve, "step": s + 1, "seconds": spent + time.time() - t0}, stem + ".state.pt.tmp")
            os.replace(stem + ".state.pt.tmp", stem + ".state.pt")
    tokens = steps * a.batch * cu.CTX
    res = {"args": vars(a), "groups": cu.GROUPS, "curve": curve, "bits_per_group": bits.tolist(),
           "tokens": tokens, "tokens_per_group": a.n_per_group * cu.CTX, "seconds": spent + time.time() - t0}
    json.dump(res, open(stem + ".json.tmp", "w"))
    os.replace(stem + ".json.tmp", stem + ".json")
    os.remove(stem + ".state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=ORDERS, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "exp04_curriculum", "data"))
    ap.add_argument("--runs4", default=os.path.join(HERE, "..", "exp04_curriculum", "runs"))
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n_per_group", type=int, default=12_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--cand", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--lp_scale", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=150)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run(a)


if __name__ == "__main__":
    main()
