"""Exp 06 and 07: small synthetic worlds where the order of learning is the object of study.

Exp 06 "syllabus": data comes from a hidden program library. Base functions on Z_23
(a(x)=x+1, b(x)=2x, c(x)=x^2+1) are composed to depth 1-4. Some depth-3/4 compositions are
never trained (held out). A learner that picks what to learn next by compression progress
(Exp3 bandit over depths) is compared with random, easy-first and hard-first syllabi.
Question: does compression progress alone rediscover the dependency order 1 -> 2 -> 3 -> 4?

Exp 07 "beliefs": the answer to "x y # t =" is always z = x+y mod 23. In 80% of the fixed
dataset the tag t equals z, so "copy the tag" explains it as well as the rule does. In the
other 20% t is random. The same dataset is learnt once, in different orders. Conflict
probes (t != z) then show whether the model believes the rule or the shortcut.

Usage: python worlds.py syllabus|beliefs --policy P --seed S [--device cuda]
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp01_two_speed_weights"))
from model import GPT, GPTConfig  # noqa: E402

P = 23
A, B, C, EQ, TAG, BOS = P, P + 1, P + 2, P + 3, P + 4, P + 5
FNS = {A: lambda x: (x + 1) % P, B: lambda x: (2 * x) % P, C: lambda x: (x * x + 1) % P}
LN2 = math.log(2)


def model_cfg():
    return GPTConfig(vocab_size=32, ctx=8, n_layer=4, n_head=4, d_model=128, d_ff=512, mem_layer=99)


def answer_loss(model, x, y):
    """Per-item loss on the answer token only (the last position)."""
    lg = model(x)[:, -1].float()
    return F.cross_entropy(lg, y, reduction="none"), lg


def step(model, opt, x, y):
    model.train()
    loss = answer_loss(model, x, y)[0].mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)


class Exp3:
    def __init__(self, k, gamma, scale, rng):
        self.w, self.g, self.s, self.rng, self.k = np.zeros(k), gamma, scale, rng, k

    def probs(self, avail=None):
        avail = np.ones(self.k) if avail is None else avail
        p = np.exp(self.w - self.w.max()) * avail
        return (1 - self.g) * p / p.sum() + self.g * avail / avail.sum()

    def pick(self, avail=None):
        self.p = self.probs(avail)
        self.arm = self.rng.choices(range(self.k), weights=self.p)[0]
        return self.arm

    def reward(self, before, after):
        r = float(np.clip((before - after) / max(before, 1e-6) / self.s, 0, 1))
        self.w[self.arm] += self.g * r / (self.k * self.p[self.arm])


# ------------------------------------------------------------------ Exp 06: syllabus
def compositions(d):
    out = [[]]
    for _ in range(d):
        out = [s + [f] for s in out for f in (A, B, C)]
    return out


def apply(seq, x):
    for f in reversed(seq):  # the function next to x is applied first
        x = FNS[f](x)
    return x


def syllabus(a):
    rng, nrng = random.Random(a.seed), np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    dev = a.device
    seen, held = {}, {}
    hr = random.Random(0)  # the held-out compositions are the same for every policy and seed
    for d in range(1, 5):
        comps = compositions(d)
        hr.shuffle(comps)
        n_hold = len(comps) // 4 if d >= 3 else 0
        held[d], seen[d] = comps[:n_hold], comps[n_hold:]

    def batch(d, n, pool):
        comps = pool[d]
        xs, ys = [], []
        for _ in range(n):
            s, x0 = comps[rng.randrange(len(comps))], rng.randrange(P)
            toks = [BOS] * (4 - d) + s + [x0, EQ]
            xs.append(toks)
            ys.append(apply(s, x0))
        return torch.tensor(xs, device=dev), torch.tensor(ys, device=dev)

    def evaluate():
        model.eval()
        rec = {}
        with torch.no_grad():
            for d in range(1, 5):
                for name, pool in (("seen", seen), ("held", held)):
                    if pool[d]:
                        x, y = batch(d, 1024, pool)
                        rec[f"acc_{name}_d{d}"] = float((answer_loss(model, x, y)[1].argmax(-1) == y).float().mean())
        return rec

    model = GPT(model_cfg()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    bandit = Exp3(4, a.gamma, a.lp_scale, rng)
    picks, window, curve, t0 = np.zeros(4), np.zeros(4), [], time.time()
    for s in range(1, a.steps + 1):
        if a.policy == "random":
            ds = nrng.integers(1, 5, a.batch)
        elif a.policy in ("easy_first", "hard_first"):  # oracle syllabi: a quarter of the steps per depth
            q = min(3, (s - 1) * 4 // a.steps)
            ds = np.full(a.batch, q + 1 if a.policy == "easy_first" else 4 - q)
        else:  # lp: one depth per step, chosen by compression progress
            ds = np.full(a.batch, bandit.pick() + 1)
        parts = [batch(int(d), int((ds == d).sum()), seen) for d in range(1, 5) if (ds == d).any()]
        x, y = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        if a.policy == "lp":
            with torch.no_grad():
                before = answer_loss(model, x, y)[0].mean().item()
        step(model, opt, x, y)
        if a.policy == "lp":
            with torch.no_grad():
                bandit.reward(before, answer_loss(model, x, y)[0].mean().item())
        cnt = np.bincount(ds - 1, minlength=4)
        picks += cnt
        window += cnt
        if s % a.eval_every == 0 or s == a.steps:
            rec = {"step": s, "window_share": (window / window.sum()).tolist(), **evaluate()}
            curve.append(rec)
            window[:] = 0
            if s % (a.eval_every * 10) == 0 or s == a.steps:
                print(f"[syllabus {a.policy} s{a.seed}] {s} share " + " ".join(f"{v:.2f}" for v in rec["window_share"])
                      + " acc " + " ".join(f"{rec[f'acc_seen_d{d}']:.2f}" for d in range(1, 5))
                      + f" held {rec['acc_held_d3']:.2f}/{rec['acc_held_d4']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    save(a, {"curve": curve, "picks": picks.tolist(), "seconds": time.time() - t0})


# ------------------------------------------------------------------ Exp 07: beliefs
def beliefs(a):
    rng, nrng = random.Random(a.seed), np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    dev = a.device
    N = a.n_items
    n1 = int(round(N * a.rule_only_frac))
    xs, ys_, src = nrng.integers(0, P, N), nrng.integers(0, P, N), np.zeros(N, int)
    z = (xs + ys_) % P
    tag = z.copy()
    src[:n1] = 1  # source 1: rule only, random tag
    tag[:n1] = nrng.integers(0, P, n1)
    X = torch.tensor(np.stack([np.full(N, BOS), xs, ys_, np.full(N, TAG), tag, np.full(N, EQ)], 1), device=dev)
    Y = torch.tensor(z, device=dev)
    # conflict probes: tag differs from the rule's answer
    px, py = nrng.integers(0, P, 2048), nrng.integers(0, P, 2048)
    pz = (px + py) % P
    pt = (pz + nrng.integers(1, P, 2048)) % P
    PX = torch.tensor(np.stack([np.full(2048, BOS), px, py, np.full(2048, TAG), pt, np.full(2048, EQ)], 1), device=dev)
    PZ, PT = torch.tensor(pz, device=dev), torch.tensor(pt, device=dev)

    model = GPT(model_cfg()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    if a.policy == "random":
        order = nrng.permutation(N)
    elif a.policy == "rule_first":
        order = np.concatenate([nrng.permutation(np.flatnonzero(src == 1)), nrng.permutation(np.flatnonzero(src == 0))])
    elif a.policy == "rule_last":
        order = np.concatenate([nrng.permutation(np.flatnonzero(src == 0)), nrng.permutation(np.flatnonzero(src == 1))])
    left = np.ones(N, bool)
    bandit = Exp3(2, a.gamma, a.lp_scale, rng)
    bits, curve, t0 = np.zeros(2), [], time.time()
    steps = N // a.batch

    def evaluate():
        model.eval()
        with torch.no_grad():
            pred = answer_loss(model, PX, PZ)[1].argmax(-1)
        return {"rule": float((pred == PZ).float().mean()), "shortcut": float((pred == PT).float().mean())}

    for s in range(steps):
        if a.policy == "lp":  # order chosen online by compression progress over the two sources
            avail = np.array([left[src == k].any() for k in range(2)], float)
            k = bandit.pick(avail)
            cand = np.flatnonzero(left & (src == k))
            idx = nrng.choice(cand, min(a.batch, len(cand)), replace=False)
            if len(idx) < a.batch:
                rest = np.flatnonzero(left & ~np.isin(np.arange(N), idx))
                idx = np.concatenate([idx, nrng.choice(rest, a.batch - len(idx), replace=False)])
        else:
            idx = order[s * a.batch:(s + 1) * a.batch]
        left[idx] = False
        x, y = X[idx], Y[idx]
        with torch.no_grad():
            before = answer_loss(model, x, y)[0]
        np.add.at(bits, src[idx], before.cpu().numpy() / LN2)  # prequential code, bits per answer
        step(model, opt, x, y)
        if a.policy == "lp":
            with torch.no_grad():
                bandit.reward(before.mean().item(), answer_loss(model, x, y)[0].mean().item())
        if (s + 1) % a.eval_every == 0 or s + 1 == steps:
            curve.append({"step": s + 1, "bits": bits.tolist(), **evaluate()})
    fin = curve[-1]
    print(f"[beliefs {a.policy} s{a.seed}] code {bits.sum() / N:.3f} bits/item rule {fin['rule']:.3f}"
          f" shortcut {fin['shortcut']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    save(a, {"curve": curve, "bits": bits.tolist(), "n_items": N, "n_rule_only": n1, "seconds": time.time() - t0})


def save(a, res):
    os.makedirs(os.path.join(a.runs, a.exp), exist_ok=True)
    out = os.path.join(a.runs, a.exp, f"{a.policy}_s{a.seed}.json")
    json.dump({"args": vars(a), **res}, open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["syllabus", "beliefs"])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=8000, help="syllabus only")
    ap.add_argument("--n_items", type=int, default=200_000, help="beliefs only: dataset size (one epoch)")
    ap.add_argument("--rule_only_frac", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--lp_scale", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=50)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    {"syllabus": syllabus, "beliefs": beliefs}[a.exp](a)


if __name__ == "__main__":
    main()
