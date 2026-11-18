"""Exp 04: a language model that chooses what to learn next.

A small GPT gets a prior from next-token pretraining on TinyStories, then trains for a fixed
budget of steps on a pool mixing four kinds of data:

  stories  - held-out TinyStories windows    (learnable, worth learning)
  facts    - bios of 300 new invented people (learnable, worth learning, novel)
  noise    - uniformly random tokens         (not learnable)
  dupes    - 64 story windows, repeated      (learnable, but soon learnt)

Each step a selection policy picks the training batch:
  random  - uniform over the pool
  maxloss - highest current loss among 4x candidates ("curiosity")
  rho     - highest reducible loss (current loss minus a reference model's loss), RHO-LOSS style
  lp      - Exp3 bandit over the four sources, rewarded by learning progress (Graves et al. 2017)

Usage: python curriculum.py prep|prior|run|all --policy rho --seed 1 [--device cuda]
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
import evals  # noqa: E402
import facts  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

GROUPS = ["stories", "facts", "noise", "dupes"]
TRAIN_TIDS = [1, 2, 3, 4, 5, 6]  # bios use these; templates 0 and 7 are probe-only
CTX = 128


def cfg():
    return GPTConfig(ctx=CTX, mem_layer=99)  # plain 4-layer GPT, no memory layer


# ------------------------------------------------------------------ data
def prep(a):
    from tokenizers import Tokenizer
    os.makedirs(a.data, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(HERE, "..", "exp03_scale", "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    sys.path.insert(0, os.path.join(HERE, "..", "exp01_two_speed_weights"))
    from data import read_stories, encode_docs

    def stream(stories, n_tok):
        ids, n = [], 0
        for i in range(0, len(stories), 20000):
            for d in encode_docs(tok, stories[i:i + 20000], eot):
                ids.append(d)
                n += len(d)
            if n >= n_tok:
                break
        return np.concatenate(ids)[:n_tok]

    train = read_stories(os.path.join(a.raw, "train200.txt"), 200 * 2**20)
    random.Random(0).shuffle(train)
    half = len(train) // 2
    np.save(os.path.join(a.data, "prior.npy"), stream(train[:half], a.prior_tokens))
    np.save(os.path.join(a.data, "pool_stories.npy"), stream(train[half:], 4_000_000))
    np.save(os.path.join(a.data, "val.npy"), stream(read_stories(os.path.join(a.raw, "valid.txt")), 600_000))

    english = set()
    for s in train[:20000]:
        english.update(w.lower() for w in s.split())
    names = facts.name_pool(a.n_people, english, seed=4242)
    people = facts.old_entities(names, seed=7)
    rng = random.Random(1)

    def bios(n_each):  # templates 1-6 only, relations in random order
        docs = []
        for e in people:
            for _ in range(n_each):
                rels = facts.REL_NAMES[:]
                rng.shuffle(rels)
                docs.append(" ".join(facts.render(r, rng.choice(TRAIN_TIDS), e["name"], e["attrs"][r]) for r in rels))
        rng.shuffle(docs)
        return np.concatenate(encode_docs(tok, docs, eot))

    np.save(os.path.join(a.data, "pool_facts.npy"), bios(40))
    np.save(os.path.join(a.data, "hold_facts.npy"), bios(10))  # reference model's holdout (same people)
    json.dump(people, open(os.path.join(a.data, "people.json"), "w"))
    print("prepared", {f: np.load(os.path.join(a.data, f)).shape[0] for f in os.listdir(a.data) if f.endswith(".npy")})


class Pool:
    """Windows of CTX+1 tokens per source; noise is fresh random tokens each draw."""

    def __init__(self, data, seed, vocab=4096):
        self.rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.src = {"stories": np.load(os.path.join(data, "pool_stories.npy")),
                    "facts": np.load(os.path.join(data, "pool_facts.npy"))}
        st = self.src["stories"]
        starts = np.random.default_rng(123).integers(0, len(st) - CTX - 1, 64)
        self.dupes = np.stack([st[s:s + CTX + 1] for s in starts])

    def draw(self, group, n):
        if group == "noise":
            return self.rng.integers(0, self.vocab, (n, CTX + 1))
        if group == "dupes":
            return self.dupes[self.rng.integers(0, len(self.dupes), n)]
        d = self.src[group]
        s = self.rng.integers(0, len(d) - CTX - 1, n)
        return np.stack([d[i:i + CTX + 1] for i in s])

    def mixed(self, n):
        g = self.rng.integers(0, len(GROUPS), n)
        w = np.concatenate([self.draw(GROUPS[k], int((g == k).sum())) for k in range(len(GROUPS))])
        lab = np.concatenate([[k] * int((g == k).sum()) for k in range(len(GROUPS))]).astype(np.int64)
        return w, lab


def to_xy(w, dev):
    t = torch.as_tensor(np.asarray(w, dtype=np.int64), device=dev)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def item_loss(model, x, y, bs=256):
    out = []
    for i in range(0, len(x), bs):
        lg = model(x[i:i + bs]).float()
        out.append(F.cross_entropy(lg.transpose(1, 2), y[i:i + bs], reduction="none").mean(1))
    return torch.cat(out)


def train_step(model, opt, x, y):
    model.train()
    loss = F.cross_entropy(model(x).float().reshape(-1, 4096), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item()


# ------------------------------------------------------------------ prior and reference
def prior(a):
    torch.manual_seed(0)
    model = GPT(cfg()).to(a.device)
    d = np.load(os.path.join(a.data, "prior.npy"))
    W = CTX + 1
    order = np.random.default_rng(0).permutation(len(d) // W)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.1)
    steps = len(order) // a.batch
    t0 = time.time()
    for s in range(steps):
        lr = 1.5e-3 * min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = to_xy(np.stack([d[i * W:(i + 1) * W] for i in order[s * a.batch:(s + 1) * a.batch]]), a.device)
        l = train_step(model, opt, x, y)
        if s % 500 == 0:
            print(f"prior {s}/{steps} loss {l:.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), os.path.join(a.runs, "prior.pt"))
    # Reference model for RHO-LOSS: the prior trained on holdout data from each learnable source.
    hold = {"stories": np.load(os.path.join(a.data, "val.npy"))[:300_000],  # not the val windows used for eval
            "facts": np.load(os.path.join(a.data, "hold_facts.npy"))}
    rng = np.random.default_rng(5)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    for s in range(a.ref_steps):
        src = hold["stories" if s % 2 else "facts"]
        st = rng.integers(0, len(src) - CTX - 1, a.batch)
        train_step(model, opt, *to_xy(np.stack([src[i:i + CTX + 1] for i in st]), a.device))
    torch.save(model.state_dict(), os.path.join(a.runs, "reference.pt"))
    print("prior and reference saved", flush=True)


# ------------------------------------------------------------------ selection run
def evaluate(model, val, probes, dev):
    model.eval()
    x, y = to_xy(val, dev)
    rec = {"val_loss": float(item_loss(model, x, y).mean())}
    for k, p in probes.items():
        rec[k] = float(np.mean(p.run(model)[0]))
    return rec


def run(a):
    from tokenizers import Tokenizer
    torch.manual_seed(a.seed)
    dev = a.device
    model = GPT(cfg()).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.runs, "prior.pt"), map_location=dev))
    ref = None
    if a.policy == "rho":
        ref = GPT(cfg()).to(dev)
        ref.load_state_dict(torch.load(os.path.join(a.runs, "reference.pt"), map_location=dev))
        ref.eval()
    tok = Tokenizer.from_file(os.path.join(HERE, "..", "exp03_scale", "tokenizer.json"))
    people = json.load(open(os.path.join(a.data, "people.json")))
    items = [(e["name"], r, e["attrs"][r]) for e in people for r in facts.REL_NAMES]
    probes = {"fact_acc": evals.Probe(tok, items, 0), "fact_acc_heldout": evals.Probe(tok, items, 7)}
    v = np.load(os.path.join(a.data, "val.npy"))[300_000:]
    vs = np.random.default_rng(0).integers(0, len(v) - CTX - 1, 256)
    val = np.stack([v[i:i + CTX + 1] for i in vs])
    pool = Pool(a.data, a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(a.seed)
    logw = np.zeros(len(GROUPS))  # Exp3 log-weights
    picks, curve, t0 = np.zeros(len(GROUPS)), [], time.time()
    curve.append({"step": 0, **evaluate(model, val, probes, dev), "picks": picks.tolist()})
    for s in range(1, a.steps + 1):
        if a.policy == "lp":
            p = np.exp(logw - logw.max())
            p = (1 - a.gamma) * p / p.sum() + a.gamma / len(GROUPS)
            k = rng.choices(range(len(GROUPS)), weights=p)[0]
            w, lab = pool.draw(GROUPS[k], a.batch), np.full(a.batch, k)
        else:
            w, lab = pool.mixed(a.batch * (1 if a.policy == "random" else a.cand))
        x, y = to_xy(w, dev)
        if a.policy in ("maxloss", "rho"):
            score = item_loss(model, x, y)
            if a.policy == "rho":
                score = score - item_loss(ref, x, y)
            top = torch.topk(score, a.batch).indices
            x, y, lab = x[top], y[top], lab[top.cpu().numpy()]
        before = item_loss(model, x, y).mean().item() if a.policy == "lp" else None
        train_step(model, opt, x, y)
        if a.policy == "lp":  # reward: relative prediction gain on the batch just learnt
            after = item_loss(model, x, y).mean().item()
            r = float(np.clip((before - after) / max(before, 1e-6) / a.lp_scale, 0, 1))
            logw[k] += a.gamma * r / (len(GROUPS) * p[k])
        picks += np.bincount(lab, minlength=len(GROUPS))
        if s % a.eval_every == 0 or s == a.steps:
            rec = {"step": s, **evaluate(model, val, probes, dev), "picks": picks.tolist()}
            curve.append(rec)
            frac = picks / picks.sum()
            print(f"[{a.policy} s{a.seed}] {s} val {rec['val_loss']:.3f} facts {rec['fact_acc']:.3f}"
                  f"/{rec['fact_acc_heldout']:.3f} picks " + " ".join(f"{g}={f:.2f}" for g, f in zip(GROUPS, frac))
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(a.runs, "sel"), exist_ok=True)
    out = os.path.join(a.runs, "sel", f"{a.policy}_s{a.seed}.json")
    json.dump({"args": vars(a), "groups": GROUPS, "curve": curve, "seconds": time.time() - t0}, open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "prior", "run"])
    ap.add_argument("--raw", default="/home/user/data_ts")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prior_tokens", type=int, default=20_000_000)
    ap.add_argument("--n_people", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ref_steps", type=int, default=1500)
    ap.add_argument("--policy", choices=["random", "maxloss", "rho", "lp"], default="random")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--cand", type=int, default=4, help="candidates per slot for maxloss/rho")
    ap.add_argument("--gamma", type=float, default=0.1, help="Exp3 exploration")
    ap.add_argument("--lp_scale", type=float, default=0.05, help="relative gain that counts as reward 1")
    ap.add_argument("--eval_every", type=int, default=150)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.runs, exist_ok=True)
    {"prep": prep, "prior": prior, "run": run}[a.stage](a)


if __name__ == "__main__":
    main()
