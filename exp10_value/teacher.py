"""Exp 10: a teacher with a value function that learns what to teach, with reward = -prequential code.

Environment: Exp 05's one-epoch setup at a smaller size (n_per_group items from each of
stories / facts / noise / dupes; every item learnt exactly once). Each step the teacher picks a
source; the student (Exp 04's prior GPT) trains on a batch from it.
  reward  = -(bits of that batch, coded just before learning it) / 1e5
  return  = -(prequential code length of the whole dataset): maximising it IS minimising MDL.
State: the student's loss on a small sample of each source's remaining items, the fraction of
each source left, and the fraction of the episode left (9 numbers).
The teacher is a small MLP Q-function trained by fitted Q-iteration over episodes
(epsilon-greedy, each episode on a freshly drawn dataset). Baselines run in the same
environment: random source, myopic greedy (highest current loss), and the Exp3 bandit (lp).

Usage: python teacher.py train --seed 1 --device cuda      (trains, then evaluates on held-out datasets)
       python teacher.py baseline --policy random --seed 1
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
sys.path.insert(0, os.path.join(HERE, "..", "exp05_order_compression"))
sys.path.insert(0, os.path.join(HERE, "..", "exp04_curriculum"))
import curriculum as cu  # noqa: E402
import evals  # noqa: E402
import facts  # noqa: E402
from model import GPT  # noqa: E402
from order import build_dataset  # noqa: E402

LN2 = math.log(2)
K = 4


class Env:
    def __init__(self, a, data_seed, probes, val):
        self.a, self.dev = a, a.device
        self.D, self.lab = build_dataset(a.data, a.n_per_group, data_seed)
        self.N = len(self.D)
        self.left = np.ones(self.N, bool)
        self.rng = np.random.default_rng(data_seed)
        self.model = GPT(cu.cfg()).to(self.dev)
        self.model.load_state_dict(torch.load(os.path.join(a.runs4, "prior.pt"), map_location=self.dev))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
        self.steps, self.t = self.N // a.batch, 0
        self.bits = np.zeros(K)
        self.picks = []
        self.probes, self.val = probes, val

    def avail(self):
        return np.array([(self.left & (self.lab == k)).sum() >= 1 for k in range(K)])

    def state(self):
        f = []
        for k in range(K):
            pool = np.flatnonzero(self.left & (self.lab == k))
            if len(pool):
                idx = self.rng.choice(pool, min(16, len(pool)), replace=False)
                f.append(float(cu.item_loss(self.model, *cu.to_xy(self.D[idx], self.dev)).mean()) / 10)
            else:
                f.append(0.0)
        f += [(self.left & (self.lab == k)).mean() * K for k in range(K)]
        f.append(1 - self.t / self.steps)
        return np.array(f, np.float32)

    def step(self, k):
        pool = np.flatnonzero(self.left & (self.lab == k))
        idx = self.rng.choice(pool, min(self.a.batch, len(pool)), replace=False)
        if len(idx) < self.a.batch:  # source nearly exhausted: top up at random
            rest = np.flatnonzero(self.left & ~np.isin(np.arange(self.N), idx))
            idx = np.concatenate([idx, self.rng.choice(rest, self.a.batch - len(idx), replace=False)])
        self.left[idx] = False
        x, y = cu.to_xy(self.D[idx], self.dev)
        before = cu.item_loss(self.model, x, y)
        b = before.cpu().numpy() * cu.CTX / LN2
        np.add.at(self.bits, self.lab[idx], b)
        cu.train_step(self.model, self.opt, x, y)
        self.t += 1
        self.picks.append(k)
        after = cu.item_loss(self.model, x, y).mean().item()
        return -b.sum() / 1e5, before.mean().item(), after

    def result(self):
        tpg = self.a.n_per_group * cu.CTX
        learn = [0, 1, 3]
        ev = cu.evaluate(self.model, self.val, self.probes, self.dev)
        pk = np.array(self.picks)
        pos = {cu.GROUPS[k]: float(np.mean(np.flatnonzero(pk == k)) / len(pk)) if (pk == k).any() else None for k in range(K)}
        return {"code_learnable": float(self.bits[learn].sum() / (tpg * 3)), "code_noise": float(self.bits[2] / tpg),
                "mean_position": pos, **ev}


class QNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(9, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                     torch.nn.Linear(64, K))

    def forward(self, s):
        return self.f(s)


def setup(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(HERE, "..", "exp03_scale", "tokenizer.json"))
    people = json.load(open(os.path.join(a.data, "people.json")))
    items = [(e["name"], r, e["attrs"][r]) for e in people for r in facts.REL_NAMES]
    probes = {"fact_acc": evals.Probe(tok, items, 0), "fact_acc_heldout": evals.Probe(tok, items, 7)}
    v = np.load(os.path.join(a.data, "val.npy"))[300_000:]
    vs = np.random.default_rng(0).integers(0, len(v) - cu.CTX - 1, 256)
    return probes, np.stack([v[i:i + cu.CTX + 1] for i in vs])


def masked_argmax(q, avail):
    q = q.copy()
    q[~avail] = -1e9
    return int(np.argmax(q))


def episode(a, env, policy, rng, eps=0.0, q=None, bandit=None, buf=None):
    s = env.state()
    while env.t < env.steps:
        av = env.avail()
        if policy == "random":
            k = rng.choice([i for i in range(K) if av[i]])
        elif policy == "greedy":  # myopic: highest current loss among available sources
            k = masked_argmax(s[:K].copy(), av)
        elif policy == "lp":
            w = np.exp(bandit["w"] - bandit["w"].max()) * av
            p = (1 - a.gamma) * w / w.sum() + a.gamma * av / av.sum()
            k = rng.choices(range(K), weights=p)[0]
        else:  # q teacher
            if rng.random() < eps:
                k = rng.choice([i for i in range(K) if av[i]])
            else:
                with torch.no_grad():
                    k = masked_argmax(q(torch.as_tensor(s, device=a.device)).cpu().numpy(), av)
        r, before, after = env.step(k)
        if policy == "lp":
            rr = float(np.clip((before - after) / max(before, 1e-6) / a.lp_scale, 0, 1))
            bandit["w"][k] += a.gamma * rr / (K * p[k])
        s2 = env.state()
        if buf is not None:
            buf.append((s, k, r, s2, env.avail(), env.t >= env.steps))
        s = s2
    return env.result()


def fit_q(a, q, qt, opt, buf, iters):
    S = torch.tensor(np.stack([b[0] for b in buf]), device=a.device)
    A = torch.tensor([b[1] for b in buf], device=a.device)
    R = torch.tensor([b[2] for b in buf], device=a.device, dtype=torch.float32)
    S2 = torch.tensor(np.stack([b[3] for b in buf]), device=a.device)
    AV2 = torch.tensor(np.stack([b[4] for b in buf]), device=a.device)
    DONE = torch.tensor([b[5] for b in buf], device=a.device, dtype=torch.float32)
    for it in range(iters):
        i = torch.randint(0, len(buf), (512,), device=a.device)
        with torch.no_grad():
            q2 = qt(S2[i]).masked_fill(~AV2[i], -1e9).max(1).values
            target = R[i] + (1 - DONE[i]) * q2
        pred = q(S[i]).gather(1, A[i][:, None])[:, 0]
        loss = torch.nn.functional.smooth_l1_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0:
            qt.load_state_dict(q.state_dict())
    return float(loss)


def train(a):
    os.makedirs(a.runs, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    probes, val = setup(a)
    q, qt = QNet().to(a.device), QNet().to(a.device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf, log, t0 = [], [], time.time()
    for ep in range(a.episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * a.episodes))
        env = Env(a, 1000 * a.seed + ep, probes, val)
        res = episode(a, env, "q", rng, eps=eps, q=q, buf=buf)
        buf = buf[-a.buffer:]
        loss = fit_q(a, q, qt, opt, buf, a.fit_iters)
        log.append({"episode": ep, "eps": eps, "td_loss": loss, **res})
        print(f"[teacher s{a.seed}] ep {ep} eps {eps:.2f} code {res['code_learnable']:.3f} val {res['val_loss']:.3f}"
              f" noise_pos {res['mean_position']['noise']:.2f} td {loss:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(q.state_dict(), os.path.join(a.runs, f"q_s{a.seed}.pt"))
    evals_ = [episode(a, Env(a, ds, probes, val), "q", rng, eps=0.0, q=q) for ds in a.eval_seeds]
    save(a, f"teacher_s{a.seed}", {"train_log": log, "eval": evals_})


def baseline(a):
    rng = random.Random(a.seed)
    probes, val = setup(a)
    evals_ = []
    for ds in a.eval_seeds:
        bandit = {"w": np.zeros(K)}
        evals_.append(episode(a, Env(a, ds, probes, val), a.policy, rng, bandit=bandit))
    save(a, f"{a.policy}_s{a.seed}", {"eval": evals_})


def save(a, name, res):
    os.makedirs(a.runs, exist_ok=True)
    out = os.path.join(a.runs, name + ".json")
    json.dump({"args": vars(a), **res}, open(out + ".tmp", "w"), default=float)
    os.replace(out + ".tmp", out)
    print("saved", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "baseline"])
    ap.add_argument("--policy", default="random", choices=["random", "greedy", "lp"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "exp04_curriculum", "data"))
    ap.add_argument("--runs4", default=os.path.join(HERE, "..", "exp04_curriculum", "runs"))
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--n_per_group", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--fit_iters", type=int, default=600)
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--lp_scale", type=float, default=0.05)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    (train if a.mode == "train" else baseline)(a)


if __name__ == "__main__":
    main()
