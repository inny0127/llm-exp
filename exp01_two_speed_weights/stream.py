"""Continual-learning stream: learn new facts one at a time, measure learning vs forgetting.

Arms (wake-phase write): A = full FT, B = LoRA, C = TF-IDF sparse memory FT.
--sleep adds periodic consolidation: self-quiz on every cued fact, reset written slots
(sparse arm), then distil into the dense weights with pretraining replay.

Usage: python stream.py --arm C --lr 4 --seed 0 [--sleep] --out runs/stream
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from tokenizers import Tokenizer

import evals
import facts
from model import GPT, GPTConfig, add_lora, lm_loss

EOT = evals.EOT


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m


def encode(tok, texts):
    """Right-padded (x, y, mask); each text is wrapped as EOT text EOT like a pretraining doc."""
    eot = tok.token_to_id(EOT)
    seqs = [[eot] + e.ids + [eot] for e in tok.encode_batch(texts)]
    L = max(map(len, seqs)) - 1
    x = torch.full((len(seqs), L), eot, dtype=torch.long)
    y = torch.full((len(seqs), L), eot, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = len(s) - 1
        x[i, :n], y[i, :n], mask[i, :n] = torch.tensor(s[:-1]), torch.tensor(s[1:]), True
    return x, y, mask


class Replay:
    """Random pretraining windows (for sleep replay and the IDF background)."""

    def __init__(self, path, ctx, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        s = self.rng.integers(0, len(self.data) - self.ctx - 1, size=n)
        w = torch.from_numpy(np.stack([self.data[i:i + self.ctx + 1].astype(np.int64) for i in s]))
        return w[:, :-1], w[:, 1:]


@torch.no_grad()
def background_df(model, train_bin, n_batches, windows_per_batch, cache):
    """Per-slot count of background batches that access the slot (the 'document frequency')."""
    if os.path.exists(cache):
        return torch.load(cache)
    rp = Replay(train_bin, model.cfg.ctx, seed=12345)
    df = torch.zeros(model.mem.n_slots)
    model.mem.record = True
    for _ in range(n_batches):
        x, _ = rp.sample(windows_per_batch)
        model(x)
        df[torch.unique(model.mem.last_idx)] += 1
    model.mem.record = False
    torch.save(df, cache)
    return df


class SparseWriter:
    """Arm C: SGD on the values of the top-t TF-IDF slots of each fact batch."""

    def __init__(self, model, lr, t, idf):
        self.model, self.lr, self.t, self.idf = model, lr, t, idf
        for p in model.parameters():
            p.requires_grad_(False)
        model.mem.values.requires_grad_(True)
        self.written = torch.zeros(model.mem.n_slots, dtype=torch.bool)  # since last reset
        self.ever = torch.zeros(model.mem.n_slots, dtype=torch.bool)

    def select(self, x, mask):
        mem = self.model.mem
        mem.record = True
        with torch.no_grad():
            self.model(x)
        mem.record = False
        counts = torch.bincount(mem.last_idx[mask].reshape(-1), minlength=mem.n_slots).float()
        score = counts / counts.sum() * self.idf
        return score.topk(self.t).indices

    def write(self, x, y, mask, steps, done=None):
        sel = self.select(x, mask)
        overlap = float(self.ever[sel].float().mean())
        v = self.model.mem.values
        for n in range(1, steps + 1):
            lm_loss(self.model, x, y, mask).backward()
            with torch.no_grad():
                v[sel] -= self.lr * v.grad[sel]
            v.grad = None
            if done is not None and done():
                break
        self.written[sel] = True
        self.ever[sel] = True
        return {"overlap": overlap, "steps": n}


class AdamWriter:
    """Arms A (all parameters) and B (LoRA parameters only)."""

    def __init__(self, model, params, lr, beta1=0.9):
        self.model, self.params = model, params
        self.opt = torch.optim.AdamW(params, lr=lr, betas=(beta1, 0.999), weight_decay=0.0,
                                     fused=True)

    def write(self, x, y, mask, steps, done=None):
        for n in range(1, steps + 1):
            lm_loss(self.model, x, y, mask).backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if done is not None and done():
                break
        return {"steps": n}


def sleep(model, args, cues, n_recent, stream_probe, writer, pretrained_values, replay, rng):
    """Self-quiz on every cue, reset written slots (sparse arm), distil into dense weights."""
    model.eval()
    _, _, pred = stream_probe.run(model, idx=list(range(len(cues))))
    labels = [facts.RELATIONS[r][int(p)] for (_, r), p in zip(cues, pred)]
    if isinstance(writer, SparseWriter):
        w = writer.written
        with torch.no_grad():
            model.mem.values[w] = pretrained_values[w]
        writer.written.zero_()
    sents = []
    for i, ((name, rel), val) in enumerate(zip(cues, labels)):
        k = args.sleep_p_recent if i >= len(cues) - n_recent else args.sleep_p_old
        tids = rng.sample(facts.STREAM_TEMPLATES, k) if k <= len(facts.STREAM_TEMPLATES) \
            else [rng.choice(facts.STREAM_TEMPLATES) for _ in range(k)]
        sents += [facts.render(rel, t, name, val) for t in tids]
    rng.shuffle(sents)
    dense = model.dense_parameters()
    saved = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in dense:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(dense, lr=args.sleep_lr, betas=(0.9, 0.999), weight_decay=0.0,
                            fused=True)
    model.train()
    steps = 0
    for _ in range(args.sleep_epochs):
        for i in range(0, len(sents), args.sleep_batch):
            x, y, m = encode(args.tok, sents[i:i + args.sleep_batch])
            rx, ry = replay.sample(args.sleep_replay)
            loss = lm_loss(model, x, y, m) + lm_loss(model, rx, ry)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
    for p in model.parameters():
        p.requires_grad_(saved[id(p)])
    return {"self_label_acc": float(np.mean([labels[i] == args.stream[i]["value"]
                                             for i in range(len(cues))])),
            "sleep_steps": steps, "sleep_sents": len(sents)}


@torch.no_grad()
def checkpoint(model, n_seen, probes, vx, vy, n_probe=None):
    """Forgetting metrics plus recall of the first n_probe stream facts (default: those seen)."""
    model.eval()
    rec = {"n_seen": n_seen, "val_loss": evals.lm_eval(model, vx, vy)}
    for k in ("old_in", "old_out"):
        rec[k] = float(probes[k].run(model)[0].mean())
    idx = list(range(n_seen if n_probe is None else n_probe))
    rec["ret_in"] = probes["stream_in"].run(model, idx)[0].tolist()
    rec["ret_out"] = probes["stream_out"].run(model, idx)[0].tolist()
    model.train()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", action="store_true")
    ap.add_argument("--n_facts", type=int, default=500)
    ap.add_argument("--steps", type=int, default=20,
                    help="max gradient steps per fact; each fact trains until its probe is right")
    ap.add_argument("--fixed_steps", action="store_true",
                    help="always take --steps steps (the pre-registered protocol)")
    ap.add_argument("--beta1", type=float, default=0.0,
                    help="Adam beta1 for arms A and B; momentum across unrelated facts hurts")
    ap.add_argument("--bg_windows", type=int, default=32,
                    help="128-token windows per IDF background batch (32 = Meta's 64x64 shape)")
    ap.add_argument("--batch", type=int, default=16, help="paraphrases per fact")
    ap.add_argument("--t", type=int, default=128, help="slots written per fact (arm C)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--sleep_every", type=int, default=100)
    ap.add_argument("--sleep_lr", type=float, default=1e-4)
    ap.add_argument("--sleep_epochs", type=int, default=3)
    ap.add_argument("--sleep_p_recent", type=int, default=7)
    ap.add_argument("--sleep_p_old", type=int, default=2)
    ap.add_argument("--sleep_batch", type=int, default=32)
    ap.add_argument("--sleep_replay", type=int, default=16, help="pretraining windows per step")
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--pretrained", default="runs/pretrain/model.pt")
    ap.add_argument("--out", default="runs/stream")
    ap.add_argument("--name", default="")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)  # paraphrase batches: identical across arms
    sleep_rng = random.Random(args.seed + 1000)
    name = args.name or f"{args.arm}{'+sleep' if args.sleep else ''}_lr{args.lr:g}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, name + ".json")

    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    with open(os.path.join(args.data_dir, "facts.json")) as f:
        fj = json.load(f)
    stream = fj["streams"][args.seed][:args.n_facts]
    args.tok, args.stream = tok, stream
    model = load_model(args.pretrained)
    pretrained_values = model.mem.values.detach().clone()
    replay = Replay(os.path.join(args.data_dir, "train.bin"), model.cfg.ctx, seed=args.seed)
    vx, vy = evals.val_windows(os.path.join(args.data_dir, "val.bin"), 128, model.cfg.ctx)
    items = [(f["name"], f["rel"], f["value"]) for f in stream]
    old_items = evals.old_fact_items(fj["old"], 100)
    probes = {"stream_in": evals.Probe(tok, items, facts.PROBE_IN),
              "stream_out": evals.Probe(tok, items, facts.PROBE_HELDOUT),
              "old_in": evals.Probe(tok, old_items, facts.PROBE_IN),
              "old_out": evals.Probe(tok, old_items, facts.PROBE_HELDOUT)}

    if args.arm == "C":
        model.eval()  # no dropout anywhere; eval() only matters for record hooks
        df = background_df(model, os.path.join(args.data_dir, "train.bin"), 1000, args.bg_windows,
                           os.path.join(os.path.dirname(args.pretrained),
                                        f"bg_df_w{args.bg_windows}.pt"))
        idf = torch.log((1000 + 1) / (df + 1))
        writer = SparseWriter(model, args.lr, args.t, idf)
    elif args.arm == "B":
        for p in model.parameters():
            p.requires_grad_(False)
        writer = AdamWriter(model, add_lora(model, args.lora_r, 2 * args.lora_r), args.lr,
                            args.beta1)
    else:
        writer = AdamWriter(model, list(model.parameters()), args.lr, args.beta1)

    t0 = time.time()
    ckpts = [checkpoint(model, 0, probes, vx, vy, n_probe=len(stream))]  # baseline
    acq_in, acq_out, acq_logp, overlaps, sleeps, write_steps = [], [], [], [], [], []
    model.train()
    for i, fact in enumerate(stream):
        x, y, m = encode(tok, facts.paraphrase_batch(fact, args.batch, rng))
        done = None
        if not args.fixed_steps:
            def done(i=i):
                with torch.no_grad():
                    return bool(probes["stream_in"].run(model, [i])[0][0])
        info = writer.write(x, y, m, args.steps, done)
        write_steps.append(info["steps"])
        if "overlap" in info:
            overlaps.append(info["overlap"])
        with torch.no_grad():
            c_in, lp_in, _ = probes["stream_in"].run(model, [i])
            c_out, _, _ = probes["stream_out"].run(model, [i])
        acq_in.append(bool(c_in[0]))
        acq_out.append(bool(c_out[0]))
        acq_logp.append(float(lp_in[0]))
        n = i + 1
        if args.sleep and n % args.sleep_every == 0 and n < len(stream):
            idx = list(range(n))
            pre_in = probes["stream_in"].run(model, idx)[0]
            cues = [(f["name"], f["rel"]) for f in stream[:n]]
            rec = sleep(model, args, cues, args.sleep_every, probes["stream_in"], writer,
                        pretrained_values, replay, sleep_rng)
            post_in = probes["stream_in"].run(model, idx)[0]
            r = slice(n - args.sleep_every, n)
            rec.update({"n_seen": n, "pre_recent": float(pre_in[r].mean()),
                        "post_recent": float(post_in[r].mean()),
                        "pre_all": float(pre_in.mean()), "post_all": float(post_in.mean())})
            sleeps.append(rec)
            print(json.dumps(rec), flush=True)
        if n % args.ckpt_every == 0 or n == len(stream):
            c = checkpoint(model, n, probes, vx, vy)
            ckpts.append(c)
            print(f"[{name}] {n} facts: acq_in {np.mean(acq_in[-args.ckpt_every:]):.2f}"
                  f" ret_in {np.mean(c['ret_in']):.2f} ret_out {np.mean(c['ret_out']):.2f}"
                  f" val {c['val_loss']:.3f} old_in {c['old_in']:.3f}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    res = {"name": name, "args": {k: v for k, v in vars(args).items() if k not in ("tok", "stream")},
           "acq_in": acq_in, "acq_out": acq_out, "acq_logp_in": acq_logp,
           "slot_overlap": overlaps, "write_steps": write_steps, "checkpoints": ckpts,
           "sleeps": sleeps,
           "seconds": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
