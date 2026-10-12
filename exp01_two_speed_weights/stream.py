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
import re
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


def loss_mask(tok, texts, names, values, y, mask, kind):
    """Which target positions a write trains on: all, all but entity-name tokens, or the answer.

    Targets y[:, j] are the text's tokens j (then the closing EOT), so token offsets index them.
    """
    if kind == "all":
        return mask
    if kind == "answer":
        vid = torch.tensor([tok.encode(" " + v).ids[0] for v in values])
        return mask & (y == vid[:, None])
    assert kind == "noname", kind
    out = mask.clone()
    for i, (text, name) in enumerate(zip(texts, names)):
        spans, s = [], text.find(name)
        while s != -1:
            spans.append((s, s + len(name)))
            s = text.find(name, s + 1)
        for j, (a, b) in enumerate(tok.encode(text).offsets):
            if any(a < e and b > s0 for s0, e in spans):
                out[i, j] = False
    return out


class Replay:
    """Random pretraining windows (for sleep replay and the IDF background)."""

    def __init__(self, path, ctx, seed, reject=None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)
        self.reject = reject  # optional predicate on a window's start index
        self.rejected = 0

    def sample(self, n):
        rows = []
        while len(rows) < n:
            for i in self.rng.integers(0, len(self.data) - self.ctx - 1, size=n - len(rows)):
                if self.reject is not None and self.reject(i):
                    self.rejected += 1
                    continue
                rows.append(self.data[i:i + self.ctx + 1].astype(np.int64))
        w = torch.from_numpy(np.stack(rows))
        return w[:, :-1], w[:, 1:]


def eval_doc_mask(train_bin, tok, names, cache):
    """Boolean mask over train.bin tokens: True inside any document (EOT-delimited bio or story)
    that mentions one of these names. Rejecting windows that overlap it also catches windows
    that start or end mid-name, which a regex on the window text misses."""
    if os.path.exists(cache):
        mask = np.load(cache)
        assert len(mask) == os.path.getsize(train_bin) // 2, f"{cache} is for another train.bin; delete it"
        return mask
    data = np.fromfile(train_bin, dtype=np.uint16)
    eot = tok.token_to_id(EOT)
    ends = np.flatnonzero(data == eot)
    starts = np.concatenate([[0], ends[:-1] + 1])
    pat = re.compile("|".join(r"\s+".join(map(re.escape, n.split())) for n in names))
    mask = np.zeros(len(data), dtype=bool)
    for i in range(0, len(starts), 20000):
        docs = [data[a:b].tolist() for a, b in zip(starts[i:i + 20000], ends[i:i + 20000])]
        for (a, b), text in zip(zip(starts[i:i + 20000], ends[i:i + 20000]), tok.decode_batch(docs)):
            if pat.search(text):
                mask[a:b + 1] = True
    np.save(cache, mask)
    return mask


def touches_mask(mask, ctx):
    """Predicate on a window start index: does [i, i + ctx] touch the mask?"""
    # int32 halves the RAM of each parallel run on a 300M-token corpus; the counts are identical
    csum = np.zeros(len(mask) + 1, dtype=np.int32 if len(mask) < 2**31 else np.int64)
    np.cumsum(mask, dtype=csum.dtype, out=csum[1:])
    return lambda i: csum[i + ctx + 1] - csum[i] > 0


@torch.no_grad()
def protected_counts(model, tok, pool, cache, tids=range(8)):
    """Slot reads by the frozen pretrained model on cloze prompts of non-eval old entities
    (the given templates, every real position). Used to down-weight shared slots."""
    if os.path.exists(cache):
        return torch.load(cache)
    P = torch.zeros(model.mem.n_slots)
    items = [(e["name"], r, e["attrs"][r]) for e in pool for r in facts.REL_NAMES]
    model.mem.record = True
    for t in tids:
        p = evals.Probe(tok, items, t)
        for i in range(0, len(p.x), 512):
            x, lens = p.x[i:i + 512], p.lens[i:i + 512]
            model(x[:, :int(lens.max())])
            valid = (torch.arange(int(lens.max()))[None, :] < lens[:, None]).to(model.mem.last_idx.device)
            P += torch.bincount(model.mem.last_idx[valid].reshape(-1), minlength=model.mem.n_slots).cpu()
    model.mem.record = False
    torch.save(P, cache)
    return P


def fisher_diag(model, params, replay, n_windows):
    """Empirical-free diagonal Fisher (labels sampled from the model), normalised to mean 1."""
    F = [torch.zeros_like(p) for p in params]
    saved = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    for _ in range(n_windows):
        x, _ = replay.sample(1)
        logits = model(x)
        yhat = torch.distributions.Categorical(logits=logits.detach()).sample()
        torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yhat.view(-1)).backward()
        for f, p in zip(F, params):
            f += p.grad.detach() ** 2 / n_windows
            p.grad = None
    for p in model.parameters():
        p.requires_grad_(saved[id(p)])
    mean = torch.cat([f.flatten() for f in F]).mean()
    return [f / mean for f in F]


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
        df[torch.unique(model.mem.last_idx).cpu()] += 1
    model.mem.record = False
    torch.save(df, cache)
    return df


class SparseWriter:
    """Arm C: SGD on the values of the top-t TF-IDF slots of each fact batch."""

    def __init__(self, model, lr, t, idf, protect=None):
        self.model, self.lr, self.t = model, lr, t
        dev = model.mem.values.device
        # Soft protection divides by (1 + gamma * P); P = 0 reproduces Exp 01 exactly.
        self.idf = (idf if protect is None else idf / (1 + protect)).to(dev)
        for p in model.parameters():
            p.requires_grad_(False)
        model.mem.values.requires_grad_(True)
        self.written = torch.zeros(model.mem.n_slots, dtype=torch.bool, device=dev)  # since last reset
        self.ever = torch.zeros(model.mem.n_slots, dtype=torch.bool, device=dev)

    def state(self):
        return {"written": self.written, "ever": self.ever}

    def load_state(self, st):
        self.written.copy_(st["written"])
        self.ever.copy_(st["ever"])

    def select(self, x, mask):
        mem = self.model.mem
        mem.record = True
        with torch.no_grad():
            self.model(x)
        mem.record = False
        counts = torch.bincount(mem.last_idx[mask.to(mem.last_idx.device)].reshape(-1),
                                minlength=mem.n_slots).float()
        score = counts / counts.sum() * self.idf
        return score.topk(self.t).indices

    def write(self, x, y, mask, steps, done=None, lmask=None):
        sel = self.select(x, mask)
        overlap = float(self.ever[sel].float().mean())
        v = self.model.mem.values
        for n in range(1, steps + 1):
            lm_loss(self.model, x, y, mask if lmask is None else lmask).backward()
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

    def state(self):
        return self.opt.state_dict()

    def load_state(self, st):
        self.opt.load_state_dict(st)

    def write(self, x, y, mask, steps, done=None, lmask=None):
        for n in range(1, steps + 1):
            lm_loss(self.model, x, y, mask if lmask is None else lmask).backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if done is not None and done():
                break
        return {"steps": n}


def sleep(model, args, cues, n_recent, stream_probe, writer, pretrained_values, replay, rng,
          old_probe=None, ewc=None, rrng=None):
    """Self-quiz on every cue, reset written slots (sparse arm), distil into dense weights."""
    model.eval()
    _, _, pred = stream_probe.run(model, idx=list(range(len(cues))))
    labels = [facts.RELATIONS[r][int(p)] for (_, r), p in zip(cues, pred)]
    diag = {}
    if old_probe is not None:
        diag["old_before"] = float(old_probe.run(model)[0].mean())
    if isinstance(writer, SparseWriter):
        w = writer.written
        with torch.no_grad():
            model.mem.values[w] = pretrained_values[w]
        writer.written.zero_()
    if old_probe is not None:
        diag["old_after_reset"] = float(old_probe.run(model)[0].mean())
    sents = []  # (text, name, value)
    for i, ((name, rel), val) in enumerate(zip(cues, labels)):
        k = args.sleep_p_recent if i >= len(cues) - n_recent else args.sleep_p_old
        tids = rng.sample(facts.STREAM_TEMPLATES, k) if k <= len(facts.STREAM_TEMPLATES) \
            else [rng.choice(facts.STREAM_TEMPLATES) for _ in range(k)]
        sents += [(facts.render(rel, t, name, val), name, val) for t in tids]
    rng.shuffle(sents)
    dense = model.dense_parameters()
    if args.sleep_freeze != "none":  # keep embeddings (tied with the output head) fixed
        frozen = {id(model.tok.weight), id(model.pos.weight)}
        if args.sleep_freeze == "lower":
            for b in model.blocks[:model.cfg.mem_layer]:
                frozen |= {id(p) for p in b.parameters()}
        dense = [p for p in dense if id(p) not in frozen]
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
            batch = sents[i:i + args.sleep_batch]
            # Rehearsal: old entities the model already knows, never the probed (eval) ones.
            # Own RNG, so the fact sentences match a sleep without rehearsal draw for draw.
            for name, rel, val in (rrng.sample(args.rehearse_pool, args.sleep_rehearse)
                                   if args.sleep_rehearse else []):
                t = rrng.choice(args.rehearse_tids)
                batch.append((facts.render(rel, t, name, val), name, val))
            texts, names, vals = zip(*batch)
            x, y, m = encode(args.tok, list(texts))
            m = loss_mask(args.tok, texts, names, vals, y, m, args.sleep_fact_loss)
            rx, ry = replay.sample(args.sleep_replay)
            loss = lm_loss(model, x, y, m) + lm_loss(model, rx, ry)
            if ewc is not None:  # 0.5 * lambda * sum F (theta - theta_pretrained)^2
                F, theta0, ids = ewc
                loss = loss + 0.5 * args.sleep_ewc * sum(
                    (F[k] * (p - theta0[k]) ** 2).sum() for k, p in enumerate(ids) if p.requires_grad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
    for p in model.parameters():
        p.requires_grad_(saved[id(p)])
    if old_probe is not None:
        model.eval()
        diag["old_after_sleep"] = float(old_probe.run(model)[0].mean())
    return {"self_label_acc": float(np.mean([labels[i] == args.stream[i]["value"]
                                             for i in range(len(cues))])),
            "sleep_steps": steps, "sleep_sents": len(sents), **diag}


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
    ap.add_argument("--write_loss", choices=["all", "noname", "answer"], default="all",
                    help="target tokens a wake write trains on")
    ap.add_argument("--sleep_fact_loss", choices=["all", "noname", "answer"], default="all")
    ap.add_argument("--sleep_rehearse", type=int, default=0,
                    help="old-entity statements (entities >= --n_eval_old) added per sleep step")
    ap.add_argument("--sleep_freeze", choices=["none", "embed", "lower"], default="none",
                    help="embed: freeze token/position embeddings; lower: also blocks below memory")
    ap.add_argument("--n_eval_old", type=int, default=100, help="old entities reserved for eval")
    ap.add_argument("--protect_gamma", type=float, default=0.0,
                    help="down-weight slot scores by 1 + gamma * reads by non-eval old entities")
    ap.add_argument("--protect_templates", choices=["all", "stream", "nonprobe"], default="all",
                    help="templates whose cloze reads define the protected counts (stream = 0-6)")
    ap.add_argument("--rehearse_labels", choices=["data", "teacher"], default="teacher",
                    help="rehearsal answers: bio facts, or the frozen pretrained model's argmax")
    ap.add_argument("--rehearse_templates", choices=["all", "nonprobe"], default="all",
                    help="nonprobe: rehearse old entities only in templates 1-6, never the probes")
    ap.add_argument("--clean_replay", action="store_true",
                    help="reject replay/Fisher windows that mention an eval old entity")
    ap.add_argument("--sleep_ewc", type=float, default=0.0,
                    help="EWC strength, Fisher from clean pretraining windows, anchor = pretrained")
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--pretrained", default="runs/pretrain/model.pt")
    ap.add_argument("--out", default="runs/stream")
    ap.add_argument("--name", default="")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--state", default="", help="resume file: saved every --save_every facts, picked up on restart")
    ap.add_argument("--save_every", type=int, default=100)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)  # paraphrase batches: identical across arms
    sleep_rng = random.Random(args.seed + 1000)
    rehearse_rng = random.Random(args.seed + 2000)
    name = args.name or f"{args.arm}{'+sleep' if args.sleep else ''}_lr{args.lr:g}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, name + ".json")

    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    with open(os.path.join(args.data_dir, "facts.json")) as f:
        fj = json.load(f)
    stream = fj["streams"][args.seed][:args.n_facts]
    args.tok, args.stream = tok, stream
    model = load_model(args.pretrained).to(args.device)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # only sleep's slot reset reads it (0.5 GB at the small scale)
    pretrained_values = model.mem.values.detach().clone() if args.sleep and args.arm == "C" else None
    assert not (args.sleep_ewc > 0 and not args.clean_replay), "EWC Fisher needs --clean_replay"
    args.rehearse_tids = list(range(8)) if args.rehearse_templates == "all" else \
        [t for t in range(8) if t not in (facts.PROBE_IN, facts.PROBE_HELDOUT)]
    eval_names = [e["name"] for e in fj["old"][:args.n_eval_old]]
    reject = None
    if args.clean_replay:
        dm = eval_doc_mask(os.path.join(args.data_dir, "train.bin"), tok, eval_names,
                           os.path.join(args.data_dir, f"eval_doc_mask{args.n_eval_old}.npy"))
        reject = touches_mask(dm, model.cfg.ctx)
    replay = Replay(os.path.join(args.data_dir, "train.bin"), model.cfg.ctx, seed=args.seed,
                    reject=reject)
    assert args.n_eval_old == 100, "eval probes and leakage splits assume 100 eval entities"
    pool = fj["old"][args.n_eval_old:]  # never the probed old entities
    args.rehearse_pool = [(e["name"], r, e["attrs"][r]) for e in pool for r in facts.REL_NAMES]
    if args.sleep and args.sleep_rehearse and args.rehearse_labels == "teacher":
        tp = evals.Probe(tok, args.rehearse_pool, facts.PROBE_IN)
        model.eval()
        pred = tp.run(model)[2]  # the model is still the frozen pretrained one here
        args.rehearse_pool = [(n, r, facts.RELATIONS[r][int(p)])
                              for (n, r, _), p in zip(args.rehearse_pool, pred)]
    ewc = None
    if args.sleep and args.sleep_ewc > 0:
        ids = model.dense_parameters()
        fr = Replay(os.path.join(args.data_dir, "train.bin"), model.cfg.ctx, seed=99, reject=reject)
        ewc = (fisher_diag(model, ids, fr, 64), [p.detach().clone() for p in ids], ids)
    vx, vy = evals.val_windows(os.path.join(args.data_dir, "val.bin"), 128, model.cfg.ctx)
    items = [(f["name"], f["rel"], f["value"]) for f in stream]
    old_items = evals.old_fact_items(fj["old"], args.n_eval_old)
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
        protect = None
        if args.protect_gamma > 0:
            tids = {"all": range(8), "stream": facts.STREAM_TEMPLATES,
                    "nonprobe": [t for t in range(8) if t not in (facts.PROBE_IN, facts.PROBE_HELDOUT)]}
            suffix = "" if args.protect_templates == "all" else "_" + args.protect_templates
            protect = args.protect_gamma * protected_counts(
                model, tok, pool, os.path.join(os.path.dirname(args.pretrained),
                                               f"protect_noneval{args.n_eval_old}{suffix}.pt"),
                tids[args.protect_templates])
        writer = SparseWriter(model, args.lr, args.t, idf, protect)
    elif args.arm == "B":
        for p in model.parameters():
            p.requires_grad_(False)
        writer = AdamWriter(model, add_lora(model, args.lora_r, 2 * args.lora_r), args.lr,
                            args.beta1)
    else:
        writer = AdamWriter(model, list(model.parameters()), args.lr, args.beta1)

    t0, spent, start = time.time(), 0.0, 0
    if args.state and os.path.exists(args.state):  # resume an interrupted run where it stopped
        st = torch.load(args.state, map_location=args.device, weights_only=False)
        model.load_state_dict(st["model"])
        writer.load_state(st["writer"])
        rng.setstate(st["rng"]); sleep_rng.setstate(st["sleep_rng"]); rehearse_rng.setstate(st["rehearse_rng"])
        torch.set_rng_state(st["torch_rng"])
        replay.rng.bit_generator.state, replay.rejected = st["replay_rng"], st["replay_rejected"]
        ckpts, acq_in, acq_out, acq_logp, overlaps, sleeps, write_steps = st["lists"]
        start, spent = st["n"], st["seconds"]
        print(f"[{name}] resumed at {start} facts", flush=True)
    else:
        ckpts = [checkpoint(model, 0, probes, vx, vy, n_probe=len(stream))]  # baseline
        acq_in, acq_out, acq_logp, overlaps, sleeps, write_steps = [], [], [], [], [], []
    model.train()
    for i in range(start, len(stream)):
        fact = stream[i]
        texts = facts.paraphrase_batch(fact, args.batch, rng)
        x, y, m = encode(tok, texts)
        lm = loss_mask(tok, texts, [fact["name"]] * len(texts), [fact["value"]] * len(texts),
                       y, m, args.write_loss)
        done = None
        if not args.fixed_steps:
            def done(i=i):
                with torch.no_grad():
                    return bool(probes["stream_in"].run(model, [i])[0][0])
        info = writer.write(x, y, m, args.steps, done, lm)
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
                        pretrained_values, replay, sleep_rng, probes["old_in"], ewc, rehearse_rng)
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
        if args.state and n % args.save_every == 0 and n < len(stream):
            torch.save({"n": n, "seconds": spent + time.time() - t0, "model": model.state_dict(),
                        "writer": writer.state(), "rng": rng.getstate(), "sleep_rng": sleep_rng.getstate(),
                        "rehearse_rng": rehearse_rng.getstate(), "torch_rng": torch.get_rng_state(),
                        "replay_rng": replay.rng.bit_generator.state, "replay_rejected": replay.rejected,
                        "lists": (ckpts, acq_in, acq_out, acq_logp, overlaps, sleeps, write_steps)},
                       args.state + ".tmp")
            os.replace(args.state + ".tmp", args.state)

    res = {"name": name, "args": {k: v for k, v in vars(args).items()
                                  if k not in ("tok", "stream", "rehearse_pool")},
           "acq_in": acq_in, "acq_out": acq_out, "acq_logp_in": acq_logp,
           "slot_overlap": overlaps, "write_steps": write_steps, "checkpoints": ckpts,
           "sleeps": sleeps,
           "replay_rejected": replay.rejected, "seconds": spent + time.time() - t0}
    with open(out_path + ".tmp", "w") as f:  # atomic: a copy taken mid-write never looks finished
        json.dump(res, f)
    os.replace(out_path + ".tmp", out_path)
    if args.state and os.path.exists(args.state):
        os.remove(args.state)


if __name__ == "__main__":
    main()
