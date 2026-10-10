"""Pretrain the memory-layer GPT on TinyStories + old-entity bios (one shuffled epoch).

Usage: python pretrain.py --data_dir data --out runs/pretrain
"""
import argparse
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer

import evals
from model import GPT, GPTConfig, lm_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = one full epoch")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    cfg = GPTConfig()
    model = GPT(cfg)
    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    with open(os.path.join(args.data_dir, "facts.json")) as f:
        old = json.load(f)["old"]

    data = np.fromfile(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16)
    W = cfg.ctx + 1
    n_win = len(data) // W
    order = np.random.default_rng(0).permutation(n_win)
    steps = n_win // args.batch if not args.max_steps else args.max_steps
    vx, vy = evals.val_windows(os.path.join(args.data_dir, "val.bin"), 128, cfg.ctx)
    items = evals.old_fact_items(old, 100)
    probe_in = evals.Probe(tok, items, 0)
    probe_out = evals.Probe(tok, items, 7)

    mem_names = model.memory_param_names()
    decay = [p for n, p in model.named_parameters() if p.dim() == 2 and n not in mem_names]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 and n not in mem_names]
    mem = [p for n, p in model.named_parameters() if n in mem_names]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay + mem, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log = open(os.path.join(args.out, "log.jsonl"), "w")
    t0, tl = time.time(), 0.0
    for step in range(steps):
        ids = order[step * args.batch:(step + 1) * args.batch]
        w = torch.from_numpy(np.stack([data[i * W:(i + 1) * W] for i in ids]).astype(np.int64))
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = lm_loss(model, w[:, :-1], w[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tl += loss.item()
        if (step + 1) % 50 == 0:
            el = time.time() - t0
            print(f"step {step + 1}/{steps} loss {tl / 50:.3f} lr {lr_at(step):.2e}"
                  f" {(step + 1) * args.batch * cfg.ctx / el:,.0f} tok/s"
                  f" eta {el / (step + 1) * (steps - step - 1) / 60:.0f}m", flush=True)
            tl = 0.0
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            model.eval()
            model.mem.record = True
            vl = evals.lm_eval(model, vx, vy)
            used = torch.unique(model.mem.last_idx).numel() / model.mem.n_slots
            model.mem.record = False
            c_in = probe_in.run(model)[0]
            c_out = probe_out.run(model)[0]
            model.train()
            rec = {"step": step + 1, "tokens": (step + 1) * args.batch * cfg.ctx,
                   "val_loss": vl, "old_acc_in": float(c_in.mean()),
                   "old_acc_heldout": float(c_out.mean()), "slot_use_last_batch": used}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict()},
                       os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
