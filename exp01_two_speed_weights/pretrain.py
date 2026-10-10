"""Pretrain the memory-layer GPT on TinyStories + old-entity bios (one shuffled epoch).

Resumable: a checkpoint with the optimizer state is written every --eval_every steps and
picked up again on restart (Colab sessions disconnect).

Usage: python pretrain.py --data_dir data --out runs/pretrain [--size small --device cuda --amp]
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
from model import GPT, GPTConfig, PRESETS, lm_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--size", choices=list(PRESETS), default="tiny")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    ap.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto",
                    help="auto: bf16 where the GPU has it (Ampere+), else fp16 with loss scaling (T4)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = one full epoch")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = GPTConfig(**PRESETS[args.size])
    model = GPT(cfg).to(args.device)
    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    with open(os.path.join(args.data_dir, "facts.json")) as f:
        old = json.load(f)["old"]

    data = np.fromfile(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16)
    W = cfg.ctx + 1
    n_win = len(data) // W
    order = np.random.default_rng(0).permutation(n_win)
    steps = n_win // args.batch if not args.max_steps else min(args.max_steps, n_win // args.batch)
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
                            lr=args.lr, betas=(0.9, 0.95),
                            fused=args.device.startswith("cuda"))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    start = 0
    resume = os.path.join(args.out, "resume.pt")
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=args.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed at step {start}/{steps}", flush=True)
    if start >= steps:
        print("already done", flush=True)
        return

    def save(step):
        torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict()},
                   os.path.join(args.out, "model.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "scaler": scaler.state_dict()}, resume + ".tmp")
        os.replace(resume + ".tmp", resume)

    dt = args.amp_dtype
    if args.amp and dt == "auto":
        dt = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    amp = torch.autocast("cuda", dtype=torch.bfloat16 if dt == "bf16" else torch.float16) if args.amp \
        else torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dt == "fp16")
    if args.amp:
        print("mixed precision:", dt, flush=True)
    if start and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    t0, tl = time.time(), 0.0
    for step in range(start, steps):
        ids = order[step * args.batch:(step + 1) * args.batch]
        w = torch.from_numpy(np.stack([data[i * W:(i + 1) * W] for i in ids]).astype(np.int64))
        w = w.to(args.device, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with amp:
            loss = lm_loss(model, w[:, :-1], w[:, 1:])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        tl += loss.item()
        if (step + 1) % 50 == 0:
            el = time.time() - t0
            done = step + 1 - start
            print(f"step {step + 1}/{steps} loss {tl / 50:.3f} lr {lr_at(step):.2e}"
                  f" {done * args.batch * cfg.ctx / el:,.0f} tok/s"
                  f" eta {el / done * (steps - step - 1) / 60:.0f}m", flush=True)
            tl = 0.0
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            model.eval()
            model.mem.record = True
            with amp:
                vl = evals.lm_eval(model, vx, vy)
            used = torch.unique(model.mem.last_idx).numel() / model.mem.n_slots
            model.mem.record = False
            c_in = probe_in.run(model)[0]
            c_out = probe_out.run(model)[0]
            model.train()
            rec = {"step": step + 1, "tokens": (step + 1) * args.batch * cfg.ctx,
                   "val_loss": vl, "old_acc_in": float(c_in.mean()),
                   "old_acc_heldout": float(c_out.mean()), "slot_use_last_batch": used,
                   "elapsed_s": time.time() - t0}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            save(step + 1)


if __name__ == "__main__":
    main()
