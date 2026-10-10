"""Where do TF-IDF-selected slots come from, and do they collide with old facts' slots?

For each of the first N stream facts: which token roles (entity name, template words,
the position that predicts the value) access the top-t slots, and what fraction of
those slots are also read when the pretrained model recalls old facts.

Usage: python slot_probe.py --bg_windows 3 --t 128 --n 50
"""
import argparse
import json
import os
import random

import torch
from tokenizers import Tokenizer

import evals
import facts
from stream import encode, load_model


def roles(tok, text, name, value):
    """Per-token role for EOT + text: 'name', 'value_pred' (position predicting the value),
    or 'template'."""
    enc = tok.encode(text)
    name_spans = []
    start = text.find(name)
    while start != -1:
        name_spans.append((start, start + len(name)))
        start = text.find(name, start + 1)
    v0 = text.rfind(" " + value)
    out = ["template"]  # the EOT position
    for k, (a, b) in enumerate(enc.offsets):
        if any(a < e and b > s for s, e in name_spans):
            out.append("name")
        else:
            out.append("template")
    # input position whose next token is the value token predicts the value
    for k, (a, b) in enumerate(enc.offsets):
        if a <= v0 < b or (a == v0 + 1 and enc.tokens[k].startswith("Ġ")) or a == v0:
            out[k] = "value_pred"  # position k (EOT shifts inputs by one)
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--pretrained", default="runs/pretrain/model.pt")
    ap.add_argument("--bg_windows", type=int, default=3)
    ap.add_argument("--t", type=int, default=128)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    fj = json.load(open(os.path.join(args.data_dir, "facts.json")))
    model = load_model(args.pretrained)
    model.eval()
    mem = model.mem
    df = torch.load(os.path.join(os.path.dirname(args.pretrained), f"bg_df_w{args.bg_windows}.pt"))
    idf = torch.log((1000 + 1) / (df + 1))

    # Slots read at the value-predicting position when recalling old facts.
    old_items = evals.old_fact_items(fj["old"], 100)
    p = evals.Probe(tok, old_items, facts.PROBE_IN)
    mem.record = True
    with torch.no_grad():
        model(p.x)
    last = mem.last_idx[torch.arange(len(p.x)), p.lens - 1]
    old_slots = torch.zeros(mem.n_slots, dtype=torch.bool)
    old_slots[last.reshape(-1)] = True
    # ...and anywhere in the old-fact prompts (mostly the entity-name tokens).
    valid = torch.arange(p.x.shape[1])[None, :] < p.lens[:, None]
    old_any = torch.zeros(mem.n_slots, dtype=torch.bool)
    old_any[mem.last_idx[valid].reshape(-1)] = True

    rng = random.Random(0)
    agg = {"name": 0.0, "template": 0.0, "value_pred": 0.0}
    hit_old, hit_any = [], []
    for f in fj["streams"][0][:args.n]:
        texts = facts.paraphrase_batch(f, 16, rng)
        x, y, m = encode(tok, texts)
        with torch.no_grad():
            model(x)
        idx = mem.last_idx
        counts = torch.bincount(idx[m].reshape(-1), minlength=mem.n_slots).float()
        sel = (counts / counts.sum() * idf).topk(args.t).indices
        in_sel = torch.zeros(mem.n_slots, dtype=torch.bool)
        in_sel[sel] = True
        for b, text in enumerate(texts):
            r = roles(tok, text, f["name"], f["value"])
            for pos in range(int(m[b].sum())):
                agg[r[pos]] += float(in_sel[idx[b, pos]].sum())
        hit_old.append(float(old_slots[sel].float().mean()))
        hit_any.append(float(old_any[sel].float().mean()))
    tot = sum(agg.values())
    print(json.dumps({"bg_windows": args.bg_windows, "t": args.t,
                      "selected_slot_accesses_by_role": {k: v / tot for k, v in agg.items()},
                      "frac_selected_slots_also_read_by_old_fact_recall": sum(hit_old) / len(hit_old),
                      "old_recall_slot_fraction_of_memory": float(old_slots.float().mean()),
                      "frac_selected_slots_read_anywhere_in_old_prompts": sum(hit_any) / len(hit_any),
                      "old_prompt_slot_fraction_of_memory": float(old_any.float().mean())},
                     indent=2))


if __name__ == "__main__":
    main()
