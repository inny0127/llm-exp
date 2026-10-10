"""Build tokenizer, pretraining token stream (TinyStories + old-entity bios) and fact files.

Usage: python data.py --data_dir data --pretrain_tokens 30_000_000
"""
import argparse
import json
import os
import random
import re

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import facts

EOT = "<|endoftext|>"


def read_stories(path, max_bytes=None):
    """Stories (EOT-separated) from the first max_bytes characters, streamed line by line: one
    non-ASCII character would make a single 1.7 GB string take 2-4 bytes per character."""
    stories, buf, pos = [], [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            cut = bool(max_bytes) and pos + len(line) >= max_bytes
            if cut:
                line = line[:max_bytes - pos]
            pos += len(line)
            parts = line.split(EOT)
            buf.append(parts[0])
            for p in parts[1:]:
                stories.append("".join(buf).strip())
                buf = [p]
            if cut:
                break
    if not max_bytes:
        stories.append("".join(buf).strip())  # the text after the last EOT
    return [s for s in stories if s]  # with max_bytes the trailing piece may be cut: dropped


def train_tokenizer(texts, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_docs(tok, docs, eot_id):
    out = []
    for i in range(0, len(docs), 4096):
        for enc in tok.encode_batch(docs[i:i + 4096]):
            out.append(np.array(enc.ids + [eot_id], dtype=np.uint16))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--pretrain_tokens", type=int, default=24_000_000)
    ap.add_argument("--n_old", type=int, default=500)
    ap.add_argument("--bios_per_old", type=int, default=100)
    ap.add_argument("--n_streams", type=int, default=5)
    ap.add_argument("--stream_len", type=int, default=1000)
    ap.add_argument("--train_file", default="TinyStoriesV2-GPT4-train-head200MB.txt")
    ap.add_argument("--max_mb", type=int, default=200, help="read at most this much (0 = all)")
    args = ap.parse_args()
    raw = os.path.join(args.data_dir, "raw")

    train_stories = read_stories(os.path.join(raw, args.train_file),
                                 max_bytes=args.max_mb * 2**20 or None)
    val_stories = read_stories(os.path.join(raw, "TinyStoriesV2-GPT4-valid.txt"))
    print(f"stories: train {len(train_stories)}, val {len(val_stories)}")

    # Tokenizer sees TinyStories only, so old and new invented names split the same way.
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.from_file(tok_path)
    else:
        tok = train_tokenizer(train_stories[:60000], args.vocab_size)
        tok.save(tok_path)
    eot_id = tok.token_to_id(EOT)
    for rel, vals in facts.RELATIONS.items():
        for v in vals:  # probes score one token per candidate, so values must be one token
            assert len(tok.encode(" " + v).ids) == 1, (rel, v)

    english = set()
    for s in val_stories:
        english.update(w.lower() for w in re.findall(r"[A-Za-z]+", s))
    for vals in facts.RELATIONS.values():
        english.update(v.lower() for v in vals)
    names = facts.name_pool(args.n_old + args.n_streams * args.stream_len, english)
    old = facts.old_entities(names[:args.n_old], seed=0)
    streams = []
    for s in range(args.n_streams):
        lo = args.n_old + s * args.stream_len
        streams.append(facts.stream_facts(names[lo:lo + args.stream_len], seed=100 + s))

    rng = random.Random(0)
    bios = [facts.bio(e, rng) for e in old for _ in range(args.bios_per_old)]
    bio_ids = encode_docs(tok, bios, eot_id)
    bio_tokens = sum(map(len, bio_ids))
    story_budget = args.pretrain_tokens - bio_tokens
    rng.shuffle(train_stories)
    story_ids, n = [], 0
    for chunk_start in range(0, len(train_stories), 20000):
        for ids in encode_docs(tok, train_stories[chunk_start:chunk_start + 20000], eot_id):
            if n >= story_budget:
                break
            story_ids.append(ids)
            n += len(ids)
        if n >= story_budget:
            break
    docs = story_ids + bio_ids
    rng.shuffle(docs)
    train = np.concatenate(docs)
    assert len(train) >= 0.99 * args.pretrain_tokens, "not enough stories read; raise --max_mb"
    val = np.concatenate(encode_docs(tok, val_stories, eot_id))
    train.tofile(os.path.join(args.data_dir, "train.bin"))
    val.tofile(os.path.join(args.data_dir, "val.bin"))

    # Background corpus for IDF: same mixture as pretraining, held-out bios and stories.
    with open(os.path.join(args.data_dir, "facts.json"), "w") as f:
        json.dump({"old": old, "streams": streams}, f)
    toks_per_story = n / max(1, len(story_ids))
    print(f"train tokens {len(train):,} (bios {bio_tokens:,} = {bio_tokens / len(train):.1%}),"
          f" val tokens {len(val):,}, stories used {len(story_ids):,}"
          f" ({toks_per_story:.0f} tok/story)")
    ex = facts.render("home", 0, old[0]["name"], old[0]["attrs"]["home"])
    p = facts.probe_prompt("home", 0, old[0]["name"])
    assert tok.encode(ex).ids[:len(tok.encode(p).ids) + 1] == \
        tok.encode(p).ids + tok.encode(" " + old[0]["attrs"]["home"]).ids
    print("example:", ex, "->", tok.encode(ex).tokens)
    print("bio:", bios[0])


if __name__ == "__main__":
    main()
