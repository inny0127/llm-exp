"""Shared evaluation: cloze probes over facts and held-out TinyStories loss."""
import numpy as np
import torch
import torch.nn.functional as F

import facts

EOT = "<|endoftext|>"


class Probe:
    """Batched single-token cloze probes.

    items: list of (name, rel, value). A probe is correct when the value token has the
    highest logit among that relation's candidate tokens (chance = 1 / #candidates).
    """

    def __init__(self, tok, items, tidx):
        eot = tok.token_to_id(EOT)
        self.rel_cands = {r: torch.tensor([tok.encode(" " + v).ids[0] for v in vals])
                          for r, vals in facts.RELATIONS.items()}
        seqs = [[eot] + tok.encode(facts.probe_prompt(r, tidx, n)).ids for n, r, _ in items]
        self.lens = torch.tensor([len(s) for s in seqs])
        L = int(self.lens.max())
        self.x = torch.full((len(seqs), L), eot, dtype=torch.long)
        for i, s in enumerate(seqs):
            self.x[i, :len(s)] = torch.tensor(s)
        self.rels = [r for _, r, _ in items]
        self.ans = torch.tensor([facts.RELATIONS[r].index(v) for _, r, v in items])
        self.cand = torch.stack([self._padded(self.rel_cands[r]) for r in self.rels])
        self.cand_mask = torch.stack([self._padded(torch.ones(len(self.rel_cands[r]), dtype=torch.bool),
                                                   False) for r in self.rels])

    @staticmethod
    def _padded(t, fill=0, n=20):
        out = torch.full((n,), fill, dtype=t.dtype)
        out[:len(t)] = t
        return out

    @torch.no_grad()
    def run(self, model, idx=None, batch=512):
        """Returns (correct bool[n], log-prob of answer among candidates, predicted index)."""
        sel = torch.arange(len(self.x)) if idx is None else torch.as_tensor(idx).reshape(-1)
        correct, logp, pred = [], [], []
        for i in range(0, len(sel), batch):
            b = sel[i:i + batch]
            L = int(self.lens[b].max())
            logits = model(self.x[b, :L])
            # Only the last prompt position matters; select it before leaving the device.
            pos = (self.lens[b] - 1).to(logits.device)
            last = logits[torch.arange(len(b), device=logits.device), pos].float().cpu()
            cl = last.gather(1, self.cand[b]).masked_fill(~self.cand_mask[b], -1e9)
            lp = F.log_softmax(cl, dim=-1)
            ans = self.ans[b]
            pred.append(cl.argmax(-1))
            correct.append(pred[-1] == ans)
            logp.append(lp.gather(1, ans[:, None]).squeeze(1))
        return torch.cat(correct).numpy(), torch.cat(logp).numpy(), torch.cat(pred).numpy()


def val_windows(path, n, ctx, seed=0):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - ctx - 1, size=n)
    w = torch.from_numpy(np.stack([data[s:s + ctx + 1].astype(np.int64) for s in starts]))
    return w[:, :-1], w[:, 1:]


@torch.no_grad()
def lm_eval(model, x, y, batch=64):
    tot = 0.0
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch]).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[i:i + batch].reshape(-1).to(logits.device),
                               reduction="sum").item()
    return tot / y.numel()


def old_fact_items(old, n_entities):
    return [(e["name"], r, e["attrs"][r]) for e in old[:n_entities] for r in facts.REL_NAMES]
