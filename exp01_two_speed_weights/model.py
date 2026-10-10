"""Small GPT whose middle FFN is a product-key memory layer, plus LoRA wrappers.

The memory layer follows Meta's sparse-memory-finetuning setup at toy scale:
product keys (Lample et al. 2019), top-k slots per head, softmax over the
selected scores, and a gated output (y * silu(x W1)) W2.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    ctx: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 1024
    mem_layer: int = 2      # block whose FFN becomes the memory layer
    mem_n_keys: int = 256   # slots = mem_n_keys ** 2
    mem_heads: int = 4
    mem_topk: int = 16
    mem_k_dim: int = 128    # query dim, split into two product-key halves


# "tiny" is Exp 01/02 (3.8M dense params, 65K slots); "small" is the ~30M scale of the
# thread's tier-1 plan, with Meta's ratio of memory to dense parameters.
PRESETS = {
    "tiny": {},
    "small": dict(ctx=256, n_layer=8, n_head=8, d_model=512, d_ff=2048, mem_layer=4,
                  mem_n_keys=512, mem_heads=4, mem_topk=32, mem_k_dim=256),
}


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.out = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.out(F.gelu(self.fc(x)))


class ProductKeyMemory(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.k, self.n_keys = c.mem_heads, c.mem_topk, c.mem_n_keys
        self.half = c.mem_k_dim // 2
        self.query = nn.Linear(c.d_model, c.mem_heads * c.mem_k_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(c.mem_heads, 2, c.mem_n_keys, self.half)
                                 / math.sqrt(self.half))
        self.values = nn.Parameter(torch.randn(c.mem_n_keys ** 2, c.d_model)
                                   / math.sqrt(c.d_model))
        self.gate = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)
        self.record = False   # when True, keep the slot indices of the last forward
        self.last_idx = None  # (B, T, heads * topk)

    @property
    def n_slots(self):
        return self.values.shape[0]

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(B * T, C)
        # Slot routing in fp32 even under bf16 autocast, so pretraining (bf16) and the
        # streams (fp32) pick the same slots; a no-op on CPU.
        with torch.autocast(x.device.type, enabled=False):
            q = F.layer_norm(self.query(xf.float()).view(B * T, self.h, 2, self.half), (self.half,))
            s = torch.einsum("nhpd,hpkd->nhpk", q, self.keys)
            s1, i1 = s[:, :, 0].topk(self.k, dim=-1)
            s2, i2 = s[:, :, 1].topk(self.k, dim=-1)
            cand = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(-2)
            sc, ci = cand.topk(self.k, dim=-1)
        idx = i1.gather(-1, ci // self.k) * self.n_keys + i2.gather(-1, ci % self.k)
        if self.record:
            self.last_idx = idx.view(B, T, self.h * self.k).detach()
        w = F.softmax(sc, dim=-1)
        y = F.embedding_bag(idx.view(-1, self.k), self.values,
                            per_sample_weights=w.reshape(-1, self.k).to(self.values.dtype), mode="sum")
        y = y.view(B * T, self.h, C).sum(1)
        return self.out(y * F.silu(self.gate(xf))).view(B, T, C)


class Block(nn.Module):
    def __init__(self, c, use_mem):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attn = Attention(c)
        self.ffn = ProductKeyMemory(c) if use_mem else MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList(Block(c, i == c.mem_layer) for i in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        for name, p in self.named_parameters():
            if name.endswith("keys") or name.endswith("values"):
                continue
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * c.n_layer) if name.endswith(("proj.weight", "out.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    @property
    def mem(self):
        return self.blocks[self.cfg.mem_layer].ffn

    def forward(self, idx):
        idx = idx.to(self.tok.weight.device)  # callers keep data on the CPU
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x) @ self.tok.weight.t()

    def memory_param_names(self):
        p = f"blocks.{self.cfg.mem_layer}.ffn."
        return {p + "keys", p + "values", p + "query.weight"}

    def dense_parameters(self):
        """Everything outside the memory lookup (keys, values, query): what sleep trains."""
        skip = self.memory_param_names()
        return [p for n, p in self.named_parameters() if n not in skip and "lora_" not in n]


def lm_loss(model, x, y, mask=None):
    logits = model(x)
    y = y.to(logits.device)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.reshape(-1).float().to(logits.device)
    return (loss * mask).sum() / mask.sum()


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def add_lora(model, r, alpha):
    """LoRA on every attention and FFN matrix (memory gate/out included, lookup excluded)."""
    params = []
    for b in model.blocks:
        targets = [(b.attn, "qkv"), (b.attn, "proj")]
        targets += [(b.ffn, "gate"), (b.ffn, "out")] if isinstance(b.ffn, ProductKeyMemory) \
            else [(b.ffn, "fc"), (b.ffn, "out")]
        for mod, name in targets:
            wrapped = LoRALinear(getattr(mod, name), r, alpha)
            setattr(mod, name, wrapped)
            params += [wrapped.lora_A, wrapped.lora_B]
    return params
