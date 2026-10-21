"""Engram: an n-gram hash lookup added into the residual stream, gated by the stream itself.

The layout and hash follow DeepSeek-V4.1-Flash's inference/engram.py, and the gate follows its
model.py `Engram`, with one residual stream instead of hyper-connection copies:

* each (n-gram size, head) pair owns a prime-sized row range of the table, primes drawn in order
  above `rows_per_head - 1` and never reused, so the ranges are disjoint;
* an n-gram's id is the XOR of `compressed_id * multiplier` over its tokens, the multipliers
  odd and drawn from a per-layer seeded generator, taken modulo the range's prime;
* the rows read for a position are joined into e_t; `wkv` turns e_t into a key and a value,
  the gate is sigmoid(signed sqrt(<h, q_weight * k_weight * key> * rstd(h) * rstd(key) / sqrt(d))),
  and h + gate * value is the output.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def next_unused_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while True:
        if candidate not in seen and is_prime(candidate):
            return candidate
        candidate += 1


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for k in range(3, math.isqrt(n) + 1, 2):
        if n % k == 0:
            return False
    return True


@dataclass(frozen=True)
class EngramLayout:
    """The row ranges of the tables of all Engram layers."""

    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    primes: tuple[tuple[int, ...], ...]  # [layer][column], columns in (n, head) order

    @classmethod
    def build(cls, layer_ids, max_ngram_size: int, n_heads: int, rows_per_head: int) -> "EngramLayout":
        seen: set[int] = set()
        primes = []
        for _ in layer_ids:
            per_layer = []
            for _ in range(max_ngram_size - 1):
                current = rows_per_head - 1
                for _ in range(n_heads):
                    current = next_unused_prime(current, seen)
                    seen.add(current)
                    per_layer.append(current)
            primes.append(tuple(per_layer))
        return cls(tuple(layer_ids), max_ngram_size, n_heads, tuple(primes))

    @property
    def n_columns(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    def column_ngram(self, column: int) -> int:
        return 2 + column // self.n_heads


def hash_multipliers(layer_ids, max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    """[n_layers, max_ngram_size] odd int64 multipliers that cannot overflow when multiplied by an id."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = [
        np.random.default_rng(10007 * layer_id).integers(0, bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1
        for layer_id in layer_ids
    ]
    return torch.tensor(np.stack(rows))


class NgramHash(nn.Module):
    """Maps a batch of token ids to the row ids each Engram layer reads, [B, L, n_layers, n_columns]."""

    def __init__(self, layout: EngramLayout, token_map: list[int], compressed_vocab_size: int, pad_token_id: int):
        super().__init__()
        self.layout = layout
        offsets = [np.cumsum([0, *p[:-1]]) for p in layout.primes]
        self.register_buffer("token_map", torch.tensor(token_map, dtype=torch.int64), persistent=False)
        self.register_buffer("primes", torch.tensor(layout.primes, dtype=torch.int64), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets), dtype=torch.int64), persistent=False)
        self.register_buffer(
            "multipliers", hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_vocab_size), persistent=False
        )
        self.pad_id = token_map[pad_token_id]

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        compressed = self.token_map[input_ids]  # [B, L]
        n_max = self.layout.max_ngram_size
        # history[..., k] is the token k positions back, the pad id before the start
        history = torch.stack(
            [F.pad(compressed, (k, 0), value=self.pad_id)[:, : compressed.shape[1]] for k in range(n_max)], dim=-1
        )
        products = history.unsqueeze(2) * self.multipliers  # [B, L, n_layers, n_max]
        rolling, columns = products[..., 0], []
        heads = self.layout.n_heads
        for i in range(1, n_max):
            rolling = torch.bitwise_xor(rolling, products[..., i])  # the hash of the (i + 1)-gram
            primes = self.primes[:, (i - 1) * heads : i * heads]  # [n_layers, heads]
            columns.append(rolling.unsqueeze(-1) % primes)
        return torch.cat(columns, dim=-1) + self.offsets


class Engram(nn.Module):
    """One Engram layer: reads n_columns rows and adds the gated value to the residual stream."""

    def __init__(self, d_model: int, n_rows: int, n_columns: int, head_dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.table = nn.Embedding(n_rows, head_dim)
        self.wkv = nn.Linear(n_columns * head_dim, 2 * d_model, bias=False)
        # two factors, as in the reference: only their product is used, but the factored form trains differently
        self.q_weight = nn.Parameter(torch.ones(d_model))
        self.k_weight = nn.Parameter(torch.ones(d_model))
        self.eps = norm_eps
        self.clamp_value = 1e-6
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, h: torch.Tensor, rows: torch.Tensor, return_gate: bool = False):
        """h: [B, L, d]; rows: [B, L, n_columns] row ids into this layer's table."""
        e = self.table(rows).flatten(-2)  # [B, L, n_columns * head_dim]
        key, value = self.wkv(e).split(self.d_model, dim=-1)
        hf, kf = h.float(), key.float()
        rstd = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(kf.square().mean(-1) + self.eps)
        weight = self.q_weight.float() * self.k_weight.float()
        dot = (hf * weight * kf).sum(-1) * rstd * self.d_model**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        out = (hf + gate.unsqueeze(-1) * value.float()).to(h.dtype)
        return (out, gate) if return_gate else out
