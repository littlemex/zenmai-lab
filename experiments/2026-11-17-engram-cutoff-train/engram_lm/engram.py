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
        self.compressed_vocab_size = compressed_vocab_size

    @torch.no_grad()
    def trigram_keys(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, L] int64 key of the 3-gram of compressed ids ending at each position (oldest first,
        base compressed_vocab_size), with the pad id before the window start; the same key
        counts.py stores for the training text."""
        c = self.token_map[input_ids]
        h1 = F.pad(c, (1, 0), value=self.pad_id)[:, : c.shape[1]]
        h2 = F.pad(c, (2, 0), value=self.pad_id)[:, : c.shape[1]]
        k = self.compressed_vocab_size
        return (h2 * k + h1) * k + c

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

    def __init__(self, d_model: int, n_rows: int, n_columns: int, head_dim: int, norm_eps: float = 1e-6,
                 n_tables: int = 1, n_low_columns: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_low_columns = n_low_columns  # the 2-gram columns; the columns after them are the 3-gram ones
        self.table = nn.Embedding(n_rows, head_dim)
        # cross tables: n_tables > 1 splits the training text into n_tables parts and each training position
        # reads only the tables that do not learn from its part (see read); evaluation reads the mean of all.
        # The second table keeps its own name so the two-table checkpoints load unchanged.
        self.table_b = nn.Embedding(n_rows, head_dim) if n_tables >= 2 else None
        self.extra_tables = nn.ModuleList(nn.Embedding(n_rows, head_dim) for _ in range(max(0, n_tables - 2)))
        self.wkv = nn.Linear(n_columns * head_dim, 2 * d_model, bias=False)
        # two factors, as in the reference: only their product is used, but the factored form trains differently
        self.q_weight = nn.Parameter(torch.ones(d_model))
        self.k_weight = nn.Parameter(torch.ones(d_model))
        self.eps = norm_eps
        self.clamp_value = 1e-6
        nn.init.normal_(self.table.weight, std=0.02)
        for t in self.tables[1:]:
            nn.init.normal_(t.weight, std=0.02)

    @property
    def tables(self) -> list[nn.Embedding]:
        return [self.table] + ([self.table_b] if self.table_b is not None else []) + list(self.extra_tables)

    def read(self, rows: torch.Tensor, fold: torch.Tensor | None) -> torch.Tensor:
        """[B, L, n_columns, head_dim]. fold [B, L] int64, training only: the part of the training text the
        position's token is in. A position in part f gets the mean of the other tables' values and sends its
        gradient to table f, so a table is never read at a position it learns from. Without fold (evaluation)
        the mean of all tables is read."""
        tables = self.tables
        if len(tables) == 1:
            return self.table(rows)
        e = torch.stack([t(rows) for t in tables])  # [N, B, L, C, D]
        if fold is None:
            return e.mean(0)
        own = torch.gather(e, 0, fold[None, ..., None, None].expand(1, *e.shape[1:]))[0]
        others = (e.sum(0) - own) / (len(tables) - 1)
        return others.detach() + own - own.detach()

    def forward(self, h: torch.Tensor, rows: torch.Tensor, return_gate: bool = False,
                scale3: torch.Tensor | None = None, fold: torch.Tensor | None = None):
        """h: [B, L, d]; rows: [B, L, n_columns] row ids into this layer's table. scale3 [B, L] multiplies
        the 3-gram rows (the count cutoff); fold selects the cross-table read during training."""
        e = self.read(rows, fold)
        if scale3 is not None:
            low = e[:, :, : self.n_low_columns]
            high = e[:, :, self.n_low_columns :] * scale3[..., None, None].to(e.dtype)
            e = torch.cat([low, high], dim=2)
        e = e.flatten(-2)  # [B, L, n_columns * head_dim]
        key, value = self.wkv(e).split(self.d_model, dim=-1)
        hf, kf = h.float(), key.float()
        rstd = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(kf.square().mean(-1) + self.eps)
        weight = self.q_weight.float() * self.k_weight.float()
        dot = (hf * weight * kf).sum(-1) * rstd * self.d_model**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        out = (hf + gate.unsqueeze(-1) * value.float()).to(h.dtype)
        return (out, gate) if return_gate else out
