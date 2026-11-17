"""A GPT-2 style decoder with optional Engram layers.

With `engram_layer_ids` empty the model is a plain GPT; with it set, Engram adds its gated
n-gram lookup to the residual stream at the start of each listed block, before attention, as
DeepSeek-V4.1-Flash does. Everything else is shared, so the two variants differ only in Engram.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .engram import Engram, EngramLayout, NgramHash


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.0
    engram_layer_ids: list[int] = field(default_factory=list)
    engram_max_ngram_size: int = 3
    engram_n_heads: int = 8
    engram_rows_per_head: int = 65536
    engram_head_dim: int = 64
    # count cutoff on the 3-gram rows: 0 is off. A position's weight is a function of c, the number of times
    # its 3-gram occurs in the training text: "hard" is 1 if c >= k else 0, "soft" is c / (c + k). While
    # training, c - 1 is used (leave-one-out), so a 3-gram seen once in training is treated as an unseen one.
    engram_cut_k: int = 0
    engram_cut_mode: str = "hard"
    # cross tables: 0, 1 or false is one table; true is 2; an integer n splits the training text into n parts
    # ("article": articles in turn, "chunk512": 512-token pieces of the token stream in turn) and keeps n tables
    engram_cross_tables: int | bool = 0
    engram_cross_split: str = "article"

    @property
    def n_cross_tables(self) -> int:
        return 2 if self.engram_cross_tables is True else max(1, int(self.engram_cross_tables))

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False),
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln_1(x)))
        return x + self.drop(self.mlp(self.ln_2(x)))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig, token_map: list[int] | None = None, compressed_vocab_size: int = 0,
                 pad_token_id: int = 50256):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

        # Engram is built after the backbone is initialised and from its own random stream, so a
        # baseline and an Engram model made with the same seed start from the same backbone.
        self.engram_hash = None
        self.engrams = nn.ModuleDict()
        if config.engram_layer_ids:
            if token_map is None:
                raise ValueError("an Engram model needs the compressed token map")
            ids = config.engram_layer_ids
            if len(set(ids)) != len(ids) or not all(0 <= i < config.n_layer for i in ids):
                raise ValueError(f"engram_layer_ids {ids} must be distinct block indices below n_layer={config.n_layer}")
            layout = EngramLayout.build(ids, config.engram_max_ngram_size, config.engram_n_heads,
                                        config.engram_rows_per_head)
            self.engram_hash = NgramHash(layout, token_map, compressed_vocab_size, pad_token_id)
            # a CPU-only stream derived from the seed, distinct from every other run's seed
            engram_seed = int(np.random.SeedSequence([torch.initial_seed(), 1]).generate_state(1)[0])
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(engram_seed)
                for index, layer_id in enumerate(layout.layer_ids):
                    engram = Engram(config.d_model, sum(layout.primes[index]), layout.n_columns, config.engram_head_dim,
                                    n_tables=config.n_cross_tables, n_low_columns=layout.n_heads)
                    nn.init.normal_(engram.wkv.weight, mean=0.0, std=0.02)
                    self.engrams[str(layer_id)] = engram

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def engram_parameters(self) -> list[nn.Parameter]:
        return [t.weight for engram in self.engrams.values() for t in engram.tables]

    def attach_counts(self, keys: np.ndarray, counts: np.ndarray) -> None:
        """The training-text 3-gram counts (counts.py) the cutoff reads; not part of the checkpoint."""
        device = self.wte.weight.device
        self.register_buffer("count_keys", torch.from_numpy(keys).to(device), persistent=False)
        self.register_buffer("count_values", torch.from_numpy(counts.astype(np.int64)).to(device), persistent=False)

    @torch.no_grad()
    def cut_scale(self, idx: torch.Tensor) -> torch.Tensor | None:
        k = self.config.engram_cut_k
        if k <= 0 or self.engram_hash is None:
            return None
        if not hasattr(self, "count_keys"):
            raise RuntimeError("engram_cut_k is set but attach_counts was not called")
        q = self.engram_hash.trigram_keys(idx)
        i = torch.searchsorted(self.count_keys, q).clamp(max=self.count_keys.numel() - 1)
        c = torch.where(self.count_keys[i] == q, self.count_values[i], torch.zeros_like(q))
        if self.training:
            c = (c - 1).clamp(min=0)
        c = c.float()
        if self.config.engram_cut_mode == "hard":
            return (c >= k).float()
        if self.config.engram_cut_mode == "soft":
            return c / (c + k)
        raise ValueError(f"unknown engram_cut_mode {self.config.engram_cut_mode!r}")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, record: dict | None = None,
                engram_off: bool = False, fold: torch.Tensor | None = None):
        """record, when given, receives the residual stream after every block ("hidden", a list) and the
        gate of every Engram layer ("gates", by layer id), for analysis. engram_off skips the Engram
        layers of a trained model, to measure what they contribute."""
        b, t = idx.shape
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(t, device=idx.device)))
        rows = self.engram_hash(idx) if self.engram_hash is not None else None
        scale3 = self.cut_scale(idx) if rows is not None else None
        fold = fold if self.training else None
        if record is not None:
            record.setdefault("hidden", [])
            record.setdefault("gates", {})
            record["rows"] = rows
        for layer_id, block in enumerate(self.blocks):
            key = str(layer_id)
            if key in self.engrams and not engram_off:
                index = self.config.engram_layer_ids.index(layer_id)
                if record is not None:
                    x, gate = self.engrams[key](x, rows[:, :, index], return_gate=True, scale3=scale3, fold=fold)
                    record["gates"][layer_id] = gate
                else:
                    x = self.engrams[key](x, rows[:, :, index], scale3=scale3, fold=fold)
            x = block(x)
            if record is not None:
                record["hidden"].append(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_parameters(self) -> dict:
        engram = sum(p.numel() for p in self.engrams.parameters())
        tables = sum(p.numel() for p in self.engram_parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "backbone": total - engram, "engram": engram, "engram_tables": tables}
