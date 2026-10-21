"""The Engram model hashes like DeepSeek-V4.1-Flash, stays causal, and differs from the baseline only in Engram."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engram_lm.model import GPT, ModelConfig  # noqa: E402
from engram_lm.tokenizer import EOT_ID, build_compressed_token_map, load_tokenizer  # noqa: E402
from engram_lm.train import build_model, load_config, optimizers  # noqa: E402

REFERENCE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REFERENCE_REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"


@pytest.fixture(scope="module")
def token_map():
    return build_compressed_token_map(load_tokenizer())


@pytest.fixture(scope="module")
def reference():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REFERENCE_REPO, "inference/engram.py", revision=REFERENCE_REVISION)
    spec = importlib.util.spec_from_file_location("engram_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_config(**overrides) -> ModelConfig:
    base = dict(n_layer=3, n_head=2, d_model=32, block_size=64, engram_layer_ids=[1], engram_n_heads=4,
                engram_rows_per_head=97, engram_head_dim=8)
    return ModelConfig(**{**base, **overrides})


def test_token_map_matches_the_reference(token_map, reference):
    from transformers import PreTrainedTokenizerFast

    ours, size = token_map
    theirs, their_size = reference.build_compressed_token_map(PreTrainedTokenizerFast(tokenizer_object=load_tokenizer()))
    assert size == their_size and ours == theirs
    assert size < 50257  # the compression merges tokens


def test_row_ids_match_the_reference(token_map, reference):
    from transformers import PreTrainedTokenizerFast

    config = ModelConfig(**json.loads((ROOT / "configs/engram.json").read_text())["model"])
    config.engram_layer_ids = [1, 5]  # two layers, so per-layer multipliers and prime ranges are exercised
    model = GPT(config, *token_map, EOT_ID)
    ids = torch.randint(0, 50257, (2, 40), generator=torch.Generator().manual_seed(0))
    ours = model.engram_hash(ids)

    layout = model.engram_hash.layout
    args = types.SimpleNamespace(
        engram_layer_ids=config.engram_layer_ids, engram_max_ngram_size=config.engram_max_ngram_size,
        engram_vocab_size=config.engram_rows_per_head, engram_n_heads=config.engram_n_heads,
        engram_num_embeddings=[sum(p) for p in layout.primes], engram_head_dim=config.engram_head_dim,
        engram_pad_id=EOT_ID, engram_compressed_vocab_size=token_map[1], max_batch_size=2, max_seq_len=40)
    their_layout = reference.EngramLayout.from_args(args)
    assert tuple(p for per_n in their_layout.primes[0] for p in per_n) == layout.primes[0]
    state = reference.NgramHashState(args, their_layout, PreTrainedTokenizerFast(tokenizer_object=load_tokenizer()))
    assert torch.equal(ours, state(ids, 0))


def test_engram_is_causal(token_map):
    torch.manual_seed(0)
    model = GPT(small_config(), *token_map, EOT_ID).eval()
    ids = torch.randint(0, 50257, (1, 32))
    changed = ids.clone()
    changed[0, 20] = (changed[0, 20] + 1) % 50257
    with torch.no_grad():
        a, _ = model(ids)
        b, _ = model(changed)
    assert torch.allclose(a[0, :20], b[0, :20])
    assert not torch.allclose(a[0, 20:23], b[0, 20:23])  # the token and the n-grams that contain it


def test_variants_share_the_backbone(token_map):
    engram = GPT(small_config(), *token_map, EOT_ID)
    baseline = GPT(small_config(engram_layer_ids=[]))
    assert engram.num_parameters()["backbone"] == baseline.num_parameters()["total"]
    assert baseline.num_parameters()["engram"] == 0


def test_gates_and_hidden_states_are_recorded(token_map):
    model = GPT(small_config(), *token_map, EOT_ID).eval()
    record = {}
    with torch.no_grad():
        model(torch.randint(0, 50257, (1, 16)), record=record)
    assert len(record["hidden"]) == 3
    gate = record["gates"][1]
    assert gate.shape == (1, 16) and bool(((gate > 0) & (gate < 1)).all())


def test_a_few_steps_reduce_the_loss(token_map):
    torch.manual_seed(0)
    model = GPT(small_config(), *token_map, EOT_ID)
    _, train = load_config(ROOT / "configs/engram.json")
    train = {**train, "lr": 3e-3}
    dense, sparse = optimizers(model, train)
    assert sparse is not None and sparse.param_groups[0]["lr"] == train["lr"] * train["engram_lr_scale"]
    x = torch.randint(0, 1000, (4, 32))
    y = torch.roll(x, -1, dims=1)
    losses = []
    for _ in range(30):
        _, loss = model(x, y)
        loss.backward()
        dense.step(); sparse.step(); dense.zero_grad(); sparse.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7


def test_configs_differ_only_in_engram():
    engram, train_a = load_config(ROOT / "configs/engram.json")
    baseline, train_b = load_config(ROOT / "configs/baseline.json")
    shared = ["n_layer", "n_head", "d_model", "block_size", "vocab_size", "dropout"]
    assert all(getattr(engram, k) == getattr(baseline, k) for k in shared)
    assert train_a == train_b
    assert engram.engram_layer_ids and not baseline.engram_layer_ids
    assert build_model(baseline).engram_hash is None


def test_same_seed_gives_the_same_backbone(token_map):
    torch.manual_seed(7)
    engram = GPT(small_config(), *token_map, EOT_ID)
    torch.manual_seed(7)
    baseline = GPT(small_config(engram_layer_ids=[]))
    theirs = dict(baseline.named_parameters())
    shared = {k: v for k, v in engram.named_parameters() if not k.startswith("engrams.")}
    assert shared.keys() == theirs.keys()
    assert all(torch.equal(v, theirs[k]) for k, v in shared.items())
    ids = torch.randint(0, 50257, (1, 16))
    with torch.no_grad():
        assert torch.equal(engram(ids, engram_off=True)[0], baseline(ids)[0])
        assert not torch.equal(engram(ids)[0], baseline(ids)[0])


def test_gate_and_update_follow_the_reference_formula(token_map):
    torch.manual_seed(0)
    model = GPT(small_config(), *token_map, EOT_ID).eval()
    engram = model.engrams["1"]
    with torch.no_grad():
        engram.q_weight.uniform_(0.5, 1.5)
        engram.k_weight.uniform_(0.5, 1.5)
        h = torch.randn(1, 16, 32)
        rows = model.engram_hash(torch.randint(0, 50257, (1, 16)))[:, :, 0]
        out, gate = engram(h, rows, return_gate=True)
        key, value = engram.wkv(engram.table(rows).flatten(-2)).split(32, dim=-1)
        rms = lambda z: z / z.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()  # noqa: E731
        dot = (rms(h) * engram.q_weight * engram.k_weight * rms(key)).sum(-1) / 32**0.5
        expected = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt())
    assert torch.allclose(gate, expected, atol=1e-5)
    assert torch.allclose(out, h + expected.unsqueeze(-1) * value, atol=1e-5)


def test_optimizers_cover_every_parameter_once(token_map):
    model = GPT(small_config(), *token_map, EOT_ID)
    _, train = load_config(ROOT / "configs/engram.json")
    dense, sparse = optimizers(model, train)
    seen = [id(p) for o in (dense, sparse) for g in o.param_groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters()) and len(seen) == len(set(seen))
    assert all(g["weight_decay"] == 0 for g in sparse.param_groups)


def test_engram_gradients_do_not_shrink_the_backbone_update(token_map):
    from engram_lm.train import clip

    torch.manual_seed(0)
    model = GPT(small_config(), *token_map, EOT_ID)
    _, loss = model(torch.randint(0, 50257, (2, 16)), torch.randint(0, 50257, (2, 16)))
    loss.backward()
    for p in model.engrams.parameters():
        p.grad.mul_(1e4)  # enormous Engram gradients, projections and tables alike
    engram = {id(p) for p in model.engrams.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in engram]
    before = [p.grad.clone() for p in backbone]
    backbone_norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in before])).item()
    norms = clip(model, max_norm=backbone_norm * 1.01)  # above the backbone's norm, below Engram's
    assert all(torch.equal(p.grad, g) for p, g in zip(backbone, before))
    assert norms["engram"] > backbone_norm and norms["tables"] > 0
    clipped = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.engram_parameters()])).item()
    assert clipped <= backbone_norm * 1.01 + 1e-6


def test_a_used_run_directory_is_refused(tmp_path):
    import subprocess

    out = tmp_path / "run"
    out.mkdir()
    result = subprocess.run([sys.executable, "-m", "engram_lm.train", "--config", str(ROOT / "configs/baseline.json"),
                             "--data-dir", str(tmp_path / "data"), "--out-dir", str(out)], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0 and "exists" in result.stderr


def test_training_windows_reach_the_end_of_the_file(tmp_path):
    from engram_lm.data import TokenFile

    np.arange(11, dtype=np.uint16).tofile(tmp_path / "t.bin")
    x, y = TokenFile(tmp_path / "t.bin").batch(64, 10, np.random.default_rng(0), "cpu")
    assert x.shape == (64, 10) and bool((y[:, -1] == 10).all())  # the only valid window, 0..10
