"""The count cutoff and the cross tables on a tiny model, on CPU (no download)."""
import numpy as np
import torch
from engram_lm.model import GPT, ModelConfig


def test_cutoff_and_cross_tables():
    V, K = 100, 37
    tm = [i % K for i in range(V)]
    def mk(**kw):
        torch.manual_seed(0)
        return GPT(ModelConfig(vocab_size=V, block_size=16, n_layer=2, n_head=2, d_model=32, engram_layer_ids=[1],
                               engram_rows_per_head=101, engram_head_dim=8, **kw), tm, K, 99)
    train = np.random.default_rng(0).integers(0, V, 5000)
    c = np.array(tm)[train]; keys = (c[:-2]*K + c[1:-1])*K + c[2:]; u, n = np.unique(keys, return_counts=True)
    m = mk(engram_cut_k=3, engram_cut_mode="hard"); m.attach_counts(u, n.astype(np.int32))
    x = torch.from_numpy(train[100:116]).unsqueeze(0)
    q = m.engram_hash.trigram_keys(x)[0].numpy()
    assert (q[2:] == keys[100:114]).all(), "keys differ from the numpy count keys"
    m.eval(); se = m.cut_scale(x)[0].numpy(); m.train(); st = m.cut_scale(x)[0].numpy()
    lut = dict(zip(u, n)); cc = np.array([lut.get(k, 0) for k in q])
    assert (se == (cc >= 3)).all() and (st == (np.maximum(cc - 1, 0) >= 3)).all()
    m.config.engram_cut_mode = "soft"; m.eval(); assert np.allclose(m.cut_scale(x)[0].numpy(), cc / (cc + 3))
    # scale only touches the 3-gram columns
    e = m.engrams["1"]; rows = m.engram_hash(x)[:, :, 0]; h = torch.randn(1, 16, 32)
    z = torch.zeros(1, 16); o1 = e(h, rows, scale3=z)
    assert e.n_low_columns == 8
    o_low = e(h, rows, scale3=torch.ones(1, 16))
    assert not torch.allclose(o1, o_low), "zeroing the 3-gram rows must change the output"
    with torch.no_grad():  # the 2-gram and 3-gram row ranges are disjoint, so this leaves the 2-gram rows alone
        e.table.weight[rows[0, :, 8:].flatten()] = 0
    assert torch.allclose(o1, e(h, rows)), "scale3 = 0 must equal reading zeros in the 3-gram rows"
    # cross tables: gradient routing and eval mean
    mc = mk(engram_cross_tables=True); ec = mc.engrams["1"]
    half = torch.zeros(1, 16, dtype=torch.bool); half[0, 8:] = True
    r = mc.engram_hash(x)[:, :, 0]
    out = ec.read(r, half); out.sum().backward()
    a, b = ec.table.weight, ec.table_b.weight
    assert torch.allclose(out[0, :8], b.detach()[r[0, :8]]) and torch.allclose(out[0, 8:], a.detach()[r[0, 8:]])
    ga = torch.zeros_like(a); ga.index_add_(0, r[0, :8].flatten(), torch.ones(8 * r.shape[-1], a.shape[1]))
    assert torch.allclose(a.grad, ga), "table A must get the gradient of the half-A positions only"
    mc.eval(); assert torch.allclose(ec.read(r, None), (a[r] + b[r]) / 2)
    # full forward in train mode with half, and the parameter list
    mc.train(); _, loss = mc(x, x, half=half); loss.backward()
    assert len(mc.engram_parameters()) == 2

