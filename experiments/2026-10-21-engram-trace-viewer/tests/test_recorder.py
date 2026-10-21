"""The recorder writes the rows the released model reads, in the axes the schema promises.

The table layout (prime-sized column ranges in (n, head) order), the hash multipliers and every
row id are recomputed here from the published configuration with code of this file, without
calling the reference module, so a recorder that mislabels a column, wires a layer, column or
position to the wrong axis, or breaks the layout fails.
"""

import json
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import sympy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recorder"))
import record_trace  # noqa: E402

TEXT = "Only Alexander the Great could tame the horse. Peter the Great ruled Russia, and the great wall is long."


@pytest.fixture(scope="module")
def trace():
    return record_trace.record(TEXT, "test", record_trace.DEFAULT_REPO, record_trace.DEFAULT_REVISION, 512, False)


@pytest.fixture(scope="module")
def reference():
    paths = record_trace.fetch_sources(record_trace.DEFAULT_REPO, record_trace.DEFAULT_REVISION)
    module = record_trace.load_reference_module(paths["inference/engram.py"])
    config = json.loads(paths["inference/config.json"].read_text())
    return module, config


def test_trace_matches_the_schema(trace):
    schema = json.loads((ROOT / "viewer/schema/engram-trace.schema.json").read_text())
    jsonschema.validate(trace, schema)


def expected_layout(config):
    """Each (layer, n, head) takes the next unused prime above engram_vocab_size - 1."""
    seen, layers = set(), []
    for layer_id in config["engram_layer_ids"]:
        columns, offset = [], 0
        for n in range(2, config["engram_max_ngram_size"] + 1):
            current = config["engram_vocab_size"] - 1
            for head in range(config["engram_n_heads"]):
                current = sympy.nextprime(current)
                while current in seen:
                    current = sympy.nextprime(current)
                seen.add(current)
                columns.append({"n": n, "head": head, "offset": offset, "size": current})
                offset += current
        layers.append({"layer_id": layer_id, "num_rows": offset, "columns": columns})
    return layers


def expected_multipliers(config):
    """One odd multiplier per (layer, lookback), from a per-layer seeded generator."""
    bound = max(1, (np.iinfo(np.int64).max // config["engram_compressed_vocab_size"]) // 2)
    return [
        np.random.default_rng(10007 * layer_id).integers(0, bound, size=config["engram_max_ngram_size"], dtype=np.int64) * 2 + 1
        for layer_id in config["engram_layer_ids"]
    ]


def test_layout_matches_the_configuration(trace, reference):
    _, config = reference
    assert trace["layers"] == expected_layout(config)
    assert [layer["num_rows"] for layer in trace["layers"]] == config["engram_num_embeddings"]


def test_rows_recomputed_independently(trace, reference):
    _, config = reference
    compressed = [token["compressed_id"] for token in trace["tokens"]]
    pad = trace["model"]["pad_compressed_id"]
    max_n = config["engram_max_ngram_size"]
    multipliers = expected_multipliers(config)
    for l, layer in enumerate(trace["layers"]):
        for t in range(len(compressed)):
            history = [compressed[t - k] if t - k >= 0 else pad for k in range(max_n)]
            rolling = np.int64(history[0]) * multipliers[l][0]
            for c, column in enumerate(layer["columns"]):
                n = column["n"]
                value = rolling
                for k in range(1, n):
                    value = np.bitwise_xor(value, np.int64(history[k]) * multipliers[l][k])
                expected = int(value % column["size"]) + column["offset"]
                assert trace["hash_rows"][t][l][c] == expected, (t, l, c)


def test_same_ngram_reads_same_rows_and_case_folds(trace):
    texts = [token["text"].strip().lower() for token in trace["tokens"]]
    greats = [t for t, text in enumerate(texts) if text == "great" and texts[t - 1] == "the"]
    assert len(greats) == 3  # "the Great" twice and "the great" once
    layer = trace["layers"][0]
    two_gram = [c for c, column in enumerate(layer["columns"]) if column["n"] == 2]
    three_gram = [c for c, column in enumerate(layer["columns"]) if column["n"] == 3]
    rows = trace["hash_rows"]
    for t in greats[1:]:
        assert [rows[t][0][c] for c in two_gram] == [rows[greats[0]][0][c] for c in two_gram]
    # "Alexander the Great", "Peter the Great" and "and the great" are three different 3-grams
    assert len({tuple(rows[t][0][c] for c in three_gram) for t in greats}) == 3


def test_split_characters_get_one_label_each():
    trace = record_trace.record("庁庁", "ja", record_trace.DEFAULT_REPO, record_trace.DEFAULT_REVISION, 512, False)
    labels = [token.get("display") for token in trace["tokens"]]
    assert labels == ["庁 1/2", "庁 2/2", "庁 1/2", "庁 2/2"]


def test_a_branch_name_is_recorded_as_its_commit():
    trace = record_trace.record("hi", "x", record_trace.DEFAULT_REPO, record_trace.DEFAULT_REVISION[:12], 512, False)
    assert trace["source"]["revision"] == record_trace.DEFAULT_REVISION


def test_too_many_tokens_is_refused():
    with pytest.raises(ValueError, match="--max-tokens"):
        record_trace.record(TEXT, "x", record_trace.DEFAULT_REPO, record_trace.DEFAULT_REVISION, 4, False)
