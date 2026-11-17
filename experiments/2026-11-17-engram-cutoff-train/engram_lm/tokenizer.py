"""The GPT-2 tokenizer and the Engram token compression built on it.

The compression follows the normalisation of DeepSeek-V4.1-Flash's inference/engram.py
(`build_compressed_token_map`): tokens that differ only in case, accents, Unicode form or
surrounding whitespace share one compressed id, so " The", "the" and "THE" hash alike.
"""

from functools import lru_cache

TOKENIZER_REPO = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EOT_ID = 50256  # <|endoftext|>, also the n-gram padding before the start of a sequence


@lru_cache(maxsize=1)
def load_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION))


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space; returns the map and its size."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"  # keeps a token that is exactly one space from collapsing to ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    key_to_new: dict[str, int] = {}
    vocab_size = tokenizer.get_vocab_size()
    lookup = [0] * vocab_size
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.id_to_token(token_id)  # a partial UTF-8 byte: key it by its raw form
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)
