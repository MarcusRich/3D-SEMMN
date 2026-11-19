"""Language-model word features: surprisal and a mid-layer embedding per word.

``gpt2_word_features`` runs a causal LM over the story's words in order and returns,
for each word, its surprisal in bits (summed over its sub-word tokens) and the hidden
state of a chosen layer at the word's last token. The model and tokenizer are passed
in, so the alignment logic is tested with a stub; ``load_gpt2`` loads GPT-2 small
through ``transformers`` and fails clearly when that package is not installed.

``Pca`` reduces the embeddings to a few dimensions, fitted on training-run words only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

GPT2_BOS = 50256  # <|endoftext|>


def load_gpt2(name: str = "gpt2"):
    """GPT-2 small and its tokenizer from the local Hugging Face cache or hub."""
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "LM features need `transformers` (GPT-2 small). Install it in the venv, then rerun."
        ) from exc
    tok = GPT2TokenizerFast.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name)
    model.eval()
    return model, tok


def word_token_ids(words: Sequence[str], encode) -> list[list[int]]:
    """Token ids of each word encoded on its own with a leading space.

    GPT-2's pre-tokenizer splits on the space before a word, so encoding words one at a
    time gives the same tokens as encoding the joined text, with exact word alignment.
    """
    ids = [list(encode(" " + w)) for w in words]
    if any(len(t) == 0 for t in ids):
        raise ValueError("a word encoded to no tokens")
    return ids


def run_units(words: Sequence[str], encode, period_after: Sequence[bool] | None = None,
              period: str = ".") -> tuple[list[list[int]], list[int]]:
    """Token ids of a run's text units and the word each unit belongs to (-1: punctuation).

    The run's words, lower-cased, each followed by ``period`` (no leading space) when
    ``period_after`` marks it as the last word of a sentence. Punctuation is context only:
    it is never charged to a word's surprisal.
    """
    per_word = word_token_ids([w.lower() for w in words], encode)
    stop = list(encode(period))
    flags = [False] * len(words) if period_after is None else list(period_after)
    if len(flags) != len(words):
        raise ValueError("one period flag per word is required")
    units, owner = [], []
    for k, ids in enumerate(per_word):
        units.append(ids)
        owner.append(k)
        if flags[k]:
            units.append(stop)
            owner.append(-1)
    return units, owner


def sentence_final(sentence_ids: np.ndarray) -> np.ndarray:
    """True for a word whose next word starts a new sentence (the run's last word: False)."""
    s = np.asarray(sentence_ids)
    out = np.zeros(len(s), dtype=bool)
    out[:-1] = s[1:] != s[:-1]
    return out


def gpt2_word_features(
    words: Sequence[str],
    model,
    encode,
    layer: int,
    period_after: Sequence[bool] | None = None,
    bos: int = GPT2_BOS,
    max_context: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """(surprisal in bits (n_words,), layer-``layer`` state at each word's last token).

    One causal pass over one run's text from ``bos``: each word sees the run's earlier
    words (and sentence-final periods) and nothing after it, and no other run. The run
    must fit in ``max_context`` tokens; longer input raises rather than being windowed.
    ``model(input_ids)`` must return ``.logits`` (1, T, V) and ``.hidden_states`` (tuple
    of (1, T, H)) when called with ``output_hidden_states=True``.
    """
    import torch

    units, owner = run_units(words, encode, period_after)
    full = [bos] + [t for u in units for t in u]
    tok_owner = np.concatenate([[-1]] + [[o] * len(u) for o, u in zip(owner, units)])
    if len(full) > max_context:
        raise ValueError(f"run is {len(full)} tokens, over the {max_context}-token context")
    with torch.no_grad():
        out = model(torch.tensor([full]), output_hidden_states=True)
    lp = torch.log_softmax(out.logits[0].double(), dim=-1).numpy()
    hs = out.hidden_states[layer][0].double().numpy()
    tok_logp = np.zeros(len(full))
    tok_logp[1:] = lp[np.arange(len(full) - 1), full[1:]]
    surprisal = np.zeros(len(words))
    word_tok = tok_owner >= 0
    np.add.at(surprisal, tok_owner[word_tok], -tok_logp[word_tok] / np.log(2))
    last = np.array([np.max(np.nonzero(tok_owner == k)[0]) for k in range(len(words))])
    return surprisal, hs[last]


@dataclass
class Pca:
    """PCA fitted on training words; ``transform`` returns unit-variance scores."""

    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, k: int) -> "Pca":
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        k = min(k, vt.shape[0])
        scale = s[:k] / np.sqrt(max(len(x) - 1, 1))
        return cls(mean, vt[:k], np.where(scale > 0, scale, 1.0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mean) @ self.components.T) / self.scale


def normalise_word(w: str) -> str:
    return "".join(ch for ch in w.lower() if ch.isalnum() or ch == "'")


def ngram_surprisal(runs: dict[int, Sequence[str]], lam: float = 0.5, k: float = 0.1) -> dict[int, np.ndarray]:
    """Stand-in lexical surprisal (bits) when no pretrained LM is available.

    For each run, an interpolated bigram/unigram model with add-k smoothing is counted on
    the *other* runs of the story (leave-run-out), so a word's own occurrence never sets
    its probability. With about 10^4 training words this mostly measures word frequency
    plus the commonest word pairs; it is not a substitute for GPT-2 surprisal.
    """
    from collections import Counter

    norm = {r: [normalise_word(w) for w in ws] for r, ws in runs.items()}
    vocab = {w for ws in norm.values() for w in ws}
    v = len(vocab) + 1
    out = {}
    for r, ws in norm.items():
        uni: Counter = Counter()
        bi: Counter = Counter()
        for q, other in norm.items():
            if q == r:
                continue
            uni.update(other)
            bi.update(zip(other[:-1], other[1:]))
        total = sum(uni.values())
        s = np.zeros(len(ws))
        for i, w in enumerate(ws):
            p_uni = (uni[w] + k) / (total + k * v)
            if i == 0:
                p = p_uni
            else:
                prev = ws[i - 1]
                p_bi = (bi[(prev, w)] + k * p_uni * v) / (uni[prev] + k * v)
                p = lam * p_bi + (1 - lam) * p_uni
            s[i] = -np.log2(p)
        out[r] = s
    return out


def read_safetensors(path, name: str) -> np.ndarray:
    """Read one float32 tensor from a .safetensors file (no extra packages needed)."""
    import json
    import struct

    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        meta = header[name]
        if meta["dtype"] != "F32":
            raise ValueError(f"{name} is {meta['dtype']}, expected F32")
        start, end = meta["data_offsets"]
        fh.seek(8 + n + start)
        return np.frombuffer(fh.read(end - start), dtype="<f4").reshape(meta["shape"]).copy()


def wordpiece(word: str, vocab: dict[str, int], unk: str = "[UNK]") -> list[int]:
    """Greedy longest-match-first WordPiece ids of one lower-cased word."""
    w = normalise_word(word)
    if not w:
        return [vocab[unk]]
    ids, start = [], 0
    while start < len(w):
        end = len(w)
        while end > start:
            piece = w[start:end] if start == 0 else "##" + w[start:end]
            if piece in vocab:
                ids.append(vocab[piece])
                break
            end -= 1
        if end == start:
            return [vocab[unk]]
        start = end
    return ids


def static_word_vectors(words: Sequence[str], table: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    """Context-free word vectors: the mean input embedding of each word's WordPiece pieces."""
    return np.stack([table[wordpiece(w, vocab)].mean(axis=0) for w in words])


def semantic_dissimilarity(vectors: np.ndarray, sentence: np.ndarray) -> np.ndarray:
    """Broderick et al. (2018): 1 - r(word vector, mean vector of the preceding words of
    its sentence). A sentence's first word is compared with the mean of the previous
    sentence; the very first word of the text gets 0 (no context)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    sentence = np.asarray(sentence)
    out = np.zeros(len(vectors))
    for i in range(1, len(vectors)):
        same = np.nonzero(sentence[:i] == sentence[i])[0]
        if len(same):
            ctx = vectors[same]
        else:
            prev = sentence[i - 1]
            ctx = vectors[: i][sentence[:i] == prev]
        out[i] = 1.0 - np.corrcoef(vectors[i], ctx.mean(axis=0))[0, 1]
    return out
