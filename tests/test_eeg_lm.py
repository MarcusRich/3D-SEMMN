"""GPT-2 word-feature alignment against a stub model with known log-probabilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from semmn.eeg import lm

V, H = 20, 6


class BigramStub:
    """Logits at position i depend only on token i, so every context size gives the
    same answer; hidden states are a fixed embedding of the token."""

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(0)
        self.table = torch.randn(V, V, generator=g)
        self.emb = [torch.randn(V, H, generator=g) for _ in range(3)]

    def __call__(self, ids, output_hidden_states=True):
        tok = ids[0]
        return SimpleNamespace(
            logits=self.table[tok][None],
            hidden_states=tuple(e[tok][None] for e in self.emb),
        )


def encode(text: str) -> list[int]:
    word = text.strip()
    n_tokens = 1 + len(word) % 3
    return [(ord(word[0]) + i) % (V - 1) for i in range(n_tokens)]


def test_surprisal_and_embedding_align_to_words():
    stub = BigramStub()
    words = ["once", "upon", "a", "time", "there", "lived", "an", "old", "fisherman"] * 3
    bos = V - 1
    s, e = lm.gpt2_word_features(words, stub, encode, layer=2, bos=bos, max_context=5, stride=3)
    assert s.shape == (len(words),) and e.shape == (len(words), H)

    logp = torch.log_softmax(stub.table.double(), dim=-1).numpy()
    prev = bos
    for k, w in enumerate(words):
        ids = encode(" " + w)
        expected = 0.0
        for t in ids:
            expected -= logp[prev, t] / np.log(2)
            prev = t
        assert abs(s[k] - expected) < 1e-9
        np.testing.assert_allclose(e[k], stub.emb[2][ids[-1]].double().numpy())

    # The block size changes nothing for a model whose context does not matter.
    s2, e2 = lm.gpt2_word_features(words, stub, encode, layer=2, bos=bos, max_context=1024, stride=512)
    np.testing.assert_allclose(s, s2)
    np.testing.assert_allclose(e, e2)


def test_pca_scores_are_unit_variance_and_fit_on_train_only():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((300, 10)) @ rng.standard_normal((10, 10))
    pca = lm.Pca.fit(train, 4)
    z = pca.transform(train)
    assert z.shape == (300, 4)
    np.testing.assert_allclose(z.std(0, ddof=1), 1.0, rtol=1e-6)
    assert pca.transform(rng.standard_normal((5, 10))).shape == (5, 4)


def test_ngram_surprisal_is_leave_run_out():
    runs = {1: ["the", "old", "man"], 2: ["the", "old", "sea"], 3: ["the", "old", "boy"]}
    s = lm.ngram_surprisal(runs)
    assert all(len(s[r]) == 3 for r in runs)
    # "old" after "the" is predictable from the other runs; "man" is never seen elsewhere.
    assert s[1][1] < s[1][2]
    # A word unseen in the other runs is maximally surprising, whatever its own run says.
    assert s[1][2] == s[2][2] == s[3][2]


def test_safetensors_reader_roundtrip(tmp_path):
    import json
    import struct

    arr = np.arange(12, dtype="<f4").reshape(3, 4)
    header = json.dumps({"t": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, arr.nbytes]}}).encode()
    p = tmp_path / "m.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + arr.tobytes())
    np.testing.assert_array_equal(lm.read_safetensors(p, "t"), arr)


def test_wordpiece_and_static_vectors():
    vocab = {"[UNK]": 0, "fish": 1, "##er": 2, "##man": 3, "old": 4, "sea": 5}
    assert lm.wordpiece("FISHERMAN", vocab) == [1, 2, 3]
    assert lm.wordpiece("Old,", vocab) == [4]
    assert lm.wordpiece("xyz", vocab) == [0]
    table = np.eye(6)
    v = lm.static_word_vectors(["fisherman", "sea"], table, vocab)
    np.testing.assert_allclose(v[0], [0, 1 / 3, 1 / 3, 1 / 3, 0, 0])


def test_semantic_dissimilarity_stays_finite_when_correlation_is_undefined():
    flat = np.ones((3, 4))
    d = lm.semantic_dissimilarity(flat, np.array([0, 0, 0]))
    assert d[0] == 0.0 and np.isfinite(d).all()
    assert d[1] == 0.0  # two constant vectors: same shape, not a NaN
    mixed = np.vstack([np.ones(4), np.arange(4, dtype=float)])
    d2 = lm.semantic_dissimilarity(mixed, np.array([0, 0]))
    assert d2[1] == 1.0  # a varying word against a constant context


def test_load_gpt2_reads_the_local_cache_unless_download_is_requested(monkeypatch):
    import sys
    import types

    calls = []

    class _Tok:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False):
            calls.append(("tok", name, local_files_only))
            return cls()

    class _Mod:
        @classmethod
        def from_pretrained(cls, name, local_files_only=False):
            calls.append(("model", name, local_files_only))
            return cls()

        def eval(self):
            return self

    fake = types.ModuleType("transformers")
    fake.GPT2TokenizerFast = _Tok
    fake.GPT2LMHeadModel = _Mod
    monkeypatch.setitem(sys.modules, "transformers", fake)

    lm.load_gpt2("gpt2")
    lm.load_gpt2("gpt2", local_files_only=False)
    assert calls == [
        ("tok", "gpt2", True),
        ("model", "gpt2", True),
        ("tok", "gpt2", False),
        ("model", "gpt2", False),
    ]


def test_semantic_dissimilarity_uses_sentence_context():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(50)
    vecs = np.stack([base, base + 0.01 * rng.standard_normal(50), rng.standard_normal(50),
                     base + 0.01 * rng.standard_normal(50)])
    d = lm.semantic_dissimilarity(vecs, np.array([0, 0, 0, 1]))
    assert d[0] == 0.0
    assert d[1] < 0.01  # same as its context
    assert d[2] > 0.5  # unrelated to its context
    # Word 3 opens sentence 1, so it is compared with the mean of sentence 0.
    expected = 1 - np.corrcoef(vecs[3], vecs[:3].mean(0))[0, 1]
    assert abs(d[3] - expected) < 1e-12
