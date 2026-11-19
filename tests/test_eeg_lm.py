"""GPT-2 word-feature alignment against a stub model with known log-probabilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
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
    if text == ".":
        return [V - 2]
    word = text.strip()
    n_tokens = 1 + len(word) % 3
    return [(ord(word[0]) + i) % (V - 2) for i in range(n_tokens)]


def test_surprisal_and_embedding_align_to_words_and_skip_periods():
    stub = BigramStub()
    words = ["ONCE", "UPON", "TIME", "THERE", "LIVED", "OLD", "FISHERMAN"]
    final = [False, False, True, False, False, False, True]
    bos = V - 1
    s, e = lm.gpt2_word_features(words, stub, encode, layer=2, period_after=final, bos=bos)
    assert s.shape == (len(words),) and e.shape == (len(words), H)

    logp = torch.log_softmax(stub.table.double(), dim=-1).numpy()
    prev = bos
    for k, w in enumerate(words):
        ids = encode(" " + w.lower())
        expected = 0.0
        for t in ids:
            expected -= logp[prev, t] / np.log(2)
            prev = t
        assert abs(s[k] - expected) < 1e-9
        np.testing.assert_allclose(e[k], stub.emb[2][ids[-1]].double().numpy())
        if final[k]:
            prev = V - 2  # the period is context, never charged to a word


def test_run_units_and_sentence_final():
    units, owner = lm.run_units(["A", "B", "C"], encode, [False, True, False])
    assert owner == [0, 1, -1, 2] and units[2] == [V - 2]
    np.testing.assert_array_equal(lm.sentence_final(np.array([0, 0, 1, 1, 2])), [False, True, False, True, False])


def test_run_longer_than_the_context_raises():
    with pytest.raises(ValueError, match="context"):
        lm.gpt2_word_features(["w"] * 10, BigramStub(), encode, layer=1, bos=V - 1, max_context=5)


def test_no_lookahead_with_a_real_causal_model():
    """Changing later words leaves earlier words' features unchanged."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=V, n_positions=64, n_embd=16, n_layer=2, n_head=2,
                                       bos_token_id=V - 1, eos_token_id=V - 1)).eval()
    a = ["old", "man", "fished", "alone", "skiff"]
    b = a[:3] + ["quietly", "boat"]
    sa, ea = lm.gpt2_word_features(a, model, encode, layer=1, bos=V - 1)
    sb, eb = lm.gpt2_word_features(b, model, encode, layer=1, bos=V - 1)
    np.testing.assert_allclose(sa[:3], sb[:3], atol=1e-6)
    np.testing.assert_allclose(ea[:3], eb[:3], atol=1e-5)
    assert not np.allclose(sa[3:], sb[3:])


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
