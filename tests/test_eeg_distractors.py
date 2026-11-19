"""GPT-2 distractor sampling: lockstep batching must not change any sequence's features."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from semmn.eeg import distractors as ds
from semmn.eeg import lm

transformers = pytest.importorskip("transformers")

V = 48
PERIOD, BOS = V - 2, V - 1
# Toy vocabulary: 14 word starts, 14 continuations, 4 function words, 3 junk tokens,
# 11 capitalised tokens (neither mask), then the period and BOS.
STRINGS = ([f" k{chr(97 + i)}" for i in range(14)] + [f"q{chr(97 + i)}" for i in range(14)]
           + [" the", " and", " he", " was"] + ["1", ",", " 42"] + [f" Z{chr(97 + i)}" for i in range(11)]
           + [".", "<bos>"])


def tiny_gpt2(seed: int = 0):
    torch.manual_seed(seed)
    cfg = transformers.GPT2Config(vocab_size=V, n_positions=128, n_embd=16, n_layer=2, n_head=2,
                                  bos_token_id=BOS, eos_token_id=BOS)
    return transformers.GPT2LMHeadModel(cfg).eval()


def encode(text: str) -> list[int]:
    """Toy tokenizer: a word is one start token plus len(word) % 3 continuation tokens."""
    if text == ".":
        return [PERIOD]
    w = text.strip()
    return [ord(w[0]) % 14] + [14 + (ord(w[0]) + i) % 14 for i in range(len(w) % 3)]


def masks():
    assert len(STRINGS) == V
    return ds.content_masks(STRINGS, {"the", "and", "he", "was"})


def test_content_masks():
    start, cont = masks()
    assert start[:14].all() and not start[14:28].any()
    assert cont[14:28].all() and not cont[:14].any()
    assert not start[28:35].any() and not cont[28:35].any()  # function words, digits, punctuation
    assert not start[PERIOD] and not cont[PERIOD]


def test_forced_path_equals_the_one_pass_features():
    model = tiny_gpt2()
    words = ["old", "man", "fished", "alone", "in", "skiff", "gulf", "stream"]
    final = [False, False, False, True, False, False, True, False]
    s1, e1 = lm.gpt2_word_features(words, model, encode, layer=1, period_after=final, bos=BOS)
    units, owner = lm.run_units(words, encode, final)
    first = {o: u for u, o in enumerate(owner) if o >= 0}
    for lo in (0, 3):  # whole run, and a window after a prefix that ends on a period
        prefix = [BOS] + [t for u in units[: first[lo]] for t in u]
        f = ds.sample_window(model, prefix, final[lo:], [PERIOD], layer=1,
                             forced=[units[first[q]] for q in range(lo, len(words))])
        np.testing.assert_allclose(f["surprisal"][0], s1[lo:], atol=1e-5)
        np.testing.assert_allclose(f["hidden"][0], e1[lo:], atol=1e-5)


def test_sampled_sequences_are_scored_as_if_alone():
    """Ragged sequences advance in lockstep with masked pads; each must get the features
    it would get on its own, and obey the content-word masks."""
    model = tiny_gpt2(1)
    start, cont = masks()
    prefix = [BOS, 3, 17, 5, PERIOD, 9]
    flags = [False, True, False, False]
    g = ds.sample_window(model, prefix, flags, [PERIOD], layer=2, start_mask=start, cont_mask=cont,
                         k=12, seed=3, top_p=0.95)
    assert g["surprisal"].shape == (12, 4) and g["hidden"].shape == (12, 4, 16)
    lengths = {len(t) for seq in g["tokens"] for t in seq}
    assert len(lengths) > 1, "the test needs words of different token lengths"
    for b, seq in enumerate(g["tokens"]):
        for word in seq:
            assert start[word[0]] and all(cont[t] for t in word[1:]) and len(word) <= 4
        f = ds.sample_window(model, prefix, flags, [PERIOD], layer=2, forced=seq)
        np.testing.assert_allclose(g["surprisal"][b], f["surprisal"][0], atol=1e-5)
        np.testing.assert_allclose(g["hidden"][b], f["hidden"][0], atol=1e-5)
    again = ds.sample_window(model, prefix, flags, [PERIOD], layer=2, start_mask=start, cont_mask=cont,
                             k=12, seed=3, top_p=0.95)
    assert again["tokens"] == g["tokens"]


def test_nucleus_keeps_the_top_mass_only():
    p = torch.tensor([[0.5, 0.3, 0.15, 0.05]] * 2000, dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    draws = ds._nucleus_sample(p, 0.7, gen)
    assert set(draws.tolist()) == {0, 1}  # 0.5 + 0.3 reaches 0.7; 0.15 and 0.05 are cut


def test_context_between_slots_matches_the_one_pass_full_stream():
    """Full transcript: slots are content words, the function words between them are
    forced context; the forced path must reproduce the one-pass features."""
    model = tiny_gpt2(2)
    full = ["the", "old", "man", "and", "the", "sea", "were", "alone", "in", "skiff"]
    flags = [False, False, False, False, False, True, False, False, False, False]
    content = [1, 2, 5, 7, 9]  # positions of the content words in the full stream
    s1, e1 = lm.gpt2_word_features(full, model, encode, layer=1, period_after=flags, bos=BOS)
    units, owner = lm.run_units(full, encode, flags)
    first = {o: u for u, o in enumerate(owner) if o >= 0}
    u = [first[c] for c in content]
    prefix = [BOS] + [t for x in units[: u[0]] for t in x]
    after = [[t for x in units[u[i] + 1 : u[i + 1]] for t in x] for i in range(len(u) - 1)] + [[]]
    f = ds.sample_window(model, prefix, [False] * len(u), [PERIOD], layer=1,
                         forced=[units[i] for i in u], context_after=after)
    np.testing.assert_allclose(f["surprisal"][0], s1[content], atol=1e-5)
    np.testing.assert_allclose(f["hidden"][0], e1[content], atol=1e-5)
    start, cont = masks()
    g = ds.sample_window(model, prefix, [False] * len(u), [PERIOD], layer=1, start_mask=start,
                         cont_mask=cont, k=6, seed=1, context_after=after)
    for b, seq in enumerate(g["tokens"]):
        h = ds.sample_window(model, prefix, [False] * len(u), [PERIOD], layer=1, forced=seq, context_after=after)
        np.testing.assert_allclose(g["surprisal"][b], h["surprisal"][0], atol=1e-5)
