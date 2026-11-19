"""GPT-2-plausible distractor word sequences for candidate ranking.

For a held-out EEG window, the true candidate is the words actually heard in it. A
distractor keeps the same word slots (same onsets, same acoustics, same sentence-final
periods) but replaces each word with one sampled from GPT-2 given the run's true text
before the window plus the distractor's own earlier words. Each distractor word then gets
the same features as a true word: its surprisal under GPT-2's full (unmasked)
next-token distribution, summed over its tokens, and the layer-``layer`` state at its
last token.

Because the release lists content words only, sampled words are restricted to
lower-case alphabetic words outside a function-word list (the words the release itself
removed). The first token of a word is drawn from the nucleus (top-p) of the masked
distribution; the word then continues with a no-space alphabetic token with the
probability GPT-2 gives such tokens, or ends.

All sequences of a window advance in lockstep, one token per step, sharing one cached
pass over the prefix. A sequence with nothing to emit at a step feeds a masked pad token
(attention mask 0, position unchanged), which leaves its own computation untouched.
``forced`` scores given token sequences through the same code, so the true candidate
can be checked against the one-pass features.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Function words the release removed from its word lists. ``content_masks`` also drops any
# of these that the story's own word list does contain, so the filter matches the release.
FUNCTION_WORDS = frozenset(
    """a an the and or but nor so yet if then than that this these those there here
    of to in on at by for with from into onto upon about as off over under
    i me my mine we us our ours you your yours he him his she her hers it its they them their theirs
    who whom whose which what where when why how
    is am are was were be been being have has had do does did
    not no can shall will would should could may might must
    """.split()
)


def content_masks(token_strings: Sequence[str], function_words: frozenset[str] | set[str]) -> tuple[np.ndarray, np.ndarray]:
    """(start, cont) boolean masks over the vocabulary.

    ``start``: tokens that begin a new lower-case alphabetic word (a leading space then
    letters) that is not a function word. ``cont``: tokens that continue a word (letters
    only, no leading space).
    """
    start = np.zeros(len(token_strings), dtype=bool)
    cont = np.zeros(len(token_strings), dtype=bool)
    for i, s in enumerate(token_strings):
        if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and s[1:].islower():
            start[i] = s[1:] not in function_words
        elif s.isascii() and s.isalpha() and s.islower():
            cont[i] = True
    return start, cont


def _nucleus_sample(p, top_p: float, generator):
    """One draw per row of the probability matrix ``p`` (k, V) from its top-p nucleus."""
    import torch

    sorted_p, order = torch.sort(p, dim=-1, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    keep = (cum - sorted_p) < top_p * cum[:, -1:]  # always keeps the most likely token
    sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
    pick = torch.multinomial(sorted_p, 1, generator=generator)
    return order.gather(1, pick)[:, 0]


def sample_window(
    model,
    prefix: Sequence[int],
    period_after: Sequence[bool],
    period_ids: Sequence[int],
    layer: int,
    start_mask: np.ndarray | None = None,
    cont_mask: np.ndarray | None = None,
    k: int = 1,
    seed: int = 0,
    top_p: float = 0.9,
    max_cont: int = 3,
    forced: Sequence[Sequence[int]] | None = None,
) -> dict:
    """Sample (or, with ``forced``, score) ``k`` word sequences after ``prefix``.

    ``period_after[i]`` appends ``period_ids`` after slot i as context. Returns
    ``surprisal`` (k, n) in bits, ``hidden`` (k, n, H) and ``tokens`` (k lists of n
    token lists). ``forced`` is one token list per slot and implies k = 1.
    """
    import torch

    n = len(period_after)
    if forced is not None:
        if len(forced) != n:
            raise ValueError("one forced token list per slot is required")
        k = 1
    elif start_mask is None or cont_mask is None:
        raise ValueError("sampling needs start and continuation masks")
    gen = torch.Generator().manual_seed(int(seed))
    ln2 = float(np.log(2))
    rows = torch.arange(k)
    with torch.no_grad():
        out = model(torch.tensor([list(prefix)]), use_cache=True, output_hidden_states=True)
        cache = out.past_key_values
        if k > 1:
            cache.batch_repeat_interleave(k)
        logp = torch.log_softmax(out.logits[0, -1].double(), dim=-1).expand(k, -1).clone()
        attn = torch.ones(k, len(prefix), dtype=torch.long)
        pos = torch.full((k,), len(prefix), dtype=torch.long)
        smask = None if start_mask is None else torch.as_tensor(start_mask)
        cmask = None if cont_mask is None else torch.as_tensor(cont_mask)

        def feed(tok, act):
            nonlocal attn, pos, logp
            attn = torch.cat([attn, act.long()[:, None]], dim=1)
            o = model(tok[:, None], past_key_values=cache, attention_mask=attn,
                      position_ids=pos[:, None], use_cache=True, output_hidden_states=True)
            new = torch.log_softmax(o.logits[:, -1].double(), dim=-1)
            logp = torch.where(act[:, None], new, logp)
            pos = pos + act.long()
            return o.hidden_states[layer][:, -1].double()

        surprisal = torch.zeros(k, n, dtype=torch.float64)
        hidden: torch.Tensor | None = None
        tokens: list[list[list[int]]] = [[[] for _ in range(n)] for _ in range(k)]
        everyone = torch.ones(k, dtype=torch.bool)
        for i in range(n):
            if forced is not None:
                tok = torch.tensor([forced[i][0]])
            else:
                tok = _nucleus_sample(logp.exp() * smask, top_p, gen)
            surprisal[:, i] -= logp[rows, tok] / ln2
            last = feed(tok, everyone)
            for b in range(k):
                tokens[b][i].append(int(tok[b]))
            active = everyone.clone()
            j = 1
            while True:
                if forced is not None:
                    go = torch.tensor([j < len(forced[i])])
                    nxt = torch.tensor([forced[i][j] if j < len(forced[i]) else 0])
                else:
                    p = logp.exp() * cmask
                    mass = p.sum(dim=1)
                    go = active & (j <= max_cont) & (torch.rand(k, generator=gen, dtype=torch.float64) < mass)
                    nxt = torch.multinomial(torch.where(go[:, None], p, torch.ones_like(p)), 1, generator=gen)[:, 0]
                go = go & active
                if not bool(go.any()):
                    break
                nxt = torch.where(go, nxt, torch.zeros_like(nxt))
                surprisal[go, i] -= logp[rows[go], nxt[go]] / ln2
                h = feed(nxt, go)
                last = torch.where(go[:, None], h, last)
                for b in torch.nonzero(go)[:, 0].tolist():
                    tokens[b][i].append(int(nxt[b]))
                active = go
                j += 1
            if hidden is None:
                hidden = torch.zeros(k, n, last.shape[1], dtype=torch.float64)
            hidden[:, i] = last
            if period_after[i]:
                for t in period_ids:
                    feed(torch.full((k,), int(t), dtype=torch.long), everyone)
    if hidden is None:
        hidden = torch.zeros(k, 0, 0, dtype=torch.float64)
    return {"surprisal": surprisal.numpy(), "hidden": hidden.numpy(), "tokens": tokens}
