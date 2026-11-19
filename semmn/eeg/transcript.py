"""Full-transcript word timings from Praat TextGrids, aligned to the release's content words.

The Broderick release lists content words only. A forced-alignment TextGrid of the same
audio has every word with its start and end time. ``read_textgrid`` parses Praat's long
and short text formats (UTF-8 or UTF-16) without executing anything from the file;
``word_intervals`` keeps the word tier minus silence labels; ``align_content`` maps each
content word to its position in the full word sequence by sequence matching on the
normalised spelling, then reports how well the two timelines agree.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SILENCE = frozenset({"", "sp", "sil", "<sil>", "spn", "<unk>", "{sl}", "{lg}", "{ns}"})
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


@dataclass(frozen=True)
class Tier:
    name: str
    kind: str
    intervals: tuple[tuple[float, float, str], ...]


def _decode(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def _values(text: str) -> list[str | float]:
    """Praat's values in file order: the right-hand side of ``key = value`` lines (long
    format) or bare lines (short format). Quoted strings are unquoted ("" -> ")."""
    out: list[str | float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line and not line.startswith('"'):
            line = line.split("=", 1)[1].strip()
        elif line.endswith(":") or line.startswith("item") or line.startswith("intervals ["):
            continue
        if line in ("<exists>", "<absent>") or line.startswith("tiers?"):
            continue
        if line.startswith('"'):
            end = line.rfind('"')
            if end <= 0:
                raise ValueError(f"unterminated string: {line[:40]}")
            out.append(line[1:end].replace('""', '"'))
        elif _NUMBER.match(line):
            out.append(float(line))
        else:
            raise ValueError(f"unexpected TextGrid line: {line[:40]}")
    return out


def read_textgrid(path: str | Path) -> list[Tier]:
    """Tiers of a Praat TextGrid in long or short text format."""
    vals = _values(_decode(Path(path).read_bytes()))
    if len(vals) < 5 or vals[0] != "ooTextFile" or vals[1] != "TextGrid":
        raise ValueError(f"{Path(path).name} is not a Praat TextGrid text file")
    i = 4  # file type, object class, xmin, xmax
    n_tiers = int(vals[i])
    i += 1
    tiers = []
    for _ in range(n_tiers):
        kind, name = str(vals[i]), str(vals[i + 1])
        n = int(vals[i + 4])
        i += 5
        items = []
        if kind == "IntervalTier":
            for _ in range(n):
                items.append((float(vals[i]), float(vals[i + 1]), str(vals[i + 2])))
                i += 3
        elif kind == "TextTier":
            for _ in range(n):
                items.append((float(vals[i]), float(vals[i]), str(vals[i + 1])))
                i += 2
        else:
            raise ValueError(f"unknown tier class {kind}")
        tiers.append(Tier(name, kind, tuple(items)))
    return tiers


def word_intervals(tiers: list[Tier]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(words, onsets, offsets) of the word tier, silences dropped, in time order."""
    cands = [t for t in tiers if t.kind == "IntervalTier" and t.name.lower() in ("words", "word")]
    cands = cands or [t for t in tiers if t.kind == "IntervalTier" and "word" in t.name.lower()]
    if len(cands) != 1:
        raise ValueError(f"expected one word tier, found {[t.name for t in tiers]}")
    keep = [(a, b, w.strip()) for a, b, w in cands[0].intervals if w.strip().lower() not in SILENCE]
    keep.sort(key=lambda x: x[0])
    return [w for _, _, w in keep], np.array([a for a, _, _ in keep]), np.array([b for _, b, _ in keep])


def normalise(w: str) -> str:
    return "".join(ch for ch in w.lower() if ch.isalnum() or ch == "'")


@dataclass(frozen=True)
class Alignment:
    """``index[k]``: position of content word k in the full word list, or -1."""

    index: np.ndarray
    match_rate: float
    onset_diff: np.ndarray  # full-word onset minus content-word onset, matched words only


def align_content(content: list[str], content_onsets: np.ndarray, full: list[str],
                  full_onsets: np.ndarray, max_shift: float = 0.5, min_similarity: float = 0.6) -> Alignment:
    """Map content words to the full word sequence, in order.

    1. Identical spellings are matched by longest-common-subsequence blocks, kept only
       when their onsets differ by at most ``max_shift`` seconds.
    2. Each word still unmatched takes the nearest-in-time full word within
       ``max_shift`` that lies between its matched neighbours and is spelled the same
       or similarly (difflib ratio >= ``min_similarity``, e.g. gray/grey).
    A word is never paired with a same-spelled word elsewhere in the run.
    """
    a = [normalise(w) for w in content]
    b = [normalise(w) for w in full]
    con = np.asarray(content_onsets, dtype=float)
    fon = np.asarray(full_onsets, dtype=float)
    index = np.full(len(a), -1, dtype=int)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for i, j in zip(range(blk.a, blk.a + blk.size), range(blk.b, blk.b + blk.size)):
            if abs(fon[j] - con[i]) <= max_shift:
                index[i] = j
    used = set(index[index >= 0].tolist())
    for i in np.nonzero(index < 0)[0]:
        lo = max([index[k] for k in range(i) if index[k] >= 0], default=-1)
        hi = min([index[k] for k in range(i + 1, len(a)) if index[k] >= 0], default=len(b))
        best, best_key = -1, None
        for j in range(lo + 1, hi):
            if j in used or abs(fon[j] - con[i]) > max_shift:
                continue
            same = a[i] == b[j]
            if not same and difflib.SequenceMatcher(None, a[i], b[j]).ratio() < min_similarity:
                continue
            key = (not same, abs(fon[j] - con[i]))
            if best_key is None or key < best_key:
                best, best_key = j, key
        if best >= 0:
            index[i] = best
            used.add(best)
    ok = index >= 0
    diff = fon[index[ok]] - con[ok]
    return Alignment(index, float(ok.mean()) if len(a) else 1.0, diff)


def period_after_full(full_onsets: np.ndarray, sentence_close: np.ndarray) -> np.ndarray:
    """True for the last full word starting before each sentence-close time."""
    out = np.zeros(len(full_onsets), dtype=bool)
    for b in np.atleast_1d(sentence_close):
        before = np.nonzero(full_onsets < b)[0]
        if len(before) and before[-1] < len(full_onsets) - 1:
            out[before[-1]] = True
    return out
