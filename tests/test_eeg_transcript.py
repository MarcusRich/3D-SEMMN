"""TextGrid parsing and content-word alignment for the full-transcript features."""

from __future__ import annotations

import numpy as np
import pytest

from semmn.eeg import transcript as tx

LONG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 3.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 3.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 3.0
            text = "AH0"
    item [2]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 3.0
        intervals: size = 5
        intervals [1]:
            xmin = 0
            xmax = 0.4
            text = "sil"
        intervals [2]:
            xmin = 0.4
            xmax = 0.9
            text = "THE"
        intervals [3]:
            xmin = 0.9
            xmax = 1.5
            text = "OLD"
        intervals [4]:
            xmin = 1.5
            xmax = 2.2
            text = "MAN'S"
        intervals [5]:
            xmin = 2.2
            xmax = 3.0
            text = ""
'''

SHORT = '''File type = "ooTextFile"
Object class = "TextGrid"

0
3.0
<exists>
1
"IntervalTier"
"words"
0
3.0
3
0
1.0
"sp"
1.0
2.0
"SAID ""HI"""
2.0
3.0
"BOY"
'''


def test_long_format(tmp_path):
    p = tmp_path / "a.TextGrid"
    p.write_text(LONG)
    tiers = tx.read_textgrid(p)
    assert [t.name for t in tiers] == ["phones", "words"] and len(tiers[1].intervals) == 5
    words, on, off = tx.word_intervals(tiers)
    assert words == ["THE", "OLD", "MAN'S"]
    np.testing.assert_allclose(on, [0.4, 0.9, 1.5])
    np.testing.assert_allclose(off, [0.9, 1.5, 2.2])


def test_short_format_and_utf16(tmp_path):
    p = tmp_path / "b.TextGrid"
    p.write_bytes(SHORT.encode("utf-16"))
    words, on, _ = tx.word_intervals(tx.read_textgrid(p))
    assert words == ['SAID "HI"', "BOY"] and list(on) == [1.0, 2.0]


def test_rejects_non_textgrid(tmp_path):
    p = tmp_path / "c.TextGrid"
    p.write_text('File type = "ooTextFile"\nObject class = "Sound"\n0\n1\n1\n')
    with pytest.raises(ValueError, match="not a Praat TextGrid"):
        tx.read_textgrid(p)


def test_alignment_handles_variants_gaps_and_distant_repeats():
    full = ["the", "old", "man", "and", "the", "grey", "sea", "of", "old", "times"]
    fon = np.arange(len(full)) * 0.5
    content = ["OLD", "MAN", "GRAY", "SEA", "MISSING", "TIMES"]
    con = np.array([0.5, 1.0, 2.5, 3.0, 3.6, 4.5])
    al = tx.align_content(content, con, full, fon)
    # GRAY/grey is a same-length replace at the same time; MISSING has no counterpart.
    np.testing.assert_array_equal(al.index, [1, 2, 5, 6, -1, 9])
    assert abs(al.match_rate - 5 / 6) < 1e-12 and np.all(al.onset_diff == 0)
    # A same-spelled word far away in time is not taken as the match.
    far = tx.align_content(["OLD"], np.array([4.0]), full, fon)
    assert far.index[0] == 8 and abs(far.onset_diff[0]) < 0.5
    none = tx.align_content(["OLD"], np.array([10.0]), full, fon)
    assert none.index[0] == -1


def test_period_after_the_last_word_before_each_sentence_close():
    fon = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_array_equal(tx.period_after_full(fon, np.array([1.2, 2.5])), [0, 0, 1, 0, 0])
