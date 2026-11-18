"""BioSemi-128 to 10-20 mapping and starter montages."""

from __future__ import annotations

import numpy as np
import pytest

from semmn.eeg import montages as mt


def test_known_biosemi_equivalents():
    for ten20, biosemi in {"Cz": "A1", "Pz": "A19", "Oz": "A23", "Fz": "C21", "C3": "D19", "C4": "B22"}.items():
        assert mt.BIOSEMI_1020[ten20] == biosemi
    assert mt.biosemi_index("A1") == 0 and mt.biosemi_index("D32") == 127


def test_starter_montages_are_distinct_and_in_range():
    for name, chans in mt.STARTER_MONTAGES.items():
        idx = mt.montage_indices(name)
        assert len(set(idx)) == len(chans)
        assert idx.min() >= 0 and idx.max() < 128
        assert set(mt.N400_ROI_1020[name]) <= set(chans)
    assert len(mt.montage_indices("cyton8")) == 8
    assert len(mt.montage_indices("cyton_daisy16")) == 16
    assert set(mt.STARTER_MONTAGES["cyton8"]) < set(mt.STARTER_MONTAGES["cyton_daisy16"])


def test_mapping_regenerates_from_mne_positions():
    mne = pytest.importorskip("mne")
    pos = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    assert mt.nearest_biosemi(pos) == mt.BIOSEMI_1020
    # Left/right pairs mirror across the midline.
    for left, right in [("C3", "C4"), ("P7", "P8"), ("O1", "O2"), ("Fp1", "Fp2"), ("PO7", "PO8")]:
        a, b = pos[mt.BIOSEMI_1020[left]], pos[mt.BIOSEMI_1020[right]]
        np.testing.assert_allclose(a * [-1, 1, 1], b, atol=1e-6)
    roi = mt.n400_roi_128(pos)
    assert mt.biosemi_index("A1") in roi and mt.biosemi_index("A19") in roi
    assert mt.biosemi_index("A23") not in roi  # Oz is outside the centro-parietal region
    assert mt.scalp_region(pos["A23"]) == "occipital-mid"
    assert mt.scalp_region(pos["C21"]) == "frontal-mid"
    assert mt.scalp_region(pos["D23"]) == "temporal-L"
