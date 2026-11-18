"""Loader for the Broderick et al. (2018) natural-speech EEG dataset.

Dataset: Broderick, Anderson, Di Liberto, Crosse & Lalor (2018), "Electrophysiological
correlates of semantic dissimilarity reflect the comprehension of natural, narrative
speech", Current Biology. Dryad doi:10.5061/dryad.070jc (CC0).

Expected layout under a condition directory (for example ``Natural Speech/``)::

    EEG/Subject<s>/Subject<s>_Run<r>.mat     eegData (n x 128), mastoids (n x 2), fs
    Stimuli/Envelopes/audio<r>_128Hz.mat     env (n,), fs
    Stimuli/Text/Run<r>.mat                  wordVec, onset_time, offset_time (seconds)

Each EEG file starts at the onset of its run's audio, so word onsets (seconds from the
start of the audio) index the EEG directly.

Preprocessing (``preprocess_eeg``): re-reference to the mastoid mean, zero-phase
Butterworth band-pass (1-8 Hz by default, the band used for semantic TRFs in the source
paper), polyphase resampling to the model rate (64 Hz by default), and z-scoring per
channel within the run. The envelope gets the same band-pass and resampling
(``preprocess_envelope``), so stimulus and response see the same filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import io as sio
from scipy import signal

EEG_KEY = "eegData"
MASTOID_KEY = "mastoids"
FS_KEY = "fs"
ENV_KEY = "env"
WORDS_KEY = "wordVec"
ONSET_KEY = "onset_time"
OFFSET_KEY = "offset_time"

N_CHANNELS = 128

_EEG_RE = re.compile(r"Subject(\d+)_Run(\d+)\.mat$")
_TEXT_RE = re.compile(r"Run(\d+)\.mat$")
_ENV_RE = re.compile(r"audio(\d+)_\d+Hz\.mat$")


@dataclass(frozen=True)
class Words:
    """Words of one run with onset and offset times in seconds from the run start."""

    words: tuple[str, ...]
    onsets: np.ndarray
    offsets: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.words) == len(self.onsets) == len(self.offsets)):
            raise ValueError("words, onsets and offsets must have the same length")
        if np.any(np.diff(self.onsets) < 0):
            raise ValueError("word onsets must be non-decreasing")

    def __len__(self) -> int:
        return len(self.words)

    def time_reversed(self, duration: float) -> "Words":
        """The same words as heard in the time-reversed audio of length ``duration`` s.

        A word spanning [on, off] in the forward audio spans [duration - off,
        duration - on] when the audio is played backwards. The words keep their
        identity (and so their language-model features) but appear in reverse order.
        """
        on = duration - self.offsets[::-1]
        off = duration - self.onsets[::-1]
        return Words(tuple(reversed(self.words)), np.asarray(on, float), np.asarray(off, float))


def load_mat(path: str | Path) -> dict:
    """Load a MATLAB v5 file; fail clearly on v7.3 (HDF5), which needs h5py."""
    path = Path(path)
    with path.open("rb") as fh:
        head = fh.read(128)
    if b"MATLAB 7.3" in head:
        raise ValueError(f"{path.name} is a MATLAB v7.3 (HDF5) file; reading it needs h5py")
    return sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def load_eeg_run(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (eeg (n x 128), mastoids (n x 2), fs) for one subject and run."""
    m = load_mat(path)
    eeg = np.asarray(m[EEG_KEY], dtype=np.float64)
    mast = np.asarray(m[MASTOID_KEY], dtype=np.float64)
    if eeg.ndim != 2 or mast.ndim != 2:
        raise ValueError(f"{Path(path).name}: expected 2-D eegData and mastoids")
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T
    if mast.shape[0] < mast.shape[1]:
        mast = mast.T
    if mast.shape[0] != eeg.shape[0]:
        raise ValueError(f"{Path(path).name}: eegData and mastoids differ in length")
    return eeg, mast, _scalar(m[FS_KEY])


def load_words(path: str | Path) -> Words:
    """Return the words of one run with their onset and offset times (seconds)."""
    m = load_mat(path)
    raw = np.atleast_1d(m[WORDS_KEY])
    words = tuple(str(np.asarray(w).reshape(-1)[0]) if np.ndim(w) else str(w) for w in raw)
    on = np.atleast_1d(np.asarray(m[ONSET_KEY], dtype=float)).reshape(-1)
    off = np.atleast_1d(np.asarray(m[OFFSET_KEY], dtype=float)).reshape(-1)
    return Words(words, on, off)


def load_envelope(path: str | Path) -> tuple[np.ndarray, float]:
    """Return (envelope (n,), fs) for one run."""
    m = load_mat(path)
    return np.asarray(m[ENV_KEY], dtype=np.float64).reshape(-1), _scalar(m[FS_KEY])


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0. ``band[0] <= 0`` gives a low-pass."""
    lo, hi = band
    if lo > 0:
        sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    else:
        sos = signal.butter(order, hi, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def resample(x: np.ndarray, fs: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling along axis 0 from ``fs`` to ``fs_out``."""
    if fs == fs_out:
        return x
    ratio = Fraction(fs_out / fs).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    sd = x.std(axis=axis, keepdims=True)
    return (x - x.mean(axis=axis, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def preprocess_eeg(
    eeg: np.ndarray,
    mastoids: np.ndarray,
    fs: float,
    band: tuple[float, float] = (1.0, 8.0),
    fs_out: float = 64.0,
    standardise: bool = True,
) -> np.ndarray:
    """Mastoid reference, band-pass, resample, z-score per channel. Returns float32."""
    x = eeg - mastoids.mean(axis=1, keepdims=True)
    x = bandpass(x, fs, band)
    x = resample(x, fs, fs_out)
    if standardise:
        x = zscore(x)
    return x.astype(np.float32)


def preprocess_envelope(
    env: np.ndarray, fs: float, band: tuple[float, float] = (1.0, 8.0), fs_out: float = 64.0
) -> np.ndarray:
    """Same band-pass and resampling as the EEG, then z-score. Returns float32 (n,)."""
    x = bandpass(env.reshape(-1, 1), fs, band)
    x = resample(x, fs, fs_out)
    return zscore(x)[:, 0].astype(np.float32)


def onset_samples(times: np.ndarray, fs: float, n: int) -> np.ndarray:
    """Seconds to sample indices at ``fs``; drops events at or beyond ``n`` samples."""
    idx = np.round(np.asarray(times) * fs).astype(int)
    return idx[(idx >= 0) & (idx < n)]


def discover_eeg(condition_dir: str | Path) -> dict[int, dict[int, Path]]:
    """Map subject -> run -> EEG file path under ``condition_dir``."""
    out: dict[int, dict[int, Path]] = {}
    for p in sorted(Path(condition_dir).rglob("*.mat")):
        m = _EEG_RE.search(p.name)
        if m:
            out.setdefault(int(m.group(1)), {})[int(m.group(2))] = p
    return out


def discover_stimuli(condition_dir: str | Path) -> tuple[dict[int, Path], dict[int, Path]]:
    """Return (run -> text file, run -> envelope file) under ``condition_dir``."""
    text: dict[int, Path] = {}
    env: dict[int, Path] = {}
    for p in sorted(Path(condition_dir).rglob("*.mat")):
        if _EEG_RE.search(p.name):
            continue
        m = _ENV_RE.search(p.name)
        if m:
            env[int(m.group(1))] = p
            continue
        m = _TEXT_RE.search(p.name)
        if m:
            text[int(m.group(1))] = p
    return text, env
