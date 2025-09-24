import os
import numpy as np
import soundfile as sf
import librosa


def load_audio(path: str, sr: int | None = None, mono: bool | None = None):
    """Load audio using librosa; returns (y, sr).
    If mono is None, keep native shape.
    """
    if mono is None:
        # librosa can't return native multichannel without mono=True/False;
        # load stereo with mono=False
        y, srr = librosa.load(path, sr=sr, mono=False)
    else:
        y, srr = librosa.load(path, sr=sr, mono=mono)
    return y, srr


def save_audio(path: str, y: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.T if y.ndim == 2 else y, sr)


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    return librosa.to_mono(y)


def ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.vstack([y, y])
    if y.shape[0] == 2:
        return y
    # If more channels, keep first two
    return y[:2]


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    if y.ndim == 1:
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    # channel-wise
    return np.vstack(
        [
            librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr)
            for ch in y
        ]
    )
