from __future__ import annotations
import numpy as np
from typing import Tuple


def separate_ica(
    y_stereo: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple and robust mid/side baseline.
    Returns two mono components: center (approx vocals) and side (approx instrumental).
    If the input is mono or highly correlated stereo, side will be near-silent.
    """
    if y_stereo.ndim != 2 or y_stereo.shape[0] != 2:
        raise ValueError("Expected stereo audio array of shape (2, N)")

    left = y_stereo[0]
    right = y_stereo[1]

    # Mid/Side transform
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    # If channels are almost identical (mono or near-mono), side ~ 0
    # This is expected; true source separation from mono requires a model like Demucs.
    return mid.astype(np.float32), side.astype(np.float32)


def separate_demucs(path_in: str, path_vocals: str, path_other: str) -> bool:
    """Optional: if Demucs is installed, run demucs to get vocals vs rest.
    Returns True if succeeded, else False.
    """
    try:
        import subprocess
        import tempfile
        import shutil
        import glob
        import soundfile as sf

        tmp = tempfile.mkdtemp()
        cmd = [
            "python",
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "-o",
            tmp,
            path_in,
        ]
        subprocess.run(cmd, check=True)
        # find stems
        song_dir = glob.glob(f"{tmp}/*/*")[0]
        vocals = glob.glob(f"{song_dir}/*vocals*.wav")[0]
        other_candidates = [
            p
            for p in glob.glob(f"{song_dir}/*.wav")
            if "vocals" not in p.lower()
        ]
        # mix non-vocals
        v, sr = sf.read(vocals)
        mix = np.zeros_like(v)
        for p in other_candidates:
            x, _ = sf.read(p)
            mix = mix + x[: mix.shape[0]]
        sf.write(path_vocals, v, sr)
        sf.write(path_other, mix, sr)
        shutil.rmtree(tmp)
        return True
    except Exception:
        return False
