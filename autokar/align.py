from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import librosa
from librosa.sequence import dtw


def _mels(y, sr, n_mels=80):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S).T  # (T, n_mels)


def _tts_pyttsx3(text: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Very simple offline TTS to build an acoustic proxy for DTW.
    Quality varies by OS. If unavailable, raise to let caller skip DTW.
    """
    try:
        import tempfile
        import os
        import pyttsx3

        tmpwav = tempfile.mktemp(suffix=".wav")
        engine = pyttsx3.init()
        engine.save_to_file(text, tmpwav)
        engine.runAndWait()
        y, srr = librosa.load(tmpwav, sr=sr, mono=True)
        os.remove(tmpwav)
        return y, srr
    except Exception as e:
        raise RuntimeError("pyttsx3 TTS path unavailable.") from e


def pass_through_alignment(
    words: List[Dict[str, Any]],
) -> List[Tuple[float, float, str]]:
    """Use existing ASR word timestamps if provided."""
    aligned = []
    for w in words:
        s = float(w.get("start", 0.0))
        e = float(w.get("end", s))
        t = str(w.get("word", "")).strip()
        if t:
            aligned.append((s, e, t))
    return aligned


def dtw_refine(
    vocals: np.ndarray,
    sr: int,
    transcript_text: str,
    chunk_sec: float = 10.0,
) -> List[Tuple[float, float, str]]:
    """Crude DTW alignment using TTS proxy.
    Splits text into chunks by punctuation/newlines.
    """
    import re

    # Prepare chunks
    parts = [
        p.strip()
        for p in re.split(r"[\n\.,;:!?]+", transcript_text)
        if p.strip()
    ]
    if not parts:
        return [(0.0, 0.0, transcript_text)]

    # Feature for full vocals once
    A = _mels(vocals, sr)
    hop = 512
    frame_dur = hop / sr  # approx; acceptable for demo

    out: List[Tuple[float, float, str]] = []
    cursor_frame = 0

    for part in parts:
        try:
            tts, srr = _tts_pyttsx3(part, sr=sr)
        except RuntimeError:
            # fallback: naive duration proportional to text length
            dur = max(0.3 * len(part.split()), 0.5)
            start = cursor_frame * frame_dur
            end = start + dur
            out.append((start, end, part))
            cursor_frame += int(dur / frame_dur)
            continue

        B = _mels(tts, sr)
        # local window of A around cursor to keep path reasonable
        a_start = max(0, cursor_frame - int(3.0 / frame_dur))
        a_end = min(A.shape[0], a_start + int(20.0 / frame_dur) + B.shape[0])
        Awin = A[a_start:a_end]
        if Awin.shape[0] < 2 or B.shape[0] < 2:
            continue
        D, wp = dtw(Awin.T, B.T, metric="euclidean", subseq=True)
        # map B start/end to A frames
        a_idx_start = a_start + int(wp[-1, 0])
        a_idx_end = a_start + int(wp[0, 0])
        if a_idx_end < a_idx_start:
            a_idx_start, a_idx_end = a_idx_end, a_idx_start
        start = a_idx_start * frame_dur
        end = max(start + 0.2, a_idx_end * frame_dur)
        out.append((start, end, part))
        cursor_frame = a_idx_end
    return out
