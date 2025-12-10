from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import librosa
import os
import tempfile
import soundfile as sf


def _transcribe_faster_whisper(
    y: np.ndarray, sr: int, model_size: str = "small"
) -> Dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "faster-whisper not installed. Install it or change ASR backend."
        ) from e

    # faster-whisper expects 16k mono float32
    mono = y if y.ndim == 1 else librosa.to_mono(y)
    if sr != 16000:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
        sr = 16000

    model = WhisperModel(model_size, device="auto")
    segments, info = model.transcribe(mono, word_timestamps=True)

    segs = []
    words = []
    for s in segments:
        segs.append(
            {"start": float(s.start), "end": float(s.end), "text": s.text}
        )
        if s.words:
            for w in s.words:
                words.append(
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word,
                    }
                )
    text = " ".join([s["text"] for s in segs]).strip()
    return {
        "text": text,
        "segments": segs,
        "words": words,
        "language": getattr(info, "language", None),
    }


def _transcribe_openai_whisper(
    y: np.ndarray, sr: int, model_size: str = "small"
) -> Dict[str, Any]:
    try:
        import whisper
    except Exception as e:
        raise RuntimeError(
            "openai-whisper not installed. Install it or change ASR backend."
        ) from e

    mono = y if y.ndim == 1 else librosa.to_mono(y)
    if sr != 16000:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
        sr = 16000

    # write to a temp wav because whisper API prefers file paths
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, mono.astype(np.float32), sr)

    try:
        model = whisper.load_model(model_size, device="cpu")
        res = model.transcribe(tmp_path, word_timestamps=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    text = res.get("text", "").strip()
    segments = res.get("segments", []) or []
    segs = []
    words = []
    for s in segments:
        segs.append(
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": s.get("text", ""),
            }
        )
        for w in s.get("words", []) or []:
            words.append(
                {
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "word": w.get("word", ""),
                }
            )

    return {
        "text": text,
        "segments": segs,
        "words": words,
        "language": res.get("language"),
    }


def transcribe_and_score(
    y: np.ndarray,
    sr: int,
    backend: str = "faster-whisper",
    model_size: str = "small",
) -> Tuple[Dict[str, Any], float]:
    """Run ASR and return (result, score).
    Score is a heuristic combining WPM and word count.
    """
    if backend == "faster-whisper":
        res = _transcribe_faster_whisper(y, sr, model_size=model_size)
    elif backend == "openai-whisper":
        res = _transcribe_openai_whisper(y, sr, model_size=model_size)
    else:
        raise ValueError(
            "Unsupported ASR backend. Use 'openai-whisper' or 'faster-whisper'."
        )

    words = res.get("words", [])
    if len(words) == 0:
        return res, 0.0
    duration = words[-1]["end"] - words[0]["start"]
    duration = max(duration, 1e-3)
    wpm = 60.0 * len(words) / duration
    score = 0.7 * min(wpm / 200.0, 1.0) + 0.3 * min(len(words) / 100.0, 1.0)
    return res, float(score)
