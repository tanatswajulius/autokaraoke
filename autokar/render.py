from __future__ import annotations
from typing import List, Tuple
import os


def _fmt(ts: float) -> str:
    h = int(ts // 3600)
    ts -= 3600 * h
    m = int(ts // 60)
    ts -= 60 * m
    s = int(ts)
    ms = int((ts - s) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: str, aligned: List[Tuple[float, float, str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(aligned, 1):
            f.write(f"{i}\n{_fmt(start)} --> { _fmt(end)}\n{text.strip()}\n\n")
