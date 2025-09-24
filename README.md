# AutoKaraoke 

End-to-end pipeline:
1) Separate audio into components (baseline: frequency-domain ICA).
2) Detect which component is speech via ASR confidence.
3) Export vocals + instrumental.
4) Align transcript to audio (uses word timestamps if ASR provides, else optional DTW+TTS proxy).
5) Render SRT subtitles.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Separate
python -m autokar.cli separate input.wav --vocals out/vocals.wav --instrumental out/instrumental.wav

# Transcribe (faster-whisper)
python -m autokar.cli transcribe out/vocals.wav --json out/transcript.json

# Align + export SRT
python -m autokar.cli align out/vocals.wav out/transcript.json --srt out/lyrics.srt
