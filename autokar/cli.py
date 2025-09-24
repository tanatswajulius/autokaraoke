from __future__ import annotations
import json
import typer
from pathlib import Path
from .io import load_audio, save_audio, to_mono, ensure_stereo
from .separate import separate_ica, separate_demucs
from .asr import transcribe_and_score
from .align import pass_through_alignment, dtw_refine
from .render import write_srt

app = typer.Typer(add_completion=False)


@app.command()
def separate(
    input_wav: str,
    vocals: str = typer.Option("out/vocals.wav"),
    instrumental: str = typer.Option("out/instrumental.wav"),
    method: str = typer.Option("ica", help="ica | demucs"),
):
    y, sr = load_audio(input_wav, sr=None, mono=False)
    y = ensure_stereo(y)

    if method == "demucs":
        ok = separate_demucs(input_wav, vocals, instrumental)
        if ok:
            typer.echo(
                f"Saved vocals to {vocals} and instrumental to {instrumental}"
            )
            raise typer.Exit()
        else:
            typer.echo("Demucs not available; falling back to ICA.")

    c1, c2 = separate_ica(y, sr)
    # Save both components; we'll pick speech during transcription.
    save_audio(vocals, c1, sr)
    save_audio(instrumental, c2, sr)
    typer.echo(
        "Saved two components to "
        f"{vocals} and {instrumental}. Use transcribe to pick vocals."
    )


@app.command()
def transcribe(
    maybe_vocals: str,
    other_component: str = typer.Option(
        "",
        help="Optional: if provided, we score both and pick speech track.",
    ),
    json_out: str = typer.Option("out/transcript.json"),
    model: str = typer.Option(
        "small",
        help="faster-whisper model size (tiny|base|small|medium|large)",
    ),
):
    y1, sr1 = load_audio(maybe_vocals, sr=None, mono=False)
    y1m = to_mono(y1)
    res1, s1 = transcribe_and_score(
        y1m, sr1, backend="faster-whisper", model_size=model
    )
    best = {
        "which": "first",
        "result": res1,
        "score": s1,
        "sr": sr1,
        "path": maybe_vocals,
    }

    if other_component:
        y2, sr2 = load_audio(other_component, sr=None, mono=False)
        y2m = to_mono(y2)
        res2, s2 = transcribe_and_score(
            y2m, sr2, backend="faster-whisper", model_size=model
        )
        if s2 > s1:
            best = {
                "which": "second",
                "result": res2,
                "score": s2,
                "sr": sr2,
                "path": other_component,
            }

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    typer.echo(
        "Saved transcription + pick info to "
        f"{json_out} (speech component='{best['which']}')."
    )


@app.command()
def align(
    vocals_wav: str,
    transcript_json: str,
    srt: str = typer.Option("out/lyrics.srt"),
    use_dtw: bool = typer.Option(
        False,
        help="If True, attempt DTW+TTS refinement when word times missing.",
    ),
):

    y, sr = load_audio(vocals_wav, sr=None, mono=False)
    y = to_mono(y)
    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = data.get("result", {}).get("words", [])
    text = data.get("result", {}).get("text", "")

    if words:
        aligned = pass_through_alignment(words)
    elif use_dtw and text:
        aligned = dtw_refine(y, sr, text)
    else:
        # naive single-chunk
        dur = len(y) / sr
        aligned = [(0.0, dur, text if text else "(no transcript)")]

    write_srt(srt, aligned)
    typer.echo(f"Wrote SRT to {srt}")


if __name__ == "__main__":
    app()
