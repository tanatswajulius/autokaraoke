# import io  # unused
import json
from pathlib import Path
import numpy as np
import streamlit as st
# soundfile is not used directly here; kept for potential future use
# import soundfile as sf
import matplotlib.pyplot as plt

# Import our pipeline modules
from autokar.io import load_audio, save_audio, to_mono, ensure_stereo
from autokar.separate import separate_ica, separate_demucs
from autokar.asr import transcribe_and_score
from autokar.align import pass_through_alignment, dtw_refine
from autokar.render import write_srt

st.set_page_config(page_title="AutoKaraoke UI", page_icon="🎤", layout="wide")
st.title("🎤 AutoKaraoke")

WORK = Path("ui_out")
WORK.mkdir(exist_ok=True, parents=True)

with st.sidebar:
    st.header("Pipeline Options")
    sep_method = st.selectbox(
        "Separation method", ["ica", "demucs (if installed)"]
    )
    asr_model = st.selectbox(
        "ASR model (faster-whisper)", ["tiny", "base", "small", "medium"]
    )
    use_dtw = st.checkbox(
        "Refine alignment with DTW + TTS proxy (demo)", value=False
    )
    st.markdown("---")
    st.caption("Tip: ICA is a baseline.")
    st.caption("If Demucs is installed, choose demucs for cleaner vocals.")

uploaded = st.file_uploader(
    "Upload a WAV/MP3/FLAC audio file",
    type=["wav", "mp3", "flac", "m4a", "ogg"],
)


def plot_waveform(y, sr, title=""):
    if y.ndim == 2:
        y = to_mono(y)
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amp")
    ax.set_title(title)
    st.pyplot(fig)


if uploaded is not None:
    # Save upload to disk
    raw_path = WORK / f"input_{uploaded.name}"
    with open(raw_path, "wb") as f:
        f.write(uploaded.read())

    st.success(f"Loaded: {uploaded.name}")
    y, sr = load_audio(str(raw_path), sr=None, mono=False)
    st.subheader("Original")
    st.audio(str(raw_path))
    plot_waveform(y, sr, "Original waveform")

    # Separate
    st.subheader("1) Separation")
    vocals_path = WORK / "vocals.wav"
    instr_path = WORK / "instrumental.wav"

    go_sep = st.button("Run separation")
    if go_sep:
        if sep_method.startswith("demucs"):
            ok = separate_demucs(
                str(raw_path), str(vocals_path), str(instr_path)
            )
            if not ok:
                st.warning("Demucs not available, falling back to ICA.")
                c1, c2 = separate_ica(ensure_stereo(y), sr)
                save_audio(str(vocals_path), c1, sr)
                save_audio(str(instr_path), c2, sr)
        else:
            c1, c2 = separate_ica(ensure_stereo(y), sr)
            save_audio(str(vocals_path), c1, sr)
            save_audio(str(instr_path), c2, sr)
        st.success("Separation done.")

    if vocals_path.exists():
        st.columns(2)[0].metric(
            "Vocals file", vocals_path.name
        )
        st.audio(str(vocals_path))
        vy, vsr = load_audio(str(vocals_path), sr=None, mono=False)
        plot_waveform(vy, vsr, "Vocals")
    if instr_path.exists():
        st.columns(2)[1].metric(
            "Instrumental file", instr_path.name
        )
        st.audio(str(instr_path))
        by, bsr = load_audio(str(instr_path), sr=None, mono=False)
        plot_waveform(by, bsr, "Instrumental")

    # Transcribe
    st.subheader("2) Transcribe & pick speech track")
    json_out = WORK / "transcript.json"
    if st.button("Run transcription"):
        # Score vocals and (optionally) the other component if present
        vy, vsr = load_audio(str(vocals_path), sr=None, mono=False)
        res1, s1 = transcribe_and_score(
            to_mono(vy),
            vsr,
            backend="faster-whisper",
            model_size=asr_model,
        )
        best = {
            "which": "vocals.wav",
            "result": res1,
            "score": s1,
            "sr": vsr,
            "path": str(vocals_path),
        }

        if instr_path.exists():
            by, bsr = load_audio(str(instr_path), sr=None, mono=False)
            res2, s2 = transcribe_and_score(
                to_mono(by),
                bsr,
                backend="faster-whisper",
                model_size=asr_model,
            )
            if s2 > s1:
                best = {
                    "which": "instrumental.wav",
                    "result": res2,
                    "score": s2,
                    "sr": bsr,
                    "path": str(instr_path),
                }

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        st.success(
            f"Transcription done. Picked speech component: {best['which']}"
        )

    if json_out.exists():
        data = json.loads(json_out.read_text())
        st.json(data["result"].get("segments", [])[:3])  # preview
        st.text_area("Transcript", data["result"].get("text", ""), height=120)

    # Align + SRT
    st.subheader("3) Align & export subtitles")
    srt_out = WORK / "lyrics.srt"
    if st.button("Align and write SRT"):
        target = vocals_path if vocals_path.exists() else raw_path
        yy, ssr = load_audio(str(target), sr=None, mono=False)
        yy = to_mono(yy)
        data = (
            json.loads(json_out.read_text())
            if json_out.exists()
            else {"result": {"words": [], "text": ""}}
        )
        words = data.get("result", {}).get("words", [])
        text = data.get("result", {}).get("text", "")

        if words:
            aligned = pass_through_alignment(words)
        elif use_dtw and text:
            aligned = dtw_refine(yy, ssr, text)
        else:
            dur = len(yy) / ssr
            aligned = [(0.0, dur, text if text else "(no transcript)")]

        write_srt(str(srt_out), aligned)
        st.success(f"Wrote SRT: {srt_out.name}")
        st.download_button(
            "Download SRT",
            data=open(srt_out, "rb").read(),
            file_name="lyrics.srt",
            mime="application/x-subrip",
        )

        # Simple karaoke preview: highlight current line by slider time
        st.markdown("#### Karaoke preview")
        cur_t = st.slider(
            "Preview time (s)", 0.0, float(len(yy) / ssr), 0.0, 0.1
        )
        # find active line
        active = None
        for (a, b, textline) in aligned:
            if a <= cur_t <= b:
                active = (a, b, textline)
                break
        if active:
            st.markdown(
                f"**▶ {active[2]}**  \n_{active[0]:.2f}s → {active[1]:.2f}s_"
            )
        else:
            st.markdown("_(No active line at this time)_")

st.markdown("---")
st.caption(
    ""
    ""
)
