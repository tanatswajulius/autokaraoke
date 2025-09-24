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

st.set_page_config(page_title="AutoKaraoke", page_icon="🎤", layout="wide")

# Custom CSS for prettier styling
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-title h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .section-header {
        background: #f8f9fa;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1.5rem 0 1rem 0;
    }
    .section-header h3 {
        margin: 0;
        color: #333;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .karaoke-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .karaoke-display h2 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .karaoke-display p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Pretty title
st.markdown("""
<div class="main-title">
    <h1>🎤 AutoKaraoke</h1>
</div>
""", unsafe_allow_html=True)

WORK = Path("ui_out")
WORK.mkdir(exist_ok=True, parents=True)

with st.sidebar:
    st.header("🎛️ Pipeline Options")
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
    st.caption("💡 Tip: ICA is a baseline.")
    st.caption("If Demucs is installed, choose demucs for cleaner vocals.")

uploaded = st.file_uploader(
    "📁 Upload a WAV/MP3/FLAC audio file",
    type=["wav", "mp3", "flac", "m4a", "ogg"],
)


def plot_waveform(y, sr, title=""):
    if y.ndim == 2:
        y = to_mono(y)
    t = np.arange(len(y)) / sr
    
    # Prettier plot styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, y, color='#667eea', linewidth=1, alpha=0.8)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#333')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)


if uploaded is not None:
    # Save upload to disk
    raw_path = WORK / f"input_{uploaded.name}"
    with open(raw_path, "wb") as f:
        f.write(uploaded.read())

    st.success(f"✅ Loaded: {uploaded.name}")
    y, sr = load_audio(str(raw_path), sr=None, mono=False)
    
    st.markdown('<div class="section-header"><h3>🎵 Original</h3></div>', unsafe_allow_html=True)
    st.audio(str(raw_path))
    plot_waveform(y, sr, "Original waveform")

    # Separate
    st.markdown('<div class="section-header"><h3>1️⃣ Separation</h3></div>', unsafe_allow_html=True)
    vocals_path = WORK / "vocals.wav"
    instr_path = WORK / "instrumental.wav"

    go_sep = st.button("🚀 Run separation")
    if go_sep:
        if sep_method.startswith("demucs"):
            ok = separate_demucs(
                str(raw_path), str(vocals_path), str(instr_path)
            )
            if not ok:
                st.warning("⚠️ Demucs not available, falling back to ICA.")
                c1, c2 = separate_ica(ensure_stereo(y), sr)
                save_audio(str(vocals_path), c1, sr)
                save_audio(str(instr_path), c2, sr)
        else:
            c1, c2 = separate_ica(ensure_stereo(y), sr)
            save_audio(str(vocals_path), c1, sr)
            save_audio(str(instr_path), c2, sr)
        st.success("✅ Separation done.")

    if vocals_path.exists():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎤 Vocals file", vocals_path.name)
            st.audio(str(vocals_path))
            vy, vsr = load_audio(str(vocals_path), sr=None, mono=False)
            plot_waveform(vy, vsr, "Vocals")
        
    if instr_path.exists():
        with col2:
            st.metric("🎼 Instrumental file", instr_path.name)
            st.audio(str(instr_path))
            by, bsr = load_audio(str(instr_path), sr=None, mono=False)
            plot_waveform(by, bsr, "Instrumental")

    # Transcribe
    st.markdown('<div class="section-header"><h3>2️⃣ Transcribe & pick speech track</h3></div>', unsafe_allow_html=True)
    json_out = WORK / "transcript.json"
    if st.button("🎯 Run transcription"):
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
            f"✅ Transcription done. Picked speech component: {best['which']}"
        )

    if json_out.exists():
        data = json.loads(json_out.read_text())
        st.json(data["result"].get("segments", [])[:3])  # preview
        st.text_area("📝 Transcript", data["result"].get("text", ""), height=120)

    # Align + SRT
    st.markdown('<div class="section-header"><h3>3️⃣ Align & export subtitles</h3></div>', unsafe_allow_html=True)
    srt_out = WORK / "lyrics.srt"
    if st.button("🎬 Align and write SRT"):
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
        st.success(f"✅ Wrote SRT: {srt_out.name}")
        st.download_button(
            "📥 Download SRT",
            data=open(srt_out, "rb").read(),
            file_name="lyrics.srt",
            mime="application/x-subrip",
        )

        # Simple karaoke preview: highlight current line by slider time
        st.markdown("#### 🎤 Karaoke preview")
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
            st.markdown(f"""
            <div class="karaoke-display">
                <h2>🎤 {active[2]}</h2>
                <p>{active[0]:.2f}s → {active[1]:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🎵 No active line at this time")

st.markdown("---")
