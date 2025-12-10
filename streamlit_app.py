# import io  # unused
import json
import base64
from pathlib import Path
import numpy as np
import streamlit as st
# soundfile is not used directly here; kept for potential future use
# import soundfile as sf
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Import our pipeline modules
from autokar.io import load_audio, save_audio, to_mono, ensure_stereo
from autokar.separate import separate_demucs
from autokar.asr import transcribe_and_score
from autokar.align import words_or_segments_to_aligned
from autokar.render import write_srt

st.set_page_config(page_title="AutoKaraoke", page_icon="🎤", layout="wide")

# Custom CSS for calmer styling
st.markdown("""
<style>
    body { background: #f7f8fb; }
    .main-title {
        background: #ffffff;
        padding: 1.3rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid #e5e7ef;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }
    .main-title h1 {
        color: #1f2a44;
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: 0.3px;
    }
    .section-header {
        background: #ffffff;
        padding: 0.85rem 1.1rem;
        border-radius: 10px;
        border-left: 4px solid #5c6ac4;
        margin: 1.2rem 0 0.9rem 0;
        border: 1px solid #e5e7ef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .section-header h3 {
        margin: 0;
        color: #1f2a44;
        font-size: 1.05rem;
    }
    .stButton > button {
        background: #5c6ac4;
        color: white;
        border: none;
        padding: 0.55rem 1.35rem;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.15s, box-shadow 0.2s;
        box-shadow: 0 6px 14px rgba(92,106,196,0.22);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(92,106,196,0.28);
    }
    .karaoke-display {
        background: #ffffff;
        padding: 1.4rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 1px solid #e5e7ef;
    }
    .karaoke-display h2 {
        color: #1f2a44;
        margin: 0;
        font-size: 1.6rem;
        letter-spacing: 0.3px;
    }
    .karaoke-display p {
        color: #4c5570;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
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
    st.caption("Using Demucs (htdemucs) for separation")
    asr_model = st.selectbox(
        "ASR model (Whisper)", ["tiny", "base", "small", "medium"]
    )
    st.markdown("---")
    st.caption("💡 Demucs gives cleaner vocals/instrumental splits.")

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


def render_karaoke_component(aligned, audio_bytes: bytes, mime: str = "audio/wav"):
    """Embed a lightweight JS karaoke player that updates without Streamlit reruns."""
    if not aligned or not audio_bytes:
        st.info("No alignment available for karaoke preview.")
        return

    b64_audio = base64.b64encode(audio_bytes).decode("ascii")
    aligned_json = json.dumps(
        [{"s": a, "e": b, "t": txt} for (a, b, txt) in aligned]
    )

    html = f"""
    <style>
      .karaoke-box {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px;
        border-radius: 12px;
        color: white;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
      }}
      .karaoke-line {{
        margin: 6px 0;
        opacity: 0.6;
        transition: opacity 0.2s;
        font-size: 1.0rem;
      }}
      .karaoke-line.active {{
        opacity: 1;
        font-weight: 800;
        font-size: 1.35rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.35);
      }}
      .karaoke-progress {{
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.2);
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0 12px 0;
      }}
      .karaoke-progress-inner {{
        height: 100%;
        background: #00e7ff;
        width: 0%;
        transition: width 0.05s linear;
      }}
    </style>
    <div class="karaoke-box">
      <audio id="karaudio" controls preload="metadata" style="width:100%; margin-bottom:8px;">
        <source src="data:{mime};base64,{b64_audio}">
      </audio>
      <div class="karaoke-progress"><div id="kaprog" class="karaoke-progress-inner"></div></div>
      <div id="kalines">
        <div id="kaline-prev" class="karaoke-line"></div>
        <div id="kaline-cur" class="karaoke-line active"></div>
        <div id="kaline-next" class="karaoke-line"></div>
      </div>
    </div>
    <script>
      const lines = {aligned_json};
      const audio = document.getElementById("karaudio");
      const prevEl = document.getElementById("kaline-prev");
      const curEl = document.getElementById("kaline-cur");
      const nextEl = document.getElementById("kaline-next");
      const progEl = document.getElementById("kaprog");

      function update(t) {{
        if (!lines.length) return;
        let idx = -1;
        for (let i = 0; i < lines.length; i++) {{
          if (t >= lines[i].s && t <= lines[i].e) {{ idx = i; break; }}
        }}
        const lineAt = (k) => (k >=0 && k < lines.length) ? lines[k].t : "";
        if (idx >= 0) {{
          const l = lines[idx];
          const span = Math.max(l.e - l.s, 1e-6);
          const p = Math.min(Math.max((t - l.s) / span, 0), 1);
          progEl.style.width = (p * 100).toFixed(1) + "%";
          curEl.textContent = lineAt(idx);
          prevEl.textContent = lineAt(idx - 1);
          nextEl.textContent = lineAt(idx + 1);
        }} else {{
          progEl.style.width = "0%";
          curEl.textContent = "";
          prevEl.textContent = "";
          nextEl.textContent = "";
        }}
      }}

      function tick() {{
        update(audio.currentTime);
        requestAnimationFrame(tick);
      }}
      requestAnimationFrame(tick);
    </script>
    """
    components.html(html, height=260, scrolling=False)

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
    st.markdown('<div class="section-header"><h3>1️⃣ Separation (Demucs)</h3></div>', unsafe_allow_html=True)
    vocals_path = WORK / "vocals.wav"
    instr_path = WORK / "instrumental.wav"

    go_sep = st.button("🚀 Run separation")
    if go_sep:
        ok = separate_demucs(
            str(raw_path), str(vocals_path), str(instr_path)
        )
        if not ok:
            st.error("Demucs separation failed. Please try another track or redeploy with Demucs installed.")
        else:
            st.success("✅ Demucs separation done.")

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
            else {"result": {"words": [], "segments": [], "text": ""}}
        )
        words = data.get("result", {}).get("words", [])
        segments = data.get("result", {}).get("segments", [])
        text = data.get("result", {}).get("text", "")
        dur = len(yy) / ssr

        aligned = words_or_segments_to_aligned(words, segments, text, dur)
        # cache for interactive preview without recomputing
        st.session_state["aligned_lines"] = aligned
        st.session_state["aligned_duration"] = dur

        write_srt(str(srt_out), aligned)
        st.success(f"✅ Wrote SRT: {srt_out.name}")
        st.download_button(
            "📥 Download SRT",
            data=open(srt_out, "rb").read(),
            file_name="lyrics.srt",
            mime="application/x-subrip",
        )

    # Karaoke preview with client-side, smooth playback synced to audio
    if "aligned_lines" in st.session_state and "aligned_duration" in st.session_state:
        aligned = st.session_state["aligned_lines"]
        # Use instrumental for karaoke playback when available
        if instr_path.exists():
            target = instr_path
            mime_type = "audio/wav"
        elif vocals_path.exists():
            target = vocals_path
            mime_type = "audio/wav"
        else:
            target = raw_path
            mime_type = "audio/wav"
        try:
            audio_bytes = open(target, "rb").read()
        except Exception:
            audio_bytes = b""
        st.markdown("#### 🎤 Karaoke preview (live sync)")
        render_karaoke_component(aligned, audio_bytes, mime=mime_type)

st.markdown("---")
