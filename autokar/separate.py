from __future__ import annotations
import numpy as np
from typing import Tuple


def separate_ica(
    y_stereo: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """REPET-based separation optimized for pop music like Taylor Swift."""
    if y_stereo.ndim != 2 or y_stereo.shape[0] != 2:
        raise ValueError("Expected stereo audio array of shape (2, N)")

    try:
        import librosa

        # Convert to mono for processing
        mono = 0.5 * (y_stereo[0] + y_stereo[1])

        # Step 1: REPET - Find repeating patterns (typical in pop music)
        S = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
        mag = np.abs(S)

        # Find the repeating period (usually 4, 8, or 16 beats in pop)
        beat_track = librosa.beat.tempo(y=mono, sr=sr, hop_length=hop)[0]
        period_samples = int(60 * sr / beat_track * 4)  # 4-beat periods
        period_frames = period_samples // hop

        # Create repeating background model
        n_frames = mag.shape[1]
        n_periods = n_frames // period_frames

        if n_periods > 1:
            # Reshape into periods and take median
            # (removes non-repeating vocals)
            reshaped = mag[:, :n_periods * period_frames].reshape(
                mag.shape[0], n_periods, period_frames
            )
            background_period = np.median(reshaped, axis=1)

            # Tile the background pattern
            background = np.tile(background_period, (1, n_periods))
            if background.shape[1] < n_frames:
                # Pad the last partial period
                remaining = n_frames - background.shape[1]
                background = np.concatenate([
                    background,
                    background_period[:, :remaining]
                ], axis=1)
            background = background[:, :n_frames]
        else:
            # Fallback if song is too short
            background = mag * 0.5

        # Step 2: Spectral subtraction with adaptive masking
        vocal_mask = np.maximum(0, (mag - background) / (mag + 1e-8))
        vocal_mask = np.minimum(vocal_mask, 0.8)  # Limit artifacts

        # Step 3: Apply frequency-dependent processing
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Enhance vocal mask in vocal frequency range
        vocal_freq_mask = ((freqs >= 80) & (freqs <= 8000)).astype(float)
        vocal_freq_mask = vocal_freq_mask.reshape(-1, 1)

        # Apply frequency weighting
        vocal_mask_enhanced = vocal_mask * (0.3 + 0.7 * vocal_freq_mask)
        instrumental_mask_enhanced = 1 - vocal_mask_enhanced

        # Step 4: Reconstruct signals
        vocal_spec = S * vocal_mask_enhanced
        instrumental_spec = S * instrumental_mask_enhanced

        # Convert back to time domain
        vocals = librosa.istft(vocal_spec, hop_length=hop)
        instrumental = librosa.istft(instrumental_spec, hop_length=hop)

        # Step 5: Stereo enhancement
        left_spec = librosa.stft(y_stereo[0], n_fft=n_fft, hop_length=hop)
        right_spec = librosa.stft(y_stereo[1], n_fft=n_fft, hop_length=hop)

        # Apply same masks to stereo channels
        left_vocal = librosa.istft(
            left_spec * vocal_mask_enhanced, hop_length=hop
        )
        right_vocal = librosa.istft(
            right_spec * vocal_mask_enhanced, hop_length=hop
        )
        left_instr = librosa.istft(
            left_spec * instrumental_mask_enhanced, hop_length=hop
        )
        right_instr = librosa.istft(
            right_spec * instrumental_mask_enhanced, hop_length=hop
        )

        # Mix stereo channels
        vocals_final = 0.5 * (left_vocal + right_vocal)
        instrumental_final = 0.5 * (left_instr + right_instr)

        # Step 6: Post-processing
        target_length = y_stereo.shape[1]

        def process_output(signal):
            if len(signal) > target_length:
                signal = signal[:target_length]
            elif len(signal) < target_length:
                signal = np.pad(signal, (0, target_length - len(signal)))

            # Gentle compression to avoid clipping
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val * 0.8

            return signal.astype(np.float32)

        vocals_processed = process_output(vocals_final)
        instrumental_processed = process_output(instrumental_final)

        return vocals_processed, instrumental_processed

    except Exception as e:
        print(f"REPET separation failed: {e}, using mid/side fallback")
        # Enhanced mid/side with high-frequency emphasis for vocals
        left = y_stereo[0]
        right = y_stereo[1]

        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)

        # Simple high-pass for vocals, low-pass for instrumental
        try:
            from scipy import signal
            # High-pass filter for vocals (remove bass)
            sos_hp = signal.butter(4, 100, 'hp', fs=sr, output='sos')
            vocals = signal.sosfilt(sos_hp, mid)

            # Enhanced side for instrumental
            instrumental = side + 0.3 * mid

            return vocals.astype(np.float32), instrumental.astype(np.float32)
        except Exception:
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