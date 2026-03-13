import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_bio(filepath, uid):
    results = {
        'pitch_jitter': None,
        'shimmer': None,
        'mfcc_variance': None,
        'mfcc_delta_variance': None,
        'hnr': None,
        'voiced_fraction': None,
        'voiced_f0_count': None,
        'spectral_flatness': None,
        'biological_score': 0,
        'flags': []
    }

    y, sr = librosa.load(filepath, sr=16000, mono=True)

    # --- Generate Spectrogram ---
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(f'static/plots/{uid}_spectrogram.png', dpi=100)
    plt.close()

    # --- Generate MFCC plot ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.title('MFCC (13 coefficients)')
    plt.tight_layout()
    plt.savefig(f'static/plots/{uid}_mfcc.png', dpi=100)
    plt.close()

    # --- Feature extraction ---
    mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
    results['mfcc_variance'] = round(mfcc_var, 4)

    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta_var = float(np.mean(np.var(mfcc_delta, axis=1)))
    results['mfcc_delta_variance'] = round(mfcc_delta_var, 4)

    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    results['spectral_flatness'] = round(spec_flatness, 6)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=2048
    )
    voiced_f0 = f0[voiced_flag]
    results['voiced_fraction'] = round(float(np.mean(voiced_flag)), 3)
    results['voiced_f0_count'] = int(len(voiced_f0))

    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic
    harmonic_energy = np.mean(harmonic ** 2) + 1e-10
    noise_energy = np.mean(noise ** 2) + 1e-10
    hnr = 10 * np.log10(harmonic_energy / noise_energy)
    results['hnr'] = round(float(hnr), 3)

    # Start at 50 — neutral
    bio_score = 50

    if len(voiced_f0) > 10:
        pitch_diffs = np.abs(np.diff(voiced_f0))
        mean_pitch = np.mean(voiced_f0)
        jitter = float(np.mean(pitch_diffs) / mean_pitch) if mean_pitch > 0 else 0
        results['pitch_jitter'] = round(jitter, 6)

        rms_frames = librosa.feature.rms(y=y)[0]
        min_len = min(len(rms_frames), len(voiced_flag))
        voiced_rms = rms_frames[:min_len][voiced_flag[:min_len]]
        if len(voiced_rms) > 1:
            shimmer_val = float(np.std(voiced_rms) / (np.mean(voiced_rms) + 1e-9))
            results['shimmer'] = round(shimmer_val, 6)
        else:
            shimmer_val = 0.0
            results['shimmer'] = 0.0

        # -------------------------------------------------------
        # SCORING — calibrated from your actual audio data
        # Human:  shimmer=0.924, jitter=0.021, hnr=2.059,
        #         voiced_f0_count=87,  mfcc_var=1150, spec_flat=0.005
        # AI:     shimmer=0.496, jitter=0.048, hnr=-2.718,
        #         voiced_f0_count=4082, mfcc_var=1272, spec_flat=0.029
        # -------------------------------------------------------

        # 1. SHIMMER — strongest separator in your data
        # Human=0.924 (very high), AI=0.496 (lower)
        # Threshold at 0.70 — above is human, below is synthetic
        if shimmer_val > 0.70:
            bio_score += 30
        elif shimmer_val > 0.50:
            bio_score += 10
        elif shimmer_val > 0.30:
            bio_score -= 15
            results['flags'].append(
                f"Low shimmer ({shimmer_val:.4f}) — amplitude too consistent, synthetic indicator")
        else:
            bio_score -= 30
            results['flags'].append(
                f"Very low shimmer ({shimmer_val:.4f}) — strong synthetic indicator")

        # 2. VOICED_F0_COUNT — second strongest separator
        # Human=87 (short natural speech), AI=4082 (unnaturally long continuous voicing)
        # Normalize by duration to get voiced frames per second
        duration = len(y) / sr
        voiced_rate = len(voiced_f0) / duration if duration > 0 else 0
        results['voiced_rate_per_sec'] = round(voiced_rate, 2)

        if voiced_rate < 50:
            bio_score += 25
        elif voiced_rate < 100:
            bio_score += 10
        elif voiced_rate < 200:
            bio_score -= 10
            results['flags'].append(
                f"High voiced rate ({voiced_rate:.1f}/sec) — unusually continuous voicing")
        else:
            bio_score -= 30
            results['flags'].append(
                f"Very high voiced rate ({voiced_rate:.1f}/sec) — strong synthetic indicator, "
                f"TTS voices sustain pitch unnaturally long")

        # 3. SPECTRAL FLATNESS
        # Human=0.005 (tonal), AI=0.029 (flatter/noisier — unexpected but real)
        # AI is actually flatter here which is unusual — use it as a flag
        if spec_flatness > 0.020:
            bio_score -= 20
            results['flags'].append(
                f"High spectral flatness ({spec_flatness:.5f}) — unnatural noise distribution")
        elif spec_flatness > 0.010:
            bio_score -= 8
        elif spec_flatness < 0.008:
            bio_score += 15

        # 4. HNR
        # Human=2.059, AI=-2.718
        # Human has slightly higher HNR here — mild signal
        if hnr > 1.5:
            bio_score += 10
        elif hnr > 0:
            bio_score += 5
        elif hnr < -1.5:
            bio_score -= 10
            results['flags'].append(
                f"Negative HNR ({hnr:.2f} dB) — unusual harmonic structure")

        # 5. MFCC variance — both are high in your data, not a strong separator
        # Only flag extreme lows
        if mfcc_var < 200:
            bio_score -= 15
            results['flags'].append(
                f"Low MFCC variance ({mfcc_var:.2f}) — unnaturally uniform timbre")
        elif mfcc_var > 800:
            bio_score += 5

        # 6. MFCC delta variance
        # Human=27.27, AI=45.31 — AI has more variation here (unexpected)
        # Flag very high delta variance as suspicious
        if mfcc_delta_var > 40:
            bio_score -= 10
            results['flags'].append(
                f"High MFCC delta variance ({mfcc_delta_var:.2f}) — erratic spectral transitions")
        elif mfcc_delta_var > 20:
            bio_score += 5
        elif mfcc_delta_var < 10:
            bio_score -= 5

    else:
        results['flags'].append(
            "Insufficient voiced segments — audio too short or non-speech")
        bio_score = 20

    results['biological_score'] = max(0, min(100, bio_score))
    return results