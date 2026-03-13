import os, struct
import mutagen
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.flac import FLAC
import numpy as np
import soundfile as sf

def analyze_metadata(filepath):
    findings = {
        'flags': [],
        'bitrate_inconsistent': False,
        'size_duration_anomaly': False,
        'editing_software_detected': False,
        'silence_anomalies': 0,
        'silence_segments': [],
        'metadata_tags': {}
    }

    ext = filepath.rsplit('.', 1)[1].lower()

    # --- Mutagen metadata read ---
    try:
        audio = mutagen.File(filepath, easy=True)
        if audio and audio.tags:
            tags = dict(audio.tags)
            findings['metadata_tags'] = {k: str(v) for k, v in tags.items()}
            # Check for editing software fingerprints
            software_keys = ['encoded_by', 'encodedby', 'encoder', 'tool', 'tsse', 'software']
            suspicious_tools = ['audacity', 'adobe audition', 'garageband', 'reaper',
                                'protools', 'ffmpeg', 'sox', 'wavosaur']
            for key in software_keys:
                val = tags.get(key, '')
                if isinstance(val, list): val = ' '.join(val)
                val = str(val).lower()
                if any(tool in val for tool in suspicious_tools):
                    findings['editing_software_detected'] = True
                    findings['flags'].append(f"Editing software found in metadata: {val}")
    except Exception as e:
        findings['flags'].append(f"Metadata parse warning: {str(e)}")

    # --- Bitrate and size/duration check ---
    try:
        file_size = os.path.getsize(filepath)
        if ext == 'mp3':
            a = MP3(filepath)
            duration = a.info.length
            declared_bitrate = a.info.bitrate  # bits per second
            expected_size = (declared_bitrate / 8) * duration
            ratio = file_size / expected_size if expected_size > 0 else 1
            if ratio < 0.7 or ratio > 1.4:
                findings['size_duration_anomaly'] = True
                findings['flags'].append(
                    f"Size/duration mismatch: expected ~{int(expected_size/1024)}KB, got {int(file_size/1024)}KB (ratio {ratio:.2f})"
                )
        elif ext in ('wav', 'flac', 'ogg'):
            data, sr = sf.read(filepath)
            duration = len(data) / sr
            findings['duration_seconds'] = round(duration, 2)
    except Exception as e:
        findings['flags'].append(f"Bitrate check warning: {str(e)}")

    # --- Silence anomaly detection ---
    try:
        data, sr = sf.read(filepath)
        if data.ndim > 1:
            data = data.mean(axis=1)  # mono
        findings['duration_seconds'] = round(len(data) / sr, 2)

        # Detect digital silences: windows of near-zero amplitude
        window = int(sr * 0.05)  # 50ms windows
        threshold = 1e-4
        silence_count = 0
        in_silence = False
        silence_start = 0

        for i in range(0, len(data) - window, window):
            chunk = data[i:i + window]
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms < threshold:
                if not in_silence:
                    in_silence = True
                    silence_start = i / sr
            else:
                if in_silence:
                    silence_end = i / sr
                    duration_s = silence_end - silence_start
                    # Only flag unnatural silences > 200ms but perfectly zero
                    max_amp = np.max(np.abs(data[int(silence_start*sr):int(silence_end*sr)]))
                    if duration_s > 0.2 and max_amp < 1e-6:
                        silence_count += 1
                        findings['silence_segments'].append({
                            'start': round(silence_start, 3),
                            'end': round(silence_end, 3),
                            'duration': round(duration_s, 3)
                        })
                    in_silence = False

        findings['silence_anomalies'] = silence_count
        if silence_count > 0:
            findings['flags'].append(f"Found {silence_count} digital silence segment(s) — possible splicing")
    except Exception as e:
        findings['flags'].append(f"Silence analysis warning: {str(e)}")

    return findings