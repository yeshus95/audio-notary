import librosa
import numpy as np
import soundfile as sf
import os
from scipy.spatial.distance import cosine

from speechbrain.inference.speaker import SpeakerRecognition


# Load model once
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)


# ---------------------------------------------------
# AUDIO PREPROCESS
# ---------------------------------------------------

def preprocess_audio(path):
    y, sr = librosa.load(path, sr=16000)

    # normalize audio
    y = y / (np.max(np.abs(y)) + 1e-9)

    processed_path = path + "_processed.wav"
    sf.write(processed_path, y, 16000)

    return processed_path


# ---------------------------------------------------
# MFCC FEATURE EXTRACTION
# ---------------------------------------------------

def extract_mfcc(path):

    y, sr = librosa.load(path, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


# ---------------------------------------------------
# MFCC SIMILARITY
# ---------------------------------------------------

def mfcc_similarity(file1, file2):

    mfcc1 = extract_mfcc(file1)
    mfcc2 = extract_mfcc(file2)

    similarity = 1 - cosine(mfcc1, mfcc2)

    return float(similarity)


# ---------------------------------------------------
# MAIN VERIFICATION
# ---------------------------------------------------

def verify_speaker(reference_path, test_path, threshold=0.60):

    try:

        ref = preprocess_audio(reference_path)
        test = preprocess_audio(test_path)

        # --------------------------
        # ECAPA similarity
        # --------------------------

        score, _ = verification.verify_files(ref, test)
        ecapa_score = float(score)

        # --------------------------
        # MFCC similarity
        # --------------------------

        mfcc_score = mfcc_similarity(ref, test)

        # --------------------------
        # Weighted fusion
        # --------------------------

        fused_score = (0.7 * ecapa_score) + (0.3 * mfcc_score)

        is_same = fused_score >= threshold

        verdict = "Same Speaker" if is_same else "Different Speaker"

        os.remove(ref)
        os.remove(test)

        return {

            "ecapa_similarity": round(ecapa_score, 4),
            "mfcc_similarity": round(mfcc_score, 4),
            "fused_similarity": round(fused_score, 4),

            "threshold": threshold,
            "is_same_speaker": is_same,
            "verdict": verdict
        }

    except Exception as e:

        return {
            "ecapa_similarity": 0,
            "mfcc_similarity": 0,
            "fused_similarity": 0,
            "threshold": threshold,
            "is_same_speaker": False,
            "verdict": "Verification Failed",
            "error": str(e)
        }