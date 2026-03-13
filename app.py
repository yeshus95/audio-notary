import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS

from analysis.metadata import analyze_metadata
from analysis.bio_features import analyze_bio
from analysis.scorer import compute_trust_score

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

os.makedirs('uploads', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debugtest')
def debugtest():
    return render_template('debugtest.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    uid = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{uid}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        layer1 = analyze_metadata(filepath)
        layer2 = analyze_bio(filepath, uid)
        result = compute_trust_score(layer1, layer2)

        result['layer1'] = layer1
        result['layer2'] = layer2
        result['spectrogram_url'] = f'/static/plots/{uid}_spectrogram.png'
        result['mfcc_url'] = f'/static/plots/{uid}_mfcc.png'

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/verify-speaker', methods=['POST'])
def verify_speaker_route():
    from analysis.speaker_verifier import verify_speaker

    if 'reference_audio' not in request.files or 'test_audio' not in request.files:
        return jsonify({'error': 'Both reference_audio and test_audio are required'}), 400

    reference_file = request.files['reference_audio']
    test_file = request.files['test_audio']

    if reference_file.filename == '' or test_file.filename == '':
        return jsonify({'error': 'Both files must be selected'}), 400

    if not allowed_file(reference_file.filename) or not allowed_file(test_file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    uid = str(uuid.uuid4())[:8]

    reference_filename = secure_filename(f"{uid}_ref_{reference_file.filename}")
    test_filename = secure_filename(f"{uid}_test_{test_file.filename}")

    reference_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
    test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)

    reference_file.save(reference_path)
    test_file.save(test_path)

    try:
        result = verify_speaker(reference_path, test_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(reference_path):
            os.remove(reference_path)
        if os.path.exists(test_path):
            os.remove(test_path)


@app.route('/debug', methods=['POST'])
def debug():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(f"debug_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(filepath, sr=16000, mono=True)

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=2048
        )

        voiced_f0 = f0[voiced_flag]

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_var = float(np.mean(np.var(mfcc_delta, axis=1)))

        spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

        harmonic = librosa.effects.harmonic(y)
        noise = y - harmonic
        hnr = 10 * np.log10(
            (np.mean(harmonic ** 2) + 1e-10) /
            (np.mean(noise ** 2) + 1e-10)
        )

        jitter = 0
        shimmer_val = 0

        if len(voiced_f0) > 10:
            pitch_diffs = np.abs(np.diff(voiced_f0))
            mean_pitch = np.mean(voiced_f0)
            jitter = float(np.mean(pitch_diffs) / mean_pitch) if mean_pitch > 0 else 0

            rms_frames = librosa.feature.rms(y=y)[0]
            min_len = min(len(rms_frames), len(voiced_flag))
            voiced_rms = rms_frames[:min_len][voiced_flag[:min_len]]

            if len(voiced_rms) > 1:
                shimmer_val = float(
                    np.std(voiced_rms) / (np.mean(voiced_rms) + 1e-9)
                )

        return jsonify({
            'jitter': round(jitter, 6),
            'shimmer': round(shimmer_val, 6),
            'mfcc_variance': round(mfcc_var, 4),
            'mfcc_delta_variance': round(mfcc_delta_var, 4),
            'spectral_flatness': round(spec_flatness, 6),
            'hnr': round(float(hnr), 3),
            'voiced_fraction': round(float(np.mean(voiced_flag)), 3),
            'voiced_f0_count': int(len(voiced_f0))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    app.run(debug=True)