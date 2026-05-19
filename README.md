# 🔏 Digital Audio Notary

A web-based forensic tool that analyzes audio files to detect AI-generated 
deepfake voices and verify speaker identity using multi-layer audio analysis.

Built for a national level hackathon solving the problem of audio-based fraud 
and manipulated evidence in the era of high-fidelity deepfakes.

---

## 🔍 What it Does

- Detects whether an audio file contains a genuine human voice or an AI generated synthetic voice
- Produces a Weighted Trust Score from 0 to 100% instead of a simple Real or Fake label
- Verifies whether two audio recordings belong to the same speaker
- Generates visual Spectrograms and MFCC plots for forensic review
- Explains every verdict in plain human readable language

---

## 🧱 How it Works

The system performs a three layer Trust Audit on every uploaded audio file:

**Layer 1 — Metadata & Structural Integrity**
Scans the file for editing software traces, size and duration mismatches,
and unnatural silent gaps left behind by audio splicing.

**Layer 2 — Biological Feature Extraction**
Measures biological voice fingerprints like shimmer, pitch jitter, voiced rate,
MFCC variance and spectral flatness that AI voices cannot perfectly replicate.

**Layer 3 — Weighted Trust Score Engine**
Combines all findings into a single Trust Score with one of four verdicts:
- ✅ Likely Authentic
- ⚠️ Authentic Voice — Potentially Edited
- ❌ Synthetic Voice Detected
- ❌ Synthetic / Manipulated

**Speaker Verification**
Compares voice features like MFCCs, pitch distribution, spectral centroid
and zero crossing rate between two audio files using cosine similarity
to confirm or deny whether both recordings belong to the same speaker.

---

## 🛠️ Tech Stack

- **Backend** — Python, Flask
- **Audio Analysis** — Librosa, NumPy, SciPy, Soundfile
- **Metadata Analysis** — Mutagen
- **Visualization** — Matplotlib
- **Frontend** — HTML, CSS, JavaScript

---

## ⚙️ How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/yourusername/audio-notary.git
cd audio-notary
```

**Step 2 — Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install flask librosa numpy scipy matplotlib mutagen soundfile werkzeug flask-cors
```

**Step 4 — Run the application**
```bash
python app.py
```

**Step 5 — Open in browser**
http://localhost:5000

---

## 📁 Project Structure
audio-notary/
├── app.py                  — Flask routes and API
├── analysis/
│   ├── init.py
│   ├── metadata.py         — Layer 1 metadata analysis
│   ├── bio_features.py     — Layer 2 biological feature extraction
│   └── scorer.py           — Layer 3 trust score engine
├── templates/
│   └── index.html          — Frontend UI
├── static/
│   ├── style.css
│   ├── app.js
│   └── plots/              — Generated spectrograms and MFCC plots
└── uploads/                — Temporary audio file storage

---

## 🎯 Supported Audio Formats

WAV, MP3, FLAC, OGG, M4A

---

## 🌍 Real World Use Cases

- Journalists verifying leaked audio recordings before publishing
- Law enforcement screening audio evidence for tampering
- Banks and cybersecurity teams detecting voice fraud
- Legal proceedings verifying authenticity of audio evidence
- Individuals proving a fabricated audio clip is fake

---

## 📸 Screenshots

[Add screenshots of your results dashboard here]

---

## 👨‍💻 Author

Yeshus S H

[https://www.linkedin.com/in/yeshus-sh-19550235b/]
