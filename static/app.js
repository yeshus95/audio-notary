const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const fileName = document.getElementById('file-name');
const analyzeBtn = document.getElementById('analyze-btn');

let selectedFile = null;

// Drag & drop
dropArea.addEventListener('dragover', e => {
  e.preventDefault();
  dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
  dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', e => {
  e.preventDefault();
  dropArea.classList.remove('dragover');
  handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = file.name;
  filePreview.classList.remove('hidden');
}

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  document.getElementById('results').classList.add('hidden');
  document.getElementById('loading').classList.remove('hidden');

  const formData = new FormData();
  formData.append('audio', selectedFile);

  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.error) throw new Error(data.error);

    renderResults(data);
  } catch (err) {
    alert('Analysis failed: ' + err.message);
  } finally {
    document.getElementById('loading').classList.add('hidden');
  }
});

function renderResults(data) {
  const score = data.trust_score;

  // Gauge arc animation
  const arc = document.getElementById('gauge-arc');
  const circumference = 251;
  const offset = circumference - (score / 100) * circumference;
  arc.style.strokeDashoffset = offset;
  arc.style.stroke = score >= 70 ? '#10b981' : score >= 40 ? '#f59e0b' : '#ef4444';

  document.getElementById('score-value').textContent = score + '%';

  const badge = document.getElementById('verdict-badge');
  badge.textContent = data.verdict;
  badge.className = `verdict-badge ${data.verdict_class}`;

  // Sub-score bars
  const ss = data.sub_scores;
  setBar('meta', ss.metadata_integrity);
  setBar('bio', ss.biological_features);
  setBar('splice', ss.splice_integrity);

  // Bio metrics
  const l2 = data.layer2;
  document.getElementById('m-jitter').textContent = l2.pitch_jitter ?? 'N/A';
  document.getElementById('m-shimmer').textContent = l2.shimmer ?? 'N/A';
  document.getElementById('m-mfcc').textContent = l2.mfcc_variance ?? 'N/A';
  document.getElementById('m-voiced').textContent = l2.voiced_fraction ?? 'N/A';
  document.getElementById('m-silence').textContent = data.layer1.silence_anomalies ?? 0;
  document.getElementById('m-duration').textContent = (data.layer1.duration_seconds ?? '--') + 's';

  // Flags
  const allFlags = [...(data.layer1.flags || []), ...(data.layer2.flags || [])];
  const list = document.getElementById('flags-list');
  list.innerHTML = '';

  if (allFlags.length === 0) {
    list.innerHTML = '<li class="ok">No anomalies detected across all layers</li>';
  } else {
    allFlags.forEach(f => {
      const li = document.createElement('li');
      li.textContent = f;
      list.appendChild(li);
    });
  }

  // Plots
  document.getElementById('spectrogram-img').src = data.spectrogram_url + '?t=' + Date.now();
  document.getElementById('mfcc-img').src = data.mfcc_url + '?t=' + Date.now();

  document.getElementById('results').classList.remove('hidden');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function setBar(id, val) {
  document.getElementById('bar-' + id).style.width = val + '%';
  document.getElementById('val-' + id).textContent = val + '%';
}

/* ==============================
   Speaker Verification
============================== */

const verifySpeakerBtn = document.getElementById('verify-speaker-btn');

if (verifySpeakerBtn) {
  verifySpeakerBtn.addEventListener('click', verifySpeaker);
}

async function verifySpeaker() {
  const referenceFile = document.getElementById('reference-audio').files[0];
  const testFile = document.getElementById('test-audio').files[0];
  const loading = document.getElementById('speaker-loading');
  const resultBox = document.getElementById('speaker-result');

  if (!referenceFile || !testFile) {
    alert('Please upload both reference and test audio files.');
    return;
  }

  const formData = new FormData();
  formData.append('reference_audio', referenceFile);
  formData.append('test_audio', testFile);

  loading.classList.remove('hidden');
  resultBox.classList.add('hidden');

  try {
    const response = await fetch('/verify-speaker', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    loading.classList.add('hidden');

    if (data.error) {
      alert('Speaker verification failed: ' + data.error);
      return;
    }

    document.getElementById('speaker-ecapa').textContent =
  Number(data.ecapa_similarity ?? 0).toFixed(4);

document.getElementById('speaker-mfcc').textContent =
  Number(data.mfcc_similarity ?? 0).toFixed(4);

document.getElementById('speaker-fused').textContent =
  Number(data.fused_similarity ?? 0).toFixed(4);

document.getElementById('speaker-threshold').textContent =
  data.threshold ?? '--';

document.getElementById('speaker-match').textContent =
  data.is_same_speaker ? 'Yes' : 'No';

document.getElementById('speaker-verdict').textContent =
  data.verdict ?? '--';

    resultBox.classList.remove('hidden');
    resultBox.scrollIntoView({ behavior: 'smooth' });

  } catch (error) {
    loading.classList.add('hidden');
    alert('Speaker verification failed: ' + error.message);
  }
}