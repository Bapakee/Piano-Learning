<?php
// =========================================================
// CONFIG
// =========================================================
define('FLASK_URL',   'http://localhost:5000');
define('MAX_FILE_MB', 200);

// =========================================================
// HELPERS
// =========================================================
function call_flask_api(array $file): array {
    $ch = curl_init(FLASK_URL . '/predict');

    curl_setopt_array($ch, [
        CURLOPT_POST           => true,
        CURLOPT_POSTFIELDS     => [
            'video' => new CURLFile($file['tmp_name'], $file['type'], $file['name']),
        ],
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => 600,   // maks 10 menit untuk video panjang
    ]);

    $body = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err  = curl_error($ch);
    curl_close($ch);

    if ($err) {
        return ['error' => 'cURL error: ' . $err .
            '. Pastikan Flask API sudah berjalan: python flask_api.py'];
    }

    $data = json_decode($body, true);
    if ($data === null) {
        return ['error' => 'Response tidak valid dari Flask API. HTTP ' . $code];
    }

    return $data;
}

function label_css(string $label): string {
    return match ($label) {
        'good'             => 'result-good',
        'needs_improvement'=> 'result-needs',
        'poor'             => 'result-poor',
        default            => '',
    };
}

function label_icon(string $label): string {
    return match ($label) {
        'good'             => '✅',
        'needs_improvement'=> '⚠️',
        'poor'             => '❌',
        default            => '🎹',
    };
}

// =========================================================
// PROCESS UPLOAD
// =========================================================
$result       = null;
$error        = null;
$video_url    = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['video'])) {
    $file = $_FILES['video'];
    $maxB = MAX_FILE_MB * 1024 * 1024;

    if ($file['error'] !== UPLOAD_ERR_OK) {
        $error = 'Upload gagal (kode error: ' . $file['error'] . ').';
    } elseif ($file['size'] > $maxB) {
        $error = 'File terlalu besar. Maksimum ' . MAX_FILE_MB . ' MB.';
    } elseif (!in_array(
        strtolower(pathinfo($file['name'], PATHINFO_EXTENSION)),
        ['mp4', 'mov', 'avi', 'mkv', 'webm'], true
    )) {
        $error = 'Format file tidak didukung. Gunakan MP4, MOV, AVI, MKV, atau WEBM.';
    } else {
        $data = call_flask_api($file);

        if (isset($data['error'])) {
            $error = htmlspecialchars($data['error']);
        } else {
            $result    = $data;
            $video_url = FLASK_URL . '/video/' . urlencode($data['annotated_video']);
        }
    }
}
?>
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Piano Playing Level Classifier</title>
<style>
  /* ======================================================
     RESET & VARIABLES
  ====================================================== */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0d0f14;
    --surface:  #161a24;
    --card:     #1e2332;
    --border:   #2a3048;
    --accent:   #6c63ff;
    --accent2:  #a78bfa;
    --text:     #e2e8f0;
    --muted:    #7c8399;
    --good:     #22c55e;
    --needs:    #f59e0b;
    --poor:     #ef4444;
    --radius:   14px;
  }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 16px 80px;
  }

  /* ======================================================
     HEADER
  ====================================================== */
  .header { text-align: center; margin-bottom: 40px; }
  .header .piano-icon {
    font-size: 56px;
    display: block;
    margin-bottom: 12px;
    filter: drop-shadow(0 0 18px rgba(108,99,255,.65));
  }
  .header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -.5px;
  }
  .header p { color: var(--muted); margin-top: 8px; font-size: .95rem; }

  /* badge kecil "Ensemble 5 Model" */
  .ensemble-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(108,99,255,.15);
    border: 1px solid rgba(108,99,255,.35);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: .78rem;
    color: var(--accent2);
    margin-top: 12px;
  }

  /* ======================================================
     CARD
  ====================================================== */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
    width: 100%;
    max-width: 720px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,.4);
  }
  .card h2 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent2);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ======================================================
     UPLOAD AREA
  ====================================================== */
  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s, background .2s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(108,99,255,.06);
  }
  .drop-zone input[type="file"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  .drop-zone .dz-icon  { font-size: 42px; margin-bottom: 10px; }
  .drop-zone .dz-label { font-size: .95rem; color: var(--muted); }
  .drop-zone .dz-label span { color: var(--accent2); font-weight: 600; }
  .drop-zone .dz-hint  { font-size: .8rem; color: var(--muted); margin-top: 6px; }
  .file-name-display   { margin-top: 14px; font-size: .88rem; color: var(--accent2);
                         word-break: break-all; min-height: 1.4em; }

  /* ======================================================
     BUTTON
  ====================================================== */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 13px 28px;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    margin-top: 20px;
    transition: transform .15s, opacity .15s;
  }
  .btn:active  { transform: scale(.97); }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), #818cf8);
    color: #fff;
  }
  .btn-primary:hover    { opacity: .88; }
  .btn-primary:disabled { opacity: .45; cursor: not-allowed; }

  /* ======================================================
     LOADING
  ====================================================== */
  .loading-wrap {
    display: none;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    padding: 20px 0 8px;
  }
  .spinner {
    width: 48px; height: 48px;
    border: 4px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-wrap p { color: var(--muted); font-size: .9rem; text-align: center; }

  /* ======================================================
     ERROR
  ====================================================== */
  .error-box {
    background: rgba(239,68,68,.12);
    border: 1px solid rgba(239,68,68,.35);
    border-radius: 10px;
    padding: 16px 20px;
    color: #fca5a5;
    font-size: .9rem;
    line-height: 1.6;
    margin-bottom: 20px;
  }

  /* ======================================================
     RESULT — badge
  ====================================================== */
  .result-badge {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 22px 26px;
    border-radius: 12px;
    margin-bottom: 24px;
    font-weight: 700;
  }
  .result-good  { background: rgba(34,197,94,.15);  border:1px solid rgba(34,197,94,.4);  color:var(--good); }
  .result-needs { background: rgba(245,158,11,.15); border:1px solid rgba(245,158,11,.4); color:var(--needs); }
  .result-poor  { background: rgba(239,68,68,.15);  border:1px solid rgba(239,68,68,.4);  color:var(--poor); }

  .badge-icon    { font-size: 2.4rem; }
  .badge-info    { display:flex; flex-direction:column; gap:4px; }
  .badge-label   { font-size: 1.4rem; }
  .badge-conf    { font-size: .85rem; opacity: .8; font-weight:500; }
  .badge-models  { font-size: .78rem; opacity: .65; font-weight:400; }

  /* ======================================================
     PROBABILITY BARS
  ====================================================== */
  .prob-section h3 { font-size:.9rem; color:var(--muted); margin-bottom:12px; }
  .prob-bars  { display:flex; flex-direction:column; gap:10px; margin-bottom:28px; }
  .prob-row   { display:flex; align-items:center; gap:12px; }
  .prob-name  { width:140px; font-size:.85rem; color:var(--muted); flex-shrink:0; }
  .prob-track { flex:1; height:10px; background:var(--border); border-radius:99px; overflow:hidden; }
  .prob-fill  { height:100%; border-radius:99px; }
  .fill-good  { background:var(--good); }
  .fill-needs { background:var(--needs); }
  .fill-poor  { background:var(--poor); }
  .prob-pct   { width:52px; text-align:right; font-size:.85rem; }

  /* per-model detail toggle */
  .detail-toggle {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: .8rem;
    cursor: pointer;
    margin-bottom: 20px;
    transition: border-color .2s, color .2s;
  }
  .detail-toggle:hover { border-color: var(--accent2); color: var(--accent2); }
  .model-detail { display:none; }
  .model-detail.open { display:block; }
  .model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 10px;
    margin-bottom: 22px;
  }
  .model-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: .82rem;
  }
  .model-card .mc-title { color: var(--accent2); font-weight:600; margin-bottom:6px; }
  .model-card .mc-row   { display:flex; justify-content:space-between; color:var(--muted); margin-top:3px; }
  .model-card .mc-pred  { font-weight:600; }
  .mc-good  { color: var(--good); }
  .mc-needs { color: var(--needs); }
  .mc-poor  { color: var(--poor); }

  /* ======================================================
     VIDEO
  ====================================================== */
  .video-section h3 { font-size:.95rem; color:var(--muted); margin-bottom:12px; }
  .video-wrap {
    border-radius:10px;
    overflow:hidden;
    border:1px solid var(--border);
    background:#000;
  }
  .video-wrap video { width:100%; max-height:480px; display:block; }

  /* ======================================================
     FOOTER
  ====================================================== */
  footer { margin-top:10px; color:var(--muted); font-size:.8rem; text-align:center; }
</style>
</head>
<body>

<header class="header">
  <span class="piano-icon">🎹</span>
  <h1>Piano Playing Level Classifier</h1>
  <p>Upload video bermain piano — BiLSTM + MediaPipe akan menganalisis tingkatan permainan Anda.</p>
  <span class="ensemble-badge">🤝 Ensemble 5 Model (K-Fold)</span>
</header>

<!-- =====================================================
     FORM UPLOAD  (disembunyikan setelah ada hasil)
===================================================== -->
<?php if (!$result): ?>
<div class="card">
  <h2>📤 Upload Video</h2>

  <?php if ($error): ?>
  <div class="error-box"><?= $error ?></div>
  <?php endif; ?>

  <form id="uploadForm" method="POST" enctype="multipart/form-data">
    <div class="drop-zone" id="dropZone">
      <input type="file" name="video" id="videoInput" accept="video/*" required>
      <div class="dz-icon">🎬</div>
      <p class="dz-label"><span>Klik untuk memilih</span> atau seret video ke sini</p>
      <p class="dz-hint">Format: MP4, MOV, AVI, MKV, WEBM &nbsp;·&nbsp; Maks. <?= MAX_FILE_MB ?> MB</p>
    </div>
    <div class="file-name-display" id="fileNameDisplay"></div>

    <button type="submit" class="btn btn-primary" id="submitBtn">
      🔍 Analisis Video
    </button>

    <div class="loading-wrap" id="loadingWrap">
      <div class="spinner"></div>
      <p>Memproses video dengan MediaPipe &amp; ensemble 5 model BiLSTM...<br>
         Mohon tunggu, proses ini mungkin membutuhkan beberapa menit.</p>
    </div>
  </form>
</div>
<?php endif; /* !$result */ ?>

<?php if ($result): ?>
<?php
  $css   = label_css($result['label']);
  $icon  = label_icon($result['label']);
  $probs = $result['probabilities'];
  $n_models = $result['model_count'] ?? 5;

  $fill_map = [
    'good'             => 'fill-good',
    'needs_improvement'=> 'fill-needs',
    'poor'             => 'fill-poor',
  ];
  $name_map = [
    'good'             => 'Good',
    'needs_improvement'=> 'Needs Improvement',
    'poor'             => 'Poor',
  ];
  $pred_css = [
    'good'             => 'mc-good',
    'needs_improvement'=> 'mc-needs',
    'poor'             => 'mc-poor',
  ];
?>
<!-- =====================================================
     HASIL KLASIFIKASI
===================================================== -->
<div class="card">
  <h2>🎯 Hasil Klasifikasi</h2>

  <!-- Badge utama -->
  <div class="result-badge <?= $css ?>">
    <span class="badge-icon"><?= $icon ?></span>
    <div class="badge-info">
      <span class="badge-label"><?= htmlspecialchars($result['display']) ?></span>
      <span class="badge-conf">Confidence: <?= $result['confidence'] ?>%</span>
      <span class="badge-models">Hasil ensemble dari <?= $n_models ?> model</span>
    </div>
  </div>

  <!-- Probability bars -->
  <div class="prob-section">
    <h3>Distribusi Probabilitas (rata-rata ensemble)</h3>
    <div class="prob-bars">
      <?php foreach ($probs as $key => $pct): ?>
      <div class="prob-row">
        <span class="prob-name"><?= $name_map[$key] ?></span>
        <div class="prob-track">
          <div class="prob-fill <?= $fill_map[$key] ?>" style="width:<?= $pct ?>%"></div>
        </div>
        <span class="prob-pct <?= $pred_css[$key] ?>"><?= $pct ?>%</span>
      </div>
      <?php endforeach; ?>
    </div>
  </div>

  <!-- Per-model detail (dari Flask jika tersedia) -->
  <?php if (!empty($result['per_model'])): ?>
  <button class="detail-toggle" onclick="toggleDetail(this)">▼ Lihat detail per model</button>
  <div class="model-detail" id="modelDetail">
    <div class="model-grid">
      <?php foreach ($result['per_model'] as $i => $m): ?>
      <div class="model-card">
        <div class="mc-title">Fold <?= $i + 1 ?></div>
        <div class="mc-row">
          <span>Prediksi</span>
          <span class="mc-pred <?= $pred_css[$m['label']] ?>"><?= $name_map[$m['label']] ?></span>
        </div>
        <?php foreach ($m['probabilities'] as $k => $v): ?>
        <div class="mc-row">
          <span><?= $name_map[$k] ?></span>
          <span><?= $v ?>%</span>
        </div>
        <?php endforeach; ?>
      </div>
      <?php endforeach; ?>
    </div>
  </div>
  <?php endif; ?>

  <!-- Annotated video -->
  <?php if ($video_url): ?>
  <div class="video-section">
    <h3>🖐️ Video dengan Anotasi MediaPipe</h3>
    <div class="video-wrap">
      <video controls playsinline>
        <source src="<?= htmlspecialchars($video_url) ?>" type="video/mp4">
        Browser Anda tidak mendukung pemutaran video inline.
      </video>
    </div>
  </div>
  <?php endif; ?>
</div>
<?php endif; ?>

<?php if ($result): ?>
<div style="max-width:720px;width:100%;text-align:center;margin-bottom:32px;">
  <a href="index.php" style="
    display:inline-flex;align-items:center;gap:8px;
    padding:12px 32px;border-radius:8px;
    background:rgba(108,99,255,.15);border:1px solid rgba(108,99,255,.4);
    color:#a78bfa;font-size:.95rem;font-weight:600;text-decoration:none;
    transition:background .2s;">
    ← Analisis Video Lain
  </a>
</div>
<?php endif; ?>

<footer>Piano Learning Classification &nbsp;·&nbsp; MediaPipe + BiLSTM Ensemble &nbsp;·&nbsp; Tugas Akhir Patrick</footer>

<script>
// =========================================================
// DRAG & DROP
// =========================================================
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('videoInput');
const fileLabel = document.getElementById('fileNameDisplay');
const form      = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const loading   = document.getElementById('loadingWrap');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) {
    fileLabel.textContent = '📁 ' + fileInput.files[0].name;
  }
});

['dragenter', 'dragover'].forEach(e =>
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); })
);
['dragleave', 'drop'].forEach(e =>
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); })
);
dropZone.addEventListener('drop', e => {
  const f = e.dataTransfer.files;
  if (f[0]) { fileInput.files = f; fileLabel.textContent = '📁 ' + f[0].name; }
});

// =========================================================
// SHOW LOADING
// =========================================================
form.addEventListener('submit', () => {
  submitBtn.disabled    = true;
  submitBtn.textContent = 'Memproses...';
  loading.style.display = 'flex';
});

// =========================================================
// TOGGLE DETAIL PER MODEL
// =========================================================
function toggleDetail(btn) {
  const box = document.getElementById('modelDetail');
  const open = box.classList.toggle('open');
  btn.textContent = (open ? '▲ ' : '▼ ') + 'Lihat detail per model';
}
</script>

</body>
</html>
