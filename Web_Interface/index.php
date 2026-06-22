<?php
// Base URL of the Flask API and the maximum allowed upload size in megabytes
define('FLASK_URL',   'http://localhost:5000');
define('MAX_FILE_MB', 200);

// Sends the uploaded video to the Flask /predict endpoint via cURL
// and returns the decoded JSON response as an associative array.
function call_flask_api(array $file): array {
    $ch = curl_init(FLASK_URL . '/predict');

    curl_setopt_array($ch, [
        CURLOPT_POST           => true,
        CURLOPT_POSTFIELDS     => [
            'video' => new CURLFile($file['tmp_name'], $file['type'], $file['name']),
        ],
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => 600,
    ]);

    $body = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err  = curl_error($ch);
    curl_close($ch);

    if ($err) {
        return ['error' => 'cURL error: ' . $err . '. Pastikan Flask API sudah berjalan: python flask_api.py'];
    }

    $data = json_decode($body, true);
    if ($data === null) {
        return ['error' => 'Response tidak valid dari Flask API. HTTP ' . $code];
    }

    return $data;
}

// Maps a label key to the CSS class used to style the result badge
function label_css(string $label): string {
    return match ($label) {
        'good'             => 'result-good',
        'needs_improvement'=> 'result-needs',
        'poor'             => 'result-poor',
        default            => '',
    };
}

$result    = null;
$error     = null;
$video_url = null;

// Validate the upload and forward it to the Flask API on form submission

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
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #ffffff;
    color: #1a1a1a;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 16px 80px;
  }

  /* HEADER */
  .header {
    text-align: center;
    margin-bottom: 40px;
  }
  .header h1 {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1a1a1a;
    letter-spacing: -0.3px;
  }
  .header p {
    margin-top: 6px;
    font-size: .875rem;
    color: #666;
  }
  .header .sub {
    display: inline-block;
    margin-top: 10px;
    font-size: .75rem;
    color: #999;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 3px 10px;
  }

  /* CARD */
  .card {
    width: 100%;
    max-width: 640px;
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 28px;
    margin-bottom: 20px;
  }
  .card h2 {
    font-size: .9rem;
    font-weight: 600;
    color: #444;
    margin-bottom: 18px;
    text-transform: uppercase;
    letter-spacing: .05em;
  }

  /* ERROR */
  .error-box {
    border: 1px solid #f5c6c6;
    background: #fff5f5;
    border-radius: 6px;
    padding: 12px 16px;
    color: #c0392b;
    font-size: .875rem;
    margin-bottom: 16px;
  }

  /* DROP ZONE */
  .drop-zone {
    border: 1px dashed #bbb;
    border-radius: 6px;
    padding: 32px 20px;
    text-align: center;
    cursor: pointer;
    position: relative;
    transition: border-color .15s, background .15s;
  }
  .drop-zone:hover,
  .drop-zone.drag-over {
    border-color: #888;
    background: #fafafa;
  }
  .drop-zone input[type="file"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  .drop-zone .dz-label {
    font-size: .875rem;
    color: #555;
  }
  .drop-zone .dz-label span {
    color: #1a1a1a;
    font-weight: 600;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .drop-zone .dz-hint {
    font-size: .78rem;
    color: #999;
    margin-top: 6px;
  }
  .file-name-display {
    margin-top: 10px;
    font-size: .8rem;
    color: #555;
    min-height: 1.2em;
    word-break: break-all;
  }

  /* VIDEO PREVIEW */
  .preview-section {
    display: none;
    margin-top: 16px;
  }
  .preview-section h3 {
    font-size: .78rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 8px;
  }
  .preview-wrap {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    overflow: hidden;
    background: #000;
  }
  .preview-wrap video {
    width: 100%;
    max-height: 360px;
    display: block;
  }

  /* BUTTON */
  .btn {
    display: block;
    width: 100%;
    padding: 11px;
    margin-top: 16px;
    border: none;
    border-radius: 6px;
    font-size: .9rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .15s;
  }
  .btn-primary {
    background: #1a1a1a;
    color: #fff;
  }
  .btn-primary:hover    { background: #333; }
  .btn-primary:disabled { background: #aaa; cursor: not-allowed; }

  /* LOADING */
  .loading-wrap {
    display: none;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 16px 0 4px;
  }
  .spinner {
    width: 32px; height: 32px;
    border: 3px solid #e0e0e0;
    border-top-color: #1a1a1a;
    border-radius: 50%;
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-wrap p { font-size: .8rem; color: #888; text-align: center; }

  /* RESULT BADGE */
  .result-badge {
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 20px;
    border-left: 4px solid;
  }
  .result-good  { border-color: #2e7d32; background: #f6fbf6; }
  .result-needs { border-color: #e65100; background: #fffaf5; }
  .result-poor  { border-color: #c62828; background: #fff5f5; }

  .badge-label {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1a1a1a;
  }
  .result-good  .badge-label { color: #2e7d32; }
  .result-needs .badge-label { color: #e65100; }
  .result-poor  .badge-label { color: #c62828; }

  .badge-meta {
    margin-top: 4px;
    font-size: .8rem;
    color: #777;
  }

  /* PROBABILITY BARS */
  .prob-section { margin-bottom: 24px; }
  .prob-section h3 {
    font-size: .78rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 12px;
  }
  .prob-bars { display: flex; flex-direction: column; gap: 8px; }
  .prob-row  { display: flex; align-items: center; gap: 10px; }
  .prob-name {
    width: 140px;
    font-size: .8rem;
    color: #555;
    flex-shrink: 0;
  }
  .prob-track {
    flex: 1;
    height: 6px;
    background: #ebebeb;
    border-radius: 99px;
    overflow: hidden;
  }
  .prob-fill  { height: 100%; border-radius: 99px; background: #1a1a1a; }
  .fill-good  { background: #2e7d32; }
  .fill-needs { background: #e65100; }
  .fill-poor  { background: #c62828; }
  .prob-pct {
    width: 44px;
    text-align: right;
    font-size: .8rem;
    color: #444;
  }

  /* VIDEO */
  .video-section h3 {
    font-size: .78rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 10px;
  }
  .video-wrap {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    overflow: hidden;
    background: #000;
  }
  .video-wrap video {
    width: 100%;
    max-height: 460px;
    display: block;
  }

  /* BACK LINK */
  .back-link {
    max-width: 640px;
    width: 100%;
    margin-bottom: 24px;
  }
  .back-link a {
    font-size: .875rem;
    color: #555;
    text-decoration: none;
    border-bottom: 1px solid #ccc;
    padding-bottom: 1px;
  }
  .back-link a:hover { color: #1a1a1a; border-color: #1a1a1a; }

  /* FOOTER */
  footer {
    font-size: .75rem;
    color: #bbb;
    text-align: center;
    margin-top: 8px;
  }
</style>
</head>
<body>

<header class="header">
  <h1>Piano Playing Level Classifier</h1>
  <p>Upload video bermain piano untuk mendapatkan hasil klasifikasi tingkatan permainan.</p>
</header>

<?php if (!$result): ?>
<!-- FORM UPLOAD -->
<div class="card">
  <h2>Upload Video</h2>

  <?php if ($error): ?>
  <div class="error-box"><?= $error ?></div>
  <?php endif; ?>

  <form id="uploadForm" method="POST" enctype="multipart/form-data">
    <div class="drop-zone" id="dropZone">
      <input type="file" name="video" id="videoInput" accept="video/*" required>
      <p class="dz-label"><span>Klik untuk memilih file</span> atau seret ke sini</p>
      <p class="dz-hint">MP4, MOV, AVI, MKV, WEBM &nbsp;&mdash;&nbsp; Maks. <?= MAX_FILE_MB ?> MB</p>
    </div>
    <div class="file-name-display" id="fileNameDisplay"></div>

    <div class="preview-section" id="previewSection">
      <h3>Preview Video</h3>
      <div class="preview-wrap">
        <video id="previewVideo" controls playsinline muted></video>
      </div>
    </div>

    <button type="submit" class="btn btn-primary" id="submitBtn">Analisis Video</button>

    <div class="loading-wrap" id="loadingWrap">
      <div class="spinner"></div>
      <p>Memproses video, mohon tunggu...</p>
    </div>
  </form>
</div>
<?php endif; ?>

<?php if ($result): ?>
<?php
  $css      = label_css($result['label']);
  $probs    = $result['probabilities'];
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
?>
<!-- HASIL KLASIFIKASI -->
<div class="card">
  <h2>Hasil Klasifikasi</h2>

  <div class="result-badge <?= $css ?>">
    <div class="badge-label"><?= htmlspecialchars($result['display']) ?></div>
    <div class="badge-meta">
      Confidence: <?= $result['confidence'] ?>% &nbsp;&mdash;&nbsp; Ensemble dari <?= $n_models ?> model
    </div>
  </div>

  <div class="prob-section">
    <h3>Distribusi Probabilitas</h3>
    <div class="prob-bars">
      <?php foreach ($probs as $key => $pct): ?>
      <div class="prob-row">
        <span class="prob-name"><?= $name_map[$key] ?></span>
        <div class="prob-track">
          <div class="prob-fill <?= $fill_map[$key] ?>" style="width:<?= $pct ?>%"></div>
        </div>
        <span class="prob-pct"><?= $pct ?>%</span>
      </div>
      <?php endforeach; ?>
    </div>
  </div>

  <?php if ($video_url): ?>
  <div class="video-section">
    <h3>Video Anotasi MediaPipe</h3>
    <div class="video-wrap">
      <video controls playsinline>
        <source src="<?= htmlspecialchars($video_url) ?>" type="video/mp4">
        Browser Anda tidak mendukung pemutaran video.
      </video>
    </div>
  </div>
  <?php endif; ?>
</div>

<div class="back-link">
  <a href="index.php">Kembali &amp; analisis video lain</a>
</div>
<?php endif; ?>

<footer>Patrick Andersen Kanginan &nbsp;&mdash;&nbsp; 160421123</footer>

<script>
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('videoInput');
const fileLabel = document.getElementById('fileNameDisplay');
const form      = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const loading   = document.getElementById('loadingWrap');

// Upload form is only rendered when there is no result to display
if (fileInput) {
  const previewSection = document.getElementById('previewSection');
  const previewVideo   = document.getElementById('previewVideo');
  let   previewObjectURL = null;

  // Show the selected filename and a local video preview beneath the drop zone
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;

    fileLabel.textContent = file.name;

    // Release the previous object URL to free browser memory
    if (previewObjectURL) URL.revokeObjectURL(previewObjectURL);
    previewObjectURL    = URL.createObjectURL(file);
    previewVideo.src    = previewObjectURL;
    previewSection.style.display = 'block';
  });

  // Highlight the drop zone while the user is dragging a file over it
  ['dragenter','dragover'].forEach(e =>
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); })
  );
  ['dragleave','drop'].forEach(e =>
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); })
  );

  // Accept a file dropped directly onto the zone and trigger the same preview logic
  dropZone.addEventListener('drop', e => {
    const f = e.dataTransfer.files;
    if (f[0]) {
      fileInput.files = f;
      fileInput.dispatchEvent(new Event('change'));
    }
  });

  // Disable the button and show the spinner while the server is processing
  form.addEventListener('submit', () => {
    submitBtn.disabled    = true;
    submitBtn.textContent = 'Memproses...';
    loading.style.display = 'flex';
  });
}
</script>

</body>
</html>
