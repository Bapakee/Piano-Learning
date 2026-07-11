<?php
// ============================================================
// index.php — Piano Playing Level Classifier (Web Interface)
// ============================================================
// This file is the web page that the user opens in the browser.
// It does three things:
//
//   1. UPLOAD FORM — lets the user pick a piano video file and
//      send it to the Flask API for analysis.
//
//   2. PHP VALIDATION — checks that the file is the right type
//      and size before sending it, and shows error messages if
//      something is wrong.
//
//   3. RESULT DISPLAY — once the API replies, it renders the
//      classification result (Good / Needs Improvement / Poor),
//      a probability chart, detection statistics, and the
//      annotated video with MediaPipe landmarks.
//
// How the page and the API communicate:
//   • The user submits the form → PHP sends the video to
//     http://localhost:5000/predict using cURL.
//   • The API returns a JSON object with the result.
//   • PHP reads the JSON and builds the HTML result section.
// ============================================================


// ── Configuration constants ─────────────────────────────────
// FLASK_URL    — address where the Python Flask API is running
// MAX_FILE_MB  — maximum allowed video file size in megabytes
define('FLASK_URL',   'http://localhost:5000');
define('MAX_FILE_MB', 50);


// ── call_flask_api() ────────────────────────────────────────
// Sends the uploaded video file to the Flask API and returns
// the decoded JSON response as a PHP array.
//
// Uses cURL — a built-in PHP tool for making HTTP requests.
// The video is attached as a file upload (multipart form data),
// exactly as if a browser were submitting a form.
//
// If the connection fails or the response is not valid JSON,
// an ['error' => '...'] array is returned instead.
function call_flask_api(array $file): array {
    // Open a cURL connection to the /predict endpoint
    $ch = curl_init(FLASK_URL . '/predict');

    curl_setopt_array($ch, [
        CURLOPT_POST           => true,                     // use POST method
        CURLOPT_POSTFIELDS     => [
            // Attach the video file under the field name "video"
            // (Flask API reads request.files["video"])
            'video' => new CURLFile($file['tmp_name'], $file['type'], $file['name']),
        ],
        CURLOPT_RETURNTRANSFER => true,    // return the response as a string
        CURLOPT_TIMEOUT        => 600,     // wait up to 10 minutes (video analysis is slow)
    ]);

    $body = curl_exec($ch);                             // send the request
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);      // HTTP status code (e.g. 200, 422)
    $err  = curl_error($ch);                            // cURL-level error (e.g. connection refused)
    curl_close($ch);

    // If cURL itself failed (API not running, network error, timeout)
    if ($err) {
        return ['error' => 'cURL error: ' . $err . '. Make sure the Flask API is running: python flask_api.py'];
    }

    // Try to decode the JSON body returned by the API
    $data = json_decode($body, true);
    if ($data === null) {
        return ['error' => 'Invalid response from Flask API. HTTP ' . $code];
    }

    return $data;
}


// ── label_css() ─────────────────────────────────────────────
// Maps each classification label to a CSS class name used to
// style the result badge with the correct colour scheme:
//   good             → green border / green text
//   needs_improvement→ orange border / orange text
//   poor             → red border / red text
function label_css(string $label): string {
    return match ($label) {
        'good'             => 'result-good',
        'needs_improvement'=> 'result-needs',
        'poor'             => 'result-poor',
        default            => '',
    };
}


// ── Page-level state variables ──────────────────────────────
// $result    — the decoded API response (null if no upload yet)
// $error     — any error message to show to the user
// $video_url — URL of the annotated video file to play back
$result    = null;
$error     = null;
$video_url = null;


// ── Form submission handling ────────────────────────────────
// This block runs only when the user submits the upload form
// (REQUEST_METHOD is POST) and a file was attached.
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['video'])) {
    $file = $_FILES['video'];
    $maxB = MAX_FILE_MB * 1024 * 1024;   // convert MB → bytes

    // Check 1: did the file upload itself succeed?
    // UPLOAD_ERR_OK means no PHP-level error during upload.
    if ($file['error'] !== UPLOAD_ERR_OK) {
        $error = 'Upload failed (error code: ' . $file['error'] . ').';

    // Check 2: is the file within the size limit?
    } elseif ($file['size'] > $maxB) {
        $error = 'File too large. Maximum ' . MAX_FILE_MB . ' MB.';

    // Check 3: is the file extension mp4 or mov?
    } elseif (!in_array(
        strtolower(pathinfo($file['name'], PATHINFO_EXTENSION)),
        ['mp4', 'mov'], true
    )) {
        $error = 'Unsupported file format. Please use MP4 or MOV.';

    // All checks passed → send the file to the Flask API
    } else {
        $data = call_flask_api($file);

        if (isset($data['error'])) {
            // The API returned an error (e.g. no hand detected, video too short)
            $error = htmlspecialchars($data['error']);
        } else {
            // Success — store the result and build the annotated video URL
            $result    = $data;
            $video_url = FLASK_URL . '/video/' . urlencode($data['annotated_video']);
        }
    }
}
?>
<!DOCTYPE html>
<!-- lang="en" tells browsers and screen readers this page is in English -->
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Piano Playing Level Classifier</title>
<style>
  /* ── CSS RESET ─────────────────────────────────────────── */
  /* Remove default margin/padding and use border-box sizing  */
  /* so padding does not add to element width unexpectedly.   */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* ── PAGE LAYOUT ────────────────────────────────────────── */
  /* Centers all content vertically and horizontally,          */
  /* with comfortable top/bottom padding.                       */
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

  /* ── HEADER ─────────────────────────────────────────────── */
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

  /* ── CARD ───────────────────────────────────────────────── */
  /* White bordered box used to group related content sections */
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

  /* ── ERROR BOX ──────────────────────────────────────────── */
  /* Red-tinted box shown when validation or the API returns   */
  /* an error message.                                          */
  .error-box {
    border: 1px solid #f5c6c6;
    background: #fff5f5;
    border-radius: 6px;
    padding: 12px 16px;
    color: #c0392b;
    font-size: .875rem;
    margin-bottom: 16px;
  }

  /* ── DROP ZONE ──────────────────────────────────────────── */
  /* The dashed rectangle where the user drops or selects a    */
  /* video file. The actual <input type="file"> is invisible   */
  /* and sits on top of the whole zone via absolute positioning.*/
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
  /* The real file input is invisible but covers the drop zone */
  .drop-zone input[type="file"] {
    position: absolute;
    inset: 0;       /* shorthand for top/right/bottom/left = 0 */
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

  /* ── VIDEO PREVIEW ──────────────────────────────────────── */
  /* Shown after the user selects a file, before submitting.   */
  /* Hidden by default; JavaScript makes it visible.           */
  .preview-section {
    display: none;
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

  /* ── BUTTONS ────────────────────────────────────────────── */
  /* .btn        — shared base style for all buttons           */
  /* .btn-primary— dark "Analyze Video" button                 */
  /* .btn-secondary — white "Change Video" button              */
  .btn-secondary {
    background: #fff;
    color: #1a1a1a;
    border: 1px solid #ccc;
    margin-top: 10px;
  }
  .btn-secondary:hover { background: #f5f5f5; border-color: #999; }

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

  /* ── LOADING SPINNER ────────────────────────────────────── */
  /* Shown while the Flask API is processing the video.        */
  /* The spinning animation is created with a CSS keyframe.    */
  .loading-wrap {
    display: none;     /* hidden until the form is submitted */
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 16px 0 4px;
  }
  .spinner {
    width: 32px; height: 32px;
    border: 3px solid #e0e0e0;
    border-top-color: #1a1a1a;    /* only the top is dark → creates rotation effect */
    border-radius: 50%;
    animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-wrap p { font-size: .8rem; color: #888; text-align: center; }

  /* ── RESULT BADGE ───────────────────────────────────────── */
  /* Coloured box showing the classification label.             */
  /* The left border colour changes based on the label:         */
  /*   Good             → green   (.result-good)               */
  /*   Needs Improvement → orange  (.result-needs)             */
  /*   Poor             → red     (.result-poor)               */
  .result-badge {
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 20px;
    border-left: 4px solid;
  }
  .result-good  { border-color: #2e7d32; background: #f6fbf6; }
  .result-needs { border-color: #e65100; background: #fffaf5; }
  .result-poor  { border-color: #c62828; background: #fff5f5; }

  /* Large bold label text (e.g. "Good") coloured to match border */
  .badge-label {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1a1a1a;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .result-good  .badge-label { color: #2e7d32; }
  .result-needs .badge-label { color: #e65100; }
  .result-poor  .badge-label { color: #c62828; }

  /* Small meta line below the label: "Confidence: 91% — Ensemble of 5 models" */
  .badge-meta {
    margin-top: 4px;
    font-size: .8rem;
    color: #777;
  }

  /* ── PROBABILITY BARS ───────────────────────────────────── */
  /* Horizontal bars showing the probability for each class.   */
  /* The bar width is set inline by PHP as a percentage.       */
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
    flex-shrink: 0;    /* do not shrink the name column */
  }
  .prob-track {
    flex: 1;           /* the track expands to fill the remaining space */
    height: 6px;
    background: #ebebeb;
    border-radius: 99px;
    overflow: hidden;
  }
  /* The filled portion of each bar */
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

  /* ── VIDEO PLAYER ───────────────────────────────────────── */
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

  /* ── BADGE TOP ROW ──────────────────────────────────────── */
  /* Flex row that places the label text and the ⓘ icon       */
  /* side by side inside the result badge.                     */
  .badge-top-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── INFO ICON BUTTON (ⓘ) ──────────────────────────────── */
  /* Small circular icon next to the label.  On hover it       */
  /* reveals a tooltip with a short plain-language explanation. */
  .info-btn {
    position: relative;      /* anchor for the tooltip's absolute position */
    display: inline-flex;
    align-items: center;
    cursor: pointer;
    color: #888;
    flex-shrink: 0;
  }
  .info-btn:hover { color: #333; }

  /* The tooltip bubble — hidden by default, shown on hover/focus */
  .info-tooltip {
    display: none;
    position: absolute;
    left: 26px;             /* appears to the right of the icon */
    top: 50%;
    transform: translateY(-50%);
    background: #1a1a1a;
    color: #f0f0f0;
    font-size: .8rem;
    font-weight: 400;
    line-height: 1.6;
    padding: 12px 15px;
    border-radius: 8px;
    width: 280px;
    z-index: 100;           /* float above all other content */
    box-shadow: 0 6px 20px rgba(0,0,0,.25);
    white-space: normal;
    pointer-events: none;   /* tooltip itself cannot be hovered (avoids flicker) */
  }
  /* Small left-pointing triangle connecting the tooltip to the icon */
  .info-tooltip::before {
    content: '';
    position: absolute;
    right: 100%;            /* sits just to the left of the tooltip box */
    top: 50%;
    transform: translateY(-50%);
    border: 6px solid transparent;
    border-right-color: #1a1a1a;
  }
  /* Show the tooltip when the icon is hovered or focused (keyboard) */
  .info-btn:hover .info-tooltip,
  .info-btn:focus .info-tooltip { display: block; }

  /* ── WHY-BOX ────────────────────────────────────────────── */
  /* Expanded explanation panel shown directly on the page,    */
  /* below the result badge.  Shows the full dynamic text      */
  /* from build_explanation() with bold highlights.            */
  .why-box {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 20px;
    background: #fafafa;
    font-size: .82rem;
    color: #333;
    line-height: 1.65;
  }
  /* Small header inside the why-box: "ⓘ WHY IS THE RESULT GOOD?" */
  .why-box-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: .72rem;
    font-weight: 700;
    color: #888;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: 8px;
  }
  .why-box-title svg { flex-shrink: 0; color: #aaa; }
  .why-box p { margin: 0; }

  /* ── DETECTION STATISTICS ───────────────────────────────── */
  /* Row of small stat boxes: Total Frames / Detected Frames / */
  /* Detection Rate / Key Press Frames.                         */
  .detect-stats {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;     /* wraps to a second row on narrow screens */
    margin-bottom: 24px;
  }
  .stat-box {
    flex: 1;
    min-width: 100px;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 14px 12px;
    text-align: center;
  }
  .stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a1a1a;
  }
  .stat-lbl {
    font-size: .72rem;
    color: #999;
    margin-top: 4px;
  }

  /* ── PER-FINGER ACTIVITY ────────────────────────────────── */
  /* Horizontal bars (same style as probability bars) showing  */
  /* what percentage of the video each finger was pressing.    */
  .finger-section { margin-bottom: 24px; }
  .finger-section h3 {
    font-size: .78rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 12px;
  }

  /* ── ANNOTATION HINT TEXT ───────────────────────────────── */
  /* Small grey text above the annotated video explaining what  */
  /* the colours and numbers mean.                              */
  .annot-hint {
    font-size: .8rem;
    color: #777;
    margin-bottom: 4px;
  }

  /* ── BACK LINK ──────────────────────────────────────────── */
  /* Link at the bottom of the result page to go back and      */
  /* analyse another video.                                     */
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

  /* ── FOOTER ─────────────────────────────────────────────── */
  footer {
    font-size: .75rem;
    color: #bbb;
    text-align: center;
    margin-top: 8px;
  }
</style>
</head>
<body>

<!-- ── PAGE HEADER ────────────────────────────────────────── -->
<header class="header">
  <h1>Piano Playing Level Classifier</h1>
  <p>Upload a piano playing video to get a classification of your playing level.</p>
</header>

<?php if (!$result): ?>
<!-- ── UPLOAD FORM ────────────────────────────────────────── -->
<!-- Shown only when no result has been returned yet.          -->
<div class="card">
  <h2>Upload Video</h2>

  <?php if ($error): ?>
  <!-- Error box appears above the form if validation failed   -->
  <div class="error-box"><?= $error ?></div>
  <?php endif; ?>

  <!-- enctype="multipart/form-data" is required for file uploads -->
  <form id="uploadForm" method="POST" enctype="multipart/form-data">

    <!-- The real file input is hidden; it is triggered by clicking
         the drop zone label or the re-upload button via JavaScript. -->
    <input type="file" name="video" id="videoInput" accept="video/*" required style="display:none">

    <!-- Visible drop zone — clicking it opens the file picker -->
    <div id="dropZone" class="drop-zone">
      <label for="videoInput" style="display:block;cursor:pointer;">
        <p class="dz-label"><span>Click to select a file</span> or drag it here</p>
        <p class="dz-hint">MP4, MOV &nbsp;&mdash;&nbsp; Max. <?= MAX_FILE_MB ?> MB</p>
      </label>
    </div>

    <!-- Preview section: hidden until a file is selected -->
    <div class="preview-section" id="previewSection">
      <h3>Video Preview</h3>
      <div class="preview-wrap">
        <video id="previewVideo" controls playsinline muted></video>
      </div>
      <!-- Clicking this button resets the form back to the drop zone -->
      <button type="button" class="btn btn-secondary" id="reuploadBtn">Change Video</button>
    </div>

    <!-- Submit button: hidden until a file is chosen -->
    <button type="submit" class="btn btn-primary" id="submitBtn" style="display:none">Analyze Video</button>

    <!-- Spinner shown while the API is processing (hides after the page reloads) -->
    <div class="loading-wrap" id="loadingWrap">
      <div class="spinner"></div>
      <p>Processing video, please wait...</p>
    </div>
  </form>
</div>
<?php endif; ?>

<?php if ($result): ?>
<?php
  // ── Extract values from the API response ──────────────────
  $css      = label_css($result['label']);            // CSS class for the badge colour
  $probs    = $result['probabilities'];               // probability for each label (0–100%)
  $n_models = $result['model_count'] ?? 5;            // number of AI models in the ensemble
  $expl     = htmlspecialchars($result['short_explanation'] ?? '');  // short tooltip text (HTML-safe)
  $full_expl = $result['explanation'] ?? '';          // long explanation (may contain <strong> tags)
  $analysis = $result['analysis'] ?? [];              // frame stats and finger activity data

  // Colour class for each probability bar fill
  $fill_map = [
    'good'             => 'fill-good',
    'needs_improvement'=> 'fill-needs',
    'poor'             => 'fill-poor',
  ];

  // Human-readable label name for the probability bar rows
  $name_map = [
    'good'             => 'Good',
    'needs_improvement'=> 'Needs Improvement',
    'poor'             => 'Poor',
  ];
?>

<!-- ── CLASSIFICATION RESULT CARD ────────────────────────── -->
<div class="card">
  <h2>Classification Result</h2>

  <!-- Result badge: coloured box with the label and confidence -->
  <div class="result-badge <?= $css ?>">
    <div class="badge-top-row">

      <!-- Large label text (e.g. "Good") -->
      <div class="badge-label"><?= htmlspecialchars($result['display']) ?></div>

      <!-- ⓘ info icon — hover to see the short tooltip explanation -->
      <div class="info-btn" tabindex="0" aria-label="Why this result">
        <svg width="18" height="18" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="10" cy="10" r="9" stroke="currentColor" stroke-width="1.5"/>
          <path d="M10 9v5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          <circle cx="10" cy="6.5" r="0.75" fill="currentColor"/>
        </svg>
        <!-- Tooltip bubble: contains the short one-paragraph explanation -->
        <div class="info-tooltip"><?= $expl ?></div>
      </div>

    </div>
    <!-- Confidence % and ensemble size shown below the label -->
    <div class="badge-meta">
      Confidence: <?= $result['confidence'] ?>% &nbsp;&mdash;&nbsp; Ensemble of <?= $n_models ?> models
    </div>
  </div>

  <?php if ($full_expl): ?>
  <!-- ── WHY-BOX: full personalised explanation ─────────── -->
  <!-- Shows a longer dynamic text built from the video's     -->
  <!-- actual data (detection rate, finger activity, etc.)    -->
  <div class="why-box">
    <div class="why-box-title">
      <!-- Small ⓘ icon matching the badge tooltip icon -->
      <svg width="14" height="14" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="10" cy="10" r="9" stroke="currentColor" stroke-width="1.5"/>
        <path d="M10 9v5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        <circle cx="10" cy="6.5" r="0.75" fill="currentColor"/>
      </svg>
      Why is the result <em><?= htmlspecialchars($name_map[$result['label']] ?? $result['label']) ?></em>?
    </div>
    <!-- Output the full explanation directly — it contains <strong> tags for bold highlights -->
    <p><?= $full_expl ?></p>
  </div>
  <?php endif; ?>

  <!-- ── PROBABILITY BARS ──────────────────────────────── -->
  <!-- One bar per class showing how confident the ensemble  -->
  <!-- was for each possible label.                          -->
  <div class="prob-section">
    <h3>Class Probability Distribution</h3>
    <div class="prob-bars">
      <?php foreach ($probs as $key => $pct): ?>
      <div class="prob-row">
        <!-- Label name column (e.g. "Good") -->
        <span class="prob-name"><?= $name_map[$key] ?></span>
        <!-- Grey track; the coloured fill width = probability % -->
        <div class="prob-track">
          <div class="prob-fill <?= $fill_map[$key] ?>" style="width:<?= $pct ?>%"></div>
        </div>
        <!-- Numeric percentage on the right -->
        <span class="prob-pct"><?= $pct ?>%</span>
      </div>
      <?php endforeach; ?>
    </div>
  </div>
</div>

<?php if (!empty($analysis)): ?>
<!-- ── DETECTION PROCESS CARD ────────────────────────────── -->
<!-- Shows statistics gathered during the MediaPipe analysis.  -->
<div class="card">
  <h2>Detection Process</h2>

  <!-- ── Frame statistics ───────────────────────────────── -->
  <!-- Four stat boxes showing video and detection summary.   -->
  <div class="detect-stats">
    <div class="stat-box">
      <!-- Total number of frames in the uploaded video -->
      <div class="stat-val"><?= $analysis['total_frames'] ?></div>
      <div class="stat-lbl">Total Frames</div>
    </div>
    <div class="stat-box">
      <!-- Frames where MediaPipe successfully found a hand -->
      <div class="stat-val"><?= $analysis['detection_frames'] ?></div>
      <div class="stat-lbl">Detected Frames</div>
    </div>
    <div class="stat-box">
      <!-- Percentage of frames with a detected hand -->
      <div class="stat-val"><?= $analysis['detection_rate'] ?>%</div>
      <div class="stat-lbl">Detection Rate</div>
    </div>
    <div class="stat-box">
      <!-- Frames where at least one finger was pressing a key -->
      <div class="stat-val"><?= $analysis['press_frames'] ?></div>
      <div class="stat-lbl">Key Press Frames</div>
    </div>
  </div>

  <?php if (!empty($analysis['finger_activity'])): ?>
  <!-- ── Per-finger activity bars ──────────────────────── -->
  <!-- Shows which fingers were used most during the video.  -->
  <!-- The percentage = frames that finger pressed / total.  -->
  <div class="finger-section">
    <h3>Per-Finger Activity</h3>
    <div class="prob-bars">
      <?php foreach ($analysis['finger_activity'] as $finger => $pct): ?>
      <div class="prob-row">
        <!-- Finger name and note, e.g. "Do/Fa (Thumb)" -->
        <span class="prob-name" style="width:140px"><?= htmlspecialchars($finger) ?></span>
        <div class="prob-track">
          <div class="prob-fill" style="width:<?= $pct ?>%; background:#1a1a1a;"></div>
        </div>
        <span class="prob-pct"><?= $pct ?>%</span>
      </div>
      <?php endforeach; ?>
    </div>
  </div>
  <?php endif; ?>
</div>
<?php endif; ?>

<?php if ($video_url): ?>
<!-- ── MEDIAPIPE ANNOTATED VIDEO CARD ────────────────────── -->
<!-- Shows the same video with hand landmarks drawn on top.   -->
<div class="card">
  <h2>MediaPipe Annotated Video</h2>
  <!-- Explanation of what the landmark numbers and colours mean -->
  <p class="annot-hint">
    MediaPipe detects 21 hand landmarks (numbered 0&ndash;20) that are tracked across every frame.
    Fingertip markers change color to indicate key-press activity:
  </p>
  <!-- Colour legend: yellow = not pressing, red = pressing -->
  <div style="display:flex; gap:16px; flex-wrap:wrap; margin:10px 0 4px; font-size:.8rem; color:#555;">
    <span style="display:flex; align-items:center; gap:6px;">
      <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#ffff00; border:1px solid #bbb; flex-shrink:0;"></span>
      Yellow &mdash; finger is not pressing a key
    </span>
    <span style="display:flex; align-items:center; gap:6px;">
      <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#ff3c00; border:1px solid #bbb; flex-shrink:0;"></span>
      Red &mdash; finger is actively pressing a key
    </span>
  </div>
  <!-- The annotated video served by Flask's /video/<filename> endpoint -->
  <div class="video-wrap" style="margin-top:12px">
    <video controls playsinline>
      <source src="<?= htmlspecialchars($video_url) ?>" type="video/mp4">
      Your browser does not support video playback.
    </video>
  </div>
</div>
<?php endif; ?>

<!-- Link to go back and upload a different video -->
<div class="back-link">
  <a href="index.php">Go back &amp; analyze another video</a>
</div>
<?php endif; ?>

<footer>Patrick Andersen Kanginan &nbsp;&mdash;&nbsp; 160421123</footer>

<!-- ============================================================
     JAVASCRIPT — CLIENT-SIDE INTERACTIVITY
     ============================================================
     This script handles everything that happens in the browser
     before the form is submitted:
       • Showing a preview when a file is selected
       • Validating file type and size (instant feedback)
       • Drag-and-drop support
       • Showing the loading spinner when the form is submitted

     The upload form only exists in the DOM when $result is null
     (i.e. no result has been shown yet), so all code is wrapped
     in an "if (fileInput)" guard.
     ============================================================ -->
<script>
// Get references to the key DOM elements
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('videoInput');
const form      = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const loading   = document.getElementById('loadingWrap');

// All upload-related logic is skipped when the result page is shown
// (because the form elements do not exist on the result page)
if (fileInput) {
  const previewSection = document.getElementById('previewSection');
  const previewVideo   = document.getElementById('previewVideo');
  const reuploadBtn    = document.getElementById('reuploadBtn');

  // Holds the temporary browser URL created for the selected file preview
  let previewObjectURL = null;

  // ── showPreview(file) ──────────────────────────────────────
  // Creates a local browser URL for the selected file, attaches
  // it to the <video> element, and reveals the preview section.
  function showPreview(file) {
    // Release the previous object URL to free browser memory
    if (previewObjectURL) URL.revokeObjectURL(previewObjectURL);
    previewObjectURL = URL.createObjectURL(file);

    previewVideo.src = previewObjectURL;
    dropZone.style.display       = 'none';   // hide the drop zone
    previewSection.style.display = 'block';  // show the preview
    submitBtn.style.display      = 'block';  // show the submit button
  }

  // ── resetToDropZone() ──────────────────────────────────────
  // Clears the file selection and reverts the UI back to the
  // original drop zone state.  Called when "Change Video" is clicked.
  function resetToDropZone() {
    fileInput.value = '';   // clear the file input so a new selection can be made
    if (previewObjectURL) {
      URL.revokeObjectURL(previewObjectURL);   // free memory
      previewObjectURL = null;
    }
    previewVideo.src             = '';
    previewSection.style.display = 'none';
    submitBtn.style.display      = 'none';
    dropZone.style.display       = 'block';
  }

  // ── Allowed MIME types ─────────────────────────────────────
  // Validated client-side for instant feedback; also validated
  // server-side in PHP for security.
  const ALLOWED_MIME = ['video/mp4', 'video/quicktime'];

  // ── validateFile(file) ─────────────────────────────────────
  // Checks the selected file before showing a preview.
  // Returns false (and shows an alert) if the file is invalid.
  function validateFile(file) {
    // file.type may be empty for some browsers on certain platforms
    if (!ALLOWED_MIME.includes(file.type) && file.type !== '') {
      alert('Unsupported file format: ' + file.type + '\nPlease use MP4 or MOV.');
      return false;
    }
    // MAX_FILE_MB is printed into the JavaScript by PHP at page render time
    if (file.size > <?= MAX_FILE_MB ?> * 1024 * 1024) {
      alert('File size exceeds the maximum limit of <?= MAX_FILE_MB ?> MB.');
      return false;
    }
    return true;
  }

  // ── File input change event ────────────────────────────────
  // Triggered when the user picks a file through the file picker.
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file && validateFile(file)) showPreview(file);
    else fileInput.value = '';   // clear invalid selection
  });

  // ── Change Video button ────────────────────────────────────
  reuploadBtn.addEventListener('click', resetToDropZone);

  // ── Drag-over visual feedback ──────────────────────────────
  // Add the 'drag-over' class while the user is dragging a file
  // over the drop zone to give visual feedback.
  ['dragenter', 'dragover'].forEach(e =>
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); })
  );
  ['dragleave', 'drop'].forEach(e =>
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); })
  );

  // ── Drop event ─────────────────────────────────────────────
  // When a file is dropped onto the zone, copy it into the
  // hidden file input and fire the same change event so
  // showPreview() is called.
  dropZone.addEventListener('drop', e => {
    const f = e.dataTransfer.files;
    if (f[0] && validateFile(f[0])) {
      fileInput.files = f;
      fileInput.dispatchEvent(new Event('change'));
    }
  });

  // ── Form submit event ──────────────────────────────────────
  // Disables the submit button and shows the spinner to prevent
  // double-submission and to let the user know the upload is
  // in progress (analysis can take 30–120 seconds).
  form.addEventListener('submit', () => {
    submitBtn.disabled    = true;
    submitBtn.textContent = 'Processing...';
    loading.style.display = 'flex';
  });
}
</script>

</body>
</html>
