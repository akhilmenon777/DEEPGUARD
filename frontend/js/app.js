/**
 * Deepfake Detection System - Frontend Client Application
 * Interacts with FastAPI backend for GenD ViT and Mel-Spectrogram CNN
 */

// Application State
let currentTab = "image";
let selectedFile = null;
let currentResult = null;

// DOM Elements
const tabBtns = document.querySelectorAll(".tab-btn");
const uploadTitle = document.getElementById("uploadTitle");
const mediaTypeTag = document.getElementById("mediaTypeTag");
const dropAcceptHint = document.getElementById("dropAcceptHint");
const pipelineInfoList = document.getElementById("pipelineInfoList");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const dropContent = document.getElementById("dropContent");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const videoPreview = document.getElementById("videoPreview");
const audioPreviewWrapper = document.getElementById("audioPreviewWrapper");
const audioPreview = document.getElementById("audioPreview");
const audioFilename = document.getElementById("audioFilename");
const btnClearFile = document.getElementById("btnClearFile");
const btnAnalyze = document.getElementById("btnAnalyze");
const analyzeSpinner = document.getElementById("analyzeSpinner");
const analyzeBtnText = document.getElementById("analyzeBtnText");

const resultsCard = document.getElementById("resultsCard");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultBody = document.getElementById("resultBody");
const btnExportReport = document.getElementById("btnExportReport");
const verdictBadge = document.getElementById("verdictBadge");
const confidenceVal = document.getElementById("confidenceVal");
const analysisTimestamp = document.getElementById("analysisTimestamp");
const probPercentage = document.getElementById("probPercentage");
const probBarFill = document.getElementById("probBarFill");

const explainTitle = document.getElementById("explainTitle");
const explainDesc = document.getElementById("explainDesc");
const heatmapComparison = document.getElementById("heatmapComparison");
const faceCropImg = document.getElementById("faceCropImg");
const heatmapImg = document.getElementById("heatmapImg");
const sliderControl = document.getElementById("sliderControl");
const heatmapOpacity = document.getElementById("heatmapOpacity");
const opacityVal = document.getElementById("opacityVal");
const videoTimelineContainer = document.getElementById("videoTimelineContainer");
const videoFramesAnalyzedText = document.getElementById("videoFramesAnalyzedText");
const framesBarChart = document.getElementById("framesBarChart");
const audioSpectrogramContainer = document.getElementById("audioSpectrogramContainer");
const spectrogramImg = document.getElementById("spectrogramImg");
const spectralNoteText = document.getElementById("spectralNoteText");
const safetyAdviceText = document.getElementById("safetyAdviceText");
const featuresList = document.getElementById("featuresList");

const historySection = document.getElementById("historySection");
const historyTableBody = document.getElementById("historyTableBody");
const workspaceGrid = document.querySelector(".workspace-grid");

// Tab configurations
const TAB_CONFIGS = {
  image: {
    title: "Ingest Image Media",
    tag: "IMAGE",
    accept: ".jpg,.jpeg,.png,.webp",
    hint: "Supports JPG, PNG, WEBP (Max 25MB)",
    pipeline: [
      "<strong>Face Crop:</strong> OpenCV Haar Cascade (224×224)",
      "<strong>Core Vision:</strong> GenD ViT (LayerNorm-only fine-tuning)",
      "<strong>Manifold:</strong> L2-Normalized Hyperspherical Head",
      "<strong>Explainability:</strong> Grad-CAM Visual Heatmap"
    ]
  },
  video: {
    title: "Ingest Video Media",
    tag: "VIDEO",
    accept: ".mp4,.avi,.mov,.mkv,.webm",
    hint: "Supports MP4, AVI, MOV (Max 100MB)",
    pipeline: [
      "<strong>Sampling:</strong> Up to 20 frames evenly spaced",
      "<strong>Frame Classifier:</strong> GenD ViT on detected face crops",
      "<strong>Aggregation:</strong> Mean Risk Score + Peak Frame Detection",
      "<strong>Explainability:</strong> Grad-CAM on most suspicious frame"
    ]
  },
  audio: {
    title: "Ingest Audio Clip",
    tag: "AUDIO",
    accept: ".wav,.mp3,.flac,.ogg,.m4a",
    hint: "Supports WAV, MP3, FLAC (Max 25MB)",
    pipeline: [
      "<strong>Preprocessing:</strong> Resampled to 16 kHz, 4s duration",
      "<strong>Spectrogram:</strong> 128-band Log-Mel Spectrogram (z-score)",
      "<strong>Classifier:</strong> 3-Block 2D SpectrogramCNN",
      "<strong>Acoustic Logic:</strong> High-frequency vocoder diagnostics"
    ]
  },
  blur: {
    title: "Blur Quality Detection",
    tag: "BLUR",
    accept: ".jpg,.jpeg,.png,.webp",
    hint: "Supports JPG, PNG, WEBP (Max 25MB)",
    pipeline: [
      "<strong>Preprocessing:</strong> Grayscale conversion + Laplacian filter",
      "<strong>Variance Analysis:</strong> Edge sharpness calculation",
      "<strong>Quality Score:</strong> Blur detection threshold (< 100 = Blurry)",
      "<strong>Feature Extraction:</strong> Frequency domain analysis"
    ]
  },
  "noisy-audio": {
    title: "Noisy Audio Analysis",
    tag: "NOISY",
    accept: ".wav,.mp3,.flac,.ogg,.m4a",
    hint: "Supports WAV, MP3, FLAC (Max 25MB)",
    pipeline: [
      "<strong>Noise Reduction:</strong> Spectral subtraction preprocessing",
      "<strong>SNR Calculation:</strong> Signal-to-Noise Ratio analysis",
      "<strong>Enhanced Detection:</strong> Robust deepfake detection on noisy audio",
      "<strong>Confidence Adjustment:</strong> Noise-aware confidence scoring"
    ]
  }
};

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupDropZone();
  setupOpacitySlider();
  checkApiHealth();
  loadHistory();
});

// Setup Mode Tabs
function setupTabs() {
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      if (targetTab === currentTab) return;

      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentTab = targetTab;

      if (currentTab === "history") {
        workspaceGrid.classList.add("hidden");
        historySection.classList.remove("hidden");
        loadHistory();
      } else {
        workspaceGrid.classList.remove("hidden");
        historySection.classList.add("hidden");
        applyTabConfig(currentTab);
        clearSelectedFile();
      }
    });
  });
}

function applyTabConfig(tabKey) {
  const cfg = TAB_CONFIGS[tabKey];
  uploadTitle.innerText = cfg.title;
  mediaTypeTag.innerText = cfg.tag;
  fileInput.setAttribute("accept", cfg.accept);
  dropAcceptHint.innerText = cfg.hint;
  pipelineInfoList.innerHTML = cfg.pipeline.map(li => `<li>${li}</li>`).join("");
}

// Drag & Drop Handling
function setupDropZone() {
  ["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", e => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", e => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  btnClearFile.addEventListener("click", e => {
    e.stopPropagation();
    clearSelectedFile();
  });

  btnAnalyze.addEventListener("click", () => {
    executeAnalysis();
  });
}

function handleFileSelected(file) {
  selectedFile = file;
  dropContent.classList.add("hidden");
  previewContainer.classList.remove("hidden");
  btnAnalyze.removeAttribute("disabled");

  imagePreview.classList.add("hidden");
  videoPreview.classList.add("hidden");
  audioPreviewWrapper.classList.add("hidden");

  const url = URL.createObjectURL(file);

  if (currentTab === "image") {
    imagePreview.src = url;
    imagePreview.classList.remove("hidden");
  } else if (currentTab === "video") {
    videoPreview.src = url;
    videoPreview.classList.remove("hidden");
  } else if (currentTab === "audio") {
    audioPreview.src = url;
    audioFilename.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    audioPreviewWrapper.classList.remove("hidden");
  }
}

function clearSelectedFile() {
  selectedFile = null;
  fileInput.value = "";
  previewContainer.classList.add("hidden");
  dropContent.classList.remove("hidden");
  btnAnalyze.setAttribute("disabled", "true");
  imagePreview.src = "";
  videoPreview.src = "";
  audioPreview.src = "";
}

// Heatmap Opacity Slider
function setupOpacitySlider() {
  heatmapOpacity.addEventListener("input", e => {
    const val = e.target.value;
    opacityVal.innerText = `${val}%`;
    if (heatmapImg) {
      heatmapImg.style.opacity = (val / 100).toString();
    }
  });
}

// API Health Check
async function checkApiHealth() {
  const statusElem = document.getElementById("apiStatusText");
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      statusElem.innerText = "Engine Online (GenD & CNN)";
    } else {
      statusElem.innerText = "API Offline";
    }
  } catch {
    statusElem.innerText = "Connecting...";
  }
}

// 1-Click Preset Sample Loading
async function loadSamplePreset(filename, mediaType) {
  // Switch to the matching tab if not active
  if (currentTab !== mediaType) {
    const targetBtn = document.querySelector(`.tab-btn[data-tab="${mediaType}"]`);
    if (targetBtn) targetBtn.click();
  }

  setAnalyzingState(true);

  try {
    const response = await fetch(`/api/load-sample/${filename}`);
    if (!response.ok) {
      throw new Error(`Failed to load preset sample: ${response.statusText}`);
    }
    const data = await response.json();
    displayResults(data);
  } catch (err) {
    alert("Preset Error: " + err.message);
  } finally {
    setAnalyzingState(false);
  }
}

// Execute Analysis
async function executeAnalysis() {
  if (!selectedFile) return;

  setAnalyzingState(true);

  const formData = new FormData();
  formData.append("file", selectedFile);

  let endpoint = "/api/predict-image";
  if (currentTab === "video") endpoint = "/api/predict-video";
  else if (currentTab === "audio") endpoint = "/api/predict-audio";
  else if (currentTab === "blur") endpoint = "/api/predict-blur";
  else if (currentTab === "noisy-audio") endpoint = "/api/predict-noisy-audio";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Prediction request failed");
    }

    const data = await res.json();
    displayResults(data);
  } catch (error) {
    alert("Analysis Error: " + error.message);
  } finally {
    setAnalyzingState(false);
  }
}

function setAnalyzingState(isAnalyzing) {
  if (isAnalyzing) {
    btnAnalyze.setAttribute("disabled", "true");
    analyzeSpinner.classList.remove("hidden");
    analyzeBtnText.innerText = "Analyzing Neural Features...";
  } else {
    btnAnalyze.removeAttribute("disabled");
    analyzeSpinner.classList.add("hidden");
    analyzeBtnText.innerText = "Run Deepfake Analysis";
  }
}

// Render Results
function displayResults(data) {
  currentResult = data;
  resultPlaceholder.classList.add("hidden");
  resultBody.classList.remove("hidden");
  btnExportReport.classList.remove("hidden");

  // Verdict Badge
  const verdict = data.verdict || data.label;
  verdictBadge.innerText = verdict;
  verdictBadge.className = "verdict-badge";

  const normVerdict = verdict.toLowerCase().replace(/\s+/g, "-");
  verdictBadge.classList.add(normVerdict);

  confidenceVal.innerText = `${data.confidence}%`;
  analysisTimestamp.innerText = data.timestamp || "Just now";

  // Probability Meter
  const prob = (data.probability_pct !== undefined) ? data.probability_pct : (data.average_probability_pct || (data.probability * 100));
  probPercentage.innerText = `${prob.toFixed(1)}%`;
  probBarFill.style.width = `${Math.min(100, Math.max(0, prob))}%`;

  // Dynamic layout by media type
  if (data.media_type === "image" || currentTab === "image") {
    renderImageResults(data);
  } else if (data.media_type === "video" || currentTab === "video") {
    renderVideoResults(data);
  } else if (data.media_type === "audio" || currentTab === "audio") {
    renderAudioResults(data);
  } else if (data.media_type === "blur" || currentTab === "blur") {
    renderBlurResults(data);
  } else if (data.media_type === "noisy-audio" || currentTab === "noisy-audio") {
    renderNoisyAudioResults(data);
  }

  // Update inspected features
  if (data.features_analyzed) {
    featuresList.innerHTML = data.features_analyzed.map(f => `<li>${f}</li>`).join("");
  }
}

function renderImageResults(data) {
  explainTitle.innerText = "Visual Explainability (GenD ViT Grad-CAM)";
  explainDesc.innerText = "Grad-CAM visualizes spatial attention over 14x14 ViT patches, pinpointing pixel boundary inconsistencies and digital blending artifacts.";
  heatmapComparison.classList.remove("hidden");
  sliderControl.classList.remove("hidden");
  videoTimelineContainer.classList.add("hidden");
  audioSpectrogramContainer.classList.add("hidden");

  if (data.face_crop_base64) {
    faceCropImg.src = data.face_crop_base64;
  }
  if (data.heatmap_base64) {
    heatmapImg.src = data.heatmap_base64;
  }
}

function renderVideoResults(data) {
  explainTitle.innerText = "Multi-Frame Video Assessment (GenD ViT)";
  explainDesc.innerText = `Evaluated ${data.frames_analyzed || 20} frames across video duration. Displaying Grad-CAM on the peak anomalous frame (Frame #${data.most_suspicious_frame?.frame_index || 0}).`;
  heatmapComparison.classList.remove("hidden");
  sliderControl.classList.remove("hidden");
  videoTimelineContainer.classList.remove("hidden");
  audioSpectrogramContainer.classList.add("hidden");

  if (data.most_suspicious_frame?.face_crop_base64) {
    faceCropImg.src = data.most_suspicious_frame.face_crop_base64;
  }
  if (data.most_suspicious_frame?.heatmap_base64) {
    heatmapImg.src = data.most_suspicious_frame.heatmap_base64;
  }

  videoFramesAnalyzedText.innerText = `Analyzed ${data.frames_analyzed} frames across ${data.duration_sec}s. Average Risk: ${data.average_probability_pct}%, Peak Risk: ${(data.max_probability * 100).toFixed(1)}%.`;

  // Render Frame Chart
  framesBarChart.innerHTML = "";
  if (data.frame_timeline) {
    const maxProb = data.max_probability;
    data.frame_timeline.forEach((f, idx) => {
      const bar = document.createElement("div");
      bar.className = "frame-bar";
      const heightPct = Math.max(10, (f.fake_probability / (maxProb || 1.0)) * 100);
      bar.style.height = `${heightPct}%`;
      bar.title = `Frame #${f.frame_index} (${f.timestamp_sec}s): ${(f.fake_probability * 100).toFixed(1)}% Fake`;

      if (f.fake_probability > 0.65) bar.classList.add("high-risk");
      if (f.fake_probability === maxProb) bar.classList.add("peak");

      framesBarChart.appendChild(bar);
    });
  }
}

function renderAudioResults(data) {
  explainTitle.innerText = "Acoustic Spectrogram Analysis & Diagnostics";
  explainDesc.innerText = "Inspection of the 128-band Mel-Frequency energy spectrum and neural vocoder frequency cutoff.";
  heatmapComparison.classList.add("hidden");
  sliderControl.classList.add("hidden");
  videoTimelineContainer.classList.add("hidden");
  audioSpectrogramContainer.classList.remove("hidden");

  if (data.spectrogram_base64) {
    spectrogramImg.src = data.spectrogram_base64;
  }
  spectralNoteText.innerText = data.spectral_explanation || "Spectral distribution analyzed.";
  safetyAdviceText.innerText = data.safety_recommendation || "Maintain identity verification protocols.";
}

// History Handling
async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    const records = data.history || [];

    if (records.length === 0) {
      historyTableBody.innerHTML = `<tr><td colspan="7" class="text-muted text-center">No verifications recorded in this session.</td></tr>`;
      return;
    }

    historyTableBody.innerHTML = records.map(r => {
      const badgeClass = (r.verdict || "").toLowerCase().replace(/\s+/g, "-");
      return `
        <tr>
          <td><code>${r.id}</code></td>
          <td>${r.filename}</td>
          <td><span class="badge-tag">${(r.media_type || "").toUpperCase()}</span></td>
          <td><span class="verdict-badge ${badgeClass}" style="font-size:0.75rem; padding:0.2rem 0.6rem;">${r.verdict}</span></td>
          <td>${r.probability_pct}%</td>
          <td>${r.confidence}%</td>
          <td style="color:#94a3b8; font-size:0.75rem;">${r.timestamp}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

async function clearHistoryLog() {
  if (!confirm("Are you sure you want to clear the verification history?")) return;
  try {
    await fetch("/api/clear-history", { method: "POST" });
    loadHistory();
  } catch (err) {
    alert("Error clearing history: " + err.message);
  }
}

// Export Forensic Report
function exportReport() {
  if (!currentResult) return;
  const printWindow = window.open("", "_blank");
  const verdict = currentResult.verdict || currentResult.label;
  const prob = currentResult.probability_pct || currentResult.average_probability_pct || 0;

  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Forensic Audit Report - ${currentResult.filename}</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 40px; color: #1e293b; }
        .header { border-bottom: 2px solid #0f172a; padding-bottom: 15px; margin-bottom: 25px; }
        .badge { display: inline-block; padding: 6px 16px; border-radius: 6px; font-weight: bold; font-size: 16px; text-transform: uppercase; }
        .authentic, .genuine { background: #dcfce7; color: #15803d; }
        .suspicious { background: #fef3c7; color: #b45309; }
        .likely-deepfake, .potentially-synthetic { background: #fee2e2; color: #b91c1c; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .box { background: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; }
        img { max-width: 100%; border-radius: 6px; }
        .footer { margin-top: 40px; font-size: 12px; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 15px; }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>AI Deepfake Detection Forensic Audit Report</h1>
        <p>Generated: ${currentResult.timestamp} | Media: ${currentResult.filename} (${currentResult.media_type.toUpperCase()})</p>
      </div>

      <div class="box">
        <h3>Forensic Verdict</h3>
        <p><span class="badge ${verdict.toLowerCase().replace(/\s+/g, "-")}">${verdict}</span></p>
        <p><strong>Manipulation Probability:</strong> ${prob}% | <strong>Confidence:</strong> ${currentResult.confidence}%</p>
      </div>

      <div class="grid">
        <div class="box">
          <h3>Inspection Pipeline</h3>
          <ul>
            <li>Model: GenD Vision Transformer (LayerNorm-only adaptation)</li>
            <li>Embedding: Hyperspherical Manifold (L2 Normalized)</li>
            <li>Audio Model: 128-band Mel-Spectrogram CNN</li>
            <li>Explainability: Grad-CAM Feature Attribution</li>
          </ul>
        </div>
        <div class="box">
          <h3>Evidence Summary</h3>
          <p>${currentResult.spectral_explanation || "Visual feature anomalies and patch boundary artifacts evaluated through Vision Transformer self-attention."}</p>
          <p><strong>Recommendation:</strong> ${currentResult.safety_recommendation || "Verify media authenticity before publishing or acting upon sensitive content."}</p>
        </div>
      </div>

      <div class="footer">
        <p>Deepfake Detection System • SE Project Phase I & Design Implementation • Yash Jain (K019), Akhil Menon (K036), Taaksh Patel (K042)</p>
      </div>
    </body>
    </html>
  `);
  printWindow.document.close();
  printWindow.focus();
  setTimeout(() => printWindow.print(), 500);
}


function renderBlurResults(data) {
  explainTitle.innerText = "Image Quality & Blur Analysis";
  explainDesc.innerText = "Laplacian variance analysis detects image blur by measuring edge sharpness. Low variance indicates blurry or low-quality images that may affect deepfake detection accuracy.";
  heatmapComparison.classList.remove("hidden");
  sliderControl.classList.add("hidden");
  videoTimelineContainer.classList.add("hidden");
  audioSpectrogramContainer.classList.add("hidden");

  if (data.original_image_base64) {
    faceCropImg.src = data.original_image_base64;
  }
  if (data.edges_image_base64) {
    heatmapImg.src = data.edges_image_base64;
  }
}

function renderNoisyAudioResults(data) {
  explainTitle.innerText = "Noisy Audio Detection & Preprocessing";
  explainDesc.innerText = "Advanced noise reduction and spectral subtraction enable deepfake detection even in noisy environments. SNR analysis provides confidence adjustment based on audio quality.";
  heatmapComparison.classList.add("hidden");
  sliderControl.classList.add("hidden");
  videoTimelineContainer.classList.add("hidden");
  audioSpectrogramContainer.classList.remove("hidden");

  if (data.spectrogram_base64) {
    spectrogramImg.src = data.spectrogram_base64;
  }
  spectralNoteText.innerText = data.spectral_explanation || "Noise-reduced audio analyzed for deepfake artifacts.";
  safetyAdviceText.innerText = data.safety_recommendation || "Audio quality may impact detection confidence.";
}
