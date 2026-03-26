import { PaddleOCR } from "paddleocr-js";

const ORT_WASM_PATHS = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
const DEFAULT_RUNTIME_PARAMS = Object.freeze({
  textDetThresh: 0.3,
  textDetBoxThresh: 0.6,
  textDetUnclipRatio: 1.5,
  textRecScoreThresh: 0.1
});

function getDemoThreadCount() {
  return self.crossOriginIsolated
    ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 2) - 1))
    : 1;
}

const ui = {
  runtimeBackend: document.getElementById("runtimeBackend"),
  detThresh: document.getElementById("detThresh"),
  boxThresh: document.getElementById("boxThresh"),
  unclipRatio: document.getElementById("unclipRatio"),
  recScoreThresh: document.getElementById("recScoreThresh"),
  imageInput: document.getElementById("imageInput"),
  chooseImageBtn: document.getElementById("chooseImageBtn"),
  reinitializeBtn: document.getElementById("reinitializeBtn"),
  runBtn: document.getElementById("runBtn"),
  status: document.getElementById("status"),
  metrics: document.getElementById("metrics"),
  results: document.getElementById("results"),
  canvas: document.getElementById("canvas")
};

const canvasCtx = ui.canvas.getContext("2d");

const state = {
  imageFile: null,
  previewBitmap: null,
  ocr: null
};

function setStatus(text, isError = false) {
  ui.status.textContent = text;
  ui.status.style.color = isError ? "#b91c1c" : "";
}

function formatMs(value) {
  return `${value.toFixed(1)} ms`;
}

function deterministicColor(idx) {
  let seed = (idx + 1) * 1103515245 + 12345;
  seed >>>= 0;
  const r = (seed >> 16) & 0xff;
  seed = (seed * 1103515245 + 12345) >>> 0;
  const g = (seed >> 16) & 0xff;
  seed = (seed * 1103515245 + 12345) >>> 0;
  const b = (seed >> 16) & 0xff;
  return [r, g, b];
}

function drawPolygonPath(ctx, poly) {
  ctx.beginPath();
  ctx.moveTo(poly[0][0], poly[0][1]);
  for (let index = 1; index < poly.length; index += 1) {
    ctx.lineTo(poly[index][0], poly[index][1]);
  }
  ctx.closePath();
}

function drawPreview(bitmap, items = []) {
  ui.canvas.width = bitmap.width;
  ui.canvas.height = bitmap.height;
  canvasCtx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  canvasCtx.drawImage(bitmap, 0, 0);

  items.forEach((item, index) => {
    const [r, g, b] = deterministicColor(index);
    canvasCtx.save();
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
    canvasCtx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.22)`;
    drawPolygonPath(canvasCtx, item.poly);
    canvasCtx.fill();
    canvasCtx.stroke();
    canvasCtx.restore();
  });
}

function renderResults(items) {
  ui.results.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.text} | score=${item.score.toFixed(3)}`;
    ui.results.appendChild(li);
  });
}

function getRuntimeOptions() {
  return {
    backend: ui.runtimeBackend.value,
    wasmPaths: ORT_WASM_PATHS,
    numThreads: getDemoThreadCount(),
    simd: true
  };
}

async function initializeOcrEngine() {
  if (state.ocr) {
    await state.ocr.dispose();
  }

  state.ocr = await PaddleOCR.create({
    initialize: false,
    worker: true,
    runtime: getRuntimeOptions()
  });

  const summary = await state.ocr.initialize();
  ui.metrics.textContent = [
    `initialize: ${formatMs(summary.elapsedMs)}`,
    `backend(requested): ${summary.backend}`,
    `webgpu available: ${summary.webgpuAvailable ? "yes" : "no"}`,
    `provider(det): ${summary.detProvider}`,
    `provider(rec): ${summary.recProvider}`,
    `assets: ${summary.assets.length}`,
    `cache hits: ${summary.cacheHits}`,
    `cache misses: ${summary.cacheMisses}`
  ].join("\n");
  setStatus(`OCR engine initialized (${summary.cacheHits} cache hits).`);
  ui.runBtn.disabled = !state.imageFile;
}

async function handleImageSelection(file) {
  if (!file) return;
  state.imageFile = file;
  state.previewBitmap?.close?.();
  state.previewBitmap = await createImageBitmap(file);
  drawPreview(state.previewBitmap);
  ui.runBtn.disabled = !state.ocr;
  setStatus(`Image selected: ${file.name}`);
}

async function runOcr() {
  if (!state.ocr || !state.imageFile) {
    setStatus("Wait for OCR engine initialization to finish, then choose an image.", true);
    return;
  }

  try {
    setStatus("Running OCR...");
    const result = await state.ocr.predict(state.imageFile, {
      textDetThresh: Number(ui.detThresh.value),
      textDetBoxThresh: Number(ui.boxThresh.value),
      textDetUnclipRatio: Number(ui.unclipRatio.value),
      textRecScoreThresh: Number(ui.recScoreThresh.value)
    });

    if (!state.previewBitmap) {
      state.previewBitmap = await createImageBitmap(state.imageFile);
    }
    drawPreview(state.previewBitmap, result.items);
    renderResults(result.items);
    ui.metrics.textContent = [
      ui.metrics.textContent,
      "",
      `det infer: ${formatMs(result.metrics.detInferMs)}`,
      `rec prep: ${formatMs(result.metrics.recPrepMs)}`,
      `rec infer: ${formatMs(result.metrics.recInferMs)}`,
      `total: ${formatMs(result.metrics.totalMs)}`,
      `detected boxes: ${result.metrics.detectedBoxes}`,
      `recognized lines: ${result.metrics.recognizedCount}`
    ].join("\n");
    setStatus(`OCR complete: ${result.metrics.recognizedCount} text lines recognized.`);
  } catch (err) {
    console.error(err);
    setStatus(`OCR failed: ${err.message}`, true);
  }
}

ui.imageInput.addEventListener("change", async (event) => {
  await handleImageSelection(event.target.files?.[0]);
});

ui.chooseImageBtn.addEventListener("click", () => {
  ui.imageInput.click();
});

async function reinitializeOcrEngine() {
  try {
    ui.reinitializeBtn.disabled = true;
    ui.runBtn.disabled = true;
    setStatus("Initializing OCR engine...");
    await initializeOcrEngine();
  } catch (err) {
    console.error(err);
    setStatus(`OCR engine initialization failed: ${err.message}`, true);
    ui.runBtn.disabled = true;
  } finally {
    ui.reinitializeBtn.disabled = false;
  }
}

ui.detThresh.value = String(DEFAULT_RUNTIME_PARAMS.textDetThresh);
ui.boxThresh.value = String(DEFAULT_RUNTIME_PARAMS.textDetBoxThresh);
ui.unclipRatio.value = String(DEFAULT_RUNTIME_PARAMS.textDetUnclipRatio);
ui.recScoreThresh.value = String(DEFAULT_RUNTIME_PARAMS.textRecScoreThresh);
ui.reinitializeBtn.addEventListener("click", reinitializeOcrEngine);
ui.runtimeBackend.addEventListener("change", reinitializeOcrEngine);

ui.runBtn.addEventListener("click", runOcr);

void reinitializeOcrEngine();
