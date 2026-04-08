import { PaddleOCR } from "@paddleocr/paddleocr-js";
import type { OcrResult, OcrResultItem } from "@paddleocr/paddleocr-js";
import { OcrVisualizer } from "@paddleocr/paddleocr-js/viz";

type OcrEngine = Awaited<ReturnType<typeof PaddleOCR.create>>;

const ORT_WASM_PATHS = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
const DEFAULT_RUNTIME_PARAMS = Object.freeze({
  textDetThresh: 0.3,
  textDetBoxThresh: 0.6,
  textDetUnclipRatio: 1.5,
  textRecScoreThresh: 0.1
});

function getDemoThreadCount(): number {
  return self.crossOriginIsolated
    ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 2) - 1))
    : 1;
}

const ui = {
  runtimeBackend: document.getElementById("runtimeBackend") as HTMLSelectElement,
  detThresh: document.getElementById("detThresh") as HTMLInputElement,
  boxThresh: document.getElementById("boxThresh") as HTMLInputElement,
  unclipRatio: document.getElementById("unclipRatio") as HTMLInputElement,
  recScoreThresh: document.getElementById("recScoreThresh") as HTMLInputElement,
  imageInput: document.getElementById("imageInput") as HTMLInputElement,
  chooseImageBtn: document.getElementById("chooseImageBtn") as HTMLButtonElement,
  reinitializeBtn: document.getElementById("reinitializeBtn") as HTMLButtonElement,
  runBtn: document.getElementById("runBtn") as HTMLButtonElement,
  status: document.getElementById("status") as HTMLElement,
  metrics: document.getElementById("metrics") as HTMLPreElement,
  results: document.getElementById("results") as HTMLOListElement,
  vizImage: document.getElementById("vizImage") as HTMLImageElement
};

interface AppState {
  imageFile: File | null;
  previewBitmap: ImageBitmap | null;
  lastResult: OcrResult | null;
  ocr: OcrEngine | null;
  vizObjectUrl: string | null;
}

const state: AppState = {
  imageFile: null,
  previewBitmap: null,
  lastResult: null,
  ocr: null,
  vizObjectUrl: null
};

const visualizer = new OcrVisualizer({
  font: {
    family: "PingFang SC",
    source:
      "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/PingFang-SC-Regular.ttf"
  }
});

function setStatus(text: string, isError = false): void {
  ui.status.textContent = text;
  ui.status.style.color = isError ? "#b91c1c" : "";
}

function formatMs(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function showVizImage(blob: Blob): void {
  if (state.vizObjectUrl) {
    URL.revokeObjectURL(state.vizObjectUrl);
  }
  state.vizObjectUrl = URL.createObjectURL(blob);
  ui.vizImage.src = state.vizObjectUrl;
}

function showPreviewImage(bitmap: ImageBitmap): void {
  // For pre-OCR preview, draw to an offscreen canvas and display as image
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(bitmap, 0, 0);
  canvas.toBlob((blob) => {
    if (blob) showVizImage(blob);
  });
}

function renderResults(items: OcrResultItem[]): void {
  ui.results.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.text} | score=${item.score.toFixed(3)}`;
    ui.results.appendChild(li);
  });
}

function getRuntimeOptions() {
  return {
    backend: ui.runtimeBackend.value as "auto" | "webgpu" | "wasm",
    wasmPaths: ORT_WASM_PATHS,
    numThreads: getDemoThreadCount(),
    simd: true
  };
}

async function initializeOcrEngine(): Promise<void> {
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
    `assets: ${String(summary.assets.length)}`,
    `cache hits: ${String(summary.cacheHits)}`,
    `cache misses: ${String(summary.cacheMisses)}`
  ].join("\n");
  setStatus(`OCR engine initialized (${String(summary.cacheHits)} cache hits).`);
  ui.runBtn.disabled = !state.imageFile;
}

async function handleImageSelection(file: File | undefined): Promise<void> {
  if (!file) return;
  state.imageFile = file;
  state.previewBitmap?.close();
  state.previewBitmap = await createImageBitmap(file);
  showPreviewImage(state.previewBitmap);
  ui.runBtn.disabled = !state.ocr;
  setStatus(`Image selected: ${file.name}`);
}

async function runOcr(): Promise<void> {
  if (!state.ocr || !state.imageFile) {
    setStatus("Wait for OCR engine initialization to finish, then choose an image.", true);
    return;
  }

  try {
    setStatus("Running OCR...");
    const result: OcrResult = await state.ocr.predict(state.imageFile, {
      textDetThresh: Number(ui.detThresh.value),
      textDetBoxThresh: Number(ui.boxThresh.value),
      textDetUnclipRatio: Number(ui.unclipRatio.value),
      textRecScoreThresh: Number(ui.recScoreThresh.value)
    });

    if (!state.previewBitmap) {
      state.previewBitmap = await createImageBitmap(state.imageFile);
    }

    // Render side-by-side visualization using viz module
    const blob = await visualizer.toBlob(state.previewBitmap, result);
    showVizImage(blob);

    renderResults(result.items);
    state.lastResult = result;
    ui.metrics.textContent = [
      ui.metrics.textContent,
      "",
      `det infer: ${formatMs(result.metrics.detInferMs)}`,
      `rec prep: ${formatMs(result.metrics.recPrepMs)}`,
      `rec infer: ${formatMs(result.metrics.recInferMs)}`,
      `total: ${formatMs(result.metrics.totalMs)}`,
      `detected boxes: ${String(result.metrics.detectedBoxes)}`,
      `recognized lines: ${String(result.metrics.recognizedCount)}`
    ].join("\n");
    setStatus(`OCR complete: ${String(result.metrics.recognizedCount)} text lines recognized.`);
  } catch (err: unknown) {
    console.error(err);
    const message = err instanceof Error ? err.message : String(err);
    setStatus(`OCR failed: ${message}`, true);
  }
}

ui.imageInput.addEventListener("change", (event: Event) => {
  const target = event.target as HTMLInputElement;
  void handleImageSelection(target.files?.[0]);
});

ui.chooseImageBtn.addEventListener("click", () => {
  ui.imageInput.click();
});

async function reinitializeOcrEngine(): Promise<void> {
  try {
    ui.reinitializeBtn.disabled = true;
    ui.runBtn.disabled = true;
    setStatus("Initializing OCR engine...");
    await visualizer.loadFont();
    await initializeOcrEngine();
  } catch (err: unknown) {
    console.error(err);
    const message = err instanceof Error ? err.message : String(err);
    setStatus(`OCR engine initialization failed: ${message}`, true);
    ui.runBtn.disabled = true;
  } finally {
    ui.reinitializeBtn.disabled = false;
  }
}

ui.detThresh.value = String(DEFAULT_RUNTIME_PARAMS.textDetThresh);
ui.boxThresh.value = String(DEFAULT_RUNTIME_PARAMS.textDetBoxThresh);
ui.unclipRatio.value = String(DEFAULT_RUNTIME_PARAMS.textDetUnclipRatio);
ui.recScoreThresh.value = String(DEFAULT_RUNTIME_PARAMS.textRecScoreThresh);
ui.reinitializeBtn.addEventListener("click", () => void reinitializeOcrEngine());
ui.runtimeBackend.addEventListener("change", () => void reinitializeOcrEngine());

ui.runBtn.addEventListener("click", () => void runOcr());

void reinitializeOcrEngine();
