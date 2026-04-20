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
  modelPreset: document.getElementById("modelPreset") as HTMLSelectElement,
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
  ocrReady: boolean;
  vizObjectUrl: string | null;
}

const state: AppState = {
  imageFile: null,
  previewBitmap: null,
  lastResult: null,
  ocr: null,
  ocrReady: false,
  vizObjectUrl: null
};

function updateRunButtonState(): void {
  ui.runBtn.disabled = !state.imageFile || !state.ocrReady;
}

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
  ui.vizImage.hidden = false;
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
  state.ocrReady = false;
  updateRunButtonState();

  if (state.ocr) {
    await state.ocr.dispose();
  }

  const preset = ui.modelPreset.value;

  state.ocr = await PaddleOCR.create({
    initialize: false,
    worker: false,
    textDetectionModelName: `${preset}_det`,
    textRecognitionModelName: `${preset}_rec`,
    ortOptions: getRuntimeOptions()
  });

  const summary = await state.ocr.initialize();
  state.ocrReady = true;
  ui.metrics.textContent = [
    `model: ${preset}`,
    `initialize: ${formatMs(summary.elapsedMs)}`,
    `backend(requested): ${summary.backend}`,
    `webgpu available: ${summary.webgpuAvailable ? "yes" : "no"}`,
    `provider(det): ${summary.detProvider}`,
    `provider(rec): ${summary.recProvider}`,
    `assets: ${String(summary.assets.length)}`
  ].join("\n");
  updateRunButtonState();
}

async function handleImageSelection(file: File | undefined): Promise<void> {
  if (!file) return;
  state.imageFile = file;
  state.previewBitmap?.close();
  state.previewBitmap = await createImageBitmap(file);
  showPreviewImage(state.previewBitmap);
  updateRunButtonState();
  setStatus(`Image selected: ${file.name}`);
}

async function runOcr(): Promise<void> {
  if (!state.ocrReady || !state.ocr || !state.imageFile) {
    setStatus("Wait for OCR engine initialization to finish, then choose an image.", true);
    return;
  }

  try {
    setStatus("Running OCR...");
    const result: OcrResult = (
      await state.ocr.predict(state.imageFile, {
        textDetThresh: Number(ui.detThresh.value),
        textDetBoxThresh: Number(ui.boxThresh.value),
        textDetUnclipRatio: Number(ui.unclipRatio.value),
        textRecScoreThresh: Number(ui.recScoreThresh.value)
      })
    )[0];

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
      `det: ${formatMs(result.metrics.detMs)}`,
      `rec: ${formatMs(result.metrics.recMs)}`,
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

async function initialize(): Promise<void> {
  try {
    ui.reinitializeBtn.disabled = true;
    state.ocrReady = false;
    updateRunButtonState();

    setStatus("Initializing...");
    await initializeOcrEngine();

    setStatus("Loading visualization font...");
    try {
      await visualizer.loadFont();
    } catch (fontErr: unknown) {
      console.warn("Font load failed, using system font:", fontErr);
      setStatus("Ready (visualization will use system font).");
      updateRunButtonState();
      return;
    }

    setStatus("Ready.");
    updateRunButtonState();
  } catch (err: unknown) {
    console.error(err);
    const message = err instanceof Error ? err.message : String(err);
    setStatus(`Initialization failed: ${message}`, true);
    state.ocrReady = false;
    updateRunButtonState();
  } finally {
    ui.reinitializeBtn.disabled = false;
  }
}

ui.detThresh.value = String(DEFAULT_RUNTIME_PARAMS.textDetThresh);
ui.boxThresh.value = String(DEFAULT_RUNTIME_PARAMS.textDetBoxThresh);
ui.unclipRatio.value = String(DEFAULT_RUNTIME_PARAMS.textDetUnclipRatio);
ui.recScoreThresh.value = String(DEFAULT_RUNTIME_PARAMS.textRecScoreThresh);
ui.reinitializeBtn.addEventListener("click", () => void initialize());
ui.modelPreset.addEventListener("change", () => void initialize());
ui.runtimeBackend.addEventListener("change", () => void initialize());

ui.runBtn.addEventListener("click", () => void runOcr());

void initialize();
