import { PaddleOCR } from "@paddleocr/paddleocr-js";
import type { OcrResult, OcrResultItem, Point2D } from "@paddleocr/paddleocr-js";

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
  canvas: document.getElementById("canvas") as HTMLCanvasElement
};

function requireContext2D(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Failed to create canvas 2D context.");
  return ctx;
}

const canvasCtx = requireContext2D(ui.canvas);

interface AppState {
  imageFile: File | null;
  previewBitmap: ImageBitmap | null;
  ocr: OcrEngine | null;
}

const state: AppState = {
  imageFile: null,
  previewBitmap: null,
  ocr: null
};

function setStatus(text: string, isError = false): void {
  ui.status.textContent = text;
  ui.status.style.color = isError ? "#b91c1c" : "";
}

function formatMs(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function deterministicColor(idx: number): [number, number, number] {
  let seed = (idx + 1) * 1103515245 + 12345;
  seed >>>= 0;
  const r = (seed >> 16) & 0xff;
  seed = (seed * 1103515245 + 12345) >>> 0;
  const g = (seed >> 16) & 0xff;
  seed = (seed * 1103515245 + 12345) >>> 0;
  const b = (seed >> 16) & 0xff;
  return [r, g, b];
}

function drawPolygonPath(ctx: CanvasRenderingContext2D, poly: Point2D[]): void {
  ctx.beginPath();
  ctx.moveTo(poly[0][0], poly[0][1]);
  for (let index = 1; index < poly.length; index += 1) {
    ctx.lineTo(poly[index][0], poly[index][1]);
  }
  ctx.closePath();
}

function drawPreview(bitmap: ImageBitmap, items: OcrResultItem[] = []): void {
  ui.canvas.width = bitmap.width;
  ui.canvas.height = bitmap.height;
  canvasCtx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  canvasCtx.drawImage(bitmap, 0, 0);

  items.forEach((item, index) => {
    const [r, g, b] = deterministicColor(index);
    canvasCtx.save();
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = `rgb(${String(r)}, ${String(g)}, ${String(b)})`;
    canvasCtx.fillStyle = `rgba(${String(r)}, ${String(g)}, ${String(b)}, 0.22)`;
    drawPolygonPath(canvasCtx, item.poly);
    canvasCtx.fill();
    canvasCtx.stroke();
    canvasCtx.restore();
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
  drawPreview(state.previewBitmap);
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
    drawPreview(state.previewBitmap, result.items);
    renderResults(result.items);
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
