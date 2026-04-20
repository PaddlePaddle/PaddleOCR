import { afterEach, describe, expect, it, vi } from "vitest";

let capturedHandler = null;
const attachWorkerMessageHandler = vi.fn((handler) => {
  capturedHandler = handler;
});

const sourcePayloadToMat = vi.fn();
const ensureServedFromHttp = vi.fn();
const initialize = vi.fn();
const getModelConfig = vi.fn();
const predict = vi.fn();
const dispose = vi.fn();

const OcrPipelineRunner = vi.fn(function MockOcrPipelineRunner(options) {
  this.options = options;
  this.initialize = initialize;
  this.getModelConfig = getModelConfig;
  this.predict = predict;
  this.dispose = dispose;
});

vi.mock("../src/worker/entry", () => ({
  attachWorkerMessageHandler
}));

vi.mock("../src/platform/worker", () => ({
  sourcePayloadToMat,
  ensureServedFromHttp
}));

vi.mock("../src/pipelines/ocr/core", () => ({
  OcrPipelineRunner
}));

afterEach(() => {
  capturedHandler = null;
  vi.resetModules();
  vi.clearAllMocks();
});

async function loadWorkerEntry() {
  await import("../src/pipelines/ocr/worker-entry");
  expect(typeof capturedHandler).toBe("function");
}

const WASM_INIT_SUMMARY = Object.freeze({ backend: "wasm" });
const EMPTY_MODEL_CONFIG = Object.freeze({ det: {}, rec: {} });

const WORKER_TEST_PIPELINE = Object.freeze({
  pipelineName: "OCR",
  raw: {},
  warnings: [] as string[],
  unsupportedFeatures: [] as string[],
  modelSelection: {
    textDetectionModelName: "d",
    textRecognitionModelName: "r"
  },
  assets: {
    det: { url: "/d" },
    rec: { url: "/r" }
  },
  runtimeDefaults: {},
  pipelineBatchSize: 1,
  textDetectionBatchSize: 1,
  textRecognitionBatchSize: 1
});

function setupResolvedInitAndModelConfig() {
  initialize.mockResolvedValue(WASM_INIT_SUMMARY);
  getModelConfig.mockReturnValue(EMPTY_MODEL_CONFIG);
}

describe("OCR worker entry bootstrap", () => {
  it("registers a worker message handler on module load", async () => {
    await loadWorkerEntry();

    expect(attachWorkerMessageHandler).toHaveBeenCalledTimes(1);
    expect(typeof capturedHandler).toBe("function");
  });

  it("initializes a runner and returns summary + model config", async () => {
    setupResolvedInitAndModelConfig();

    await loadWorkerEntry();
    const result = await capturedHandler("init", {
      options: {
        pipelineConfig: WORKER_TEST_PIPELINE,
        ortOptions: { backend: "wasm" }
      }
    });

    expect(OcrPipelineRunner).toHaveBeenCalledWith({
      pipelineConfig: WORKER_TEST_PIPELINE,
      ortOptions: { backend: "wasm" },
      ensureServedFromHttp,
      sourceToMat: sourcePayloadToMat
    });
    expect(result).toEqual({
      summary: WASM_INIT_SUMMARY,
      modelConfig: EMPTY_MODEL_CONFIG
    });
  });

  it("disposes an existing runner before re-initializing", async () => {
    setupResolvedInitAndModelConfig();

    await loadWorkerEntry();
    await capturedHandler("init", { options: { pipelineConfig: WORKER_TEST_PIPELINE, id: 1 } });
    await capturedHandler("init", { options: { pipelineConfig: WORKER_TEST_PIPELINE, id: 2 } });

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(OcrPipelineRunner).toHaveBeenCalledTimes(2);
  });

  it("routes predict and dispose requests through the active runner", async () => {
    setupResolvedInitAndModelConfig();
    const predictPayload = [
      {
        image: { width: 1, height: 1 },
        items: [],
        metrics: {
          detMs: 0,
          recMs: 0,
          totalMs: 0,
          detectedBoxes: 0,
          recognizedCount: 0
        },
        runtime: {
          requestedBackend: "auto",
          detProvider: "wasm",
          recProvider: "wasm",
          webgpuAvailable: false
        }
      }
    ];
    predict.mockResolvedValue(predictPayload);
    dispose.mockResolvedValue(undefined);

    await loadWorkerEntry();
    await capturedHandler("init", { options: { pipelineConfig: WORKER_TEST_PIPELINE } });

    await expect(
      capturedHandler("predict", {
        sources: [{ kind: "imageBitmap" }],
        params: { limit: 1 }
      })
    ).resolves.toEqual(predictPayload);
    expect(predict).toHaveBeenCalledWith([{ kind: "imageBitmap" }], { limit: 1 });

    await expect(capturedHandler("dispose", {})).resolves.toEqual({});
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it("rejects predict before initialization and unknown request types", async () => {
    await loadWorkerEntry();

    await expect(capturedHandler("predict", { sources: [{}], params: {} })).rejects.toThrow(
      /not initialized/i
    );
    await expect(capturedHandler("other", {})).rejects.toThrow(/Unsupported worker request type/i);
  });
});
