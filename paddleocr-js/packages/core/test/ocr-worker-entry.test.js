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

vi.mock("../src/worker/entry.js", () => ({
  attachWorkerMessageHandler
}));

vi.mock("../src/platform/worker.js", () => ({
  sourcePayloadToMat,
  ensureServedFromHttp
}));

vi.mock("../src/pipelines/ocr/core.js", () => ({
  OcrPipelineRunner
}));

afterEach(() => {
  capturedHandler = null;
  vi.resetModules();
  vi.clearAllMocks();
});

async function loadWorkerEntry() {
  await import("../src/pipelines/ocr/worker-entry.js");
  expect(typeof capturedHandler).toBe("function");
}

const WASM_INIT_SUMMARY = Object.freeze({ backend: "wasm" });
const EMPTY_MODEL_CONFIG = Object.freeze({ det: {}, rec: {} });

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
        runtime: { backend: "wasm" }
      }
    });

    expect(OcrPipelineRunner).toHaveBeenCalledWith({
      runtime: { backend: "wasm" },
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
    await capturedHandler("init", { options: { id: 1 } });
    await capturedHandler("init", { options: { id: 2 } });

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(OcrPipelineRunner).toHaveBeenCalledTimes(2);
  });

  it("routes predict and dispose requests through the active runner", async () => {
    setupResolvedInitAndModelConfig();
    predict.mockResolvedValue({ items: [] });
    dispose.mockResolvedValue(undefined);

    await loadWorkerEntry();
    await capturedHandler("init", { options: {} });

    await expect(
      capturedHandler("predict", { source: { kind: "imageBitmap" }, params: { limit: 1 } })
    ).resolves.toEqual({ items: [] });
    expect(predict).toHaveBeenCalledWith({ kind: "imageBitmap" }, { limit: 1 });

    await expect(capturedHandler("dispose", {})).resolves.toEqual({});
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it("rejects predict before initialization and unknown request types", async () => {
    await loadWorkerEntry();

    await expect(capturedHandler("predict", { source: {}, params: {} })).rejects.toThrow(
      /not initialized/i
    );
    await expect(capturedHandler("other", {})).rejects.toThrow(/Unsupported worker request type/i);
  });
});
