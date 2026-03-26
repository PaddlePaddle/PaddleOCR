import { afterEach, describe, expect, it, vi } from "vitest";

const loadStandardModelAsset = vi.fn();
const createDetModel = vi.fn();
const createRecModel = vi.fn();
const cropByPoly = vi.fn();
const initOpenCvRuntime = vi.fn();
const initOrtRuntime = vi.fn();
const nowMs = vi.fn();
const getOcrRuntimeParams = vi.fn();
const cloneDefaultOcrConfig = vi.fn();
const validateLoadedModelName = vi.fn();

vi.mock("../src/resources/index.js", () => ({
  loadStandardModelAsset
}));

vi.mock("../src/models/index.js", () => ({
  createDetModel,
  createRecModel,
  cropByPoly
}));

vi.mock("../src/runtime/opencv.js", () => ({
  initOpenCvRuntime
}));

vi.mock("../src/runtime/ort.js", () => ({
  initOrtRuntime
}));

vi.mock("../src/utils/common.js", () => ({
  nowMs
}));

vi.mock("../src/pipelines/ocr/runtime-params.js", () => ({
  getOcrRuntimeParams
}));

vi.mock("../src/pipelines/ocr/shared.js", () => ({
  cloneDefaultOcrConfig,
  validateLoadedModelName
}));

afterEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
});

const AUTO_RUNTIME_OPTIONS = Object.freeze({
  backend: "auto"
});

function createCrop() {
  return {
    delete: vi.fn()
  };
}

function createResolvedAssets() {
  return {
    det: { id: "det" },
    rec: { id: "rec" }
  };
}

function mockEmptyDefaultOcrConfig() {
  cloneDefaultOcrConfig.mockReturnValue({ det: {}, rec: {} });
}

async function loadCoreModule() {
  return import("../src/pipelines/ocr/core.js");
}

describe("OCR pipeline core", () => {
  it("initializes runtimes, loads assets, and creates models", async () => {
    const cv = { name: "cv" };
    const ort = { name: "ort" };
    const detModel = { config: { det: true }, provider: "wasm", dispose: vi.fn() };
    const recModel = { config: { rec: true }, provider: "webgpu", dispose: vi.fn() };

    cloneDefaultOcrConfig.mockReturnValue({
      det: { marker: "default-det-config" },
      rec: { marker: "default-rec-config" }
    });
    nowMs.mockReturnValueOnce(100).mockReturnValueOnce(145);
    initOpenCvRuntime.mockResolvedValue({ cv });
    initOrtRuntime.mockResolvedValue({
      ort,
      webgpuState: { available: true, reason: "" },
      backend: "auto"
    });
    loadStandardModelAsset
      .mockResolvedValueOnce({
        modelBytes: new Uint8Array([1]),
        configText: "det-config",
        download: { cacheHit: true }
      })
      .mockResolvedValueOnce({
        modelBytes: new Uint8Array([2]),
        configText: "rec-config",
        download: { cacheHit: false }
      });
    createDetModel.mockResolvedValue(detModel);
    createRecModel.mockResolvedValue(recModel);

    const { OcrPipelineRunner } = await loadCoreModule();
    const ensureServedFromHttp = vi.fn();
    const runner = new OcrPipelineRunner({
      assets: {
        det: { id: "det" },
        rec: { id: "rec" }
      },
      modelSelection: {
        textDetectionModelName: "det-name",
        textRecognitionModelName: "rec-name"
      },
      pipelineConfig: {
        warnings: ["warning"]
      },
      runtime: AUTO_RUNTIME_OPTIONS,
      ensureServedFromHttp
    });

    const summary = await runner.initialize();

    expect(ensureServedFromHttp).toHaveBeenCalledTimes(1);
    expect(initOpenCvRuntime).toHaveBeenCalledTimes(1);
    expect(initOrtRuntime).toHaveBeenCalledWith(AUTO_RUNTIME_OPTIONS);
    expect(loadStandardModelAsset).toHaveBeenCalledTimes(2);
    expect(validateLoadedModelName).toHaveBeenNthCalledWith(1, "TextDetection", "det-name", "det-config");
    expect(validateLoadedModelName).toHaveBeenNthCalledWith(2, "TextRecognition", "rec-name", "rec-config");
    expect(createDetModel).toHaveBeenCalledWith({
      ort,
      modelBytes: new Uint8Array([1]),
      configText: "det-config",
      backend: AUTO_RUNTIME_OPTIONS.backend,
      webgpuState: { available: true, reason: "" }
    });
    expect(createRecModel).toHaveBeenCalledWith({
      ort,
      modelBytes: new Uint8Array([2]),
      configText: "rec-config",
      backend: AUTO_RUNTIME_OPTIONS.backend,
      webgpuState: { available: true, reason: "" }
    });
    expect(summary).toEqual({
      backend: AUTO_RUNTIME_OPTIONS.backend,
      webgpuAvailable: true,
      detProvider: "wasm",
      recProvider: "webgpu",
      assets: [{ cacheHit: true }, { cacheHit: false }],
      elapsedMs: 45,
      cacheHits: 1,
      cacheMisses: 1,
      pipelineConfigWarnings: ["warning"]
    });
    expect(runner.getInitializationSummary()).toEqual(summary);
    expect(runner.getModelConfig()).toEqual({
      det: { det: true },
      rec: { rec: true }
    });
  });

  it("rejects initialization when assets are not pre-resolved", async () => {
    mockEmptyDefaultOcrConfig();
    initOpenCvRuntime.mockResolvedValue({ cv: {} });
    initOrtRuntime.mockResolvedValue({
      ort: {},
      webgpuState: { available: false, reason: "" },
      backend: "wasm"
    });

    const { OcrPipelineRunner } = await loadCoreModule();
    const runner = new OcrPipelineRunner({
      assets: {
        det: null,
        rec: { id: "rec" }
      }
    });

    await expect(runner.initialize()).rejects.toThrow(/requires pre-resolved detection and recognition asset/i);
  });

  it("predicts OCR results and filters by score threshold", async () => {
    const cv = { name: "cv" };
    const sourceMat = {
      delete: vi.fn()
    };
    const sourceImage = {
      width: 640,
      height: 480,
      mat: sourceMat,
      dispose: vi.fn()
    };
    const cropA = createCrop();
    const cropB = createCrop();
    const detModel = {
      provider: "wasm",
      detect: vi.fn().mockResolvedValue({
        boxes: [{ poly: [[1, 1]] }, { poly: [[2, 2]] }]
      }),
      dispose: vi.fn()
    };
    const recModel = {
      provider: "wasm",
      prepareSample: vi
        .fn()
        .mockReturnValueOnce({ originalIndex: 1, poly: [[2, 2]], width: 40, chw: new Float32Array(1) })
        .mockReturnValueOnce({ originalIndex: 0, poly: [[1, 1]], width: 20, chw: new Float32Array(1) }),
      recognize: vi.fn().mockResolvedValue([
        { originalIndex: 1, poly: [[2, 2]], text: "low", score: 0.4 },
        { originalIndex: 0, poly: [[1, 1]], text: "high", score: 0.95 },
        { originalIndex: 2, poly: [[3, 3]], text: "", score: 0.99 }
      ]),
      dispose: vi.fn()
    };

    mockEmptyDefaultOcrConfig();
    getOcrRuntimeParams.mockReturnValue({
      text_rec_score_thresh: 0.5
    });
    cropByPoly.mockReturnValueOnce(cropA).mockReturnValueOnce(cropB);
    nowMs
      .mockReturnValueOnce(10)
      .mockReturnValueOnce(20)
      .mockReturnValueOnce(30)
      .mockReturnValueOnce(40)
      .mockReturnValueOnce(60)
      .mockReturnValueOnce(70)
      .mockReturnValueOnce(90)
      .mockReturnValueOnce(100);

    const { OcrPipelineRunner } = await loadCoreModule();
    const runner = new OcrPipelineRunner({
      runtime: AUTO_RUNTIME_OPTIONS,
      runtimeDefaults: {
        text_det_limit_side_len: 64
      },
      sourceToMat: vi.fn().mockResolvedValue(sourceImage)
    });
    runner.cv = cv;
    runner.ort = { name: "ort" };
    runner.detModel = detModel;
    runner.recModel = recModel;
    runner.webgpuState = { available: false, reason: "" };
    runner.modelConfig = { det: { conf: true }, rec: { conf: true } };

    const result = await runner.predict({ kind: "blob" }, { text_rec_score_thresh: 0.8 });

    expect(getOcrRuntimeParams).toHaveBeenCalledWith(
      { det: { conf: true }, rec: { conf: true } },
      { text_det_limit_side_len: 64 },
      { text_rec_score_thresh: 0.8 }
    );
    expect(detModel.detect).toHaveBeenCalledWith({
      cv,
      sourceMat,
      params: { text_rec_score_thresh: 0.5 }
    });
    expect(cropByPoly).toHaveBeenNthCalledWith(1, cv, sourceMat, [[1, 1]]);
    expect(cropByPoly).toHaveBeenNthCalledWith(2, cv, sourceMat, [[2, 2]]);
    expect(cropA.delete).toHaveBeenCalledTimes(1);
    expect(cropB.delete).toHaveBeenCalledTimes(1);
    expect(sourceImage.dispose).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      image: {
        width: 640,
        height: 480
      },
      items: [{ originalIndex: 0, poly: [[1, 1]], text: "high", score: 0.95 }],
      metrics: {
        detInferMs: 10,
        recPrepMs: 20,
        recInferMs: 20,
        totalMs: 90,
        detectedBoxes: 2,
        recognizedCount: 1
      },
      runtime: {
        requestedBackend: AUTO_RUNTIME_OPTIONS.backend,
        detProvider: "wasm",
        recProvider: "wasm",
        webgpuAvailable: false
      }
    });
  });

  it("auto-initializes on predict and rejects when source adapter is missing", async () => {
    const detModel = {
      provider: "wasm",
      detect: vi.fn().mockResolvedValue({ boxes: [] }),
      dispose: vi.fn()
    };
    const recModel = {
      provider: "wasm",
      prepareSample: vi.fn(),
      recognize: vi.fn().mockResolvedValue([]),
      dispose: vi.fn()
    };

    mockEmptyDefaultOcrConfig();
    nowMs.mockReturnValue(0);
    initOpenCvRuntime.mockResolvedValue({ cv: {} });
    initOrtRuntime.mockResolvedValue({
      ort: {},
      webgpuState: { available: false, reason: "" },
      backend: "wasm"
    });
    loadStandardModelAsset
      .mockResolvedValueOnce({
        modelBytes: new Uint8Array([1]),
        configText: "det-config",
        download: { cacheHit: false }
      })
      .mockResolvedValueOnce({
        modelBytes: new Uint8Array([2]),
        configText: "rec-config",
        download: { cacheHit: false }
      });
    createDetModel.mockResolvedValue(detModel);
    createRecModel.mockResolvedValue(recModel);
    getOcrRuntimeParams.mockReturnValue({ text_rec_score_thresh: 0 });

    const { OcrPipelineRunner } = await loadCoreModule();
    const noSourceRunner = new OcrPipelineRunner({});
    await expect(noSourceRunner.predict({}, {})).rejects.toThrow(/source adapter is not configured/i);

    const sourceImage = {
      width: 1,
      height: 1,
      mat: {},
      dispose: vi.fn()
    };
    const runner = new OcrPipelineRunner({
      assets: createResolvedAssets(),
      sourceToMat: vi.fn().mockResolvedValue(sourceImage)
    });

    const result = await runner.predict({}, {});

    expect(initOpenCvRuntime).toHaveBeenCalled();
    expect(result.items).toEqual([]);
    expect(sourceImage.dispose).toHaveBeenCalledTimes(1);
  });

  it("disposes models and clears references", async () => {
    mockEmptyDefaultOcrConfig();
    const detDispose = vi.fn().mockResolvedValue(undefined);
    const recDispose = vi.fn().mockResolvedValue(undefined);

    const { OcrPipelineRunner } = await loadCoreModule();
    const runner = new OcrPipelineRunner({});
    runner.detModel = { dispose: detDispose };
    runner.recModel = { dispose: recDispose };

    await runner.disposeModelsOnly();
    expect(detDispose).toHaveBeenCalledTimes(1);
    expect(recDispose).toHaveBeenCalledTimes(1);
    expect(runner.detModel).toBeNull();
    expect(runner.recModel).toBeNull();

    runner.detModel = { dispose: detDispose };
    runner.recModel = { dispose: recDispose };
    await runner.dispose();
    expect(runner.detModel).toBeNull();
    expect(runner.recModel).toBeNull();
  });
});
