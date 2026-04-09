import { afterEach, describe, expect, it, vi } from "vitest";

const loadModelAsset = vi.fn();
const createDetModel = vi.fn();
const createRecModel = vi.fn();
const cropByPoly = vi.fn();
const initOpenCvRuntime = vi.fn();
const initOrtRuntime = vi.fn();
const nowMs = vi.fn();
const getOcrRuntimeParams = vi.fn();
const cloneDefaultOcrConfig = vi.fn();
const validateLoadedModelName = vi.fn();

vi.mock("../src/resources/index", () => ({
  loadModelAsset
}));

vi.mock("../src/models/index", () => ({
  createDetModel,
  createRecModel
}));

vi.mock("../src/pipelines/ocr/crop", () => ({
  cropByPoly
}));

vi.mock("../src/runtime/opencv", () => ({
  initOpenCvRuntime
}));

vi.mock("../src/runtime/ort", () => ({
  initOrtRuntime
}));

vi.mock("../src/utils/common", () => ({
  nowMs
}));

vi.mock("../src/pipelines/ocr/runtime-params", () => ({
  getOcrRuntimeParams
}));

vi.mock("../src/pipelines/ocr/shared", () => ({
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

function createResolvedAssets() {
  return {
    det: { url: "/det.tar" },
    rec: { url: "/rec.tar" }
  };
}

function minimalPipelineConfig(overrides: Record<string, unknown> = {}) {
  return {
    pipelineName: "OCR",
    raw: {},
    warnings: [] as string[],
    unsupportedFeatures: [] as string[],
    modelSelection: {
      textDetectionModelName: "det-name",
      textRecognitionModelName: "rec-name"
    },
    assets: createResolvedAssets(),
    runtimeDefaults: {} as Record<string, unknown>,
    pipelineBatchSize: 1,
    textDetectionBatchSize: 1,
    textRecognitionBatchSize: 1,
    ...overrides
  };
}

function mockEmptyDefaultOcrConfig() {
  cloneDefaultOcrConfig.mockReturnValue({ det: {}, rec: {} });
}

async function loadCoreModule() {
  return import("../src/pipelines/ocr/core");
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
    loadModelAsset
      .mockResolvedValueOnce({
        modelBytes: new Uint8Array([1]),
        configText: "det-config",
        download: { url: "/det.tar", bytes: 100 }
      })
      .mockResolvedValueOnce({
        modelBytes: new Uint8Array([2]),
        configText: "rec-config",
        download: { url: "/rec.tar", bytes: 200 }
      });
    createDetModel.mockResolvedValue(detModel);
    createRecModel.mockResolvedValue(recModel);

    const { OcrPipelineRunner } = await loadCoreModule();
    const ensureServedFromHttp = vi.fn();
    const runner = new OcrPipelineRunner({
      pipelineConfig: minimalPipelineConfig({
        warnings: ["warning"]
      }),
      ortOptions: AUTO_RUNTIME_OPTIONS,
      ensureServedFromHttp
    });

    const summary = await runner.initialize();

    expect(ensureServedFromHttp).toHaveBeenCalledTimes(1);
    expect(initOpenCvRuntime).toHaveBeenCalledTimes(1);
    expect(initOrtRuntime).toHaveBeenCalledWith(AUTO_RUNTIME_OPTIONS);
    expect(loadModelAsset).toHaveBeenCalledTimes(2);
    expect(validateLoadedModelName).toHaveBeenNthCalledWith(
      1,
      "TextDetection",
      "det-name",
      "det-config"
    );
    expect(validateLoadedModelName).toHaveBeenNthCalledWith(
      2,
      "TextRecognition",
      "rec-name",
      "rec-config"
    );
    expect(createDetModel).toHaveBeenCalledWith({
      ort,
      modelBytes: new Uint8Array([1]),
      configText: "det-config",
      backend: AUTO_RUNTIME_OPTIONS.backend,
      webgpuState: { available: true, reason: "" },
      batchSize: 1
    });
    expect(createRecModel).toHaveBeenCalledWith({
      ort,
      modelBytes: new Uint8Array([2]),
      configText: "rec-config",
      backend: AUTO_RUNTIME_OPTIONS.backend,
      webgpuState: { available: true, reason: "" },
      batchSize: 1
    });
    expect(summary).toEqual({
      backend: AUTO_RUNTIME_OPTIONS.backend,
      webgpuAvailable: true,
      detProvider: "wasm",
      recProvider: "webgpu",
      assets: [
        { url: "/det.tar", bytes: 100 },
        { url: "/rec.tar", bytes: 200 }
      ],
      elapsedMs: 45,
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
      pipelineConfig: minimalPipelineConfig({
        assets: {
          det: null,
          rec: { id: "rec" }
        }
      })
    });

    await expect(runner.initialize()).rejects.toThrow(
      /requires pre-resolved detection and recognition asset/i
    );
  });

  it("predicts OCR results and filters by score threshold", async () => {
    const cv = { name: "cv" };
    const sourceMat = { delete: vi.fn() };
    const sourceImage = {
      width: 640,
      height: 480,
      mat: sourceMat,
      dispose: vi.fn()
    };
    const cropA = { delete: vi.fn() };
    const cropB = { delete: vi.fn() };
    const detModel = {
      provider: "wasm",
      predict: vi
        .fn()
        .mockResolvedValue([
          { boxes: [{ poly: [[1, 1]] }, { poly: [[2, 2]] }], srcW: 640, srcH: 480 }
        ]),
      dispose: vi.fn()
    };
    const recModel = {
      provider: "wasm",
      predict: vi.fn().mockResolvedValue([
        { text: "high", score: 0.95 },
        { text: "low", score: 0.4 }
      ]),
      dispose: vi.fn()
    };

    mockEmptyDefaultOcrConfig();
    getOcrRuntimeParams.mockReturnValue({
      det: {},
      pipeline: { scoreThresh: 0.5 }
    });
    cropByPoly.mockReturnValueOnce(cropA).mockReturnValueOnce(cropB);
    nowMs
      .mockReturnValueOnce(10)
      .mockReturnValueOnce(20)
      .mockReturnValueOnce(30)
      .mockReturnValueOnce(40)
      .mockReturnValueOnce(60)
      .mockReturnValueOnce(70);

    const { OcrPipelineRunner } = await loadCoreModule();
    const runner = new OcrPipelineRunner({
      pipelineConfig: minimalPipelineConfig({
        runtimeDefaults: { text_det_limit_side_len: 64 }
      }),
      ortOptions: AUTO_RUNTIME_OPTIONS,
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
    expect(detModel.predict).toHaveBeenCalledWith(cv, [sourceMat], {});
    expect(cropByPoly).toHaveBeenNthCalledWith(1, cv, sourceMat, [[1, 1]]);
    expect(cropByPoly).toHaveBeenNthCalledWith(2, cv, sourceMat, [[2, 2]]);
    expect(recModel.predict).toHaveBeenCalledWith(cv, [cropA, cropB]);
    expect(cropA.delete).toHaveBeenCalledTimes(1);
    expect(cropB.delete).toHaveBeenCalledTimes(1);
    expect(sourceImage.dispose).toHaveBeenCalledTimes(1);
    expect(result).toEqual([
      {
        image: { width: 640, height: 480 },
        items: [{ poly: [[1, 1]], text: "high", score: 0.95 }],
        metrics: {
          detMs: 10,
          recMs: 20,
          totalMs: 60,
          detectedBoxes: 2,
          recognizedCount: 1
        },
        runtime: {
          requestedBackend: AUTO_RUNTIME_OPTIONS.backend,
          detProvider: "wasm",
          recProvider: "wasm",
          webgpuAvailable: false
        }
      }
    ]);
  });

  it("auto-initializes on predict and rejects when source adapter is missing", async () => {
    const detModel = {
      provider: "wasm",
      predict: vi.fn().mockResolvedValue([{ boxes: [], srcW: 1, srcH: 1 }]),
      dispose: vi.fn()
    };
    const recModel = {
      provider: "wasm",
      predict: vi.fn().mockResolvedValue([]),
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
    loadModelAsset
      .mockResolvedValueOnce({
        modelBytes: new Uint8Array([1]),
        configText: "det-config",
        download: { url: "/det.tar", bytes: 100 }
      })
      .mockResolvedValueOnce({
        modelBytes: new Uint8Array([2]),
        configText: "rec-config",
        download: { url: "/rec.tar", bytes: 200 }
      });
    createDetModel.mockResolvedValue(detModel);
    createRecModel.mockResolvedValue(recModel);
    getOcrRuntimeParams.mockReturnValue({
      det: {},
      pipeline: { scoreThresh: 0 }
    });

    const { OcrPipelineRunner } = await loadCoreModule();
    const noSourceRunner = new OcrPipelineRunner({
      pipelineConfig: minimalPipelineConfig()
    });
    await expect(noSourceRunner.predict({}, {})).rejects.toThrow(
      /source adapter is not configured/i
    );

    const sourceImage = {
      width: 1,
      height: 1,
      mat: {},
      dispose: vi.fn()
    };
    const runner = new OcrPipelineRunner({
      pipelineConfig: minimalPipelineConfig(),
      sourceToMat: vi.fn().mockResolvedValue(sourceImage)
    });

    const result = await runner.predict({}, {});

    expect(initOpenCvRuntime).toHaveBeenCalled();
    expect(result[0].items).toEqual([]);
    expect(sourceImage.dispose).toHaveBeenCalledTimes(1);
  });

  it("disposes models and clears references", async () => {
    mockEmptyDefaultOcrConfig();
    const detDispose = vi.fn().mockResolvedValue(undefined);
    const recDispose = vi.fn().mockResolvedValue(undefined);

    const { OcrPipelineRunner } = await loadCoreModule();
    const runner = new OcrPipelineRunner({
      pipelineConfig: minimalPipelineConfig()
    });
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
