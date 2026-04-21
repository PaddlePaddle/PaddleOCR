import { afterEach, describe, expect, it, vi } from "vitest";

const sourceToWorkerPayload = vi.fn();

vi.mock("../src/platform/browser", () => ({
  sourceToWorkerPayload
}));

afterEach(() => {
  vi.clearAllMocks();
});

function basePipelineConfig() {
  return {
    pipelineName: "OCR",
    raw: {},
    warnings: [] as string[],
    unsupportedFeatures: [] as string[],
    modelSelection: {
      textDetectionModelName: "PP-OCRv5_mobile_det",
      textRecognitionModelName: "PP-OCRv5_mobile_rec"
    },
    assets: {
      det: { id: "det" },
      rec: { id: "rec" }
    },
    runtimeDefaults: {} as Record<string, unknown>,
    pipelineBatchSize: 1,
    textDetectionBatchSize: 1,
    textRecognitionBatchSize: 1
  };
}

function createWorkerBackedOptions(overrides: Record<string, unknown> = {}) {
  const { pipelineConfig: pipelineOverrides, ...rest } = overrides;
  return {
    pipelineConfig: {
      ...basePipelineConfig(),
      ...(pipelineOverrides as object)
    },
    ortOptions: {},
    ...rest
  };
}

describe("worker-backed OCR adapter", () => {
  it("initializes once and forces wasm proxy off in worker mode", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed");
    const transportClient = {
      request: vi.fn().mockResolvedValue({
        summary: { backend: "wasm" },
        modelConfig: { det: { name: "det" }, rec: { name: "rec" } }
      }),
      dispose: vi.fn()
    };

    const ocr = new WorkerBackedPaddleOCR(
      createWorkerBackedOptions({
        pipelineConfig: { runtimeDefaults: { text_det_limit_side_len: 64 } },
        ortOptions: { proxy: true }
      }),
      transportClient
    );

    const first = await ocr.initialize();
    const second = await ocr.initialize();

    expect(first).toEqual({ backend: "wasm" });
    expect(second).toEqual({ backend: "wasm" });
    expect(transportClient.request).toHaveBeenCalledTimes(1);
    expect(transportClient.request).toHaveBeenCalledWith("init", {
      options: expect.objectContaining({
        ortOptions: {
          proxy: true,
          disableWasmProxy: true
        }
      })
    });
    expect(ocr.getInitializationSummary()).toEqual({ backend: "wasm" });
    expect(ocr.getModelConfig()).toEqual({ det: { name: "det" }, rec: { name: "rec" } });
  });

  it("predicts through the worker transport using transferable payloads", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed");
    const transferables = [{ id: "bitmap" }];
    const mockOcrResult = {
      image: { width: 1, height: 1 },
      items: [{ poly: [[0, 0]], text: "hello", score: 1 }],
      metrics: {
        detMs: 1,
        recMs: 1,
        totalMs: 2,
        detectedBoxes: 1,
        recognizedCount: 1
      },
      runtime: {
        requestedBackend: "auto",
        detProvider: "wasm",
        recProvider: "wasm",
        webgpuAvailable: false
      }
    };
    const transportClient = {
      request: vi
        .fn()
        .mockResolvedValueOnce({
          summary: { backend: "wasm" },
          modelConfig: { det: {}, rec: {} }
        })
        .mockResolvedValueOnce([mockOcrResult]),
      dispose: vi.fn()
    };
    sourceToWorkerPayload.mockResolvedValue({
      payload: {
        kind: "imageBitmap",
        imageBitmap: transferables[0]
      },
      transferables
    });

    const ocr = new WorkerBackedPaddleOCR(createWorkerBackedOptions(), transportClient);

    const result = await ocr.predict({ kind: "source" }, { text_rec_score_thresh: 0.5 });

    expect(sourceToWorkerPayload).toHaveBeenCalledWith({ kind: "source" });
    expect(transportClient.request).toHaveBeenNthCalledWith(
      2,
      "predict",
      {
        sources: [
          {
            kind: "imageBitmap",
            imageBitmap: transferables[0]
          }
        ],
        params: { text_rec_score_thresh: 0.5 }
      },
      transferables
    );
    expect(result).toEqual([mockOcrResult]);
  });

  it("predicts multiple sources when input is an array", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed");
    const mockOcrResult = {
      image: { width: 10, height: 10 },
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
    };
    const transportClient = {
      request: vi
        .fn()
        .mockResolvedValueOnce({
          summary: { backend: "wasm" },
          modelConfig: { det: {}, rec: {} }
        })
        .mockResolvedValueOnce([
          mockOcrResult,
          { ...mockOcrResult, image: { width: 20, height: 20 } }
        ]),
      dispose: vi.fn()
    };
    sourceToWorkerPayload.mockResolvedValue({
      payload: { kind: "imageBitmap", imageBitmap: {} },
      transferables: []
    });

    const ocr = new WorkerBackedPaddleOCR(createWorkerBackedOptions(), transportClient);

    await ocr.predict([{ a: 1 }, { b: 2 }], {});

    expect(sourceToWorkerPayload).toHaveBeenCalledTimes(2);
    expect(transportClient.request).toHaveBeenNthCalledWith(
      2,
      "predict",
      {
        sources: [
          { kind: "imageBitmap", imageBitmap: {} },
          { kind: "imageBitmap", imageBitmap: {} }
        ],
        params: {}
      },
      []
    );
  });

  it("disposes the transport after initialization failures and allows retrying", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed");
    const transportClient = {
      request: vi
        .fn()
        .mockRejectedValueOnce(new Error("init failed"))
        .mockResolvedValueOnce({
          summary: { backend: "wasm" },
          modelConfig: { det: {}, rec: {} }
        }),
      dispose: vi.fn()
    };

    const ocr = new WorkerBackedPaddleOCR(createWorkerBackedOptions(), transportClient);

    await expect(ocr.initialize()).rejects.toThrow("init failed");
    expect(transportClient.dispose).toHaveBeenCalledTimes(1);
    await expect(ocr.initialize()).resolves.toEqual({ backend: "wasm" });
  });

  it("swallows dispose request failures and rejects use after disposal", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed");
    const transportClient = {
      request: vi.fn().mockRejectedValue(new Error("worker already gone")),
      dispose: vi.fn()
    };

    const ocr = new WorkerBackedPaddleOCR(createWorkerBackedOptions(), transportClient);

    await expect(ocr.dispose()).resolves.toBeUndefined();
    expect(transportClient.request).toHaveBeenCalledWith("dispose", {});
    expect(transportClient.dispose).toHaveBeenCalledTimes(1);
    await expect(ocr.predict({}, {})).rejects.toThrow(/worker instance has been disposed/i);
  });
});
