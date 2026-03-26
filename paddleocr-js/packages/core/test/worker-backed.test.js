import { afterEach, describe, expect, it, vi } from "vitest";

const sourceToWorkerPayload = vi.fn();

vi.mock("../src/platform/browser.js", () => ({
  sourceToWorkerPayload
}));

afterEach(() => {
  vi.clearAllMocks();
});

function createWorkerBackedOptions(overrides = {}) {
  return {
    assets: {
      det: { id: "det" },
      rec: { id: "rec" }
    },
    runtimeDefaults: {},
    runtime: {},
    ...overrides
  };
}

describe("worker-backed OCR adapter", () => {
  it("initializes once and forces wasm proxy off in worker mode", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed.js");
    const transportClient = {
      request: vi.fn().mockResolvedValue({
        summary: { backend: "wasm" },
        modelConfig: { det: { name: "det" }, rec: { name: "rec" } }
      }),
      dispose: vi.fn()
    };

    const ocr = new WorkerBackedPaddleOCR(
      createWorkerBackedOptions({
        runtimeDefaults: { text_det_limit_side_len: 64 },
        runtime: { proxy: true }
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
        runtime: {
          proxy: true,
          disableWasmProxy: true
        }
      })
    });
    expect(ocr.getInitializationSummary()).toEqual({ backend: "wasm" });
    expect(ocr.getModelConfig()).toEqual({ det: { name: "det" }, rec: { name: "rec" } });
  });

  it("predicts through the worker transport using transferable payloads", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed.js");
    const transferables = [{ id: "bitmap" }];
    const transportClient = {
      request: vi
        .fn()
        .mockResolvedValueOnce({
          summary: { backend: "wasm" },
          modelConfig: { det: {}, rec: {} }
        })
        .mockResolvedValueOnce({ text: "hello" }),
      dispose: vi.fn()
    };
    sourceToWorkerPayload.mockResolvedValue({
      payload: {
        kind: "imageBitmap",
        imageBitmap: transferables[0]
      },
      transferables
    });

    const ocr = new WorkerBackedPaddleOCR(
      createWorkerBackedOptions(),
      transportClient
    );

    const result = await ocr.predict({ kind: "source" }, { text_rec_score_thresh: 0.5 });

    expect(sourceToWorkerPayload).toHaveBeenCalledWith({ kind: "source" });
    expect(transportClient.request).toHaveBeenNthCalledWith(2, "predict", {
      source: {
        kind: "imageBitmap",
        imageBitmap: transferables[0]
      },
      params: { text_rec_score_thresh: 0.5 }
    }, transferables);
    expect(result).toEqual({ text: "hello" });
  });

  it("disposes the transport after initialization failures and allows retrying", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed.js");
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

    const ocr = new WorkerBackedPaddleOCR(
      createWorkerBackedOptions(),
      transportClient
    );

    await expect(ocr.initialize()).rejects.toThrow("init failed");
    expect(transportClient.dispose).toHaveBeenCalledTimes(1);
    await expect(ocr.initialize()).resolves.toEqual({ backend: "wasm" });
  });

  it("swallows dispose request failures and rejects use after disposal", async () => {
    const { WorkerBackedPaddleOCR } = await import("../src/pipelines/ocr/worker-backed.js");
    const transportClient = {
      request: vi.fn().mockRejectedValue(new Error("worker already gone")),
      dispose: vi.fn()
    };

    const ocr = new WorkerBackedPaddleOCR(
      createWorkerBackedOptions(),
      transportClient
    );

    await expect(ocr.dispose()).resolves.toBeUndefined();
    expect(transportClient.request).toHaveBeenCalledWith("dispose", {});
    expect(transportClient.dispose).toHaveBeenCalledTimes(1);
    await expect(ocr.predict({}, {})).rejects.toThrow(/worker instance has been disposed/i);
  });
});
