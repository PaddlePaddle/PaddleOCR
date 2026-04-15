import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockOrtTensorClass } from "./helpers/mock-ort-tensor";

const assertModelResources = vi.fn();
const createSession = vi.fn();
const getProviderCandidates = vi.fn();
const releaseSessions = vi.fn();
const clamp = vi.fn((value, min, max) => Math.max(min, Math.min(max, value)));
const withTimeout = vi.fn((promise) => promise);
const chunkArray = vi.fn((items, size) => {
  const chunks = [];
  for (let i = 0; i < items.length; i += size) {
    chunks.push(items.slice(i, i + size));
  }
  return chunks;
});
const getTransformOp = vi.fn();
const parseInferenceConfigText = vi.fn();
const toBgrFloatCHWFromBgr = vi.fn();

vi.mock("../src/resources/model-asset", () => ({
  assertModelResources
}));

vi.mock("../src/runtime/ort", () => ({
  createSession,
  getProviderCandidates,
  releaseSessions
}));

vi.mock("../src/utils/common", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/utils/common")>();
  return {
    ...actual,
    clamp,
    withTimeout,
    chunkArray
  };
});

vi.mock("../src/models/common", () => ({
  getTransformOp,
  parseInferenceConfigText,
  toBgrFloatCHWFromBgr
}));

afterEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
});

async function loadRecModule() {
  return import("../src/models/rec");
}

function createMat(channels, cols = 20, rows = 10) {
  return {
    cols,
    rows,
    data: new Uint8Array(cols * rows * 3).fill(1),
    channels: () => channels,
    copyTo: vi.fn(),
    delete: vi.fn()
  };
}

/** Minimal OpenCV-like `cv` used by `createRecModel().predict()` → internal `preprocessSample`. */
function createRecPredictCvStub() {
  return {
    Mat: class Mat {
      constructor() {
        this.deleted = false;
        this.data = new Uint8Array(8);
        this._channels = 3;
      }
      channels() {
        return this._channels;
      }
      copyTo(target) {
        target.data = this.data;
        target._channels = 3;
      }
      delete() {
        this.deleted = true;
      }
    },
    Size: class Size {
      constructor(width, height) {
        this.width = width;
        this.height = height;
      }
    },
    INTER_LINEAR: "linear",
    COLOR_RGBA2BGR: "rgba",
    COLOR_GRAY2BGR: "gray",
    resize: vi.fn((src, dst, size) => {
      dst.data = new Uint8Array(size.width * size.height * 3);
      dst._channels = src.channels();
    }),
    cvtColor: vi.fn((src, dst) => {
      dst.data = src.data;
      dst._channels = 3;
    })
  };
}

describe("recognition model", () => {
  it("parses recognition configs and validates character dictionaries", async () => {
    parseInferenceConfigText.mockReturnValue({
      PreProcess: {
        transform_ops: [{ id: "resize" }, { id: "normalize" }]
      },
      PostProcess: {
        character_dict: ["a", "b"]
      }
    });
    getTransformOp.mockReturnValueOnce({ image_shape: [3, 32, 160] });

    const { DEFAULT_REC_MODEL_PARSE_FALLBACKS, parseRecModelConfigText } = await loadRecModule();
    expect(parseRecModelConfigText("config")).toEqual({
      imageShape: [3, 32, 160],
      charDict: ["a", "b", " "]
    });

    parseInferenceConfigText.mockReturnValue({
      PreProcess: {},
      PostProcess: {}
    });
    getTransformOp.mockReturnValue(undefined);

    expect(() => parseRecModelConfigText("invalid")).toThrow(
      /RecResizeImg\.image_shape is required/i
    );
  });

  it("runs recognition batches through predict and decodes CTC output", async () => {
    parseInferenceConfigText.mockReturnValue({
      PreProcess: { transform_ops: [] },
      PostProcess: { character_dict: ["A", "B", "C"] }
    });
    getTransformOp.mockImplementation((_ops, id) => {
      if (id === "RecResizeImg") return { image_shape: [3, 4, 8] };
      return null;
    });
    clamp.mockImplementation((value, min, max) => Math.max(min, Math.min(max, value)));

    const tensorCalls = [];
    const ort = {
      Tensor: createMockOrtTensorClass(tensorCalls)
    };
    const ctcRow = new Float32Array([0.1, 0.9, 0.2, 0.1, 0.2, 0.1, 0.8, 0.1, 0.8, 0.1, 0.1, 0.0]);
    const sessionRun = vi
      .fn()
      .mockResolvedValueOnce({
        output: {
          dims: [2, 3, 4],
          data: new Float32Array([...ctcRow, ...ctcRow])
        }
      })
      .mockResolvedValueOnce({
        output: {
          dims: [1, 3, 4],
          data: ctcRow
        }
      });
    const session = {
      inputNames: ["input"],
      outputNames: ["output"],
      run: sessionRun
    };
    getProviderCandidates.mockReturnValue([["wasm"]]);
    createSession.mockResolvedValue({
      session,
      provider: "wasm"
    });

    const { createRecModel } = await loadRecModule();
    const model = await createRecModel({
      ort,
      modelBytes: new Uint8Array([1]),
      configText: "rec-batch",
      backend: "auto",
      webgpuState: { available: false, reason: "" },
      batchSize: 2
    });

    const cvFixture = createRecPredictCvStub();
    toBgrFloatCHWFromBgr.mockImplementation((data, width, height) => {
      const out = new Float32Array(3 * width * height);
      for (let i = 0; i < out.length; i += 1) out[i] = i + 1;
      return out;
    });
    const mat = createMat(3, 8, 4);
    const results = await model.predict(cvFixture, [mat, mat, mat]);

    expect(sessionRun).toHaveBeenCalledTimes(2);
    expect(tensorCalls).toEqual([
      { type: "float32", dims: [2, 3, 4, 8], size: 192 },
      { type: "float32", dims: [1, 3, 4, 8], size: 96 }
    ]);
    expect(results).toHaveLength(3);
    expect(results[0]).toMatchObject({ text: "AB" });
    expect(results[0].score).toBeCloseTo(0.85, 5);
    expect(results[1]).toMatchObject({ text: "AB" });
    expect(results[1].score).toBeCloseTo(0.85, 5);
    expect(results[2]).toMatchObject({ text: "AB" });
    expect(results[2].score).toBeCloseTo(0.85, 5);
  });

  it("allows per-predict batch size override on the recognition model", async () => {
    parseInferenceConfigText.mockReturnValue({
      PreProcess: { transform_ops: [] },
      PostProcess: { character_dict: ["A", "B", "C"] }
    });
    getTransformOp.mockImplementation((_ops, id) => {
      if (id === "RecResizeImg") return { image_shape: [3, 4, 8] };
      return null;
    });
    clamp.mockImplementation((value, min, max) => Math.max(min, Math.min(max, value)));

    const ctcRow = new Float32Array([0.1, 0.9, 0.2, 0.1, 0.2, 0.1, 0.8, 0.1, 0.8, 0.1, 0.1, 0.0]);
    const sessionRun = vi.fn().mockResolvedValue({
      output: {
        dims: [1, 3, 4],
        data: ctcRow
      }
    });
    const session = {
      inputNames: ["input"],
      outputNames: ["output"],
      run: sessionRun
    };
    getProviderCandidates.mockReturnValue([["wasm"]]);
    createSession.mockResolvedValue({
      session,
      provider: "wasm"
    });

    const { createRecModel } = await loadRecModule();
    const model = await createRecModel({
      ort: { Tensor: createMockOrtTensorClass() },
      modelBytes: new Uint8Array([1]),
      configText: "rec-override-batch",
      backend: "auto",
      webgpuState: { available: false, reason: "" },
      batchSize: 6
    });

    const cvFixture = createRecPredictCvStub();
    toBgrFloatCHWFromBgr.mockImplementation((data, width, height) => {
      const out = new Float32Array(3 * width * height);
      for (let i = 0; i < out.length; i += 1) out[i] = i + 1;
      return out;
    });
    const mat = createMat(3, 8, 4);
    await model.predict(cvFixture, [mat, mat, mat], { batchSize: 1 });

    expect(sessionRun).toHaveBeenCalledTimes(3);
  });

  it("rejects unexpected recognition output dimensions", async () => {
    parseInferenceConfigText.mockReturnValue({
      PreProcess: { transform_ops: [] },
      PostProcess: { character_dict: ["A"] }
    });
    getTransformOp.mockImplementation((_ops, id) => {
      if (id === "RecResizeImg") return { image_shape: [3, 4, 8] };
      return null;
    });
    getProviderCandidates.mockReturnValue([["wasm"]]);
    createSession.mockResolvedValue({
      session: {
        inputNames: ["input"],
        outputNames: ["output"],
        run: vi.fn().mockResolvedValue({
          output: {
            dims: [1, 4],
            data: new Float32Array([1, 2, 3, 4])
          }
        })
      },
      provider: "wasm"
    });

    const { createRecModel } = await loadRecModule();
    const model = await createRecModel({
      ort: {
        Tensor: createMockOrtTensorClass()
      },
      modelBytes: new Uint8Array([1]),
      configText: "rec-bad-out",
      backend: "auto",
      webgpuState: { available: false, reason: "" }
    });
    const cvFixture = createRecPredictCvStub();
    toBgrFloatCHWFromBgr.mockImplementation((data, width, height) => {
      const out = new Float32Array(3 * width * height);
      out.fill(1);
      return out;
    });

    await expect(model.predict(cvFixture, [createMat(3, 8, 4)])).rejects.toThrow(
      /Unexpected rec output dims/i
    );
  });

  it("creates, uses, and disposes recognition models through runtime wrappers", async () => {
    parseInferenceConfigText.mockReturnValue({
      PreProcess: {
        transform_ops: []
      },
      PostProcess: {
        character_dict: ["A"]
      }
    });
    getTransformOp.mockImplementation((_ops, id) => {
      if (id === "RecResizeImg") return { image_shape: [3, 4, 8] };
      return undefined;
    });
    getProviderCandidates.mockReturnValue([["wasm"]]);
    createSession.mockResolvedValue({
      session: {
        inputNames: ["input"],
        outputNames: ["output"],
        run: vi.fn().mockResolvedValue({
          output: {
            dims: [1, 2, 2],
            data: new Float32Array([0.1, 0.9, 0.9, 0.1])
          }
        })
      },
      provider: "wasm"
    });
    const released = [];
    releaseSessions.mockImplementation(async (session) => {
      released.push(session);
    });

    const { createRecModel, createRecModelSession } = await loadRecModule();
    const sessionState = await createRecModelSession({}, new Uint8Array([1]), "auto", {
      available: false,
      reason: ""
    });
    expect(getProviderCandidates).toHaveBeenCalledWith("auto", { available: false, reason: "" });
    expect(withTimeout).toHaveBeenCalled();
    expect(sessionState.provider).toBe("wasm");

    const model = await createRecModel({
      ort: {
        Tensor: createMockOrtTensorClass()
      },
      modelBytes: new Uint8Array([1]),
      configText: "config",
      backend: "auto",
      webgpuState: { available: false, reason: "" }
    });

    expect(assertModelResources).toHaveBeenCalled();
    expect(model.kind).toBe("rec");
    expect(model.provider).toBe("wasm");
    expect(model.config.charDict).toEqual(["A", " "]);

    const cvFixture = createRecPredictCvStub();
    toBgrFloatCHWFromBgr.mockImplementation((data, width, height) => {
      const out = new Float32Array(3 * width * height);
      for (let i = 0; i < out.length; i += 1) out[i] = 1;
      return out;
    });
    await expect(model.predict(cvFixture, [createMat(3, 8, 4)])).resolves.toSatisfy((results) => {
      expect(results).toHaveLength(1);
      expect(results[0]).toMatchObject({ text: "A" });
      expect(results[0].score).toBeCloseTo(0.9, 5);
      return true;
    });
    await expect(model.dispose()).resolves.toBeUndefined();
    expect(released.at(-1)).toBeTruthy();
    await expect(model.predict(cvFixture, [createMat(3, 8, 4)])).rejects.toThrow(
      /session is not initialized/i
    );
  });
});
