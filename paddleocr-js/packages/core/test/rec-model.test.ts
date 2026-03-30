import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockOrtTensorClass } from "./helpers/mock-ort-tensor";

const assertStandardModelResources = vi.fn();
const createSession = vi.fn();
const getProviderCandidates = vi.fn();
const releaseSessions = vi.fn();
const clamp = vi.fn((value, min, max) => Math.max(min, Math.min(max, value)));
const withTimeout = vi.fn((promise) => promise);
const getTransformOp = vi.fn();
const parseInferenceConfigText = vi.fn();
const parseScaleValue = vi.fn();
const toBgrFloatCHWFromBgr = vi.fn();

vi.mock("../src/resources/standard-model", () => ({
  assertStandardModelResources
}));

vi.mock("../src/runtime/ort", () => ({
  createSession,
  getProviderCandidates,
  releaseSessions
}));

vi.mock("../src/utils/common", () => ({
  clamp,
  withTimeout
}));

vi.mock("../src/models/common", () => ({
  getTransformOp,
  parseInferenceConfigText,
  parseScaleValue,
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

/** OpenCV-like facade for `prepareRecSample` branch tests. */
function createPrepareRecSampleCvFixture() {
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

const SAMPLE_REC_PREPARE_CONFIG = Object.freeze({
  imageShape: [3, 4, 8],
  maxWidth: 16,
  normalize: {}
});

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
    getTransformOp
      .mockReturnValueOnce({ image_shape: [3, 32, 160] })
      .mockReturnValueOnce({ mean: [0.1], std: [0.9], scale: "1./2." });
    parseScaleValue.mockReturnValue(0.5);

    const {
      DEFAULT_REC_MODEL_PARSE_FALLBACKS,
      DEFAULT_REC_RUNTIME_LIMITS,
      parseRecModelConfigText
    } = await loadRecModule();
    expect(parseRecModelConfigText("config")).toEqual({
      imageShape: [3, 32, 160],
      maxBatch: DEFAULT_REC_RUNTIME_LIMITS.maxBatch,
      maxWidth: DEFAULT_REC_RUNTIME_LIMITS.maxWidth,
      scoreThresh: DEFAULT_REC_MODEL_PARSE_FALLBACKS.scoreThresh,
      normalize: {
        mean: [0.1],
        std: [0.9],
        scale: 0.5
      },
      charDict: ["a", "b", " "]
    });

    parseInferenceConfigText.mockReturnValue({
      PreProcess: {},
      PostProcess: {}
    });
    getTransformOp.mockReturnValue(undefined);

    expect(() => parseRecModelConfigText("invalid")).toThrow(/No valid character_dict/i);
  });

  it("prepares recognition samples across copy, rgba, and grayscale branches", async () => {
    clamp.mockImplementation((value, min, max) => Math.max(min, Math.min(max, value)));
    toBgrFloatCHWFromBgr.mockImplementation((data, width, height) => {
      const out = new Float32Array(3 * width * height);
      for (let i = 0; i < out.length; i += 1) out[i] = i + 1;
      return out;
    });

    const { prepareRecSample } = await loadRecModule();

    const cvCopy = createPrepareRecSampleCvFixture();
    const sampleCopy = prepareRecSample(
      {
        cv: cvCopy,
        config: SAMPLE_REC_PREPARE_CONFIG
      },
      createMat(3, 8, 4),
      [[1, 1]],
      2
    );
    expect(sampleCopy.originalIndex).toBe(2);
    expect(sampleCopy.width).toBe(8);
    expect(cvCopy.cvtColor).not.toHaveBeenCalled();

    const cvRgba = createPrepareRecSampleCvFixture();
    const sampleRgba = prepareRecSample(
      {
        cv: cvRgba,
        config: SAMPLE_REC_PREPARE_CONFIG
      },
      createMat(4, 8, 4),
      [[2, 2]],
      3
    );
    expect(sampleRgba.width).toBe(8);
    expect(cvRgba.cvtColor).toHaveBeenCalled();

    const cvGray = createPrepareRecSampleCvFixture();
    const sampleGray = prepareRecSample(
      {
        cv: cvGray,
        config: SAMPLE_REC_PREPARE_CONFIG
      },
      createMat(1, 2, 4),
      [[3, 3]],
      4
    );
    expect(sampleGray.width).toBe(8);
    expect(cvGray.cvtColor).toHaveBeenCalled();

    expect(() =>
      prepareRecSample(
        {
          cv: createPrepareRecSampleCvFixture(),
          config: {
            ...SAMPLE_REC_PREPARE_CONFIG,
            imageShape: [1, 4, 8]
          }
        },
        createMat(3, 8, 4),
        [[0, 0]],
        0
      )
    ).toThrow(/Unexpected recognition channels/i);
  });

  it("runs recognition batches and decodes CTC output", async () => {
    const tensorCalls = [];
    const ort = {
      Tensor: createMockOrtTensorClass(tensorCalls)
    };
    const session = {
      inputNames: ["input"],
      outputNames: ["output"],
      run: vi
        .fn()
        .mockResolvedValueOnce({
          output: {
            dims: [2, 3, 4],
            data: new Float32Array([
              0.1, 0.9, 0.2, 0.1,
              0.2, 0.1, 0.8, 0.1,
              0.8, 0.1, 0.1, 0.0,
              0.2, 0.1, 0.8, 0.1,
              0.2, 0.1, 0.8, 0.1,
              0.7, 0.2, 0.1, 0.0
            ])
          }
        })
        .mockResolvedValueOnce({
          output: {
            dims: [1, 3, 4],
            data: new Float32Array([
              0.1, 0.1, 0.8, 0.1,
              0.1, 0.1, 0.8, 0.1,
              0.6, 0.2, 0.1, 0.1
            ])
          }
        })
    };

    const { runRecModel } = await loadRecModule();
    const results = await runRecModel(
      {
        ort,
        session,
        config: {
          maxBatch: 2,
          imageShape: [3, 4, 8]
        },
        charDict: ["A", "B", "C"]
      },
      [
        { originalIndex: 2, poly: [[2]], width: 5, chw: new Float32Array(3 * 4 * 5).fill(1) },
        { originalIndex: 0, poly: [[0]], width: 3, chw: new Float32Array(3 * 4 * 3).fill(2) },
        { originalIndex: 1, poly: [[1]], width: 4, chw: new Float32Array(3 * 4 * 4).fill(3) }
      ]
    );

    expect(session.run).toHaveBeenCalledTimes(2);
    expect(tensorCalls).toEqual([
      { type: "float32", dims: [2, 3, 4, 4], size: 96 },
      { type: "float32", dims: [1, 3, 4, 5], size: 60 }
    ]);
    expect(results).toHaveLength(3);
    expect(results[0]).toMatchObject({ originalIndex: 0, poly: [[0]], text: "AB" });
    expect(results[0].score).toBeCloseTo(0.85, 5);
    expect(results[1]).toMatchObject({ originalIndex: 1, poly: [[1]], text: "B" });
    expect(results[1].score).toBeCloseTo(0.8, 5);
    expect(results[2]).toMatchObject({ originalIndex: 2, poly: [[2]], text: "B" });
    expect(results[2].score).toBeCloseTo(0.8, 5);

    await expect(
      runRecModel(
        {
          ort,
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
          config: {
            maxBatch: 1,
            imageShape: [3, 4, 8]
          },
          charDict: ["A"]
        },
        [{ originalIndex: 0, poly: [], width: 1, chw: new Float32Array(12) }]
      )
    ).rejects.toThrow(/Unexpected rec output dims/i);
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
    getTransformOp.mockReturnValue(undefined);
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

    expect(assertStandardModelResources).toHaveBeenCalled();
    expect(model.kind).toBe("rec");
    expect(model.provider).toBe("wasm");
    expect(model.charDict).toEqual(["A", " "]);

    const sample = {
      originalIndex: 0,
      poly: [[0]],
      width: 2,
      chw: new Float32Array(24).fill(1)
    };
    await expect(model.recognize([sample])).resolves.toSatisfy((results) => {
      expect(results).toHaveLength(1);
      expect(results[0]).toMatchObject({ originalIndex: 0, poly: [[0]], text: "A" });
      expect(results[0].score).toBeCloseTo(0.9, 5);
      return true;
    });
    await expect(model.dispose()).resolves.toBeUndefined();
    expect(released.at(-1)).toBeTruthy();
    await expect(model.recognize([sample])).rejects.toThrow(/session is not initialized/i);
  });
});
