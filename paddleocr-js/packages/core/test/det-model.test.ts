import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockOrtTensorClass } from "./helpers/mock-ort-tensor";

const assertModelResources = vi.fn();
const createSession = vi.fn();
const getProviderCandidates = vi.fn();
const releaseSessions = vi.fn();
const clamp = vi.fn((value, min, max) => Math.max(min, Math.min(max, value)));
const withTimeout = vi.fn((promise) => promise);
const boxScoreFast = vi.fn();
const getMiniBoxFromPoints = vi.fn();
const getTransformOp = vi.fn();
const parseInferenceConfigText = vi.fn();
const parseScaleValue = vi.fn();
const toBgrFloatCHWFromBgr = vi.fn();
const unclip = vi.fn();

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
    withTimeout
  };
});

vi.mock("../src/models/common", () => ({
  boxScoreFast,
  getMiniBoxFromPoints,
  getTransformOp,
  parseInferenceConfigText,
  parseScaleValue,
  toBgrFloatCHWFromBgr,
  unclip
}));

afterEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
});

async function loadDetModule() {
  return import("../src/models/det");
}

/** CV facade for `createDetModel().predict()` integration-style test (preprocess → infer → postprocess). */
function createDetModelIntegrationCv() {
  return {
    Mat: class Mat {
      constructor() {
        this.data = new Uint8Array(1);
      }
      channels() {
        return 3;
      }
      copyTo() {}
      delete() {}
    },
    Size: class Size {
      constructor(width, height) {
        this.width = width;
        this.height = height;
      }
    },
    INTER_LINEAR: "linear",
    CV_32FC1: "float1",
    CV_8UC1: "mask1",
    RETR_LIST: "list",
    CHAIN_APPROX_SIMPLE: "chain",
    resize: vi.fn((src, dst, size) => {
      dst.data = new Uint8Array(size.width * size.height * 3);
      dst.channels = () => 3;
      dst.copyTo = vi.fn();
      dst.delete = vi.fn();
    }),
    cvtColor: vi.fn(),
    matFromArray: vi
      .fn()
      .mockImplementationOnce(() => ({ delete: vi.fn() }))
      .mockImplementationOnce(() => ({ delete: vi.fn() })),
    MatVector: class MatVector {
      size() {
        return 1;
      }
      get() {
        return {
          rows: 4,
          data32S: [0, 0, 4, 0, 4, 2, 0, 2],
          delete: vi.fn()
        };
      }
      delete() {}
    },
    findContours: vi.fn()
  };
}

describe("detection model", () => {
  it("parses detection configs with explicit values and fallbacks", async () => {
    parseInferenceConfigText.mockReturnValue({
      PreProcess: {
        transform_ops: [{ id: "resize" }, { id: "normalize" }]
      },
      PostProcess: {
        thresh: "0.22",
        box_thresh: "0.55",
        max_candidates: "200",
        unclip_ratio: "1.8"
      }
    });
    getTransformOp
      .mockReturnValueOnce({ resize_long: 736 })
      .mockReturnValueOnce({ mean: [0.1], std: [0.9], scale: "1./2." });
    parseScaleValue.mockReturnValue(0.5);

    const { DEFAULT_DET_MODEL_PARSE_FALLBACKS, parseDetModelConfigText } = await loadDetModule();
    expect(parseDetModelConfigText("config")).toEqual({
      resizeLong: 736,
      limitType: "max",
      maxSideLimit: 4000,
      normalize: {
        mean: [0.1],
        std: [0.9],
        scale: 0.5
      },
      postprocess: {
        thresh: 0.22,
        boxThresh: 0.55,
        maxCandidates: 200,
        unclipRatio: 1.8
      }
    });

    parseInferenceConfigText.mockReturnValue({});
    getTransformOp.mockReturnValue(undefined);
    parseScaleValue.mockReturnValue(1 / 255);

    expect(parseDetModelConfigText("fallback")).toEqual({
      resizeLong: DEFAULT_DET_MODEL_PARSE_FALLBACKS.resizeLong,
      limitType: DEFAULT_DET_MODEL_PARSE_FALLBACKS.limitType,
      maxSideLimit: DEFAULT_DET_MODEL_PARSE_FALLBACKS.maxSideLimit,
      normalize: {
        mean: DEFAULT_DET_MODEL_PARSE_FALLBACKS.normalize.mean,
        std: DEFAULT_DET_MODEL_PARSE_FALLBACKS.normalize.std,
        scale: 1 / 255
      },
      postprocess: {
        thresh: DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.thresh,
        boxThresh: DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.boxThresh,
        maxCandidates: DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.maxCandidates,
        unclipRatio: DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.unclipRatio
      }
    });
  });

  it("runs detection models and crops rotated boxes", async () => {
    const { cropByPoly } = await import("../src/pipelines/ocr/crop");
    parseInferenceConfigText.mockReturnValue({
      PreProcess: { transform_ops: [] },
      PostProcess: { max_candidates: "10" }
    });
    getTransformOp.mockImplementation((_ops, id) => {
      if (id === "DetResizeForTest") return { resize_long: 64 };
      return null;
    });
    parseScaleValue.mockReturnValue(1 / 255);
    clamp.mockImplementation((value, min, max) => Math.max(min, Math.min(max, value)));
    getProviderCandidates.mockReturnValue([["wasm"]]);

    const tensorCalls = [];
    const ort = {
      Tensor: createMockOrtTensorClass(tensorCalls)
    };
    const sessionRun = vi.fn().mockResolvedValue({
      output: {
        dims: [1, 1, 4, 8],
        data: new Float32Array(32).fill(0.9)
      }
    });
    const session = {
      inputNames: ["input"],
      outputNames: ["output"],
      run: sessionRun
    };
    createSession.mockResolvedValue({
      session,
      provider: "wasm"
    });
    toBgrFloatCHWFromBgr.mockReturnValue(new Float32Array(3 * 32 * 64).fill(1));

    const makeCv = () => {
      const pred = { delete: vi.fn() };
      const bitmap = { delete: vi.fn() };
      const contour = {
        rows: 4,
        data32S: [0, 0, 4, 0, 4, 2, 0, 2],
        delete: vi.fn()
      };
      const warped = {
        rows: 20,
        cols: 10,
        delete: vi.fn()
      };
      const rotated = {
        rows: 10,
        cols: 20,
        delete: vi.fn()
      };
      return {
        warped,
        rotated,
        Mat: class Mat {
          constructor() {
            return warped;
          }
        },
        Size: class Size {
          constructor(width, height) {
            this.width = width;
            this.height = height;
          }
        },
        Scalar: class Scalar {},
        INTER_LINEAR: "linear",
        INTER_CUBIC: "cubic",
        BORDER_REPLICATE: "replicate",
        COLOR_RGBA2BGR: "rgba",
        COLOR_GRAY2BGR: "gray",
        ROTATE_90_COUNTERCLOCKWISE: "ccw",
        CV_32FC1: "float1",
        CV_8UC1: "mask1",
        CV_32FC2: "float",
        RETR_LIST: "list",
        CHAIN_APPROX_SIMPLE: "chain",
        resize: vi.fn((src, dst, size) => {
          dst.data = new Uint8Array(size.width * size.height * 3);
          dst.channels = () => 3;
          dst.copyTo = vi.fn();
          dst.delete = vi.fn();
        }),
        cvtColor: vi.fn(),
        matFromArray: vi
          .fn()
          .mockImplementationOnce(() => pred)
          .mockImplementationOnce(() => bitmap)
          .mockImplementationOnce(() => ({ delete: vi.fn() }))
          .mockImplementationOnce(() => ({ delete: vi.fn() })),
        MatVector: class MatVector {
          size() {
            return 1;
          }
          get() {
            return contour;
          }
          delete() {}
        },
        findContours: vi.fn(),
        getPerspectiveTransform: vi.fn(() => ({ delete: vi.fn() })),
        warpPerspective: vi.fn(),
        rotate: vi.fn()
      };
    };
    const cv = makeCv();

    const { createDetModel } = await loadDetModule();
    getMiniBoxFromPoints
      .mockReturnValueOnce({
        side: 4,
        box: [
          [0, 0],
          [4, 0],
          [4, 2],
          [0, 2]
        ]
      })
      .mockReturnValueOnce({
        side: 6,
        box: [
          [0, 0],
          [5, 0],
          [5, 3],
          [0, 3]
        ]
      });
    boxScoreFast.mockReturnValue(0.9);
    unclip.mockReturnValue([
      [0, 0],
      [5, 0],
      [5, 3],
      [0, 3]
    ]);

    const model = await createDetModel({
      ort,
      modelBytes: new Uint8Array([1]),
      configText: "det-crop",
      backend: "auto",
      webgpuState: { available: false, reason: "" }
    });
    const [detResult] = await model.predict(
      cv,
      [
        {
          cols: 64,
          rows: 32,
          channels: () => 3
        }
      ],
      {
        thresh: 0.3,
        boxThresh: 0.5,
        unclipRatio: 1.5,
        limitSideLen: 64,
        limitType: "max",
        maxSideLimit: 96
      }
    );

    expect(sessionRun).toHaveBeenCalledTimes(1);
    expect(tensorCalls[0]).toEqual({ type: "float32", dims: [1, 3, 32, 64], size: 6144 });
    expect(detResult.boxes).toEqual([
      {
        poly: [
          [0, 0],
          [40, 0],
          [40, 24],
          [0, 24]
        ],
        score: 0.9
      }
    ]);

    const cropWarped = {
      rows: 20,
      cols: 10,
      delete: vi.fn()
    };
    const cropRotated = {
      rows: 10,
      cols: 20,
      delete: vi.fn()
    };
    let cropMatCount = 0;
    const cropCv = {
      Size: cv.Size,
      Scalar: cv.Scalar,
      INTER_CUBIC: cv.INTER_CUBIC,
      BORDER_REPLICATE: cv.BORDER_REPLICATE,
      ROTATE_90_COUNTERCLOCKWISE: cv.ROTATE_90_COUNTERCLOCKWISE,
      CV_32FC2: cv.CV_32FC2,
      Mat: class Mat {
        constructor() {
          cropMatCount += 1;
          return cropMatCount === 1 ? cropWarped : cropRotated;
        }
      },
      matFromArray: vi
        .fn()
        .mockImplementationOnce(() => ({ delete: vi.fn() }))
        .mockImplementationOnce(() => ({ delete: vi.fn() })),
      getPerspectiveTransform: vi.fn(() => ({ delete: vi.fn() })),
      warpPerspective: vi.fn(),
      rotate: vi.fn()
    };

    getMiniBoxFromPoints.mockReturnValue({
      box: [
        [0, 0],
        [10, 0],
        [10, 20],
        [0, 20]
      ]
    });
    const rotatedCrop = cropByPoly(cropCv, { id: "src" }, [[0, 0]]);
    expect(cropCv.rotate).toHaveBeenCalled();
    expect(rotatedCrop).toBe(cropRotated);
  });

  it("runs batched detection when batchSize > 1 (one session.run per chunk)", async () => {
    parseInferenceConfigText.mockReturnValue({
      PreProcess: { transform_ops: [] },
      PostProcess: { max_candidates: "10" }
    });
    getTransformOp.mockImplementation((_ops, id) => {
      if (id === "DetResizeForTest") return { resize_long: 64 };
      return null;
    });
    parseScaleValue.mockReturnValue(1 / 255);
    clamp.mockImplementation((value, min, max) => Math.max(min, Math.min(max, value)));
    getProviderCandidates.mockReturnValue([["wasm"]]);

    const tensorCalls = [];
    const ort = {
      Tensor: createMockOrtTensorClass(tensorCalls)
    };
    const sessionRun = vi.fn().mockResolvedValue({
      output: {
        dims: [2, 1, 4, 8],
        data: new Float32Array(64).fill(0.1)
      }
    });
    const session = {
      inputNames: ["input"],
      outputNames: ["output"],
      run: sessionRun
    };
    createSession.mockResolvedValue({
      session,
      provider: "wasm"
    });
    toBgrFloatCHWFromBgr.mockReturnValue(new Float32Array(3 * 32 * 64).fill(1));

    const pred = { delete: vi.fn() };
    const bitmap = { delete: vi.fn() };
    const warped = {
      rows: 20,
      cols: 10,
      delete: vi.fn()
    };
    const cv = {
      warped,
      Mat: class Mat {
        constructor() {
          return warped;
        }
      },
      Size: class Size {
        constructor(width, height) {
          this.width = width;
          this.height = height;
        }
      },
      Scalar: class Scalar {},
      INTER_LINEAR: "linear",
      INTER_CUBIC: "cubic",
      BORDER_REPLICATE: "replicate",
      COLOR_RGBA2BGR: "rgba",
      COLOR_GRAY2BGR: "gray",
      ROTATE_90_COUNTERCLOCKWISE: "ccw",
      CV_32FC1: "float1",
      CV_8UC1: "mask1",
      CV_32FC2: "float",
      RETR_LIST: "list",
      CHAIN_APPROX_SIMPLE: "chain",
      resize: vi.fn((src, dst, size) => {
        dst.data = new Uint8Array(size.width * size.height * 3);
        dst.channels = () => 3;
        dst.copyTo = vi.fn();
        dst.delete = vi.fn();
      }),
      cvtColor: vi.fn(),
      matFromArray: vi
        .fn()
        .mockImplementationOnce(() => pred)
        .mockImplementationOnce(() => bitmap)
        .mockImplementationOnce(() => pred)
        .mockImplementationOnce(() => bitmap),
      MatVector: class MatVector {
        size() {
          return 0;
        }
        delete() {}
      },
      findContours: vi.fn(),
      getPerspectiveTransform: vi.fn(() => ({ delete: vi.fn() })),
      warpPerspective: vi.fn(),
      rotate: vi.fn()
    };

    const { createDetModel } = await loadDetModule();
    const model = await createDetModel({
      ort,
      modelBytes: new Uint8Array([1]),
      configText: "det-batch",
      backend: "auto",
      webgpuState: { available: false, reason: "" },
      batchSize: 2
    });

    const mat = { cols: 64, rows: 32, channels: () => 3 };
    const results = await model.predict(cv, [mat, mat], {
      thresh: 0.3,
      boxThresh: 0.5,
      unclipRatio: 1.5,
      limitSideLen: 64,
      limitType: "max",
      maxSideLimit: 96
    });

    expect(sessionRun).toHaveBeenCalledTimes(1);
    const batchInput = tensorCalls.find((t) => t.dims[0] === 2);
    expect(batchInput).toEqual({ type: "float32", dims: [2, 3, 32, 64], size: 12288 });
    expect(results).toHaveLength(2);
    expect(results[0].srcW).toBe(64);
    expect(results[1].srcW).toBe(64);
  });

  it("creates, uses, and disposes detection models through runtime wrappers", async () => {
    parseInferenceConfigText.mockReturnValue({
      PreProcess: {
        transform_ops: []
      },
      PostProcess: {}
    });
    getTransformOp.mockReturnValue(undefined);
    parseScaleValue.mockReturnValue(1 / 255);
    getProviderCandidates.mockReturnValue([["wasm"]]);
    createSession.mockResolvedValue({
      session: {
        inputNames: ["input"],
        outputNames: ["output"],
        run: vi.fn().mockResolvedValue({
          output: {
            dims: [1, 1, 4, 8],
            data: new Float32Array(32).fill(0.9)
          }
        })
      },
      provider: "wasm"
    });
    releaseSessions.mockResolvedValue(undefined);
    toBgrFloatCHWFromBgr.mockReturnValue(new Float32Array(3 * 32 * 64).fill(1));
    getMiniBoxFromPoints
      .mockReturnValueOnce({
        side: 4,
        box: [
          [0, 0],
          [4, 0],
          [4, 2],
          [0, 2]
        ]
      })
      .mockReturnValueOnce({
        side: 6,
        box: [
          [0, 0],
          [5, 0],
          [5, 3],
          [0, 3]
        ]
      });
    boxScoreFast.mockReturnValue(0.9);
    unclip.mockReturnValue([
      [0, 0],
      [5, 0],
      [5, 3],
      [0, 3]
    ]);

    const { createDetModel, createDetModelSession } = await loadDetModule();
    const sessionState = await createDetModelSession({}, new Uint8Array([1]), "auto", {
      available: false,
      reason: ""
    });
    expect(getProviderCandidates).toHaveBeenCalledWith("auto", { available: false, reason: "" });
    expect(withTimeout).toHaveBeenCalled();
    expect(sessionState.provider).toBe("wasm");

    const model = await createDetModel({
      ort: {
        Tensor: createMockOrtTensorClass()
      },
      modelBytes: new Uint8Array([1]),
      configText: "config",
      backend: "auto",
      webgpuState: { available: false, reason: "" }
    });

    expect(assertModelResources).toHaveBeenCalled();
    expect(model.kind).toBe("det");
    expect(model.provider).toBe("wasm");
    await expect(
      model.predict(
        createDetModelIntegrationCv(),
        [
          {
            cols: 64,
            rows: 32,
            channels: () => 3
          }
        ],
        {
          thresh: 0.3,
          boxThresh: 0.5,
          unclipRatio: 1.5
        }
      )
    ).resolves.toMatchObject([
      {
        boxes: expect.any(Array),
        srcW: 64,
        srcH: 32
      }
    ]);
    await expect(model.dispose()).resolves.toBeUndefined();
    await expect(model.predict({}, [{}], {})).rejects.toThrow(/session is not initialized/i);
  });
});
