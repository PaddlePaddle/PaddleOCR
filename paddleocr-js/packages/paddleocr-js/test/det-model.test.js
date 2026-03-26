import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockOrtTensorClass } from "./helpers/mock-ort-tensor.js";

const assertStandardModelResources = vi.fn();
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

vi.mock("../src/resources/standard-model.js", () => ({
  assertStandardModelResources
}));

vi.mock("../src/runtime/ort.js", () => ({
  createSession,
  getProviderCandidates,
  releaseSessions
}));

vi.mock("../src/utils/common.js", () => ({
  clamp,
  withTimeout
}));

vi.mock("../src/models/common.js", () => ({
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
  return import("../src/models/det.js");
}

function createSourceMat(channels, cols = 100, rows = 50) {
  return {
    cols,
    rows,
    channels: () => channels
  };
}

/** OpenCV-like facade for `preprocessDet` branch tests (resize/cvtColor only). */
function createPreprocessCvFixture() {
  return {
    Mat: class Mat {
      constructor() {
        this._channels = 3;
        this.data = new Uint8Array(1);
      }
      channels() {
        return this._channels;
      }
      copyTo(target) {
        target._channels = 3;
        target.data = this.data;
      }
      delete() {}
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
      dst._channels = src.channels();
      dst.data = new Uint8Array(size.width * size.height * 3);
    }),
    cvtColor: vi.fn((src, dst) => {
      dst._channels = 3;
      dst.data = src.data;
    })
  };
}

/** CV facade for `createDetModel().detect()` integration-style test (preprocess + dbPostprocess). */
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

    const {
      DEFAULT_DET_MODEL_PARSE_FALLBACKS,
      DEFAULT_DET_RUNTIME_LIMITS,
      parseDetModelConfigText
    } = await loadDetModule();
    expect(parseDetModelConfigText("config")).toEqual({
      resizeLong: 736,
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
      },
      maxSideLimit: DEFAULT_DET_RUNTIME_LIMITS.maxSideLimit
    });

    parseInferenceConfigText.mockReturnValue({});
    getTransformOp.mockReturnValue(undefined);
    parseScaleValue.mockReturnValue(1 / 255);

    expect(parseDetModelConfigText("fallback")).toEqual({
      resizeLong: DEFAULT_DET_MODEL_PARSE_FALLBACKS.resizeLong,
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
      },
      maxSideLimit: DEFAULT_DET_RUNTIME_LIMITS.maxSideLimit
    });
  });

  it("preprocesses detection inputs across max/min and color conversion branches", async () => {
    clamp.mockImplementation((value, min, max) => Math.max(min, Math.min(max, value)));
    toBgrFloatCHWFromBgr.mockImplementation((data, width, height) => {
      const out = new Float32Array(3 * width * height);
      out.fill(1);
      return out;
    });

    const { preprocessDet } = await loadDetModule();

    const ort = {
      Tensor: createMockOrtTensorClass()
    };

    const cvMax = createPreprocessCvFixture();
    const maxResult = preprocessDet(
      {
        cv: cvMax,
        ort,
        config: {
          resizeLong: 64,
          maxSideLimit: 128,
          normalize: {}
        }
      },
      createSourceMat(3, 100, 50),
      {}
    );
    expect(maxResult.dstW).toBe(64);
    expect(maxResult.dstH).toBe(32);
    expect(cvMax.cvtColor).not.toHaveBeenCalled();

    const cvMin = createPreprocessCvFixture();
    const minResult = preprocessDet(
      {
        cv: cvMin,
        ort,
        config: {
          resizeLong: 64,
          maxSideLimit: 96,
          normalize: {}
        }
      },
      createSourceMat(4, 16, 8),
      {
        text_det_limit_type: "min",
        text_det_limit_side_len: 64
      }
    );
    expect(minResult.dstW).toBe(96);
    expect(minResult.dstH).toBe(64);
    expect(cvMin.cvtColor).toHaveBeenCalled();

    const cvGray = createPreprocessCvFixture();
    preprocessDet(
      {
        cv: cvGray,
        ort,
        config: {
          resizeLong: 64,
          maxSideLimit: 96,
          normalize: {}
        }
      },
      createSourceMat(1, 16, 16),
      {}
    );
    expect(cvGray.cvtColor).toHaveBeenCalled();
  });

  it("postprocesses detection outputs and filters contour candidates", async () => {
    clamp.mockImplementation((value, min, max) => Math.max(min, Math.min(max, value)));

    const makeContour = (rows, values) => ({
      rows,
      data32S: values,
      delete: vi.fn()
    });
    const contour0 = makeContour(3, []);
    const contour1 = makeContour(4, [0, 0, 5, 0, 5, 5, 0, 5]);
    const contour2 = makeContour(4, [1, 1, 6, 1, 6, 6, 1, 6]);
    const contour3 = makeContour(4, [2, 2, 7, 2, 7, 7, 2, 7]);
    const contour4 = makeContour(4, [3, 3, 8, 3, 8, 8, 3, 8]);
    const contour5 = makeContour(4, [4, 4, 9, 4, 9, 9, 4, 9]);
    const contours = [contour0, contour1, contour2, contour3, contour4, contour5];

    getMiniBoxFromPoints
      .mockReturnValueOnce({ side: 2, box: [[0, 0], [1, 0], [1, 1], [0, 1]] })
      .mockReturnValueOnce({ side: 4, box: [[0, 0], [5, 0], [5, 5], [0, 5]] })
      .mockReturnValueOnce({ side: 4, box: [[0, 0], [5, 0], [5, 5], [0, 5]] })
      .mockReturnValueOnce({ side: 4, box: [[0, 0], [5, 0], [5, 5], [0, 5]] })
      .mockReturnValueOnce({ side: 6, box: [[4, 4], [10, 4], [10, 10], [4, 10]] })
      .mockReturnValueOnce({ side: 4, box: [[20, 0], [26, 0], [26, 6], [20, 6]] })
      .mockReturnValueOnce({ side: 6, box: [[8, 2], [14, 2], [14, 8], [8, 8]] });
    boxScoreFast
      .mockReturnValueOnce(0.1)
      .mockReturnValueOnce(0.9)
      .mockReturnValueOnce(0.9)
      .mockReturnValueOnce(0.95);
    unclip
      .mockReturnValueOnce(null)
      .mockReturnValueOnce([[4, 4], [10, 4], [10, 10], [4, 10]])
      .mockReturnValueOnce([[8, 2], [14, 2], [14, 8], [8, 8]]);

    const pred = { delete: vi.fn() };
    const bitmap = { delete: vi.fn() };
    const cv = {
      CV_32FC1: "float",
      CV_8UC1: "mask",
      RETR_LIST: "list",
      CHAIN_APPROX_SIMPLE: "chain",
      Mat: class Mat {
        delete() {}
      },
      MatVector: class MatVector {
        size() {
          return contours.length;
        }
        get(index) {
          return contours[index];
        }
        delete() {}
      },
      matFromArray: vi
        .fn()
        .mockImplementationOnce(() => pred)
        .mockImplementationOnce(() => bitmap),
      findContours: vi.fn()
    };

    const { dbPostprocess } = await loadDetModule();
    const boxes = dbPostprocess(
      {
        cv,
        config: {
          postprocess: {
            maxCandidates: 10
          }
        }
      },
      {
        dims: [1, 1, 10, 20],
        data: new Float32Array(200).fill(0.9)
      },
      {
        srcW: 200,
        srcH: 100
      },
      0.3,
      0.5,
      1.5
    );

    expect(boxes).toHaveLength(2);
    expect(boxes.map((box) => box.score)).toEqual([0.95, 0.9]);
    expect(contour0.delete).toHaveBeenCalled();
    expect(contour5.delete).toHaveBeenCalled();
    expect(pred.delete).toHaveBeenCalled();
    expect(bitmap.delete).toHaveBeenCalled();
  });

  it("runs detection models and crops rotated boxes", async () => {
    const tensorCalls = [];
    const ort = {
      Tensor: createMockOrtTensorClass(tensorCalls)
    };
    const session = {
      inputNames: ["input"],
      outputNames: ["output"],
      run: vi.fn().mockResolvedValue({
        output: {
          dims: [1, 1, 4, 8],
          data: new Float32Array(32).fill(0.9)
        }
      })
    };
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

    const { runDetModel, cropByPoly } = await loadDetModule();
    getMiniBoxFromPoints
      .mockReturnValueOnce({ side: 4, box: [[0, 0], [4, 0], [4, 2], [0, 2]] })
      .mockReturnValueOnce({ side: 6, box: [[0, 0], [5, 0], [5, 3], [0, 3]] });
    boxScoreFast.mockReturnValue(0.9);
    unclip.mockReturnValue([[0, 0], [5, 0], [5, 3], [0, 3]]);

    const detResult = await runDetModel(
      {
        cv,
        ort,
        config: {
          resizeLong: 64,
          maxSideLimit: 96,
          normalize: {},
          postprocess: {
            maxCandidates: 10
          }
        },
        session
      },
      {
        cols: 64,
        rows: 32,
        channels: () => 3
      },
      {
        text_det_thresh: 0.3,
        text_det_box_thresh: 0.5,
        text_det_unclip_ratio: 1.5
      }
    );

    expect(session.run).toHaveBeenCalledTimes(1);
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
      .mockReturnValueOnce({ side: 4, box: [[0, 0], [4, 0], [4, 2], [0, 2]] })
      .mockReturnValueOnce({ side: 6, box: [[0, 0], [5, 0], [5, 3], [0, 3]] });
    boxScoreFast.mockReturnValue(0.9);
    unclip.mockReturnValue([[0, 0], [5, 0], [5, 3], [0, 3]]);

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

    expect(assertStandardModelResources).toHaveBeenCalled();
    expect(model.kind).toBe("det");
    expect(model.provider).toBe("wasm");
    await expect(
      model.detect({
        cv: createDetModelIntegrationCv(),
        sourceMat: {
          cols: 64,
          rows: 32,
          channels: () => 3
        },
        params: {
          text_det_thresh: 0.3,
          text_det_box_thresh: 0.5,
          text_det_unclip_ratio: 1.5
        }
      })
    ).resolves.toMatchObject({
      output: expect.any(Object)
    });
    await expect(model.dispose()).resolves.toBeUndefined();
    await expect(
      model.detect({
        cv: {},
        sourceMat: {},
        params: {}
      })
    ).rejects.toThrow(/session is not initialized/i);
  });
});
