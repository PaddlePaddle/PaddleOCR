import { describe, expect, it } from "vitest";

import {
  boxScoreFast,
  extractInferenceModelName,
  getMiniBoxFromPoints,
  getTransformOp,
  parseInferenceConfigText,
  parseScaleValue,
  toBgrFloatCHWFromBgr,
  unclip
} from "../src/models/common";

describe("model common helpers", () => {
  it("parses inference config text into plain objects", () => {
    expect(parseInferenceConfigText("Global:\n  model_name: det")).toEqual({
      Global: {
        model_name: "det"
      }
    });
    expect(parseInferenceConfigText("- item")).toEqual({});
  });

  it("parses numeric and fractional scale values with fallback behavior", () => {
    expect(parseScaleValue(0.5, 1)).toBe(0.5);
    expect(parseScaleValue("1./255.", 1)).toBeCloseTo(1 / 255);
    expect(parseScaleValue("2/0", 3)).toBe(3);
    expect(parseScaleValue({}, 4)).toBe(4);
  });

  it("finds transform ops and nested model names", () => {
    expect(
      getTransformOp([{ NormalizeImage: { mean: [0.5] } }, { Other: {} }], "NormalizeImage")
    ).toEqual({ mean: [0.5] });
    expect(getTransformOp([{ Other: {} }], "NormalizeImage")).toBeNull();

    expect(extractInferenceModelName("Global:\n  model_name: det")).toBe("det");
    expect(
      extractInferenceModelName(
        "Outer:\n  nested:\n    - item: 1\n    - model_name: rec\narray:\n  - another: thing"
      )
    ).toBe("rec");
  });

  it("converts BGR image data into CHW float tensors", () => {
    const bgr = new Uint8Array([10, 20, 30, 40, 50, 60]);
    const result = toBgrFloatCHWFromBgr(bgr, 2, 1, {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      scale: 1
    });

    expect(Array.from(result)).toEqual([10, 40, 20, 50, 30, 60]);
  });

  it("returns null when unclipping a degenerate polygon and expands valid polygons", () => {
    expect(
      unclip(
        [
          [0, 0],
          [0, 0],
          [0, 0]
        ],
        2
      )
    ).toBeNull();

    const expanded = unclip(
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
      ],
      2
    );

    expect(expanded).not.toBeNull();
    expect(expanded.length).toBeGreaterThanOrEqual(4);
  });

  it("computes mini boxes from points with a cv facade", () => {
    const contour = {
      deleteCalled: false,
      delete() {
        this.deleteCalled = true;
      }
    };
    const cv = {
      CV_32FC2: "float",
      matFromArray: () => contour,
      minAreaRect: () => ({ id: "rect" }),
      RotatedRect: {
        points: () => [
          { x: 0, y: 0 },
          { x: 10, y: 0 },
          { x: 10, y: 5 },
          { x: 0, y: 5 }
        ]
      }
    };

    const result = getMiniBoxFromPoints(cv, [
      [0, 0],
      [10, 0],
      [10, 5],
      [0, 5]
    ]);

    expect(result.box).toEqual([
      [0, 0],
      [10, 0],
      [10, 5],
      [0, 5]
    ]);
    expect(result.side).toBe(5);
    expect(contour.deleteCalled).toBe(true);
  });

  it("computes masked box scores with a cv facade", () => {
    const roi = {
      delete: () => {}
    };
    const mask = {
      delete: () => {}
    };
    const pts = {
      delete: () => {}
    };
    const predMat = {
      rows: 20,
      cols: 30,
      roi: () => roi
    };
    const cv = {
      Rect: class Rect {
        constructor(x, y, width, height) {
          this.x = x;
          this.y = y;
          this.width = width;
          this.height = height;
        }
      },
      Mat: {
        zeros: function zeros() {
          return mask;
        }
      },
      CV_8UC1: "mask",
      CV_32SC2: "int",
      MatVector: class MatVector {
        push_back() {}
        delete() {}
      },
      Scalar: class Scalar {
        constructor(value) {
          this.value = value;
        }
      },
      matFromArray: () => pts,
      fillPoly: () => {},
      mean: () => [0.75]
    };

    expect(
      boxScoreFast(cv, predMat, [
        [1.2, 2.1],
        [8.5, 2.4],
        [8.1, 9.2],
        [1.4, 9.8]
      ])
    ).toBe(0.75);
  });
});
