import { describe, expect, it } from "vitest";

import {
  normalizeModelAsset,
  normalizeAssets,
  getStandardModelEntryPath,
  assertStandardModelResourceSlot,
  assertStandardModelResources
} from "../src/resources/model-asset";

describe("model asset normalization", () => {
  it("resolves built-in model asset references", () => {
    const assets = normalizeAssets({
      det: "PP-OCRv5_mobile_det",
      rec: "PP-OCRv5_mobile_rec"
    });
    expect(assets.det.url).toMatch(/\.tar$/);
  });

  it("normalizes a single model asset directly", () => {
    const asset = normalizeModelAsset("det", {
      url: "/det.tar"
    });

    expect(asset.url).toBe("/det.tar");
  });

  it("rejects invalid assets", () => {
    expect(() =>
      normalizeAssets({
        encoder: {}
      })
    ).toThrow(/must define url/i);
  });

  it("rejects non-object asset descriptors", () => {
    expect(() => normalizeModelAsset("det", null)).toThrow(/must be an object/i);
  });

  it("rejects unknown model asset references", () => {
    expect(() =>
      normalizeAssets({
        det: "missing_model"
      })
    ).toThrow(/unknown model asset/i);
  });
});

describe("standard model protocol", () => {
  it("provides standard entry names", () => {
    expect(getStandardModelEntryPath("model")).toBe("inference.onnx");
    expect(getStandardModelEntryPath("config")).toBe("inference.yml");
    expect(getStandardModelEntryPath("other")).toBe(null);
  });

  it("rejects missing standard model binary resources", () => {
    expect(() => assertStandardModelResourceSlot("Detection", "model", new Uint8Array())).toThrow(
      /inference\.onnx/i
    );
  });

  it("rejects missing standard model config resources", () => {
    expect(() => assertStandardModelResourceSlot("Recognition", "config", "")).toThrow(
      /inference\.yml/i
    );
  });

  it("supports validating multiple standard model resources together", () => {
    expect(() =>
      assertStandardModelResources("Detection", {
        model: new Uint8Array([1]),
        config: "Global:\n  model_name: det"
      })
    ).not.toThrow();
  });

  it("rejects unsupported standard model resource slots", () => {
    expect(() => assertStandardModelResourceSlot("Detection", "labels", "abc")).toThrow(
      /Unsupported standard model resource slot/i
    );
    expect(() =>
      assertStandardModelResources("Detection", {
        labels: "abc"
      })
    ).toThrow(/Unsupported standard model resource slot/i);
  });
});
