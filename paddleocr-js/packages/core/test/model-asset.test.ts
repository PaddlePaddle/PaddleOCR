import { describe, expect, it } from "vitest";

import {
  normalizeModelAsset,
  normalizeAssets,
  getModelEntryPath,
  assertModelResourceSlot,
  assertModelResources
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

describe("model resource validation", () => {
  it("provides standard entry names", () => {
    expect(getModelEntryPath("model")).toBe("inference.onnx");
    expect(getModelEntryPath("config")).toBe("inference.yml");
    expect(getModelEntryPath("other")).toBe(null);
  });

  it("rejects missing model binary resources", () => {
    expect(() => assertModelResourceSlot("Detection", "model", new Uint8Array())).toThrow(
      /inference\.onnx/i
    );
  });

  it("rejects missing model config resources", () => {
    expect(() => assertModelResourceSlot("Recognition", "config", "")).toThrow(/inference\.yml/i);
  });

  it("supports validating multiple model resources together", () => {
    expect(() =>
      assertModelResources("Detection", {
        model: new Uint8Array([1]),
        config: "Global:\n  model_name: det"
      })
    ).not.toThrow();
  });

  it("rejects unsupported model resource slots", () => {
    expect(() => assertModelResourceSlot("Detection", "labels", "abc")).toThrow(
      /Unsupported model resource slot/i
    );
    expect(() =>
      assertModelResources("Detection", {
        labels: "abc"
      })
    ).toThrow(/Unsupported model resource slot/i);
  });
});
