import { describe, expect, it } from "vitest";

import {
  DEFAULT_MODEL_ASSETS,
  normalizeAssetDescriptor,
  normalizeAssets
} from "../src/resources/registry.js";

describe("asset normalization", () => {
  it("resolves built-in model asset references", () => {
    const assets = normalizeAssets({
      det: "PP-OCRv5_mobile_det",
      rec: "PP-OCRv5_mobile_rec"
    });
    expect(assets.det.kind).toBe("tar");
    expect(assets.det.url).toMatch(/\.tar$/);
    expect(assets.det.entries.model).toBe("inference.onnx");
  });

  it("normalizes custom assets with direct asset descriptors", () => {
    const assets = normalizeAssets({
      encoder: { id: "encoder", url: "/encoder.onnx", version: "1" },
      vocab: { id: "vocab", url: "/vocab.txt" }
    });

    expect(assets.encoder.version).toBe("1");
    expect(assets.encoder.kind).toBe("file");
    expect(assets.vocab.url).toBe("/vocab.txt");
  });

  it("normalizes a single asset descriptor directly", () => {
    const asset = normalizeAssetDescriptor("det", {
      id: "det",
      url: "/det.tar",
      kind: "tar",
      entries: {
        model: "inference.onnx",
        config: "inference.yml"
      }
    });

    expect(asset.id).toBe("det");
    expect(asset.kind).toBe("tar");
    expect(asset.entries.config).toBe("inference.yml");
  });

  it("allows overriding the model asset table", () => {
    const assets = normalizeAssets(
      {
        det: "custom_det"
      },
      {
        ...DEFAULT_MODEL_ASSETS,
        custom_det: { id: "custom-det", url: "/custom-det.onnx" }
      }
    );

    expect(assets.det.id).toBe("custom-det");
    expect(assets.det.kind).toBe("file");
  });

  it("rejects invalid assets", () => {
    expect(() =>
      normalizeAssets({
        encoder: { id: "encoder" }
      })
    ).toThrow(/must define both id and url/i);
  });

  it("rejects non-object asset descriptors and unsupported kinds", () => {
    expect(() => normalizeAssetDescriptor("det", null)).toThrow(/must be an object/i);
    expect(() =>
      normalizeAssetDescriptor("det", {
        id: "det",
        url: "/det.bin",
        kind: "zip"
      })
    ).toThrow(/unsupported kind/i);
  });

  it("rejects malformed tar entry maps", () => {
    expect(() =>
      normalizeAssetDescriptor("det", {
        id: "det",
        url: "/det.tar",
        kind: "tar",
        entries: []
      })
    ).toThrow(/must define entries as an object/i);

    expect(() =>
      normalizeAssetDescriptor("det", {
        id: "det",
        url: "/det.tar",
        kind: "tar",
        entries: {
          model: ""
        }
      })
    ).toThrow(/must map entries to file paths/i);
  });

  it("rejects unknown model asset references", () => {
    expect(() =>
      normalizeAssets({
        det: "missing_model"
      })
    ).toThrow(/unknown model asset/i);
  });

  it("rejects empty asset version strings", () => {
    expect(() =>
      normalizeAssets({
        encoder: { id: "encoder", url: "/encoder.onnx", version: "" }
      })
    ).toThrow(/non-empty version string/i);
  });

  it("ignores extra top-level metadata-like fields by using only asset entries", () => {
    const assets = normalizeAssets({
      encoder: { id: "encoder", url: "/encoder.onnx" }
    });

    expect(assets.encoder.id).toBe("encoder");
  });
});
