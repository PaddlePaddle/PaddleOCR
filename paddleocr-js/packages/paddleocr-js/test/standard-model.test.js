import { describe, expect, it, vi } from "vitest";

vi.mock("@techstark/opencv-js", () => ({
  default: {
    Mat() {}
  }
}));

import {
  assertStandardModelResources,
  assertStandardModelResourceSlot,
  getStandardModelEntryPath,
  loadStandardModelAsset
} from "../src/resources/standard-model.js";
import { assertStandardModelResources as assertStandardModelResourcesFromIndex } from "../src/resources/index.js";

describe("standard model protocol", () => {
  it("provides standard entry names", () => {
    expect(getStandardModelEntryPath("model")).toBe("inference.onnx");
    expect(getStandardModelEntryPath("config")).toBe("inference.yml");
    expect(getStandardModelEntryPath("other")).toBe(null);
  });

  it("rejects non-tar standard model assets", async () => {
    await expect(
      loadStandardModelAsset({
        id: "det",
        kind: "file",
        url: "/det.onnx"
      })
    ).rejects.toThrow(/tar bundle/i);
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

  it("re-exports standard model resource validation helpers", () => {
    expect(assertStandardModelResourcesFromIndex).toBe(assertStandardModelResources);
  });
});
