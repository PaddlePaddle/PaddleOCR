import { describe, expect, it, vi } from "vitest";

// Mock FontFace and document.fonts before importing
const mockFontFace = {
  load: vi.fn().mockResolvedValue(undefined),
  family: ""
};
(globalThis as Record<string, unknown>).FontFace = vi.fn((family: string) => {
  mockFontFace.family = family;
  return mockFontFace;
});
if (!document.fonts) {
  Object.defineProperty(document, "fonts", {
    value: { add: vi.fn(), delete: vi.fn() },
    configurable: true
  });
} else {
  vi.spyOn(document.fonts, "add").mockImplementation(() => {});
  vi.spyOn(document.fonts, "delete").mockImplementation(() => true);
}

// Mock createImageBitmap
(globalThis as Record<string, unknown>).createImageBitmap = vi
  .fn()
  .mockResolvedValue({ width: 200, height: 100, close: vi.fn() });

import { OcrVisualizer, renderOcrToBlob } from "../src/viz/ocr/renderer";

describe("OcrVisualizer", () => {
  it("can be constructed with no options", () => {
    const viz = new OcrVisualizer();
    expect(viz).toBeDefined();
    viz.dispose();
  });

  it("can be constructed with font config", () => {
    const viz = new OcrVisualizer({
      font: { family: "TestFont", source: "https://example.com/f.woff2" }
    });
    expect(viz).toBeDefined();
    viz.dispose();
  });

  it("dispose() is safe to call multiple times", () => {
    const viz = new OcrVisualizer();
    viz.dispose();
    viz.dispose(); // should not throw
  });
});

describe("renderOcrToBlob", () => {
  it("is exported as a function", () => {
    expect(typeof renderOcrToBlob).toBe("function");
  });
});
