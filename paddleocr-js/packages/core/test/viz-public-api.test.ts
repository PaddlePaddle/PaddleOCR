import { describe, expect, it, vi } from "vitest";

// Mock browser APIs needed by the module
(globalThis as Record<string, unknown>).FontFace = vi.fn(() => ({
  load: vi.fn().mockResolvedValue(undefined)
}));
if (!document.fonts) {
  Object.defineProperty(document, "fonts", {
    value: { add: vi.fn(), delete: vi.fn() },
    configurable: true
  });
}

import { OcrVisualizer, renderOcrToBlob, deterministicColor } from "../src/viz/index";

describe("viz public API", () => {
  it("exports OcrVisualizer as a class", () => {
    expect(typeof OcrVisualizer).toBe("function");
  });

  it("exports renderOcrToBlob as a function", () => {
    expect(typeof renderOcrToBlob).toBe("function");
  });

  it("exports deterministicColor as a function", () => {
    expect(typeof deterministicColor).toBe("function");
  });
});
