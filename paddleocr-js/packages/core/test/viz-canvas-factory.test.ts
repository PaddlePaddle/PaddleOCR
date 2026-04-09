import { describe, expect, it, vi, afterEach } from "vitest";
import { createCanvas } from "../src/viz/canvas-factory";

describe("viz/canvas-factory", () => {
  const origOffscreen = globalThis.OffscreenCanvas;

  afterEach(() => {
    if (origOffscreen) {
      globalThis.OffscreenCanvas = origOffscreen;
    } else {
      delete (globalThis as Record<string, unknown>).OffscreenCanvas;
    }
  });

  it("returns an OffscreenCanvas when available", () => {
    const mockCanvas = { width: 0, height: 0, getContext: vi.fn() };
    (globalThis as Record<string, unknown>).OffscreenCanvas = vi.fn((w: number, h: number) => {
      mockCanvas.width = w;
      mockCanvas.height = h;
      return mockCanvas;
    });

    const result = createCanvas(100, 200);
    expect(result.width).toBe(100);
    expect(result.height).toBe(200);
  });

  it("falls back to document.createElement when OffscreenCanvas is unavailable", () => {
    delete (globalThis as Record<string, unknown>).OffscreenCanvas;

    const mockCtx = {};
    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => mockCtx)
    };
    vi.spyOn(document, "createElement").mockReturnValue(mockCanvas as unknown as HTMLElement);

    const result = createCanvas(300, 400);
    expect(result.width).toBe(300);
    expect(result.height).toBe(400);
    expect(document.createElement).toHaveBeenCalledWith("canvas");
  });
});
