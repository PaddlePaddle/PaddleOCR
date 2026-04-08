import { describe, expect, it, vi } from "vitest";
import { drawTextPanel } from "../src/viz/ocr/draw-text";
import type { OcrResultItem } from "../src/pipelines/ocr/core";
import type { Point2D } from "../src/models/common";

function createMockCtx() {
  return {
    save: vi.fn(),
    restore: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    measureText: vi.fn().mockReturnValue({ width: 50 }),
    fillText: vi.fn(),
    translate: vi.fn(),
    rotate: vi.fn(),
    lineWidth: 0,
    strokeStyle: "",
    fillStyle: "",
    font: "",
    textBaseline: "" as CanvasTextBaseline
  } as unknown as CanvasRenderingContext2D;
}

function makeItem(poly: Point2D[], text: string): OcrResultItem {
  return { originalIndex: 0, poly, text, score: 0.95 };
}

describe("viz/draw-text", () => {
  it("fills the panel background", () => {
    const ctx = createMockCtx();
    drawTextPanel(ctx, 200, 100, [], {}, "sans-serif");

    expect(ctx.fillRect).toHaveBeenCalledWith(200, 0, 200, 100);
  });

  it("draws text for each item", () => {
    const ctx = createMockCtx();
    const items: OcrResultItem[] = [
      makeItem(
        [
          [10, 10],
          [90, 10],
          [90, 40],
          [10, 40]
        ],
        "hello"
      )
    ];

    drawTextPanel(ctx, 200, 100, items, {}, "sans-serif");

    expect(ctx.fillText).toHaveBeenCalled();
    // Verify the text content is "hello"
    const textCall = (ctx.fillText as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(textCall[0]).toBe("hello");
  });

  it("draws box outlines on the right panel for each item", () => {
    const ctx = createMockCtx();
    const items: OcrResultItem[] = [
      makeItem(
        [
          [10, 10],
          [90, 10],
          [90, 40],
          [10, 40]
        ],
        "hello"
      )
    ];

    drawTextPanel(ctx, 200, 100, items, {}, "sans-serif");

    expect(ctx.beginPath).toHaveBeenCalled();
    expect(ctx.stroke).toHaveBeenCalled();
  });

  it("uses custom textPanelBackground", () => {
    const ctx = createMockCtx();
    drawTextPanel(ctx, 200, 100, [], {}, "sans-serif", "#f0f0f0");

    // fillRect call for the background should happen
    expect(ctx.fillRect).toHaveBeenCalledWith(200, 0, 200, 100);
  });
});
