import { describe, expect, it, vi } from "vitest";
import { drawBoxesPanel } from "../src/viz/ocr/draw-boxes";
import type { OcrResultItem } from "../src/pipelines/ocr/core";
import type { Point2D } from "../src/models/common";

function createMockCtx() {
  const calls: string[] = [];
  return {
    calls,
    save: vi.fn(() => calls.push("save")),
    restore: vi.fn(() => calls.push("restore")),
    drawImage: vi.fn(() => calls.push("drawImage")),
    beginPath: vi.fn(() => calls.push("beginPath")),
    moveTo: vi.fn(() => calls.push("moveTo")),
    lineTo: vi.fn(() => calls.push("lineTo")),
    closePath: vi.fn(() => calls.push("closePath")),
    fill: vi.fn(() => calls.push("fill")),
    stroke: vi.fn(() => calls.push("stroke")),
    lineWidth: 0,
    strokeStyle: "",
    fillStyle: ""
  } as unknown as CanvasRenderingContext2D & { calls: string[] };
}

function makeItem(poly: Point2D[], text: string): OcrResultItem {
  return { originalIndex: 0, poly, text, score: 0.95 };
}

describe("viz/draw-boxes", () => {
  it("draws the source image at (0,0)", () => {
    const ctx = createMockCtx();
    const image = { width: 100, height: 50 } as ImageBitmap;

    drawBoxesPanel(ctx, image, [], {});

    expect(ctx.drawImage).toHaveBeenCalledWith(image, 0, 0);
  });

  it("draws a polygon fill and stroke for each item", () => {
    const ctx = createMockCtx();
    const image = { width: 100, height: 50 } as ImageBitmap;
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

    drawBoxesPanel(ctx, image, items, {});

    expect(ctx.beginPath).toHaveBeenCalled();
    expect(ctx.moveTo).toHaveBeenCalledWith(10, 10);
    expect(ctx.lineTo).toHaveBeenCalledWith(90, 10);
    expect(ctx.lineTo).toHaveBeenCalledWith(90, 40);
    expect(ctx.lineTo).toHaveBeenCalledWith(10, 40);
    expect(ctx.closePath).toHaveBeenCalled();
    expect(ctx.fill).toHaveBeenCalled();
  });

  it("wraps each item draw in save/restore", () => {
    const ctx = createMockCtx();
    const image = { width: 100, height: 50 } as ImageBitmap;
    const items: OcrResultItem[] = [
      makeItem(
        [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10]
        ],
        "a"
      )
    ];

    drawBoxesPanel(ctx, image, items, {});

    const saveIdx = ctx.calls.indexOf("save");
    const restoreIdx = ctx.calls.lastIndexOf("restore");
    expect(saveIdx).toBeGreaterThan(-1);
    expect(restoreIdx).toBeGreaterThan(saveIdx);
  });

  it("uses custom colorFn when provided", () => {
    const ctx = createMockCtx();
    const image = { width: 100, height: 50 } as ImageBitmap;
    const items: OcrResultItem[] = [
      makeItem(
        [
          [0, 0],
          [10, 0],
          [10, 10],
          [0, 10]
        ],
        "a"
      )
    ];
    const colorFn = vi.fn().mockReturnValue([255, 0, 0]);

    drawBoxesPanel(ctx, image, items, { colorFn });

    expect(colorFn).toHaveBeenCalledWith(0);
  });
});
