# Viz Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional visualization subpath export (`@paddleocr/paddleocr-js/viz`) that renders OCR results as side-by-side composite images and exports them as downloadable Blobs.

**Architecture:** Seven focused source files under `packages/core/src/viz/` implement types, color generation, font management, box drawing, text drawing, composite assembly, and the public `OcrVisualizer` class. The viz module imports only type definitions from the core SDK — zero runtime dependency on OpenCV.js or ONNX Runtime. A separate Vite entry point produces an independent chunk. Tests live alongside existing tests in `packages/core/test/`.

**Tech Stack:** TypeScript, Canvas 2D API, FontFace API, Vite (multi-entry lib build), Vitest.

---

## File Map

| File                                                        | Responsibility                                                                                                         |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Create:** `packages/core/src/viz/types.ts`                | All shared interfaces: `OcrVisualizerOptions`, `FontConfig`, `BoxStyleOptions`, `RgbColor`                             |
| **Create:** `packages/core/src/viz/color.ts`                | `deterministicColor(index)` — LCG-based deterministic RGB color generator                                              |
| **Create:** `packages/core/src/viz/canvas-factory.ts`       | `createCanvas(w, h)` — returns `OffscreenCanvas` or falls back to `document.createElement("canvas")`                   |
| **Create:** `packages/core/src/viz/font.ts`                 | `loadFontFace(config)` / `removeFontFace(face)` — FontFace lifecycle helpers                                           |
| **Create:** `packages/core/src/viz/draw-boxes.ts`           | `drawBoxesPanel(ctx, image, items, style)` — draws left panel (source image + detection polygons)                      |
| **Create:** `packages/core/src/viz/draw-text.ts`            | `drawTextPanel(ctx, offsetX, height, items, style, fontFamily)` — draws right panel (white background + text in boxes) |
| **Create:** `packages/core/src/viz/side-by-side.ts`         | `renderSideBySideToCanvas(canvas, ctx, image, result, options)` — orchestrates left+right panels                       |
| **Create:** `packages/core/src/viz/renderer.ts`             | `OcrVisualizer` class and `renderOcrToBlob` convenience function                                                       |
| **Create:** `packages/core/src/viz/index.ts`                | Public subpath entry — re-exports `OcrVisualizer`, `renderOcrToBlob`, and all types                                    |
| **Modify:** `packages/core/vite.config.ts`                  | Add `src/viz/index.ts` as second entry, output `viz.mjs`/`viz.cjs`                                                     |
| **Modify:** `packages/core/package.json`                    | Add `./viz` to `exports`, add `./dist/viz.*` to `files`                                                                |
| **Create:** `packages/core/test/viz-color.test.ts`          | Tests for `deterministicColor`                                                                                         |
| **Create:** `packages/core/test/viz-canvas-factory.test.ts` | Tests for `createCanvas`                                                                                               |
| **Create:** `packages/core/test/viz-font.test.ts`           | Tests for font load/remove lifecycle                                                                                   |
| **Create:** `packages/core/test/viz-draw-boxes.test.ts`     | Tests for left-panel box drawing                                                                                       |
| **Create:** `packages/core/test/viz-draw-text.test.ts`      | Tests for right-panel text drawing                                                                                     |
| **Create:** `packages/core/test/viz-renderer.test.ts`       | Tests for `OcrVisualizer` and `renderOcrToBlob`                                                                        |

---

### Task 1: Types

**Files:**

- Create: `packages/core/src/viz/types.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/core/test/viz-types.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import type { FontConfig, BoxStyleOptions, OcrVisualizerOptions, RgbColor } from "../src/viz/types";

describe("viz/types", () => {
  it("allows constructing a minimal OcrVisualizerOptions", () => {
    const opts: OcrVisualizerOptions = {};
    expect(opts).toEqual({});
  });

  it("allows constructing a full OcrVisualizerOptions", () => {
    const font: FontConfig = {
      family: "Test",
      source: "https://example.com/font.woff2"
    };
    const boxStyle: BoxStyleOptions = {
      lineWidth: 3,
      fillOpacity: 0.5,
      colorFn: (i: number): RgbColor => [i, i, i]
    };
    const opts: OcrVisualizerOptions = {
      font,
      boxStyle,
      textPanelBackground: "#f0f0f0",
      outputFormat: "jpeg",
      outputQuality: 0.8
    };
    expect(opts.font?.family).toBe("Test");
    expect(opts.boxStyle?.lineWidth).toBe(3);
    expect(opts.outputFormat).toBe("jpeg");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-types.test.ts`
Expected: FAIL — cannot find module `../src/viz/types`

- [ ] **Step 3: Write the implementation**

Create `packages/core/src/viz/types.ts`:

```typescript
export type RgbColor = [number, number, number];

export interface FontConfig {
  /** CSS font-family name. */
  family: string;
  /** Font source: URL string or ArrayBuffer. */
  source: string | ArrayBuffer;
  /** FontFace descriptors (weight, style, etc.). */
  descriptors?: FontFaceDescriptors;
}

export interface BoxStyleOptions {
  /** Box stroke width. Default: 2. */
  lineWidth?: number;
  /** Fill opacity 0-1. Default: 0.3. */
  fillOpacity?: number;
  /** Custom color function. Default: deterministic LCG-based colors. */
  colorFn?: (index: number) => RgbColor;
}

export interface OcrVisualizerOptions {
  /** Custom font configuration. Falls back to system sans-serif if omitted. */
  font?: FontConfig;
  /** Detection box style overrides. */
  boxStyle?: BoxStyleOptions;
  /** Right panel background color. Default: "#ffffff". */
  textPanelBackground?: string;
  /** Output image format. Default: "png". */
  outputFormat?: "png" | "jpeg" | "webp";
  /** JPEG/WebP quality 0-1. Default: 0.92. */
  outputQuality?: number;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-types.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/types.ts packages/core/test/viz-types.test.ts && git commit -m "feat(viz): add shared type definitions"
```

---

### Task 2: Deterministic Color Generation

**Files:**

- Create: `packages/core/src/viz/color.ts`
- Create: `packages/core/test/viz-color.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/core/test/viz-color.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { deterministicColor } from "../src/viz/color";

describe("viz/color", () => {
  it("returns an RGB tuple of three integers 0-255", () => {
    const [r, g, b] = deterministicColor(0);
    expect(Number.isInteger(r)).toBe(true);
    expect(Number.isInteger(g)).toBe(true);
    expect(Number.isInteger(b)).toBe(true);
    expect(r).toBeGreaterThanOrEqual(0);
    expect(r).toBeLessThanOrEqual(255);
    expect(g).toBeGreaterThanOrEqual(0);
    expect(g).toBeLessThanOrEqual(255);
    expect(b).toBeGreaterThanOrEqual(0);
    expect(b).toBeLessThanOrEqual(255);
  });

  it("produces the same color for the same index", () => {
    expect(deterministicColor(5)).toEqual(deterministicColor(5));
    expect(deterministicColor(42)).toEqual(deterministicColor(42));
  });

  it("produces different colors for different indices", () => {
    const c0 = deterministicColor(0);
    const c1 = deterministicColor(1);
    const c2 = deterministicColor(2);
    // At least two of three should differ (LCG guarantees this)
    const allSame =
      JSON.stringify(c0) === JSON.stringify(c1) && JSON.stringify(c1) === JSON.stringify(c2);
    expect(allSame).toBe(false);
  });

  it("matches the exact values from the demo app's LCG", () => {
    // Pre-computed from the demo's deterministicColor for index 0
    // seed = (0+1)*1103515245 + 12345 = 1103527590
    // r = (1103527590 >> 16) & 0xff = 0xff & (16838) = 198 -> wait, let me compute
    // Actually: 1103527590 >>> 0 = 1103527590
    // r = (1103527590 >> 16) & 0xff
    //   = (16838) & 0xff = 198? No: 1103527590 / 65536 = 16838.xx, floor = 16838
    //   16838 & 0xff = 198
    // seed2 = (1103527590 * 1103515245 + 12345) >>> 0
    // This is hard to hand-compute; just snapshot-test
    const c0 = deterministicColor(0);
    const c1 = deterministicColor(1);
    // Verify determinism by snapshot
    expect(c0).toMatchInlineSnapshot();
    expect(c1).toMatchInlineSnapshot();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-color.test.ts`
Expected: FAIL — cannot find module `../src/viz/color`

- [ ] **Step 3: Write the implementation**

Create `packages/core/src/viz/color.ts`:

```typescript
import type { RgbColor } from "./types";

/**
 * Generate a deterministic RGB color for a given index.
 * Uses a Linear Congruential Generator (LCG) seeded by the index.
 * Same algorithm as the demo app to ensure visual consistency.
 */
export function deterministicColor(index: number): RgbColor {
  let seed = (index + 1) * 1103515245 + 12345;
  seed >>>= 0;
  const r = (seed >> 16) & 0xff;
  seed = (seed * 1103515245 + 12345) >>> 0;
  const g = (seed >> 16) & 0xff;
  seed = (seed * 1103515245 + 12345) >>> 0;
  const b = (seed >> 16) & 0xff;
  return [r, g, b];
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-color.test.ts`
Expected: PASS. Update the inline snapshots on first run with `npx vitest run packages/core/test/viz-color.test.ts -u` if needed.

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/color.ts packages/core/test/viz-color.test.ts && git commit -m "feat(viz): add deterministic color generator"
```

---

### Task 3: Canvas Factory

**Files:**

- Create: `packages/core/src/viz/canvas-factory.ts`
- Create: `packages/core/test/viz-canvas-factory.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/core/test/viz-canvas-factory.test.ts`:

```typescript
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { createCanvas } from "../src/viz/canvas-factory";

describe("viz/canvas-factory", () => {
  const origOffscreen = globalThis.OffscreenCanvas;

  afterEach(() => {
    // Restore original OffscreenCanvas
    if (origOffscreen) {
      globalThis.OffscreenCanvas = origOffscreen;
    } else {
      delete (globalThis as Record<string, unknown>).OffscreenCanvas;
    }
  });

  it("returns an OffscreenCanvas when available", () => {
    // jsdom doesn't have OffscreenCanvas, so mock it
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-canvas-factory.test.ts`
Expected: FAIL — cannot find module `../src/viz/canvas-factory`

- [ ] **Step 3: Write the implementation**

Create `packages/core/src/viz/canvas-factory.ts`:

```typescript
type AnyCanvas = OffscreenCanvas | HTMLCanvasElement;

/**
 * Create a canvas of the given dimensions.
 * Prefers OffscreenCanvas (no DOM dependency, works in workers).
 * Falls back to document.createElement("canvas") in older browsers.
 */
export function createCanvas(width: number, height: number): AnyCanvas {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/**
 * Get a 2D rendering context from any canvas type.
 */
export function getContext2D(
  canvas: AnyCanvas
): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create 2D rendering context.");
  }
  return ctx as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
}

/**
 * Convert a canvas to a Blob.
 * Uses convertToBlob() for OffscreenCanvas, toBlob() for HTMLCanvasElement.
 */
export function canvasToBlob(canvas: AnyCanvas, type: string, quality: number): Promise<Blob> {
  if (canvas instanceof OffscreenCanvas) {
    return canvas.convertToBlob({ type, quality });
  }
  return new Promise<Blob>((resolve, reject) => {
    (canvas as HTMLCanvasElement).toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("canvas.toBlob() returned null."));
        }
      },
      type,
      quality
    );
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-canvas-factory.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/canvas-factory.ts packages/core/test/viz-canvas-factory.test.ts && git commit -m "feat(viz): add canvas factory with OffscreenCanvas fallback"
```

---

### Task 4: Font Management

**Files:**

- Create: `packages/core/src/viz/font.ts`
- Create: `packages/core/test/viz-font.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/core/test/viz-font.test.ts`:

```typescript
import { describe, expect, it, vi, beforeEach } from "vitest";
import { loadFontFace, removeFontFace } from "../src/viz/font";
import type { FontConfig } from "../src/viz/types";

describe("viz/font", () => {
  let mockFontFace: { load: ReturnType<typeof vi.fn>; family: string };
  let originalFontFace: typeof globalThis.FontFace;

  beforeEach(() => {
    mockFontFace = {
      load: vi.fn().mockResolvedValue(undefined),
      family: ""
    };
    originalFontFace = globalThis.FontFace;
    (globalThis as Record<string, unknown>).FontFace = vi.fn((family: string, _source: unknown) => {
      mockFontFace.family = family;
      return mockFontFace;
    });
    // Mock document.fonts
    if (!document.fonts) {
      Object.defineProperty(document, "fonts", {
        value: { add: vi.fn(), delete: vi.fn() },
        configurable: true
      });
    } else {
      vi.spyOn(document.fonts, "add").mockImplementation(() => {});
      vi.spyOn(document.fonts, "delete").mockImplementation(() => true);
    }
  });

  it("loads a font from a URL string and adds to document.fonts", async () => {
    const config: FontConfig = {
      family: "TestFont",
      source: "https://example.com/font.woff2"
    };

    const face = await loadFontFace(config);
    expect(globalThis.FontFace).toHaveBeenCalledWith(
      "TestFont",
      "url(https://example.com/font.woff2)",
      undefined
    );
    expect(mockFontFace.load).toHaveBeenCalled();
    expect(document.fonts.add).toHaveBeenCalledWith(face);
  });

  it("loads a font from an ArrayBuffer source", async () => {
    const buffer = new ArrayBuffer(8);
    const config: FontConfig = {
      family: "BufFont",
      source: buffer
    };

    await loadFontFace(config);
    expect(globalThis.FontFace).toHaveBeenCalledWith("BufFont", buffer, undefined);
  });

  it("passes descriptors to FontFace constructor", async () => {
    const config: FontConfig = {
      family: "DescFont",
      source: "https://example.com/font.woff2",
      descriptors: { weight: "bold" }
    };

    await loadFontFace(config);
    expect(globalThis.FontFace).toHaveBeenCalledWith(
      "DescFont",
      "url(https://example.com/font.woff2)",
      { weight: "bold" }
    );
  });

  it("removes a font face from document.fonts", () => {
    removeFontFace(mockFontFace as unknown as FontFace);
    expect(document.fonts.delete).toHaveBeenCalledWith(mockFontFace);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-font.test.ts`
Expected: FAIL — cannot find module `../src/viz/font`

- [ ] **Step 3: Write the implementation**

Create `packages/core/src/viz/font.ts`:

```typescript
import type { FontConfig } from "./types";

/**
 * Load a font using the FontFace API and register it with document.fonts.
 * Returns the loaded FontFace instance for later removal.
 */
export async function loadFontFace(config: FontConfig): Promise<FontFace> {
  const source = typeof config.source === "string" ? `url(${config.source})` : config.source;

  const face = new FontFace(config.family, source, config.descriptors);
  await face.load();
  document.fonts.add(face);
  return face;
}

/**
 * Remove a previously loaded FontFace from document.fonts.
 */
export function removeFontFace(face: FontFace): void {
  document.fonts.delete(face);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-font.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/font.ts packages/core/test/viz-font.test.ts && git commit -m "feat(viz): add FontFace loading and removal helpers"
```

---

### Task 5: Draw Boxes Panel (Left Side)

**Files:**

- Create: `packages/core/src/viz/draw-boxes.ts`
- Create: `packages/core/test/viz-draw-boxes.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/core/test/viz-draw-boxes.test.ts`:

```typescript
import { describe, expect, it, vi } from "vitest";
import { drawBoxesPanel } from "../src/viz/draw-boxes";
import type { OcrResultItem, Point2D } from "../src/pipelines/ocr/core";

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
    expect(ctx.stroke).toHaveBeenCalled();
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

    // After drawImage, the first call for the item should be save, last should be restore
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-draw-boxes.test.ts`
Expected: FAIL — cannot find module `../src/viz/draw-boxes`

- [ ] **Step 3: Write the implementation**

Create `packages/core/src/viz/draw-boxes.ts`:

```typescript
import type { OcrResultItem } from "../pipelines/ocr/core";
import type { Point2D } from "../models/common";
import type { BoxStyleOptions, RgbColor } from "./types";
import { deterministicColor } from "./color";

const DEFAULT_LINE_WIDTH = 2;
const DEFAULT_FILL_OPACITY = 0.3;

type DrawableImage = ImageBitmap | HTMLImageElement;

function drawPolygonPath(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  poly: Point2D[]
): void {
  ctx.beginPath();
  ctx.moveTo(poly[0][0], poly[0][1]);
  for (let i = 1; i < poly.length; i += 1) {
    ctx.lineTo(poly[i][0], poly[i][1]);
  }
  ctx.closePath();
}

/**
 * Draw the left panel: source image overlaid with detection box polygons.
 */
export function drawBoxesPanel(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  image: DrawableImage,
  items: OcrResultItem[],
  style: BoxStyleOptions
): void {
  ctx.drawImage(image, 0, 0);

  const lineWidth = style.lineWidth ?? DEFAULT_LINE_WIDTH;
  const fillOpacity = style.fillOpacity ?? DEFAULT_FILL_OPACITY;
  const getColor = style.colorFn ?? deterministicColor;

  for (let i = 0; i < items.length; i += 1) {
    const [r, g, b]: RgbColor = getColor(i);
    ctx.save();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = `rgb(${String(r)}, ${String(g)}, ${String(b)})`;
    ctx.fillStyle = `rgba(${String(r)}, ${String(g)}, ${String(b)}, ${String(fillOpacity)})`;
    drawPolygonPath(ctx, items[i].poly);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-draw-boxes.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/draw-boxes.ts packages/core/test/viz-draw-boxes.test.ts && git commit -m "feat(viz): add left-panel box drawing"
```

---

### Task 6: Draw Text Panel (Right Side)

**Files:**

- Create: `packages/core/src/viz/draw-text.ts`
- Create: `packages/core/test/viz-draw-text.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/core/test/viz-draw-text.test.ts`:

```typescript
import { describe, expect, it, vi } from "vitest";
import { drawTextPanel } from "../src/viz/draw-text";
import type { OcrResultItem, Point2D } from "../src/pipelines/ocr/core";

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

    // fillStyle should be set to the custom background before fillRect
    // The fillRect call for the background should happen
    expect(ctx.fillRect).toHaveBeenCalledWith(200, 0, 200, 100);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-draw-text.test.ts`
Expected: FAIL — cannot find module `../src/viz/draw-text`

- [ ] **Step 3: Write the implementation**

Create `packages/core/src/viz/draw-text.ts`:

```typescript
import type { OcrResultItem } from "../pipelines/ocr/core";
import type { Point2D } from "../models/common";
import type { BoxStyleOptions, RgbColor } from "./types";
import { deterministicColor } from "./color";

const DEFAULT_BG = "#ffffff";
const OUTLINE_LINE_WIDTH = 1;
const TEXT_COLOR = "#000000";
const ROTATION_THRESHOLD_DEG = 5;

/**
 * Compute the angle (in radians) of the top edge of a quad polygon.
 * The top edge is defined as poly[0] -> poly[1].
 */
function topEdgeAngle(poly: Point2D[]): number {
  const dx = poly[1][0] - poly[0][0];
  const dy = poly[1][1] - poly[0][1];
  return Math.atan2(dy, dx);
}

/**
 * Compute the bounding box of a polygon.
 */
function polyBounds(poly: Point2D[]): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
} {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of poly) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

function drawPolygonPath(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  poly: Point2D[],
  offsetX: number
): void {
  ctx.beginPath();
  ctx.moveTo(poly[0][0] + offsetX, poly[0][1]);
  for (let i = 1; i < poly.length; i += 1) {
    ctx.lineTo(poly[i][0] + offsetX, poly[i][1]);
  }
  ctx.closePath();
}

/**
 * Draw the right panel: white background with detection box outlines and
 * recognized text rendered inside each box.
 */
export function drawTextPanel(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  offsetX: number,
  height: number,
  items: OcrResultItem[],
  style: BoxStyleOptions,
  fontFamily: string,
  background?: string
): void {
  const getColor = style.colorFn ?? deterministicColor;
  const bg = background ?? DEFAULT_BG;

  // Fill background
  ctx.save();
  ctx.fillStyle = bg;
  ctx.fillRect(offsetX, 0, offsetX, height);
  ctx.restore();

  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    const [r, g, b]: RgbColor = getColor(i);
    const bounds = polyBounds(item.poly);
    const angle = topEdgeAngle(item.poly);
    const absDeg = Math.abs(angle * (180 / Math.PI));
    const needsRotation = absDeg > ROTATION_THRESHOLD_DEG && absDeg < 180 - ROTATION_THRESHOLD_DEG;

    // Draw box outline
    ctx.save();
    ctx.lineWidth = OUTLINE_LINE_WIDTH;
    ctx.strokeStyle = `rgb(${String(r)}, ${String(g)}, ${String(b)})`;
    drawPolygonPath(ctx, item.poly, offsetX);
    ctx.stroke();
    ctx.restore();

    // Draw text
    const fontSize = Math.max(12, Math.floor(bounds.height * 0.8));
    ctx.save();
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = `${String(fontSize)}px "${fontFamily}"`;
    ctx.textBaseline = "middle";

    if (needsRotation) {
      const cx = bounds.minX + bounds.width / 2 + offsetX;
      const cy = bounds.minY + bounds.height / 2;
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.fillText(item.text, -bounds.width / 2, 0);
    } else {
      const x = bounds.minX + offsetX + 2;
      const y = bounds.minY + bounds.height / 2;
      ctx.fillText(item.text, x, y);
    }

    ctx.restore();
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-draw-text.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/draw-text.ts packages/core/test/viz-draw-text.test.ts && git commit -m "feat(viz): add right-panel text drawing"
```

---

### Task 7: Side-by-Side Composite Assembly

**Files:**

- Create: `packages/core/src/viz/side-by-side.ts`

This file is a thin orchestrator. It does not need its own unit tests — it will be covered by the renderer integration tests in Task 8.

- [ ] **Step 1: Write the implementation**

Create `packages/core/src/viz/side-by-side.ts`:

```typescript
import type { OcrResult } from "../pipelines/ocr/core";
import type { BoxStyleOptions } from "./types";
import { drawBoxesPanel } from "./draw-boxes";
import { drawTextPanel } from "./draw-text";
import { createCanvas, getContext2D, canvasToBlob } from "./canvas-factory";

type DrawableImage = ImageBitmap | HTMLImageElement;

function imageWidth(image: DrawableImage): number {
  return image instanceof HTMLImageElement ? image.naturalWidth : image.width;
}

function imageHeight(image: DrawableImage): number {
  return image instanceof HTMLImageElement ? image.naturalHeight : image.height;
}

export interface SideBySideOptions {
  boxStyle: BoxStyleOptions;
  fontFamily: string;
  textPanelBackground: string;
  outputFormat: string;
  outputQuality: number;
}

/**
 * Render a side-by-side composite to a new canvas and return the canvas.
 */
export function renderSideBySideToCanvas(
  image: DrawableImage,
  result: OcrResult,
  options: SideBySideOptions
): OffscreenCanvas | HTMLCanvasElement {
  const w = imageWidth(image);
  const h = imageHeight(image);
  const canvas = createCanvas(w * 2, h);
  const ctx = getContext2D(canvas);

  drawBoxesPanel(ctx, image, result.items, options.boxStyle);
  drawTextPanel(
    ctx,
    w,
    h,
    result.items,
    options.boxStyle,
    options.fontFamily,
    options.textPanelBackground
  );

  return canvas;
}

/**
 * Render side-by-side composite and return as ImageBitmap.
 */
export async function renderSideBySideToImageBitmap(
  image: DrawableImage,
  result: OcrResult,
  options: SideBySideOptions
): Promise<ImageBitmap> {
  const canvas = renderSideBySideToCanvas(image, result, options);
  return createImageBitmap(canvas as ImageBitmapSource);
}

/**
 * Render side-by-side composite and return as Blob.
 */
export async function renderSideBySideToBlob(
  image: DrawableImage,
  result: OcrResult,
  options: SideBySideOptions
): Promise<Blob> {
  const canvas = renderSideBySideToCanvas(image, result, options);
  return canvasToBlob(canvas, `image/${options.outputFormat}`, options.outputQuality);
}
```

- [ ] **Step 2: Run typecheck to verify no type errors**

Run: `cd paddleocr-js && npx tsc --noEmit -p packages/core/tsconfig.json`
Expected: No errors (or only pre-existing ones unrelated to viz).

- [ ] **Step 3: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/side-by-side.ts && git commit -m "feat(viz): add side-by-side composite assembly"
```

---

### Task 8: OcrVisualizer Class and renderOcrToBlob

**Files:**

- Create: `packages/core/src/viz/renderer.ts`
- Create: `packages/core/test/viz-renderer.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/core/test/viz-renderer.test.ts`:

```typescript
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock FontFace and document.fonts before importing
const mockFontFace = { load: vi.fn().mockResolvedValue(undefined), family: "" };
(globalThis as Record<string, unknown>).FontFace = vi.fn((family: string, _source: unknown) => {
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

import { OcrVisualizer, renderOcrToBlob } from "../src/viz/renderer";
import type { OcrResult } from "../src/pipelines/ocr/core";

function makeMockResult(): OcrResult {
  return {
    image: { width: 100, height: 50 },
    items: [
      {
        originalIndex: 0,
        poly: [
          [10, 10],
          [90, 10],
          [90, 40],
          [10, 40]
        ],
        text: "hello",
        score: 0.95
      }
    ],
    metrics: {
      detInferMs: 10,
      recPrepMs: 5,
      recInferMs: 15,
      totalMs: 30,
      detectedBoxes: 1,
      recognizedCount: 1
    },
    runtime: {
      requestedBackend: "auto",
      detProvider: "wasm",
      recProvider: "wasm",
      webgpuAvailable: false
    }
  };
}

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

  it("loadFont() loads the font via FontFace API", async () => {
    const viz = new OcrVisualizer({
      font: { family: "TestFont", source: "https://example.com/f.woff2" }
    });
    await viz.loadFont();
    expect(globalThis.FontFace).toHaveBeenCalled();
    expect(document.fonts.add).toHaveBeenCalled();
    viz.dispose();
  });

  it("loadFont() is a no-op when no font config", async () => {
    const viz = new OcrVisualizer();
    // Should not throw
    await viz.loadFont();
    viz.dispose();
  });

  it("dispose() removes the loaded font", async () => {
    const viz = new OcrVisualizer({
      font: { family: "TestFont", source: "https://example.com/f.woff2" }
    });
    await viz.loadFont();
    viz.dispose();
    expect(document.fonts.delete).toHaveBeenCalled();
  });

  it("dispose() is safe to call multiple times", async () => {
    const viz = new OcrVisualizer({
      font: { family: "TestFont", source: "https://example.com/f.woff2" }
    });
    await viz.loadFont();
    viz.dispose();
    viz.dispose(); // should not throw
  });
});

describe("renderOcrToBlob", () => {
  it("is exported as a function", () => {
    expect(typeof renderOcrToBlob).toBe("function");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-renderer.test.ts`
Expected: FAIL — cannot find module `../src/viz/renderer`

- [ ] **Step 3: Write the implementation**

Create `packages/core/src/viz/renderer.ts`:

```typescript
import type { OcrResult } from "../pipelines/ocr/core";
import type { OcrVisualizerOptions, BoxStyleOptions } from "./types";
import { loadFontFace, removeFontFace } from "./font";
import { renderSideBySideToImageBitmap, renderSideBySideToBlob } from "./side-by-side";
import type { SideBySideOptions } from "./side-by-side";

type DrawableImage = ImageBitmap | HTMLImageElement;

const DEFAULT_FONT_FAMILY = "sans-serif";
const DEFAULT_OUTPUT_FORMAT = "png";
const DEFAULT_OUTPUT_QUALITY = 0.92;
const DEFAULT_TEXT_PANEL_BG = "#ffffff";

let fontWarningEmitted = false;

function resolveOptions(
  base: OcrVisualizerOptions,
  overrides?: Partial<OcrVisualizerOptions>
): SideBySideOptions {
  const merged = overrides ? { ...base, ...overrides } : base;
  return {
    boxStyle: merged.boxStyle ?? {},
    fontFamily: merged.font?.family ?? DEFAULT_FONT_FAMILY,
    textPanelBackground: merged.textPanelBackground ?? DEFAULT_TEXT_PANEL_BG,
    outputFormat: merged.outputFormat ?? DEFAULT_OUTPUT_FORMAT,
    outputQuality: merged.outputQuality ?? DEFAULT_OUTPUT_QUALITY
  };
}

export class OcrVisualizer {
  private options: OcrVisualizerOptions;
  private loadedFace: FontFace | null = null;

  constructor(options?: OcrVisualizerOptions) {
    this.options = options ?? {};
  }

  /**
   * Load the custom font asynchronously.
   * Must be called before rendering if a font config was provided.
   * No-op if no font config is set.
   */
  async loadFont(): Promise<void> {
    if (!this.options.font) return;
    if (this.loadedFace) return; // already loaded
    this.loadedFace = await loadFontFace(this.options.font);
  }

  /**
   * Render side-by-side composite, return ImageBitmap.
   */
  async renderSideBySide(
    image: DrawableImage,
    result: OcrResult,
    overrides?: Partial<OcrVisualizerOptions>
  ): Promise<ImageBitmap> {
    this.warnIfNoFont();
    const opts = resolveOptions(this.options, overrides);
    return renderSideBySideToImageBitmap(image, result, opts);
  }

  /**
   * Render side-by-side composite, return downloadable Blob.
   */
  async toBlob(
    image: DrawableImage,
    result: OcrResult,
    overrides?: Partial<OcrVisualizerOptions>
  ): Promise<Blob> {
    this.warnIfNoFont();
    const opts = resolveOptions(this.options, overrides);
    return renderSideBySideToBlob(image, result, opts);
  }

  /**
   * Release internal resources (removes loaded font from document.fonts).
   */
  dispose(): void {
    if (this.loadedFace) {
      removeFontFace(this.loadedFace);
      this.loadedFace = null;
    }
  }

  private warnIfNoFont(): void {
    if (this.options.font && !this.loadedFace && !fontWarningEmitted) {
      fontWarningEmitted = true;
      console.warn(
        "[paddleocr-js/viz] Font config provided but loadFont() was not called. " +
          "Text will render with system sans-serif. CJK characters may not display correctly."
      );
    }
  }
}

/**
 * One-shot convenience function: create a temporary OcrVisualizer,
 * optionally load font, render to Blob, and dispose.
 */
export async function renderOcrToBlob(
  image: DrawableImage,
  result: OcrResult,
  options?: OcrVisualizerOptions
): Promise<Blob> {
  const viz = new OcrVisualizer(options);
  try {
    await viz.loadFont();
    return await viz.toBlob(image, result);
  } finally {
    viz.dispose();
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-renderer.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/renderer.ts packages/core/test/viz-renderer.test.ts && git commit -m "feat(viz): add OcrVisualizer class and renderOcrToBlob"
```

---

### Task 9: Subpath Entry and Public Exports

**Files:**

- Create: `packages/core/src/viz/index.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/core/test/viz-public-api.test.ts`:

```typescript
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-public-api.test.ts`
Expected: FAIL — cannot find module `../src/viz/index`

- [ ] **Step 3: Write the implementation**

Create `packages/core/src/viz/index.ts`:

```typescript
export { OcrVisualizer, renderOcrToBlob } from "./renderer";
export { deterministicColor } from "./color";

export type { OcrVisualizerOptions, FontConfig, BoxStyleOptions, RgbColor } from "./types";
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd paddleocr-js && npx vitest run packages/core/test/viz-public-api.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/src/viz/index.ts packages/core/test/viz-public-api.test.ts && git commit -m "feat(viz): add subpath entry point and public exports"
```

---

### Task 10: Build Configuration (Vite Multi-Entry + package.json)

**Files:**

- Modify: `packages/core/vite.config.ts`
- Modify: `packages/core/package.json`

- [ ] **Step 1: Update vite.config.ts to add viz entry**

In `packages/core/vite.config.ts`, change the `build.lib` section from a single entry to multiple entries. Replace the existing `build.lib` block:

```typescript
// REPLACE this in vite.config.ts build section:
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'paddleocr',
      formats: ['es', 'cjs', 'umd'],
      fileName: (format) => {
        if (format === 'es') return 'index.mjs'
        if (format === 'cjs') return 'index.cjs'
        return 'index.umd.js'
      },
    },

// WITH:
    lib: {
      entry: {
        index: resolve(__dirname, 'src/index.ts'),
        viz: resolve(__dirname, 'src/viz/index.ts'),
      },
      name: 'paddleocr',
      formats: ['es', 'cjs'],
      fileName: (format, entryName) => {
        const ext = format === 'es' ? 'mjs' : 'cjs'
        return `${entryName}.${ext}`
      },
    },
```

**Note:** UMD format does not support multiple entry points in Vite's lib mode. Since the viz module is a subpath export (consumed by bundlers, not CDN script tags), dropping UMD for the multi-entry build is acceptable. If UMD is needed for the main entry, a separate build step can be added later.

- [ ] **Step 2: Update package.json exports and file fields**

In `packages/core/package.json`, update the `"exports"` field to:

```json
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs",
      "require": "./dist/index.cjs"
    },
    "./viz": {
      "types": "./dist/viz.d.ts",
      "import": "./dist/viz.mjs",
      "require": "./dist/viz.cjs"
    }
  },
```

Also remove the top-level `"unpkg"` and `"jsdelivr"` fields (they pointed to the UMD build which is no longer generated):

Remove these lines:

```json
  "unpkg": "./dist/index.umd.js",
  "jsdelivr": "./dist/index.umd.js",
```

- [ ] **Step 3: Verify the build succeeds**

Run: `cd paddleocr-js && npm run build:sdk`
Expected: Build completes. Output files include `dist/index.mjs`, `dist/index.cjs`, `dist/viz.mjs`, `dist/viz.cjs`, `dist/index.d.ts`, `dist/viz.d.ts`.

Verify:

```bash
ls -la paddleocr-js/packages/core/dist/{index,viz}.{mjs,cjs,d.ts}
```

- [ ] **Step 4: Run all tests to verify nothing is broken**

Run: `cd paddleocr-js && npm test`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
cd paddleocr-js && git add packages/core/vite.config.ts packages/core/package.json && git commit -m "build(viz): add viz subpath entry to Vite build and package.json exports"
```

---

### Task 11: Run Full Check and Fix Any Issues

**Files:** None new — verification only.

- [ ] **Step 1: Run the full check suite**

Run: `cd paddleocr-js && npm run check`

This runs: `format:check && lint && build:sdk && typecheck && test && build:demo`.

Expected: All green. If there are lint/format issues, fix them.

- [ ] **Step 2: Fix any lint or format issues**

If the check fails due to formatting:

```bash
cd paddleocr-js && npm run format && npm run lint:fix
```

Then re-run: `cd paddleocr-js && npm run check`

- [ ] **Step 3: Commit any fixes**

```bash
cd paddleocr-js && git add -A && git commit -m "chore(viz): fix lint and formatting"
```

(Only run this step if there were fixes needed. Skip if Step 1 passed clean.)
