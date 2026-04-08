# Viz Utility for @paddleocr/paddleocr-js

**Date:** 2026-04-08
**Status:** Approved

## Summary

Add an optional visualization module to `@paddleocr/paddleocr-js`, exported via a subpath (`@paddleocr/paddleocr-js/viz`). The module renders a side-by-side composite image (source image with detection boxes on the left, white panel with recognized text on the right) and exports it as a downloadable `Blob`. It uses pure Canvas 2D with FontFace API for custom font support, introducing zero new runtime dependencies.

## Goals

- Provide a ready-to-use visualization comparable to PaddleOCR Python's `result.save_to_img()`.
- Keep the core SDK untouched; viz is fully opt-in via subpath import.
- Support custom font loading for CJK text rendering.
- Output downloadable `Blob` images (PNG/JPEG/WebP).

## Non-Goals

- Detection-box-only overlay (no separate API for just drawing boxes).
- Node.js / SSR support (Canvas 2D and FontFace are browser APIs).
- SVG output.
- Bundling a CJK font within the package.

## Module Structure

```
packages/core/src/
  viz/
    index.ts          # Subpath export entry point
    renderer.ts       # OcrVisualizer class
    font.ts           # FontFace loading and management
    draw-boxes.ts     # Detection box drawing (left panel)
    draw-text.ts      # Text rendering (right panel)
    side-by-side.ts   # Left-right composite assembly
    color.ts          # Deterministic color generation
    types.ts          # Shared type definitions
```

### Subpath Export Configuration

`packages/core/package.json` adds:

```json
{
  "exports": {
    ".": { "types": "...", "import": "...", "require": "..." },
    "./viz": {
      "types": "./dist/viz.d.ts",
      "import": "./dist/viz.mjs",
      "require": "./dist/viz.cjs"
    }
  }
}
```

### Build Configuration

Vite config adds `src/viz/index.ts` as an additional entry point, producing a separate chunk. The viz module imports only type definitions from the core SDK (`OcrResult`, `OcrResultItem`, `Point2D`). It has no runtime dependency on OpenCV.js or ONNX Runtime.

## API Design

### Types

```typescript
interface OcrVisualizerOptions {
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

interface FontConfig {
  /** CSS font-family name. */
  family: string;
  /** Font source: URL string or ArrayBuffer. */
  source: string | ArrayBuffer;
  /** FontFace descriptors (weight, style, etc.). */
  descriptors?: FontFaceDescriptors;
}

interface BoxStyleOptions {
  /** Box stroke width. Default: 2. */
  lineWidth?: number;
  /** Fill opacity 0-1. Default: 0.3. */
  fillOpacity?: number;
  /** Custom color function. Default: deterministic LCG-based colors. */
  colorFn?: (index: number) => [number, number, number];
}
```

### OcrVisualizer Class

```typescript
class OcrVisualizer {
  constructor(options?: OcrVisualizerOptions);

  /** Load the custom font asynchronously. Must be called before rendering
      if a font config was provided. */
  async loadFont(): Promise<void>;

  /** Render side-by-side composite, return ImageBitmap. */
  async renderSideBySide(
    image: ImageBitmap | HTMLImageElement,
    result: OcrResult,
    overrides?: Partial<OcrVisualizerOptions>,
  ): Promise<ImageBitmap>;

  /** Render side-by-side composite, return downloadable Blob. */
  async toBlob(
    image: ImageBitmap | HTMLImageElement,
    result: OcrResult,
    overrides?: Partial<OcrVisualizerOptions>,
  ): Promise<Blob>;

  /** Release internal resources. */
  dispose(): void;
}
```

### One-Shot Convenience Function

```typescript
/** Create a temporary OcrVisualizer, render, and dispose in one call. */
async function renderOcrToBlob(
  image: ImageBitmap | HTMLImageElement,
  result: OcrResult,
  options?: OcrVisualizerOptions,
): Promise<Blob>;
```

## Rendering Logic

### Canvas Layout

```
+-------------------+-------------------+
|                   |                   |
|    Left Panel     |    Right Panel    |
|  (source + boxes) |  (white + text)   |
|                   |                   |
+-------------------+-------------------+
  width = imgW          width = imgW
  height = imgH         height = imgH

Total canvas: (imgW * 2, imgH)
```

### Left Panel

1. Draw the source image at `(0, 0)`.
2. For each `OcrResultItem`, draw its `poly` as:
   - Semi-transparent filled polygon (using deterministic per-index color, configurable opacity).
   - Solid stroke outline (same color, configurable line width).

### Right Panel

1. Fill with `textPanelBackground` (default white).
2. For each `OcrResultItem`:
   - Draw the detection box outline (thin stroke, same color as left panel).
   - Render recognized text inside the box:
     - **Font size**: auto-calculated from the detection box height.
     - **Placement**: horizontally left-aligned at the box's top-left corner, vertically centered within the box height.
     - **Rotation**: if the box has significant rotation (angle > 5 degrees from horizontal), apply `canvas.rotate()` along the box's primary axis.

### Differences from Python Version

The Python implementation (`OCRResult._to_img()`) uses PIL perspective transform to warp text into the exact quadrilateral shape of the detection box. Canvas 2D lacks native perspective transform capability.

Our approach uses a simplified rendering: text is drawn horizontally (or rotated for tilted boxes) within the minimum bounding rectangle of the detection box. This is simpler, reliable, and produces good results for the vast majority of text (which is near-horizontal).

### Canvas Strategy

- Prefer `OffscreenCanvas` when available (works without DOM access, compatible with workers).
- Fall back to `document.createElement("canvas")` in environments where `OffscreenCanvas` is not supported.
- `renderSideBySide()` returns `ImageBitmap` via `createImageBitmap(canvas)` (works in both `OffscreenCanvas` and regular `<canvas>` paths). `toBlob()` uses `canvas.convertToBlob()` (OffscreenCanvas) or `canvas.toBlob()` (HTMLCanvasElement) with a Promise wrapper.

## Font Management

### Loading

`OcrVisualizer.loadFont()` uses the browser [FontFace API](https://developer.mozilla.org/en-US/docs/Web/API/FontFace):

```typescript
const face = new FontFace(config.family, sourceData, config.descriptors);
await face.load();
document.fonts.add(face);
```

### Behavior Without Custom Font

If no `FontConfig` is provided, or if `loadFont()` is not called:
- Rendering proceeds with Canvas's default `"sans-serif"` font.
- A `console.warn()` is emitted once noting that CJK characters may not render correctly.
- The result is still valid; boxes are drawn correctly, text may show as fallback glyphs.

### Disposal

`OcrVisualizer.dispose()` removes the loaded `FontFace` from `document.fonts` to avoid leaking registered fonts across the page lifetime.

## Deterministic Color Generation

Reuses the same LCG-based algorithm currently in `apps/demo/src/main.ts`:

```typescript
function deterministicColor(index: number): [number, number, number] {
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

This ensures the same result always produces the same colors. Users can override via `boxStyle.colorFn`.

## Usage Examples

### Multi-Render with Renderer Instance

```typescript
import { PaddleOCR } from "@paddleocr/paddleocr-js";
import { OcrVisualizer } from "@paddleocr/paddleocr-js/viz";

const ocr = await PaddleOCR.create({ runtime: { wasmPaths: "..." } });
const viz = new OcrVisualizer({
  font: { family: "Noto Sans SC", source: "/fonts/NotoSansSC-Regular.ttf" },
});
await viz.loadFont();

const result = await ocr.predict(imageFile);
const imageBitmap = await createImageBitmap(imageFile);
const blob = await viz.toBlob(imageBitmap, result);

// Trigger browser download
const url = URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = url;
a.download = "ocr_result.png";
a.click();
URL.revokeObjectURL(url);

viz.dispose();
await ocr.dispose();
```

### One-Shot Rendering

```typescript
import { renderOcrToBlob } from "@paddleocr/paddleocr-js/viz";

const blob = await renderOcrToBlob(imageBitmap, result, {
  font: { family: "Noto Sans SC", source: "/fonts/NotoSansSC-Regular.ttf" },
  outputFormat: "jpeg",
  outputQuality: 0.85,
});
```

### System Font Fallback (No Custom Font)

```typescript
import { OcrVisualizer } from "@paddleocr/paddleocr-js/viz";

const viz = new OcrVisualizer(); // No font config
const blob = await viz.toBlob(image, result);
// Right panel text uses "sans-serif"; CJK may show as fallback glyphs
```

## Demo Integration

After implementing this module, `apps/demo/src/main.ts` can be updated to:
- Replace `deterministicColor()`, `drawPolygonPath()`, `drawPreview()` with imports from `@paddleocr/paddleocr-js/viz`.
- Add a "Download Result" button that calls `viz.toBlob()` and triggers a browser download.

This is a follow-up task, not part of the core implementation.

## Testing Strategy

- **Unit tests** for `color.ts` (deterministic output), `font.ts` (load/dispose lifecycle), and type validation.
- **Visual regression tests** are not in scope for the initial implementation. Manual verification with the demo app is sufficient.
- **Integration test**: verify that `renderOcrToBlob()` returns a valid PNG `Blob` with non-zero `size` given a mock `OcrResult` and a test image. Use `OffscreenCanvas` in jsdom/vitest environment (may require polyfill or skip if unavailable).
