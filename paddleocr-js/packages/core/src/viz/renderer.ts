// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import type { OcrResult } from "../pipelines/ocr/core";
import type { OcrVisualizerOptions } from "./types";
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
