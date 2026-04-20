/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { OcrResult } from "../../pipelines/ocr/core";
import type { OcrVisualizerOptions } from "./types";
import { loadFontFace, removeFontFace } from "../font";
import { renderSideBySideToImageBitmap, renderSideBySideToBlob } from "./side-by-side";
import type { SideBySideOptions } from "./side-by-side";

type DrawableImage = ImageBitmap | HTMLImageElement;

const DEFAULT_FONT_FAMILY = "sans-serif";
const DEFAULT_OUTPUT_FORMAT = "png";
const DEFAULT_OUTPUT_QUALITY = 0.92;
const DEFAULT_TEXT_PANEL_BG = "#ffffff";

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

  async loadFont(): Promise<void> {
    if (!this.options.font) return;
    if (this.loadedFace) return;
    this.loadedFace = await loadFontFace(this.options.font);
  }

  async renderSideBySide(
    image: DrawableImage,
    result: OcrResult,
    overrides?: Partial<OcrVisualizerOptions>
  ): Promise<ImageBitmap> {
    await this.loadFont();
    const opts = resolveOptions(this.options, overrides);
    return renderSideBySideToImageBitmap(image, result, opts);
  }

  async toBlob(
    image: DrawableImage,
    result: OcrResult,
    overrides?: Partial<OcrVisualizerOptions>
  ): Promise<Blob> {
    await this.loadFont();
    const opts = resolveOptions(this.options, overrides);
    return renderSideBySideToBlob(image, result, opts);
  }

  dispose(): void {
    if (this.loadedFace) {
      removeFontFace(this.loadedFace);
      this.loadedFace = null;
    }
  }
}

export async function renderOcrToBlob(
  image: DrawableImage,
  result: OcrResult,
  options?: OcrVisualizerOptions
): Promise<Blob> {
  const viz = new OcrVisualizer(options);
  try {
    return await viz.toBlob(image, result);
  } finally {
    viz.dispose();
  }
}
