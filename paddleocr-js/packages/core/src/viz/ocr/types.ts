/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { RgbColor, FontConfig } from "../types";

export interface BoxStyleOptions {
  /** Fill opacity 0-1. Default: 0.5. */
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
