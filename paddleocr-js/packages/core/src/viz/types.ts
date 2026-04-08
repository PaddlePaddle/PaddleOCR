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

/** An RGB color as a 3-element tuple of 0-255 integers. */
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
