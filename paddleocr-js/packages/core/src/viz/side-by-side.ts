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
  options: SideBySideOptions,
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
    options.textPanelBackground,
  );

  return canvas;
}

/**
 * Render side-by-side composite and return as ImageBitmap.
 */
export async function renderSideBySideToImageBitmap(
  image: DrawableImage,
  result: OcrResult,
  options: SideBySideOptions,
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
  options: SideBySideOptions,
): Promise<Blob> {
  const canvas = renderSideBySideToCanvas(image, result, options);
  return canvasToBlob(
    canvas,
    `image/${options.outputFormat}`,
    options.outputQuality,
  );
}
