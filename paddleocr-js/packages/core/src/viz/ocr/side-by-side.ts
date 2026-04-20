/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { OcrResult } from "../../pipelines/ocr/core";
import type { BoxStyleOptions } from "./types";
import { drawBoxesPanel } from "./draw-boxes";
import { drawTextPanel } from "./draw-text";
import { createCanvas, getContext2D, canvasToBlob } from "../canvas-factory";

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

export async function renderSideBySideToImageBitmap(
  image: DrawableImage,
  result: OcrResult,
  options: SideBySideOptions
): Promise<ImageBitmap> {
  const canvas = renderSideBySideToCanvas(image, result, options);
  return createImageBitmap(canvas as ImageBitmapSource);
}

export async function renderSideBySideToBlob(
  image: DrawableImage,
  result: OcrResult,
  options: SideBySideOptions
): Promise<Blob> {
  const canvas = renderSideBySideToCanvas(image, result, options);
  return canvasToBlob(canvas, `image/${options.outputFormat}`, options.outputQuality);
}
