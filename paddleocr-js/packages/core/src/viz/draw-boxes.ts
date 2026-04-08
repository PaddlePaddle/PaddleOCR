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

import type { OcrResultItem } from "../pipelines/ocr/core";
import type { Point2D } from "../models/common";
import type { BoxStyleOptions, RgbColor } from "./types";
import { deterministicColor } from "./color";

const DEFAULT_FILL_OPACITY = 0.5;

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
 * Draw the left panel: source image blended with detection box polygon fills.
 */
export function drawBoxesPanel(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  image: DrawableImage,
  items: OcrResultItem[],
  style: BoxStyleOptions
): void {
  const fillOpacity = style.fillOpacity ?? DEFAULT_FILL_OPACITY;
  const getColor = style.colorFn ?? deterministicColor;

  // Draw original image
  ctx.drawImage(image, 0, 0);

  // Draw filled polygons with the specified opacity.
  // Canvas source-over compositing with alpha gives:
  //   result = color * alpha + original * (1 - alpha)
  for (let i = 0; i < items.length; i += 1) {
    const [r, g, b]: RgbColor = getColor(i);
    ctx.save();
    ctx.fillStyle = `rgba(${String(r)}, ${String(g)}, ${String(b)}, ${String(fillOpacity)})`;
    drawPolygonPath(ctx, items[i].poly);
    ctx.fill();
    ctx.restore();
  }
}
