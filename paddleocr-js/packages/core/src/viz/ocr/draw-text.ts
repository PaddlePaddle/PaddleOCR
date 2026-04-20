/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { OcrResultItem } from "../../pipelines/ocr/core";
import type { Point2D } from "../../models/common";
import type { RgbColor } from "../types";
import type { BoxStyleOptions } from "./types";
import { deterministicColor } from "../color";

const DEFAULT_BG = "#ffffff";
const OUTLINE_LINE_WIDTH = 1;
const TEXT_COLOR = "#000000";
const ROTATION_THRESHOLD_DEG = 5;
const VERTICAL_LINE_SPACING = 2;

function topEdgeAngle(poly: Point2D[]): number {
  const dx = poly[1][0] - poly[0][0];
  const dy = poly[1][1] - poly[0][1];
  return Math.atan2(dy, dx);
}

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

function drawVerticalText(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  text: string,
  x: number,
  startY: number,
  fontSize: number,
  fontFamily: string
): void {
  ctx.font = `${String(fontSize)}px "${fontFamily}"`;
  let y = startY;
  for (const char of text) {
    ctx.fillText(char, x, y);
    y += fontSize + VERTICAL_LINE_SPACING;
  }
}

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

    // Detect vertical text: height > 2 * width and height > 30px
    const isVertical = bounds.height > 2 * bounds.width && bounds.height > 30;

    // Draw box outline
    ctx.save();
    ctx.lineWidth = OUTLINE_LINE_WIDTH;
    ctx.strokeStyle = `rgb(${String(r)}, ${String(g)}, ${String(b)})`;
    drawPolygonPath(ctx, item.poly, offsetX);
    ctx.stroke();
    ctx.restore();

    // Draw text
    ctx.save();
    ctx.fillStyle = TEXT_COLOR;

    if (isVertical) {
      // Vertical text: render characters one-by-one, stacked vertically
      ctx.textBaseline = "top";
      const chars = Array.from(item.text);
      const charCount = Math.max(1, chars.length);
      let fontSize = Math.max(8, Math.floor(bounds.width * 0.8));

      // Scale down if total character height exceeds box height
      const totalHeight = charCount * (fontSize + VERTICAL_LINE_SPACING);
      if (totalHeight > bounds.height) {
        fontSize = Math.max(8, Math.floor((bounds.height / charCount) * 0.8));
      }

      // Ensure each character fits within the box width
      ctx.font = `${String(fontSize)}px "${fontFamily}"`;
      const maxCharWidth = Math.max(...chars.map((c) => ctx.measureText(c).width));
      if (maxCharWidth > bounds.width) {
        fontSize = Math.max(8, Math.floor(fontSize * (bounds.width / maxCharWidth)));
      }

      const x = bounds.minX + offsetX + (bounds.width - fontSize) / 2;
      const y = bounds.minY + 2;
      drawVerticalText(ctx, item.text, x, y, fontSize, fontFamily);
    } else {
      // Horizontal text
      ctx.textBaseline = "middle";
      let fontSize = Math.max(12, Math.floor(bounds.height * 0.8));
      ctx.font = `${String(fontSize)}px "${fontFamily}"`;

      // Shrink font if text is wider than the box
      const measured = ctx.measureText(item.text);
      if (measured.width > bounds.width && bounds.width > 0) {
        fontSize = Math.max(8, Math.floor(fontSize * (bounds.width / measured.width)));
        ctx.font = `${String(fontSize)}px "${fontFamily}"`;
      }

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
    }

    ctx.restore();
  }
}
