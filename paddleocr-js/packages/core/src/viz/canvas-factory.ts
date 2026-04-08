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
  canvas: AnyCanvas,
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
export function canvasToBlob(
  canvas: AnyCanvas,
  type: string,
  quality: number,
): Promise<Blob> {
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
      quality,
    );
  });
}
