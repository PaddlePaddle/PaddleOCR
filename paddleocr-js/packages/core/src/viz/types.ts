/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
