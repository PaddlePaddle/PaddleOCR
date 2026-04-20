/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { FontConfig } from "./types";

export async function loadFontFace(config: FontConfig): Promise<FontFace> {
  const source = typeof config.source === "string" ? `url(${config.source})` : config.source;

  const face = new FontFace(config.family, source, config.descriptors);
  await face.load();
  document.fonts.add(face);
  return face;
}

export function removeFontFace(face: FontFace): void {
  document.fonts.delete(face);
}
