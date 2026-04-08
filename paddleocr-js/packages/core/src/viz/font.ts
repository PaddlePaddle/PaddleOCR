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

import type { FontConfig } from "./types";

/**
 * Load a font using the FontFace API and register it with document.fonts.
 * Returns the loaded FontFace instance for later removal.
 */
export async function loadFontFace(config: FontConfig): Promise<FontFace> {
  const source = typeof config.source === "string" ? `url(${config.source})` : config.source;

  const face = new FontFace(config.family, source, config.descriptors);
  await face.load();
  document.fonts.add(face);
  return face;
}

/**
 * Remove a previously loaded FontFace from document.fonts.
 */
export function removeFontFace(face: FontFace): void {
  document.fonts.delete(face);
}
