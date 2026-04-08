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

import type { RgbColor } from "./types";

/**
 * Generate a deterministic RGB color for a given index.
 * Uses a Linear Congruential Generator (LCG) seeded by the index.
 * Same algorithm as the demo app to ensure visual consistency.
 */
export function deterministicColor(index: number): RgbColor {
  let seed = (index + 1) * 1103515245 + 12345;
  seed >>>= 0;
  const r = (seed >> 16) & 0xff;
  seed = (seed * 1103515245 + 12345) >>> 0;
  const g = (seed >> 16) & 0xff;
  seed = (seed * 1103515245 + 12345) >>> 0;
  const b = (seed >> 16) & 0xff;
  return [r, g, b];
}
