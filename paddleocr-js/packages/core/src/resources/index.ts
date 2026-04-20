/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export type { ModelAsset, ModelAssetsMap, ModelLoadResult, ModelLoadSummary } from "./model-asset";
export {
  DEFAULT_MODEL_ASSETS,
  MODEL_ENTRY_PATHS,
  assertModelResourceSlot,
  assertModelResources,
  getModelEntryPath,
  loadModelAsset,
  normalizeAssets,
  normalizeModelAsset
} from "./model-asset";
export { extractTarEntries, pickTarEntry } from "./tar";
