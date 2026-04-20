/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ModelAsset {
  url: string;
}

export type ModelAssetsMap = Record<string, ModelAsset>;

export const DEFAULT_MODEL_ASSETS: ModelAssetsMap = {
  "PP-OCRv5_mobile_det": {
    url: "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_onnx.tar"
  },
  "PP-OCRv5_mobile_rec": {
    url: "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_onnx.tar"
  }
};

export const MODEL_ENTRY_PATHS: Readonly<Record<string, string>> = Object.freeze({
  model: "inference.onnx",
  config: "inference.yml"
});

export interface ModelLoadResult {
  modelBytes: Uint8Array;
  configText: string;
  download: ModelLoadSummary;
}

export interface ModelLoadSummary {
  url: string;
  bytes: number;
}

// --- Validation helpers ---

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

export function normalizeModelAsset(assetName: string, asset: unknown): ModelAsset {
  if (isNonEmptyString(asset)) {
    const resolvedAsset = DEFAULT_MODEL_ASSETS[asset];
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition -- runtime guard for missing Record key
    if (!resolvedAsset) {
      throw new Error(`Asset "${assetName}" references unknown model asset "${asset}".`);
    }
    return { url: resolvedAsset.url };
  }

  if (!isObject(asset)) {
    throw new Error(`Asset "${assetName}" must be an object.`);
  }
  if (!isNonEmptyString(asset.url)) {
    throw new Error(`Asset "${assetName}" must define url.`);
  }

  return {
    url: asset.url
  };
}

export function normalizeAssets(
  assets: Record<string, unknown> | undefined
): Record<string, ModelAsset> {
  const assetEntries = Object.entries(assets || {});

  if (assetEntries.length === 0) {
    throw new Error("Assets must define at least one asset.");
  }

  return Object.fromEntries(
    assetEntries.map(([assetName, asset]) => [assetName, normalizeModelAsset(assetName, asset)])
  );
}

// --- Model loading ---

export function getModelEntryPath(slot: string): string | null {
  return MODEL_ENTRY_PATHS[slot] || null;
}

export function assertModelResourceSlot(kind: string, slot: string, value: unknown): void {
  if (slot === "model") {
    if (!(value instanceof Uint8Array) || value.byteLength === 0) {
      throw new Error(`${kind} model requires a non-empty ${MODEL_ENTRY_PATHS.model} resource.`);
    }
    return;
  }

  if (slot === "config") {
    if (typeof value !== "string" || value.trim().length === 0) {
      throw new Error(`${kind} model requires a non-empty ${MODEL_ENTRY_PATHS.config} resource.`);
    }
    return;
  }

  throw new Error(`Unsupported model resource slot "${slot}".`);
}

export function assertModelResources(kind: string, resources: Record<string, unknown>): void {
  for (const [slot, value] of Object.entries(resources)) {
    assertModelResourceSlot(kind, slot, value);
  }
}

// --- Model loading (fetch + tar extraction) ---

import { extractTarEntries, pickTarEntry } from "./tar";

export async function loadModelAsset(
  asset: ModelAsset,
  fetchImpl: typeof fetch = fetch
): Promise<ModelLoadResult> {
  const response = await fetchImpl(asset.url);
  if (!response.ok) {
    throw new Error(`Failed to download ${asset.url}: HTTP ${String(response.status)}`);
  }
  const buffer = await response.arrayBuffer();
  const entries = extractTarEntries(buffer);
  const modelBytes = pickTarEntry(entries, MODEL_ENTRY_PATHS.model);
  const configBytes = pickTarEntry(entries, MODEL_ENTRY_PATHS.config);

  return {
    modelBytes,
    configText: new TextDecoder().decode(configBytes),
    download: {
      url: asset.url,
      bytes: buffer.byteLength
    }
  };
}
