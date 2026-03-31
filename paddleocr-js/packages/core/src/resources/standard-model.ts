import type { AssetDescriptor } from "./registry";
import { fetchResourceAsset, readAssetArrayBuffer, summarizeAssetResult } from "./cache";
import type { AssetDownloadSummary } from "./cache";
import { extractTarEntries, pickTarEntry } from "./tar";

export const STANDARD_MODEL_ENTRY_PATHS: Readonly<Record<string, string>> = Object.freeze({
  model: "inference.onnx",
  config: "inference.yml",
});

export interface StandardModelLoadResult {
  modelBytes: Uint8Array;
  configText: string;
  download: AssetDownloadSummary;
}

export function getStandardModelEntryPath(slot: string): string | null {
  return STANDARD_MODEL_ENTRY_PATHS[slot] || null;
}

function getStandardModelEntry(asset: AssetDescriptor, slot: string): string {
  return asset.entries?.[slot] || getStandardModelEntryPath(slot) || slot;
}

export async function loadStandardModelAsset(
  asset: AssetDescriptor,
  fetchImpl: typeof fetch = fetch,
): Promise<StandardModelLoadResult> {
  if (asset.kind !== "tar") {
    throw new Error(`Standard model asset "${asset.id}" must use a tar bundle.`);
  }

  const downloaded = await fetchResourceAsset(asset, fetchImpl);
  const buffer = await readAssetArrayBuffer(downloaded);
  const entries = extractTarEntries(buffer);
  const modelBytes = pickTarEntry(entries, getStandardModelEntry(asset, "model"));
  const configBytes = pickTarEntry(entries, getStandardModelEntry(asset, "config"));

  return {
    modelBytes,
    configText: new TextDecoder().decode(configBytes),
    download: summarizeAssetResult(asset, downloaded, buffer.byteLength),
  };
}

export function assertStandardModelResourceSlot(kind: string, slot: string, value: unknown): void {
  if (slot === "model") {
    if (!(value instanceof Uint8Array) || value.byteLength === 0) {
      throw new Error(
        `${kind} model requires a non-empty ${STANDARD_MODEL_ENTRY_PATHS.model} resource.`,
      );
    }
    return;
  }

  if (slot === "config") {
    if (typeof value !== "string" || value.trim().length === 0) {
      throw new Error(
        `${kind} model requires a non-empty ${STANDARD_MODEL_ENTRY_PATHS.config} resource.`,
      );
    }
    return;
  }

  throw new Error(`Unsupported standard model resource slot "${slot}".`);
}

export function assertStandardModelResources(kind: string, resources: Record<string, unknown>): void {
  for (const [slot, value] of Object.entries(resources)) {
    assertStandardModelResourceSlot(kind, slot, value);
  }
}
