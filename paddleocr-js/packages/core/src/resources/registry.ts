export interface AssetDescriptor {
  id: string;
  url: string;
  kind: "file" | "tar";
  version?: string;
  entries?: Record<string, string>;
}

export type ModelAssetsMap = Record<string, AssetDescriptor>;

export const DEFAULT_MODEL_ASSETS: ModelAssetsMap = {
  "PP-OCRv5_mobile_det": {
    id: "pp-ocrv5-mobile-det-tar",
    kind: "tar",
    url: "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_onnx.tar",
    entries: {
      model: "inference.onnx",
      config: "inference.yml",
    },
  },
  "PP-OCRv5_mobile_rec": {
    id: "pp-ocrv5-mobile-rec-tar",
    kind: "tar",
    url: "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_onnx.tar",
    entries: {
      model: "inference.onnx",
      config: "inference.yml",
    },
  },
};

function isAssetDescriptor(asset: unknown): asset is Record<string, unknown> {
  return Boolean(asset && typeof asset === "object" && !Array.isArray(asset));
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

export function normalizeAssetDescriptor(assetName: string, asset: unknown): AssetDescriptor {
  if (!isAssetDescriptor(asset)) {
    throw new Error(`Asset "${assetName}" must be an object.`);
  }
  if (!isNonEmptyString(asset.id) || !isNonEmptyString(asset.url)) {
    throw new Error(`Asset "${assetName}" must define both id and url.`);
  }

  const kind = (asset.kind as string) || "file";
  if (kind !== "file" && kind !== "tar") {
    throw new Error(`Asset "${assetName}" has unsupported kind "${kind}".`);
  }
  if (kind === "tar" && asset.entries && !isAssetDescriptor(asset.entries)) {
    throw new Error(`Tar asset "${assetName}" must define entries as an object.`);
  }
  if (
    kind === "tar" &&
    asset.entries &&
    Object.values(asset.entries as Record<string, unknown>).some(
      (entryPath) => !isNonEmptyString(entryPath),
    )
  ) {
    throw new Error(`Tar asset "${assetName}" must map entries to file paths.`);
  }
  if (asset.version !== undefined && !isNonEmptyString(asset.version)) {
    throw new Error(`Asset "${assetName}" must use a non-empty version string.`);
  }

  return { ...(asset as Record<string, unknown>), kind } as unknown as AssetDescriptor;
}

function resolveAssetReference(
  assetName: string,
  asset: unknown,
  modelAssets: ModelAssetsMap,
): AssetDescriptor {
  if (isNonEmptyString(asset)) {
    const resolvedAsset = modelAssets[asset];
    if (!resolvedAsset) {
      throw new Error(`Asset "${assetName}" references unknown model asset "${asset}".`);
    }
    return normalizeAssetDescriptor(assetName, resolvedAsset);
  }

  return normalizeAssetDescriptor(assetName, asset);
}

export function normalizeAssets(
  assets: Record<string, unknown> | undefined,
  modelAssets: ModelAssetsMap = DEFAULT_MODEL_ASSETS,
): Record<string, AssetDescriptor> {
  const assetEntries = Object.entries(assets || {});

  if (assetEntries.length === 0) {
    throw new Error("Assets must define at least one asset.");
  }

  return Object.fromEntries(
    assetEntries.map(([assetName, asset]) => [
      assetName,
      resolveAssetReference(assetName, asset, modelAssets),
    ]),
  );
}
