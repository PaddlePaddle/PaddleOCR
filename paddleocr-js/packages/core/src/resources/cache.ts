export interface ResourceAsset {
  id: string;
  url: string;
  version?: string;
}

export interface AssetFetchResult {
  cacheHit: boolean;
  cacheKey: string;
  response: Response;
}

export interface AssetDownloadSummary {
  id: string;
  url: string;
  cacheHit: boolean;
  bytes: number;
}

const CACHE_NAME = "paddleocr-js-model-cache";
const memoryCache = new Map<string, Response>();

function getCacheKey(asset: ResourceAsset): string {
  const cacheVersion = asset.version || "latest";
  if (typeof location === "undefined") {
    return `memory://${asset.id}?version=${encodeURIComponent(cacheVersion)}&url=${encodeURIComponent(asset.url)}`;
  }
  const url = new URL(`/__paddleocr_js_models__/${encodeURIComponent(asset.id)}`, location.origin);
  url.searchParams.set("version", cacheVersion);
  url.searchParams.set("url", asset.url);
  return url.toString();
}

async function getPersistentCache(): Promise<Cache | null> {
  if (!("caches" in globalThis)) return null;
  return caches.open(CACHE_NAME);
}

async function readCachedResponse(key: string): Promise<Response | null> {
  const persistentCache = await getPersistentCache();
  if (persistentCache) {
    const response = await persistentCache.match(key);
    if (response) return response;
  }
  return memoryCache.get(key)?.clone() || null;
}

async function storeCachedResponse(key: string, response: Response): Promise<void> {
  const persistentCache = await getPersistentCache();
  if (persistentCache) {
    await persistentCache.put(key, response.clone());
  }
  memoryCache.set(key, response.clone());
}

export async function fetchResourceAsset(
  asset: ResourceAsset,
  fetchImpl: typeof fetch = fetch,
): Promise<AssetFetchResult> {
  const cacheKey = getCacheKey(asset);
  const cachedResponse = await readCachedResponse(cacheKey);
  if (cachedResponse) {
    return {
      cacheHit: true,
      cacheKey,
      response: cachedResponse,
    };
  }

  const response = await fetchImpl(asset.url);
  if (!response.ok) {
    throw new Error(`Failed to download ${asset.id}: HTTP ${response.status}`);
  }
  await storeCachedResponse(cacheKey, response);
  return {
    cacheHit: false,
    cacheKey,
    response: response.clone(),
  };
}

export async function readAssetArrayBuffer(assetResult: AssetFetchResult): Promise<ArrayBuffer> {
  return assetResult.response.arrayBuffer();
}

export async function readAssetText(assetResult: AssetFetchResult): Promise<string> {
  return assetResult.response.text();
}

export function summarizeAssetResult(
  asset: ResourceAsset,
  assetResult: AssetFetchResult,
  byteLength = 0,
): AssetDownloadSummary {
  return {
    id: asset.id,
    url: asset.url,
    cacheHit: assetResult.cacheHit,
    bytes: byteLength,
  };
}
