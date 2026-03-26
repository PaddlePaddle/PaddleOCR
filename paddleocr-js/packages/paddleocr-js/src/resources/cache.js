const CACHE_NAME = "paddleocr-js-model-cache";
const memoryCache = new Map();

function getCacheKey(asset) {
  const cacheVersion = asset.version || "latest";
  if (typeof location === "undefined") {
    return `memory://${asset.id}?version=${encodeURIComponent(cacheVersion)}&url=${encodeURIComponent(asset.url)}`;
  }
  const url = new URL(`/__paddleocr_js_models__/${encodeURIComponent(asset.id)}`, location.origin);
  url.searchParams.set("version", cacheVersion);
  url.searchParams.set("url", asset.url);
  return url.toString();
}

async function getPersistentCache() {
  if (!("caches" in globalThis)) return null;
  return caches.open(CACHE_NAME);
}

async function readCachedResponse(key) {
  const persistentCache = await getPersistentCache();
  if (persistentCache) {
    const response = await persistentCache.match(key);
    if (response) return response;
  }
  return memoryCache.get(key)?.clone() || null;
}

async function storeCachedResponse(key, response) {
  const persistentCache = await getPersistentCache();
  if (persistentCache) {
    await persistentCache.put(key, response.clone());
  }
  memoryCache.set(key, response.clone());
}

export async function fetchResourceAsset(asset, fetchImpl = fetch) {
  const cacheKey = getCacheKey(asset);
  const cachedResponse = await readCachedResponse(cacheKey);
  if (cachedResponse) {
    return {
      cacheHit: true,
      cacheKey,
      response: cachedResponse
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
    response: response.clone()
  };
}

export async function readAssetArrayBuffer(assetResult) {
  return assetResult.response.arrayBuffer();
}

export async function readAssetText(assetResult) {
  return assetResult.response.text();
}

export function summarizeAssetResult(asset, assetResult, byteLength = 0) {
  return {
    id: asset.id,
    url: asset.url,
    cacheHit: assetResult.cacheHit,
    bytes: byteLength
  };
}
