import { afterEach, describe, expect, it, vi } from "vitest";

import {
  fetchResourceAsset,
  readAssetArrayBuffer,
  readAssetText,
  summarizeAssetResult
} from "../src/resources/cache.js";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("resource cache", () => {
  it("falls back to in-memory caching when Cache Storage is unavailable", async () => {
    let requestCount = 0;
    const fetchImpl = async () => {
      requestCount += 1;
      return new Response("demo", { status: 200 });
    };
    const asset = {
      id: "demo-asset",
      url: "https://example.com/demo.bin",
      version: "v1"
    };

    const first = await fetchResourceAsset(asset, fetchImpl);
    const second = await fetchResourceAsset(asset, fetchImpl);

    expect(first.cacheHit).toBe(false);
    expect(second.cacheHit).toBe(true);
    expect(requestCount).toBe(1);
  });

  it("does not share cache entries across assets with the same id but different urls", async () => {
    let requestCount = 0;
    const fetchImpl = async (url) => {
      requestCount += 1;
      return new Response(url, { status: 200 });
    };
    const firstAsset = {
      id: "shared-id",
      url: "https://example.com/one.bin",
      version: "v1"
    };
    const secondAsset = {
      id: "shared-id",
      url: "https://example.com/two.bin",
      version: "v1"
    };

    const first = await fetchResourceAsset(firstAsset, fetchImpl);
    const second = await fetchResourceAsset(secondAsset, fetchImpl);

    expect(first.cacheHit).toBe(false);
    expect(second.cacheHit).toBe(false);
    expect(requestCount).toBe(2);
  });

  it("uses Cache Storage when available on the web", async () => {
    const persistentResponse = new Response("cached", { status: 200 });
    const match = vi.fn().mockResolvedValueOnce(null).mockResolvedValueOnce(persistentResponse.clone());
    const put = vi.fn().mockResolvedValue(undefined);
    const open = vi.fn().mockResolvedValue({ match, put });
    const fetchImpl = vi.fn(async () => new Response("fresh", { status: 200 }));
    vi.stubGlobal("location", { origin: "https://example.com" });
    vi.stubGlobal("caches", { open });

    const asset = {
      id: "persistent-asset",
      url: "https://example.com/model.bin",
      version: "v1"
    };

    const first = await fetchResourceAsset(asset, fetchImpl);
    const second = await fetchResourceAsset(asset, fetchImpl);

    expect(first.cacheHit).toBe(false);
    expect(second.cacheHit).toBe(true);
    expect(open).toHaveBeenCalled();
    expect(put).toHaveBeenCalledTimes(1);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("rejects failed downloads and exposes asset readers", async () => {
    const asset = {
      id: "broken-asset",
      url: "https://example.com/broken.bin"
    };

    await expect(
      fetchResourceAsset(asset, async () => new Response("nope", { status: 404 }))
    ).rejects.toThrow("Failed to download broken-asset: HTTP 404");

    const response = new Response("hello", { status: 200 });
    const assetResult = {
      cacheHit: false,
      response
    };
    expect(new Uint8Array(await readAssetArrayBuffer(assetResult))).toEqual(
      new Uint8Array([104, 101, 108, 108, 111])
    );
    await expect(readAssetText({ response: new Response("text", { status: 200 }) })).resolves.toBe("text");
    expect(summarizeAssetResult(asset, { cacheHit: true }, 5)).toEqual({
      id: "broken-asset",
      url: "https://example.com/broken.bin",
      cacheHit: true,
      bytes: 5
    });
  });
});
