import { describe, expect, it, vi } from "vitest";

import { loadStandardModelAsset } from "../src/resources/standard-model";
import { createTar } from "./tar-fixture";

describe("standard model asset resolution", () => {
  it("resolves standard model tar assets", async () => {
    const asset = {
      id: "det",
      kind: "tar",
      url: "/det.tar",
      entries: {
        model: "inference.onnx",
        config: "inference.yml"
      }
    };

    const fetchImpl = vi.fn(async (url) => {
      if (url.endsWith("det.tar")) {
        return new Response(
          createTar([
            { name: "nested/inference.onnx", content: new Uint8Array([57, 56, 55]) },
            { name: "nested/inference.yml", content: "name: det" }
          ])
        );
      }
      return new Response(
        createTar([
          { name: "nested/inference.onnx", content: new Uint8Array([49, 50, 51]) },
          { name: "nested/inference.yml", content: "name: det" }
        ])
      );
    });

    const resolved = await loadStandardModelAsset(asset, fetchImpl);

    expect(Array.from(resolved.modelBytes)).toEqual([57, 56, 55]);
    expect(resolved.configText).toBe("name: det");
    expect(resolved.download.id).toBe("det");
  });

  it("uses standard inference entry names when tar entries are omitted", async () => {
    const asset = {
      id: "det-default",
      kind: "tar",
      url: "/det-default.tar"
    };

    const fetchImpl = vi.fn(
      async () =>
        new Response(
          createTar([
            { name: "nested/inference.onnx", content: new Uint8Array([1, 2, 3]) },
            { name: "nested/inference.yml", content: "name: config" }
          ])
        )
    );

    const resolved = await loadStandardModelAsset(asset, fetchImpl);

    expect(Array.from(resolved.modelBytes)).toEqual([1, 2, 3]);
    expect(resolved.configText).toBe("name: config");
  });

  it("uses asset-level version for cache invalidation", async () => {
    const fetchImpl = vi.fn(async () => new Response(new Uint8Array([1, 2, 3])));

    const createAsset = (version) => ({
      id: "encoder",
      kind: "tar",
      url: "/encoder.tar",
      version,
      entries: {
        model: "inference.onnx",
        config: "inference.yml"
      }
    });
    const response = () =>
      new Response(
        createTar([
          { name: "inference.onnx", content: new Uint8Array([1, 2, 3]) },
          { name: "inference.yml", content: "name: config" }
        ])
      );

    fetchImpl.mockImplementation(async () => response());

    await loadStandardModelAsset(createAsset("v1"), fetchImpl);
    await loadStandardModelAsset(createAsset("v1"), fetchImpl);
    await loadStandardModelAsset(createAsset("v2"), fetchImpl);

    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});
