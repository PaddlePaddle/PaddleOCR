import { describe, expect, it, vi } from "vitest";

import { loadModelAsset } from "../src/resources/model-asset";
import { createTar } from "./tar-fixture";

describe("model asset loading", () => {
  it("resolves model tar assets", async () => {
    const asset = {
      url: "/det.tar"
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

    const resolved = await loadModelAsset(asset, fetchImpl);

    expect(Array.from(resolved.modelBytes)).toEqual([57, 56, 55]);
    expect(resolved.configText).toBe("name: det");
    expect(resolved.download.url).toBe("/det.tar");
  });

  it("uses standard inference entry names", async () => {
    const asset = {
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

    const resolved = await loadModelAsset(asset, fetchImpl);

    expect(Array.from(resolved.modelBytes)).toEqual([1, 2, 3]);
    expect(resolved.configText).toBe("name: config");
  });
});
