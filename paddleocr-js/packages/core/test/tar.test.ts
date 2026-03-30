import { describe, expect, it } from "vitest";

import { extractTarEntries, pickTarEntry } from "../src/resources/tar";
import { createTar } from "./tar-fixture";

describe("tar helpers", () => {
  it("extracts nested inference files from a tar archive", () => {
    const tarBuffer = createTar([
      { name: "det/inference.onnx", content: "onnx" },
      { name: "det/inference.yml", content: "yaml" }
    ]);

    const entries = extractTarEntries(tarBuffer);

    expect(new TextDecoder().decode(pickTarEntry(entries, "inference.onnx"))).toBe("onnx");
    expect(new TextDecoder().decode(pickTarEntry(entries, "inference.yml"))).toBe("yaml");
  });

  it("ignores AppleDouble and PaxHeader metadata entries", () => {
    const tarBuffer = createTar([
      { name: "._det", content: "metadata" },
      { name: "PaxHeader/det", content: "metadata" },
      { name: "det/._inference.onnx", content: "metadata" },
      { name: "det/PaxHeader/inference.onnx", content: "metadata" },
      { name: "det/inference.onnx", content: "onnx" },
      { name: "det/._inference.yml", content: "metadata" },
      { name: "det/PaxHeader/inference.yml", content: "metadata" },
      { name: "det/inference.yml", content: "yaml" }
    ]);

    const entries = extractTarEntries(tarBuffer);

    expect([...entries.keys()]).toEqual(["det/inference.onnx", "det/inference.yml"]);
    expect(new TextDecoder().decode(pickTarEntry(entries, "inference.onnx"))).toBe("onnx");
    expect(new TextDecoder().decode(pickTarEntry(entries, "inference.yml"))).toBe("yaml");
  });
});
