import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ensureServedFromHttp,
  sourceToImageBitmap,
  sourceToMat,
  sourceToWorkerPayload
} from "../src/platform/browser";

class FakeImageBitmap {
  constructor(width = 10, height = 20) {
    this.width = width;
    this.height = height;
    this.close = vi.fn();
  }
}

class FakeImageData {
  constructor(data, width, height) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
}

class FakeCanvasElement {}
class FakeImageElement {}

function stubCanvasDocument() {
  const drawImage = vi.fn();
  const putImageData = vi.fn();
  const canvas = {
    width: 0,
    height: 0,
    getContext: vi.fn(() => ({
      drawImage,
      putImageData
    }))
  };

  vi.stubGlobal("document", {
    createElement: vi.fn(() => canvas)
  });

  return {
    canvas,
    drawImage,
    putImageData
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("platform/browser", () => {
  it("rejects file:// origins", () => {
    vi.stubGlobal("location", { protocol: "file:" });

    expect(() => ensureServedFromHttp()).toThrow(/requires an HTTP\(S\) origin/i);
  });

  it("allows http origins", () => {
    vi.stubGlobal("location", { protocol: "https:" });

    expect(() => ensureServedFromHttp()).not.toThrow();
  });

  it("reuses existing ImageBitmap sources", async () => {
    vi.stubGlobal("ImageBitmap", FakeImageBitmap);
    const bitmap = new FakeImageBitmap();

    await expect(sourceToImageBitmap(bitmap)).resolves.toBe(bitmap);
  });

  it("converts Blob and canvas sources through createImageBitmap", async () => {
    const bitmap = new FakeImageBitmap();
    const createImageBitmap = vi.fn().mockResolvedValue(bitmap);

    vi.stubGlobal("ImageBitmap", FakeImageBitmap);
    vi.stubGlobal("HTMLCanvasElement", FakeCanvasElement);
    vi.stubGlobal("createImageBitmap", createImageBitmap);

    const blob = new Blob(["hello"], { type: "text/plain" });
    await expect(sourceToImageBitmap(blob)).resolves.toBe(bitmap);

    const canvas = new FakeCanvasElement();
    await expect(sourceToImageBitmap(canvas)).resolves.toBe(bitmap);
    expect(createImageBitmap).toHaveBeenCalledTimes(2);
  });

  it("converts ImageData and img sources through createImageBitmap", async () => {
    const bitmap = new FakeImageBitmap();
    const createImageBitmap = vi.fn().mockResolvedValue(bitmap);
    const { canvas, putImageData } = stubCanvasDocument();

    vi.stubGlobal("ImageBitmap", FakeImageBitmap);
    vi.stubGlobal("ImageData", FakeImageData);
    vi.stubGlobal("HTMLImageElement", FakeImageElement);
    vi.stubGlobal("createImageBitmap", createImageBitmap);

    const imageData = new FakeImageData(new Uint8ClampedArray(16), 2, 2);
    await expect(sourceToImageBitmap(imageData)).resolves.toBe(bitmap);
    expect(canvas.width).toBe(2);
    expect(canvas.height).toBe(2);
    expect(putImageData).toHaveBeenCalledWith(imageData, 0, 0);

    const image = new FakeImageElement();
    await expect(sourceToImageBitmap(image)).resolves.toBe(bitmap);
  });

  it("rejects unsupported image sources", async () => {
    vi.stubGlobal("ImageData", FakeImageData);

    await expect(sourceToImageBitmap({})).rejects.toThrow(/Unsupported image source/i);
  });

  it("converts bitmap-backed sources into cv.Mat wrappers", async () => {
    const bitmap = new FakeImageBitmap(64, 32);
    const { drawImage } = stubCanvasDocument();
    const mat = {
      deleted: false,
      delete() {
        this.deleted = true;
      }
    };

    vi.stubGlobal("ImageBitmap", FakeImageBitmap);

    const loaded = await sourceToMat(
      {
        imread: vi.fn(() => mat)
      },
      bitmap
    );

    expect(loaded.width).toBe(64);
    expect(loaded.height).toBe(32);
    expect(drawImage).toHaveBeenCalledWith(bitmap, 0, 0);

    loaded.dispose();
    expect(mat.deleted).toBe(true);
    expect(bitmap.close).toHaveBeenCalledTimes(1);
  });

  it("creates transferable worker payloads from ImageBitmap sources", async () => {
    const sourceBitmap = new FakeImageBitmap(10, 20);
    const clonedBitmap = new FakeImageBitmap(10, 20);
    const createImageBitmap = vi.fn().mockResolvedValue(clonedBitmap);

    vi.stubGlobal("ImageBitmap", FakeImageBitmap);
    vi.stubGlobal("createImageBitmap", createImageBitmap);

    const result = await sourceToWorkerPayload(sourceBitmap);

    expect(createImageBitmap).toHaveBeenCalledWith(sourceBitmap);
    expect(result).toEqual({
      payload: {
        kind: "imageBitmap",
        imageBitmap: clonedBitmap
      },
      transferables: [clonedBitmap]
    });
  });

  it("rejects worker payload creation when ImageBitmap support is missing", async () => {
    await expect(sourceToWorkerPayload(new Blob(["x"]))).rejects.toThrow(
      /Worker mode requires ImageBitmap support/i
    );
  });
});
