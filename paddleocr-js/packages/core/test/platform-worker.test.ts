import { afterEach, describe, expect, it, vi } from "vitest";

import { ensureServedFromHttp, sourcePayloadToMat } from "../src/platform/worker";

class FakeMat {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.deleted = false;
  }

  clone() {
    return new FakeMat(this.rows, this.cols);
  }

  delete() {
    this.deleted = true;
  }
}

class FakeImageBitmap {
  constructor(width = 8, height = 6) {
    this.width = width;
    this.height = height;
    this.close = vi.fn();
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("platform/worker", () => {
  it("re-exports the HTTP origin guard", () => {
    vi.stubGlobal("location", { protocol: "file:" });

    expect(() => ensureServedFromHttp()).toThrow(/requires an HTTP\(S\) origin/i);
  });

  it("accepts cv.Mat payloads by cloning them", () => {
    const source = new FakeMat(32, 64);
    const loaded = sourcePayloadToMat({ Mat: FakeMat } as any, source);

    expect(loaded.width).toBe(64);
    expect(loaded.height).toBe(32);
    expect(loaded.mat).toBeInstanceOf(FakeMat);
    expect(loaded.mat).not.toBe(source);

    loaded.dispose();
    expect(loaded.mat.deleted).toBe(true);
    expect(source.deleted).toBe(false);
  });

  it("converts imageBitmap payloads into cv mats", () => {
    const imageBitmap = new FakeImageBitmap(20, 10);
    const getImageData = vi.fn(() => ({
      width: 20,
      height: 10,
      data: new Uint8ClampedArray(20 * 10 * 4)
    }));
    const drawImage = vi.fn();
    const mat = {
      deleted: false,
      delete() {
        this.deleted = true;
      }
    };

    vi.stubGlobal("ImageBitmap", FakeImageBitmap);
    vi.stubGlobal(
      "OffscreenCanvas",
      class {
        constructor(width, height) {
          this.width = width;
          this.height = height;
        }

        getContext() {
          return {
            drawImage,
            getImageData
          };
        }
      }
    );

    const loaded = sourcePayloadToMat(
      {
        Mat: FakeMat,
        CV_8UC4: "rgba",
        matFromArray: vi.fn(() => mat)
      } as any,
      {
        kind: "imageBitmap",
        imageBitmap
      }
    );

    expect(drawImage).toHaveBeenCalledWith(imageBitmap, 0, 0);
    expect(loaded.width).toBe(20);
    expect(loaded.height).toBe(10);

    loaded.dispose();
    expect(mat.deleted).toBe(true);
    expect(imageBitmap.close).toHaveBeenCalledTimes(1);
  });

  it("rejects imageBitmap payloads when OffscreenCanvas is unavailable", () => {
    vi.stubGlobal("ImageBitmap", FakeImageBitmap);

    expect(() =>
      sourcePayloadToMat({ Mat: FakeMat } as any, {
        kind: "imageBitmap",
        imageBitmap: new FakeImageBitmap()
      })
    ).toThrow(/requires OffscreenCanvas support/i);
  });

  it("rejects imageBitmap payloads when the canvas context cannot be created", () => {
    vi.stubGlobal("ImageBitmap", FakeImageBitmap);
    vi.stubGlobal(
      "OffscreenCanvas",
      class {
        getContext() {
          return null;
        }
      }
    );

    expect(() =>
      sourcePayloadToMat({ Mat: FakeMat } as any, {
        kind: "imageBitmap",
        imageBitmap: new FakeImageBitmap()
      })
    ).toThrow(/Failed to create a 2D canvas context/i);
  });

  it("rejects unsupported worker payloads", () => {
    expect(() => sourcePayloadToMat({ Mat: FakeMat } as any, { kind: "other" })).toThrow(
      /Unsupported worker image source payload/i
    );
  });
});
