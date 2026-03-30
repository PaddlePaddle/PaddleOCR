import { describe, expect, it } from "vitest";

import { sourceToMat } from "../src/platform/browser";

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

describe("browser source helpers", () => {
  it("accepts cv.Mat inputs by cloning them", async () => {
    const source = new FakeMat(32, 64);
    const loaded = await sourceToMat({ Mat: FakeMat }, source);

    expect(loaded.width).toBe(64);
    expect(loaded.height).toBe(32);
    expect(loaded.mat).toBeInstanceOf(FakeMat);
    expect(loaded.mat).not.toBe(source);

    loaded.dispose();
    expect(loaded.mat.deleted).toBe(true);
    expect(source.deleted).toBe(false);
  });
});
