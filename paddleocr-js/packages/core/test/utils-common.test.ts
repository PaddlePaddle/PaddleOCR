import { describe, expect, it, vi } from "vitest";

import {
  clamp,
  deepClone,
  distance2,
  formatMs,
  nowMs,
  resolveRuntimeBatchSize,
  withTimeout
} from "../src/utils/common";

describe("utils/common", () => {
  it("reads the current high-resolution timestamp", () => {
    const spy = vi.spyOn(performance, "now").mockReturnValue(123.45);

    expect(nowMs()).toBe(123.45);

    spy.mockRestore();
  });

  it("clamps values into the requested range", () => {
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(15, 0, 10)).toBe(10);
  });

  it("computes Euclidean distance between two points", () => {
    expect(distance2([0, 0], [3, 4])).toBe(5);
  });

  it("formats millisecond durations with one decimal place", () => {
    expect(formatMs(12.34)).toBe("12.3 ms");
  });

  it("resolveRuntimeBatchSize uses default when override is undefined", () => {
    expect(resolveRuntimeBatchSize(undefined, 4)).toBe(4);
  });

  it("resolveRuntimeBatchSize coerces numeric strings and enforces a minimum of 1", () => {
    expect(resolveRuntimeBatchSize("3", 1)).toBe(3);
    expect(resolveRuntimeBatchSize(0, 2)).toBe(1);
    expect(resolveRuntimeBatchSize(-1, 2)).toBe(1);
    expect(resolveRuntimeBatchSize(Number.NaN, 2)).toBe(1);
  });

  it("resolves before the timeout when the promise settles in time", async () => {
    await expect(withTimeout(Promise.resolve("ok"), 50, "load")).resolves.toBe("ok");
  });

  it("rejects with the original error before the timeout elapses", async () => {
    const error = new Error("boom");

    await expect(withTimeout(Promise.reject(error), 50, "load")).rejects.toBe(error);
  });

  it("rejects when the timeout elapses first", async () => {
    await expect(withTimeout(new Promise(() => {}), 1, "load")).rejects.toThrow(
      "load timed out after 0.001s"
    );
  });

  it("deep clones structured data", () => {
    const original = {
      nested: {
        values: [1, 2, 3]
      }
    };

    const cloned = deepClone(original);
    cloned.nested.values.push(4);

    expect(cloned).toEqual({
      nested: {
        values: [1, 2, 3, 4]
      }
    });
    expect(original).toEqual({
      nested: {
        values: [1, 2, 3]
      }
    });
  });
});
