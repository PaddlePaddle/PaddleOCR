import { describe, expect, it } from "vitest";
import { deterministicColor } from "../src/viz/color";

describe("viz/color", () => {
  it("returns an RGB tuple of three integers 0-255", () => {
    const [r, g, b] = deterministicColor(0);
    expect(Number.isInteger(r)).toBe(true);
    expect(Number.isInteger(g)).toBe(true);
    expect(Number.isInteger(b)).toBe(true);
    expect(r).toBeGreaterThanOrEqual(0);
    expect(r).toBeLessThanOrEqual(255);
    expect(g).toBeGreaterThanOrEqual(0);
    expect(g).toBeLessThanOrEqual(255);
    expect(b).toBeGreaterThanOrEqual(0);
    expect(b).toBeLessThanOrEqual(255);
  });

  it("produces the same color for the same index", () => {
    expect(deterministicColor(5)).toEqual(deterministicColor(5));
    expect(deterministicColor(42)).toEqual(deterministicColor(42));
  });

  it("produces different colors for different indices", () => {
    const c0 = deterministicColor(0);
    const c1 = deterministicColor(1);
    const c2 = deterministicColor(2);
    const allSame =
      JSON.stringify(c0) === JSON.stringify(c1) && JSON.stringify(c1) === JSON.stringify(c2);
    expect(allSame).toBe(false);
  });

  it("matches the exact values from the demo app's LCG", () => {
    const c0 = deterministicColor(0);
    const c1 = deterministicColor(1);
    // Verify determinism by snapshot
    expect(c0).toMatchInlineSnapshot(`
      [
        198,
        126,
        223,
      ]
    `);
    expect(c1).toMatchInlineSnapshot(`
      [
        140,
        33,
        85,
      ]
    `);
  });
});
