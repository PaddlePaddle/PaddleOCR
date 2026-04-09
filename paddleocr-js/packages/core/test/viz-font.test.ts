import { describe, expect, it, vi, beforeEach } from "vitest";
import { loadFontFace, removeFontFace } from "../src/viz/font";
import type { FontConfig } from "../src/viz/types";

describe("viz/font", () => {
  let mockFontFace: { load: ReturnType<typeof vi.fn>; family: string };

  beforeEach(() => {
    mockFontFace = {
      load: vi.fn().mockResolvedValue(undefined),
      family: ""
    };
    (globalThis as Record<string, unknown>).FontFace = vi.fn((family: string) => {
      mockFontFace.family = family;
      return mockFontFace;
    });
    if (!document.fonts) {
      Object.defineProperty(document, "fonts", {
        value: { add: vi.fn(), delete: vi.fn() },
        configurable: true
      });
    } else {
      vi.spyOn(document.fonts, "add").mockImplementation(() => {});
      vi.spyOn(document.fonts, "delete").mockImplementation(() => true);
    }
  });

  it("loads a font from a URL string and adds to document.fonts", async () => {
    const config: FontConfig = {
      family: "TestFont",
      source: "https://example.com/font.woff2"
    };

    const face = await loadFontFace(config);
    expect(globalThis.FontFace).toHaveBeenCalledWith(
      "TestFont",
      "url(https://example.com/font.woff2)",
      undefined
    );
    expect(mockFontFace.load).toHaveBeenCalled();
    expect(document.fonts.add).toHaveBeenCalledWith(face);
  });

  it("loads a font from an ArrayBuffer source", async () => {
    const buffer = new ArrayBuffer(8);
    const config: FontConfig = {
      family: "BufFont",
      source: buffer
    };

    await loadFontFace(config);
    expect(globalThis.FontFace).toHaveBeenCalledWith("BufFont", buffer, undefined);
  });

  it("passes descriptors to FontFace constructor", async () => {
    const config: FontConfig = {
      family: "DescFont",
      source: "https://example.com/font.woff2",
      descriptors: { weight: "bold" }
    };

    await loadFontFace(config);
    expect(globalThis.FontFace).toHaveBeenCalledWith(
      "DescFont",
      "url(https://example.com/font.woff2)",
      { weight: "bold" }
    );
  });

  it("removes a font face from document.fonts", () => {
    removeFontFace(mockFontFace as unknown as FontFace);
    expect(document.fonts.delete).toHaveBeenCalledWith(mockFontFace);
  });
});
