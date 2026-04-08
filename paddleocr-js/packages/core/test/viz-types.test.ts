import { describe, expect, it } from "vitest";
import type { FontConfig, RgbColor } from "../src/viz/types";
import type { BoxStyleOptions, OcrVisualizerOptions } from "../src/viz/ocr/types";

describe("viz/types", () => {
  it("allows constructing a minimal OcrVisualizerOptions", () => {
    const opts: OcrVisualizerOptions = {};
    expect(opts).toEqual({});
  });

  it("allows constructing a full OcrVisualizerOptions", () => {
    const font: FontConfig = {
      family: "Test",
      source: "https://example.com/font.woff2"
    };
    const boxStyle: BoxStyleOptions = {
      lineWidth: 3,
      fillOpacity: 0.5,
      colorFn: (i: number): RgbColor => [i, i, i]
    };
    const opts: OcrVisualizerOptions = {
      font,
      boxStyle,
      textPanelBackground: "#f0f0f0",
      outputFormat: "jpeg",
      outputQuality: 0.8
    };
    expect(opts.font?.family).toBe("Test");
    expect(opts.boxStyle?.lineWidth).toBe(3);
    expect(opts.outputFormat).toBe("jpeg");
  });
});
