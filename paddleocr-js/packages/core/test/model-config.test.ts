import { describe, expect, it } from "vitest";

import { parseDetModelConfigText, parseRecModelConfigText } from "../src/models/index";

const detConfig = `
PreProcess:
  transform_ops:
    - DetResizeForTest:
        resize_long: 736
        limit_type: max
        max_side_limit: 4500
    - NormalizeImage:
        mean: [0.1, 0.2, 0.3]
        std: [0.9, 0.8, 0.7]
        scale: 1./255.
PostProcess:
  thresh: 0.22
  box_thresh: 0.55
  max_candidates: 200
  unclip_ratio: 1.8
`;

const recConfig = `
PreProcess:
  transform_ops:
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - NormalizeImage:
        mean: [0.5, 0.5, 0.5]
        std: [0.5, 0.5, 0.5]
        scale: 1./255.
PostProcess:
  character_dict:
    - a
    - b
    - c
`;

describe("model config parsers", () => {
  it("parses detection model config text", () => {
    const config = parseDetModelConfigText(detConfig);

    expect(config.resizeLong).toBe(736);
    expect(config.limitType).toBe("max");
    expect(config.maxSideLimit).toBe(4500);
    expect(config.postprocess.thresh).toBe(0.22);
    expect(config.postprocess.boxThresh).toBe(0.55);
  });

  it("parses recognition model config text", () => {
    const config = parseRecModelConfigText(recConfig);

    expect(config.imageShape).toEqual([3, 48, 320]);
    expect(config.charDict.slice(0, 3)).toEqual(["a", "b", "c"]);
    expect(config.charDict.at(-1)).toBe(" ");
  });
});
