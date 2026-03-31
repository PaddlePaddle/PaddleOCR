import type { OpenCv, Mat } from "@techstark/opencv-js";
import type { InferenceSession, Tensor } from "onnxruntime-web";

import { assertStandardModelResources } from "../resources/standard-model";
import { createSession, getProviderCandidates, releaseSessions } from "../runtime/ort";
import type { OrtModule, WebGpuState, SessionState } from "../runtime/ort";
import { clamp, withTimeout } from "../utils/common";
import {
  boxScoreFast,
  getMiniBoxFromPoints,
  getTransformOp,
  parseInferenceConfigText,
  parseScaleValue,
  toBgrFloatCHWFromBgr,
  unclip,
} from "./common";
import type { Point2D, NormalizeConfig, DetBox } from "./common";
import type { OcrRuntimeParams } from "../pipelines/ocr/runtime-params";

export interface DetPostprocessConfig {
  thresh: number;
  boxThresh: number;
  maxCandidates: number;
  unclipRatio: number;
}

export interface DetModelConfig {
  resizeLong: number;
  normalize: NormalizeConfig;
  postprocess: DetPostprocessConfig;
  maxSideLimit: number;
}

export interface DetModel {
  kind: "det";
  config: DetModelConfig;
  readonly provider: string;
  detect(ctx: { cv: OpenCv; sourceMat: Mat; params: OcrRuntimeParams }): Promise<DetResult>;
  dispose(): Promise<void>;
}

export interface DetPreprocessResult {
  tensor: Tensor;
  srcW: number;
  srcH: number;
  dstW: number;
  dstH: number;
}

export interface DetResult {
  output: Tensor;
  prep: DetPreprocessResult;
  boxes: DetBox[];
}

const DET_BOX_MIN_SIZE = 3;

export const DEFAULT_DET_MODEL_PARSE_FALLBACKS: Readonly<Omit<DetModelConfig, "maxSideLimit">> = Object.freeze({
  resizeLong: 960,
  normalize: {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    scale: 1 / 255,
  },
  postprocess: {
    thresh: 0.3,
    boxThresh: 0.6,
    maxCandidates: 1000,
    unclipRatio: 1.5,
  },
});

export const DEFAULT_DET_RUNTIME_LIMITS = Object.freeze({
  maxSideLimit: 4000,
});

export const DEFAULT_DET_MODEL_CONFIG: Readonly<DetModelConfig> = Object.freeze({
  ...DEFAULT_DET_MODEL_PARSE_FALLBACKS,
  maxSideLimit: DEFAULT_DET_RUNTIME_LIMITS.maxSideLimit,
});

export function parseDetModelConfigText(text: string): DetModelConfig {
  const parsed = parseInferenceConfigText(text);
  const preProcess = parsed.PreProcess as Record<string, unknown> | undefined;
  const transformOps = preProcess?.transform_ops as Array<Record<string, unknown>> | undefined;
  const resize = getTransformOp(transformOps, "DetResizeForTest");
  const normalize = getTransformOp(transformOps, "NormalizeImage");
  const postprocess = (parsed.PostProcess || {}) as Record<string, unknown>;

  return {
    resizeLong: Number(resize?.resize_long ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.resizeLong),
    normalize: {
      mean: (normalize?.mean as number[] | undefined) ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.normalize.mean,
      std: (normalize?.std as number[] | undefined) ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.normalize.std,
      scale: parseScaleValue(normalize?.scale, DEFAULT_DET_MODEL_PARSE_FALLBACKS.normalize.scale),
    },
    postprocess: {
      thresh: Number(postprocess.thresh ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.thresh),
      boxThresh: Number(
        postprocess.box_thresh ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.boxThresh,
      ),
      maxCandidates: Number(
        postprocess.max_candidates ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.maxCandidates,
      ),
      unclipRatio: Number(
        postprocess.unclip_ratio ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.unclipRatio,
      ),
    },
    maxSideLimit: DEFAULT_DET_RUNTIME_LIMITS.maxSideLimit,
  };
}

interface CreateDetModelArgs {
  ort: OrtModule;
  modelBytes: Uint8Array;
  configText: string;
  backend: string;
  webgpuState: WebGpuState;
}

export async function createDetModel({ ort, modelBytes, configText, backend, webgpuState }: CreateDetModelArgs): Promise<DetModel> {
  assertStandardModelResources("Detection", {
    model: modelBytes,
    config: configText,
  });
  const config = parseDetModelConfigText(configText);
  let sessionState: SessionState | null = await createDetModelSession(ort, modelBytes, backend, webgpuState);

  return {
    kind: "det",
    config,
    get provider() {
      return sessionState?.provider || "";
    },
    async detect({ cv, sourceMat, params }) {
      if (!sessionState?.session) {
        throw new Error("Detection model session is not initialized.");
      }
      return runDetModel(
        {
          cv,
          ort,
          config,
          session: sessionState.session,
        },
        sourceMat,
        params,
      );
    },
    async dispose() {
      await releaseSessions(sessionState?.session);
      sessionState = null;
    },
  };
}

export async function createDetModelSession(
  ort: OrtModule,
  modelBytes: Uint8Array,
  backend: string,
  webgpuState: WebGpuState,
): Promise<SessionState> {
  const providerCandidates = getProviderCandidates(backend, webgpuState);
  return withTimeout(createSession(ort, modelBytes, providerCandidates), 60000, "Detection model");
}

interface DetContext {
  cv: OpenCv;
  ort: OrtModule;
  config: DetModelConfig;
}

interface DetRunContext extends DetContext {
  session: InferenceSession;
}

export function preprocessDet(context: DetContext, sourceMat: Mat, params: OcrRuntimeParams): DetPreprocessResult {
  const { cv, ort, config } = context;
  const srcW = sourceMat.cols;
  const srcH = sourceMat.rows;
  const limitSideLen = Math.max(32, params.text_det_limit_side_len || config.resizeLong);
  const limitType = params.text_det_limit_type === "min" ? "min" : "max";
  const maxSideLimit = Math.max(32, params.text_det_max_side_limit || config.maxSideLimit);
  let scale = 1.0;
  if (limitType === "max") {
    const maxSide = Math.max(srcW, srcH);
    if (maxSide > limitSideLen) {
      scale = limitSideLen / Math.max(1, maxSide);
    }
  } else {
    const minSide = Math.min(srcW, srcH);
    if (minSide < limitSideLen) {
      scale = limitSideLen / Math.max(1, minSide);
    }
  }
  let dstW = Math.max(32, Math.round((srcW * scale) / 32) * 32);
  let dstH = Math.max(32, Math.round((srcH * scale) / 32) * 32);
  if (Math.max(dstW, dstH) > maxSideLimit) {
    const limitScale = maxSideLimit / Math.max(dstW, dstH);
    dstW = Math.max(32, Math.floor(dstW * limitScale));
    dstH = Math.max(32, Math.floor(dstH * limitScale));
  }
  dstW = clamp(dstW, 32, maxSideLimit);
  dstH = clamp(dstH, 32, maxSideLimit);
  dstW = Math.max(32, Math.round(dstW / 32) * 32);
  dstH = Math.max(32, Math.round(dstH / 32) * 32);

  const resized = new cv.Mat();
  const bgr = new cv.Mat();
  cv.resize(sourceMat, resized, new cv.Size(dstW, dstH), 0, 0, cv.INTER_LINEAR);
  if (resized.channels() === 4) {
    cv.cvtColor(resized, bgr, cv.COLOR_RGBA2BGR);
  } else if (resized.channels() === 1) {
    cv.cvtColor(resized, bgr, cv.COLOR_GRAY2BGR);
  } else {
    resized.copyTo(bgr);
  }
  const chw = toBgrFloatCHWFromBgr(bgr.data, dstW, dstH, config.normalize);
  resized.delete();
  bgr.delete();

  return {
    tensor: new ort.Tensor("float32", chw, [1, 3, dstH, dstW]),
    srcW,
    srcH,
    dstW,
    dstH,
  };
}

function getDetMap(outputTensor: Tensor): { data: Float32Array; h: number; w: number } {
  const dims = outputTensor.dims;
  const data = outputTensor.data as Float32Array;
  if (dims.length === 4) return { data, h: dims[2], w: dims[3] };
  if (dims.length === 3) return { data, h: dims[1], w: dims[2] };
  throw new Error(`Unexpected det output dims: [${dims.join(", ")}]`);
}

export async function runDetModel(
  context: DetRunContext,
  sourceMat: Mat,
  params: OcrRuntimeParams,
): Promise<DetResult> {
  const { cv, ort, config, session } = context;
  const prep = preprocessDet({ cv, ort, config }, sourceMat, params);
  const inputName = session.inputNames[0];
  const outputMap = await session.run({ [inputName]: prep.tensor });
  const output = outputMap[session.outputNames[0]];
  return {
    output,
    prep,
    boxes: dbPostprocess(
      { cv, config },
      output,
      prep,
      params.text_det_thresh,
      params.text_det_box_thresh,
      params.text_det_unclip_ratio,
    ),
  };
}

export function dbPostprocess(
  context: { cv: OpenCv; config: DetModelConfig },
  detOutput: Tensor,
  meta: DetPreprocessResult,
  detThresh: number,
  boxThresh: number,
  unclipRatio: number,
): DetBox[] {
  const { cv, config } = context;
  const { data, h, w } = getDetMap(detOutput);
  const pred = cv.matFromArray(h, w, cv.CV_32FC1, data);
  const maskData = new Uint8Array(h * w);
  for (let i = 0; i < data.length; i += 1) {
    maskData[i] = data[i] > detThresh ? 255 : 0;
  }
  const bitmap = cv.matFromArray(h, w, cv.CV_8UC1, maskData);
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(bitmap, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  const boxes: DetBox[] = [];
  const candidateCount = Math.min(config.postprocess.maxCandidates, contours.size());
  for (let i = 0; i < candidateCount; i += 1) {
    const contour = contours.get(i);
    if (contour.rows < 4) {
      contour.delete();
      continue;
    }
    const points: Point2D[] = [];
    for (let row = 0; row < contour.rows; row += 1) {
      points.push([contour.data32S[row * 2], contour.data32S[row * 2 + 1]]);
    }
    const mini = getMiniBoxFromPoints(cv, points);
    if (mini.side < DET_BOX_MIN_SIZE) {
      contour.delete();
      continue;
    }
    const score = boxScoreFast(cv, pred, mini.box);
    if (score < boxThresh) {
      contour.delete();
      continue;
    }
    const expanded = unclip(mini.box, unclipRatio);
    if (!expanded || expanded.length < 4) {
      contour.delete();
      continue;
    }
    const miniUnclip = getMiniBoxFromPoints(cv, expanded);
    if (miniUnclip.side < DET_BOX_MIN_SIZE + 2) {
      contour.delete();
      continue;
    }

    const poly: Point2D[] = miniUnclip.box.map((point) => [
      clamp(Math.round((point[0] * meta.srcW) / Math.max(1, w)), 0, meta.srcW),
      clamp(Math.round((point[1] * meta.srcH) / Math.max(1, h)), 0, meta.srcH),
    ]);
    boxes.push({ poly, score });
    contour.delete();
  }

  pred.delete();
  bitmap.delete();
  contours.delete();
  hierarchy.delete();

  boxes.sort((a, b) => a.poly[0][1] - b.poly[0][1] || a.poly[0][0] - b.poly[0][0]);
  for (let i = 0; i < boxes.length - 1; i += 1) {
    for (let j = i; j >= 0; j -= 1) {
      if (
        Math.abs(boxes[j + 1].poly[0][1] - boxes[j].poly[0][1]) < 10 &&
        boxes[j + 1].poly[0][0] < boxes[j].poly[0][0]
      ) {
        const tmp = boxes[j];
        boxes[j] = boxes[j + 1];
        boxes[j + 1] = tmp;
      } else {
        break;
      }
    }
  }

  return boxes;
}

export function cropByPoly(cv: OpenCv, srcMat: Mat, poly: Point2D[]): Mat {
  const ordered = getMiniBoxFromPoints(cv, poly).box;
  const widthTop = Math.hypot(ordered[1][0] - ordered[0][0], ordered[1][1] - ordered[0][1]);
  const widthBottom = Math.hypot(ordered[2][0] - ordered[3][0], ordered[2][1] - ordered[3][1]);
  const heightLeft = Math.hypot(ordered[3][0] - ordered[0][0], ordered[3][1] - ordered[0][1]);
  const heightRight = Math.hypot(ordered[2][0] - ordered[1][0], ordered[2][1] - ordered[1][1]);
  const cropW = Math.max(1, Math.floor(Math.max(widthTop, widthBottom)));
  const cropH = Math.max(1, Math.floor(Math.max(heightLeft, heightRight)));

  const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
    ordered[0][0],
    ordered[0][1],
    ordered[1][0],
    ordered[1][1],
    ordered[2][0],
    ordered[2][1],
    ordered[3][0],
    ordered[3][1],
  ]);
  const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, cropW, 0, cropW, cropH, 0, cropH]);
  const transform = cv.getPerspectiveTransform(srcTri, dstTri);
  const warped = new cv.Mat();
  cv.warpPerspective(
    srcMat,
    warped,
    transform,
    new cv.Size(cropW, cropH),
    cv.INTER_CUBIC,
    cv.BORDER_REPLICATE,
    new cv.Scalar(),
  );
  srcTri.delete();
  dstTri.delete();
  transform.delete();

  if (warped.rows / Math.max(1, warped.cols) >= 1.5) {
    const rotated = new cv.Mat();
    cv.rotate(warped, rotated, cv.ROTATE_90_COUNTERCLOCKWISE);
    warped.delete();
    return rotated;
  }
  return warped;
}
