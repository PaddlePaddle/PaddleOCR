/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { OpenCv, Mat } from "@techstark/opencv-js";
import type { InferenceSession, Tensor } from "onnxruntime-web";

import { assertModelResources } from "../resources/model-asset";
import { createSession, getProviderCandidates, releaseSessions } from "../runtime/ort";
import type { OrtModule, WebGpuState, SessionState } from "../runtime/ort";
import { chunkArray, clamp, resolveRuntimeBatchSize, withTimeout } from "../utils/common";
import { runInference } from "./infer";
import {
  boxScoreFast,
  getMiniBoxFromPoints,
  getTransformOp,
  parseInferenceConfigText,
  parseScaleValue,
  toBgrFloatCHWFromBgr,
  unclip
} from "./common";
import type { Point2D, NormalizeConfig, DetBox } from "./common";

export type LimitType = "min" | "max";

export interface DetRuntimeOverrides {
  batchSize?: number;
  thresh?: number;
  boxThresh?: number;
  unclipRatio?: number;
  limitSideLen?: number;
  limitType?: LimitType;
  maxSideLimit?: number;
}

export interface DetPostprocessConfig {
  thresh: number;
  boxThresh: number;
  maxCandidates: number;
  unclipRatio: number;
}

export interface DetModelConfig {
  resizeLong: number;
  limitType: LimitType;
  maxSideLimit: number;
  normalize: NormalizeConfig;
  postprocess: DetPostprocessConfig;
}

export interface DetModel {
  readonly kind: "det";
  readonly config: DetModelConfig;
  readonly provider: string;
  predict(cv: OpenCv, mats: Mat[], overrides?: DetRuntimeOverrides): Promise<DetResult[]>;
  dispose(): Promise<void>;
}

interface DetPreprocessResult {
  tensor: Tensor;
  srcW: number;
  srcH: number;
  dstW: number;
  dstH: number;
}

export interface DetResult {
  boxes: DetBox[];
  srcW: number;
  srcH: number;
}

interface InternalDetParams {
  limitSideLen: number;
  limitType: LimitType;
  maxSideLimit: number;
  thresh: number;
  boxThresh: number;
  unclipRatio: number;
}

interface InternalDetBatchItem {
  prep: DetPreprocessResult;
  boxes: DetBox[];
}

const DET_BOX_MIN_SIZE = 3;

export const DEFAULT_DET_MODEL_PARSE_FALLBACKS: Readonly<DetModelConfig> = Object.freeze({
  resizeLong: 960,
  limitType: "max",
  maxSideLimit: 4000,
  normalize: {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    scale: 1 / 255
  },
  postprocess: {
    thresh: 0.3,
    boxThresh: 0.6,
    maxCandidates: 1000,
    unclipRatio: 2.0
  }
});

export const DEFAULT_DET_MODEL_CONFIG: Readonly<DetModelConfig> = Object.freeze({
  ...DEFAULT_DET_MODEL_PARSE_FALLBACKS
});

function parseDetLimitType(raw: unknown): LimitType {
  const v = typeof raw === "string" ? raw.trim().toLowerCase() : "";
  if (v === "min" || v === "max") {
    return v;
  }
  return DEFAULT_DET_MODEL_PARSE_FALLBACKS.limitType;
}

export function parseDetModelConfigText(text: string): DetModelConfig {
  const parsed = parseInferenceConfigText(text);
  const preProcess = parsed.PreProcess as Record<string, unknown> | undefined;
  const transformOps = preProcess?.transform_ops as Array<Record<string, unknown>> | undefined;
  const resize = getTransformOp(transformOps, "DetResizeForTest");
  const normalize = getTransformOp(transformOps, "NormalizeImage");
  const postprocess = (parsed.PostProcess || {}) as Record<string, unknown>;

  const maxSideRaw = resize?.max_side_limit;
  const maxSideLimit = Number(maxSideRaw);
  const maxSide =
    Number.isFinite(maxSideLimit) && maxSideLimit > 0
      ? maxSideLimit
      : DEFAULT_DET_MODEL_PARSE_FALLBACKS.maxSideLimit;

  return {
    resizeLong: Number(resize?.resize_long ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.resizeLong),
    limitType: parseDetLimitType(resize?.limit_type),
    maxSideLimit: maxSide,
    normalize: {
      mean:
        (normalize?.mean as number[] | undefined) ??
        DEFAULT_DET_MODEL_PARSE_FALLBACKS.normalize.mean,
      std:
        (normalize?.std as number[] | undefined) ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.normalize.std,
      scale: parseScaleValue(normalize?.scale, DEFAULT_DET_MODEL_PARSE_FALLBACKS.normalize.scale)
    },
    postprocess: {
      thresh: Number(postprocess.thresh ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.thresh),
      boxThresh: Number(
        postprocess.box_thresh ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.boxThresh
      ),
      maxCandidates: Number(
        postprocess.max_candidates ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.maxCandidates
      ),
      unclipRatio: Number(
        postprocess.unclip_ratio ?? DEFAULT_DET_MODEL_PARSE_FALLBACKS.postprocess.unclipRatio
      )
    }
  };
}

interface CreateDetModelArgs {
  ort: OrtModule;
  modelBytes: Uint8Array;
  configText: string;
  backend: string;
  webgpuState: WebGpuState;
  batchSize?: number;
}

function resolveDetParams(
  defaults: InternalDetParams,
  overrides?: DetRuntimeOverrides
): InternalDetParams {
  return {
    limitSideLen: overrides?.limitSideLen ?? defaults.limitSideLen,
    limitType: overrides?.limitType ?? defaults.limitType,
    maxSideLimit: overrides?.maxSideLimit ?? defaults.maxSideLimit,
    thresh: overrides?.thresh ?? defaults.thresh,
    boxThresh: overrides?.boxThresh ?? defaults.boxThresh,
    unclipRatio: overrides?.unclipRatio ?? defaults.unclipRatio
  };
}

export async function createDetModel({
  ort,
  modelBytes,
  configText,
  backend,
  webgpuState,
  batchSize: batchSizeArg
}: CreateDetModelArgs): Promise<DetModel> {
  assertModelResources("Detection", {
    model: modelBytes,
    config: configText
  });
  const config = parseDetModelConfigText(configText);
  const defaultBatchSize = Math.max(1, batchSizeArg ?? 1);
  const defaultParams: InternalDetParams = {
    limitSideLen: config.resizeLong,
    limitType: config.limitType,
    maxSideLimit: config.maxSideLimit,
    thresh: config.postprocess.thresh,
    boxThresh: config.postprocess.boxThresh,
    unclipRatio: config.postprocess.unclipRatio
  };
  let sessionState: SessionState | null = await createDetModelSession(
    ort,
    modelBytes,
    backend,
    webgpuState
  );

  return {
    kind: "det",
    config,
    get provider() {
      return sessionState?.provider || "";
    },
    async predict(cv, mats, overrides) {
      if (!sessionState?.session) {
        throw new Error("Detection model session is not initialized.");
      }
      const params = resolveDetParams(defaultParams, overrides);
      const batchSize = resolveRuntimeBatchSize(overrides?.batchSize, defaultBatchSize);
      const results: DetResult[] = [];
      const runCtx: DetRunContext = {
        cv,
        ort,
        config,
        session: sessionState.session
      };
      for (const chunk of chunkArray(mats, batchSize)) {
        const preps = preprocess({ cv, ort, config }, chunk, params);
        const inputTensor = packDetBatchTensor(ort, preps);
        const fullOutput = await runInference(sessionState.session, inputTensor);
        const internals = postprocess(runCtx, fullOutput, preps, params);
        for (const internal of internals) {
          results.push({
            boxes: internal.boxes,
            srcW: internal.prep.srcW,
            srcH: internal.prep.srcH
          });
        }
      }
      return results;
    },
    async dispose() {
      await releaseSessions(sessionState?.session);
      sessionState = null;
    }
  };
}

export async function createDetModelSession(
  ort: OrtModule,
  modelBytes: Uint8Array,
  backend: string,
  webgpuState: WebGpuState
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

function preprocess(
  context: DetContext,
  mats: Mat[],
  params: InternalDetParams
): DetPreprocessResult[] {
  return mats.map((mat) => preprocessSample(context, mat, params));
}

function preprocessSample(
  context: DetContext,
  sourceMat: Mat,
  params: InternalDetParams
): DetPreprocessResult {
  const { cv, ort, config } = context;
  const srcW = sourceMat.cols;
  const srcH = sourceMat.rows;
  const limitSideLen = Math.max(32, params.limitSideLen);
  const limitType: LimitType = params.limitType;
  const maxSideLimit = Math.max(32, params.maxSideLimit);
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
    dstH
  };
}

function getDetMap(outputTensor: Tensor): { data: Float32Array; h: number; w: number } {
  const dims = outputTensor.dims;
  const data = outputTensor.data as Float32Array;
  if (dims.length === 4) return { data, h: dims[2], w: dims[3] };
  if (dims.length === 3) return { data, h: dims[1], w: dims[2] };
  throw new Error(`Unexpected det output dims: [${dims.join(", ")}]`);
}

function createBatchDetTensor(
  ort: OrtModule,
  preps: DetPreprocessResult[],
  maxH: number,
  maxW: number
): Tensor {
  const batch = preps.length;
  const plane = 3 * maxH * maxW;
  const out = new Float32Array(batch * plane);
  for (let i = 0; i < batch; i += 1) {
    const prep = preps[i];
    const chw = prep.tensor.data as Float32Array;
    const { dstH, dstW } = prep;
    const base = i * plane;
    for (let c = 0; c < 3; c += 1) {
      const srcChannelBase = c * dstH * dstW;
      const dstChannelBase = base + c * maxH * maxW;
      for (let y = 0; y < dstH; y += 1) {
        const srcRow = srcChannelBase + y * dstW;
        const dstRow = dstChannelBase + y * maxW;
        out.set(chw.subarray(srcRow, srcRow + dstW), dstRow);
      }
    }
  }
  return new ort.Tensor("float32", out, [batch, 3, maxH, maxW]);
}

function packDetBatchTensor(ort: OrtModule, preps: DetPreprocessResult[]): Tensor {
  const maxH = Math.max(...preps.map((p) => p.dstH));
  const maxW = Math.max(...preps.map((p) => p.dstW));
  return createBatchDetTensor(ort, preps, maxH, maxW);
}

function batchDetOutputPlaneOffset(dims: readonly number[], batchIndex: number): number {
  const tail = dims.slice(1).reduce((a, b) => a * b, 1);
  return batchIndex * tail;
}

function detFeatureCropDims(
  dstH: number,
  dstW: number,
  maxH: number,
  maxW: number,
  ohFull: number,
  owFull: number
): { cropOh: number; cropOw: number } {
  const cropOh = Math.max(1, Math.min(ohFull, Math.round((ohFull * dstH) / maxH)));
  const cropOw = Math.max(1, Math.min(owFull, Math.round((owFull * dstW) / maxW)));
  return { cropOh, cropOw };
}

function sliceBatchedDetOutputPlane(
  ort: OrtModule,
  fullOutput: Tensor,
  batchIndex: number,
  cropOh: number,
  cropOw: number,
  ohFull: number,
  owFull: number
): Tensor {
  const data = fullOutput.data as Float32Array;
  const dims = fullOutput.dims;
  const base = batchDetOutputPlaneOffset(dims, batchIndex);
  const out = new Float32Array(cropOh * cropOw);
  for (let r = 0; r < cropOh; r += 1) {
    const rowStart = base + r * owFull;
    out.set(data.subarray(rowStart, rowStart + cropOw), r * cropOw);
  }
  return new ort.Tensor("float32", out, [1, 1, cropOh, cropOw]);
}

function postprocess(
  context: DetRunContext,
  fullOutput: Tensor,
  preps: DetPreprocessResult[],
  params: InternalDetParams
): InternalDetBatchItem[] {
  const { cv, ort, config } = context;
  const od = fullOutput.dims;
  if (od.length !== 3 && od.length !== 4) {
    throw new Error(`Unexpected det output dims: [${od.join(", ")}]`);
  }
  const ohFull = od.length === 4 ? od[2] : od[1];
  const owFull = od.length === 4 ? od[3] : od[2];
  const nOut = od.length === 4 ? od[0] : preps.length === 1 ? 1 : od[0];
  if (nOut !== preps.length) {
    throw new Error(
      `Detection batch output N=${String(nOut)} does not match input batch ${String(preps.length)}`
    );
  }

  const maxH = Math.max(...preps.map((p) => p.dstH));
  const maxW = Math.max(...preps.map((p) => p.dstW));

  const items: InternalDetBatchItem[] = [];
  for (let i = 0; i < preps.length; i += 1) {
    const prep = preps[i];
    const { cropOh, cropOw } = detFeatureCropDims(prep.dstH, prep.dstW, maxH, maxW, ohFull, owFull);
    const planeTensor = sliceBatchedDetOutputPlane(
      ort,
      fullOutput,
      i,
      cropOh,
      cropOw,
      ohFull,
      owFull
    );
    const boxes = decodeDetOutput(
      { cv, config },
      planeTensor,
      prep,
      params.thresh,
      params.boxThresh,
      params.unclipRatio
    );
    items.push({ prep, boxes });
  }
  return items;
}

function decodeDetOutput(
  context: { cv: OpenCv; config: DetModelConfig },
  detOutput: Tensor,
  meta: DetPreprocessResult,
  detThresh: number,
  boxThresh: number,
  unclipRatio: number
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
      clamp(Math.round((point[1] * meta.srcH) / Math.max(1, h)), 0, meta.srcH)
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
