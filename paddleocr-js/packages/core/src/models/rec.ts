import type { OpenCv, Mat } from "@techstark/opencv-js";
import type { InferenceSession, Tensor } from "onnxruntime-web";

import { assertModelResources } from "../resources/model-asset";
import { createSession, getProviderCandidates, releaseSessions } from "../runtime/ort";
import type { OrtModule, WebGpuState, SessionState } from "../runtime/ort";
import { chunkArray, clamp, resolveRuntimeBatchSize, withTimeout } from "../utils/common";
import { getTransformOp, parseInferenceConfigText, toBgrFloatCHWFromBgr } from "./common";
import type { NormalizeConfig } from "./common";

export interface RecModelConfig {
  imageShape: number[];
  charDict: string[];
  scoreThresh: number;
}

export interface RecResult {
  text: string;
  score: number;
}

export interface RecRuntimeOverrides {
  batchSize?: number;
}

export interface RecModel {
  readonly kind: "rec";
  readonly config: RecModelConfig;
  readonly provider: string;
  predict(cv: OpenCv, mats: Mat[], overrides?: RecRuntimeOverrides): Promise<RecResult[]>;
  dispose(): Promise<void>;
}

interface RecSample {
  inputIndex: number;
  width: number;
  chw: Float32Array;
}

const DEFAULT_REC_ALPHANUMERIC_DICT = "0123456789abcdefghijklmnopqrstuvwxyz".split("");

const REC_NORMALIZE: Readonly<NormalizeConfig> = Object.freeze({
  mean: [0.5, 0.5, 0.5],
  std: [0.5, 0.5, 0.5],
  scale: 1 / 255
});

export const DEFAULT_REC_MODEL_PARSE_FALLBACKS: Readonly<
  Pick<RecModelConfig, "imageShape" | "charDict" | "scoreThresh">
> = Object.freeze({
  imageShape: [3, 48, 320],
  charDict: [],
  scoreThresh: 0
});

export const DEFAULT_REC_RUNTIME_LIMITS = Object.freeze({
});

const MAX_REC_WIDTH = 3200;

export const DEFAULT_REC_MODEL_CONFIG: Readonly<RecModelConfig> = Object.freeze({
  ...DEFAULT_REC_MODEL_PARSE_FALLBACKS
});

export function parseRecModelConfigText(text: string): RecModelConfig {
  const parsed = parseInferenceConfigText(text);
  const preProcess = parsed.PreProcess as Record<string, unknown> | undefined;
  const transformOps = preProcess?.transform_ops as Array<Record<string, unknown>> | undefined;
  const resize = getTransformOp(transformOps, "RecResizeImg");
  const postprocess = (parsed.PostProcess || {}) as Record<string, unknown>;
  const baseCharDict = postprocess.character_dict;

  const imageShape = resize?.image_shape as number[] | undefined;
  if (!imageShape || !Array.isArray(imageShape) || imageShape.length < 3) {
    throw new Error("RecResizeImg.image_shape is required in rec inference.yml");
  }

  const charDict =
    Array.isArray(baseCharDict) && baseCharDict.length > 0
      ? [...(baseCharDict as string[]), " "]
      : [...DEFAULT_REC_ALPHANUMERIC_DICT, " "];

  return {
    imageShape,
    charDict,
    scoreThresh: 0
  };
}

interface CreateRecModelArgs {
  ort: OrtModule;
  modelBytes: Uint8Array;
  configText: string;
  backend: string;
  webgpuState: WebGpuState;
  batchSize?: number;
}

export async function createRecModel({
  ort,
  modelBytes,
  configText,
  backend,
  webgpuState,
  batchSize: batchSizeArg
}: CreateRecModelArgs): Promise<RecModel> {
  assertModelResources("Recognition", {
    model: modelBytes,
    config: configText
  });
  const config = parseRecModelConfigText(configText);
  const defaultBatchSize = Math.max(1, batchSizeArg ?? 1);
  let sessionState: SessionState | null = await createRecModelSession(
    ort,
    modelBytes,
    backend,
    webgpuState
  );

  return {
    kind: "rec",
    config,
    get provider() {
      return sessionState?.provider || "";
    },
    async predict(cv, mats, overrides) {
      if (!sessionState?.session) {
        throw new Error("Recognition model session is not initialized.");
      }
      const batchSize = resolveRuntimeBatchSize(overrides?.batchSize, defaultBatchSize);
      const samples: RecSample[] = [];
      for (let i = 0; i < mats.length; i += 1) {
        samples.push(preprocessRec({ cv, config }, mats[i], i));
      }
      return runRecInference(
        {
          ort,
          session: sessionState.session,
          config,
          charDict: config.charDict
        },
        samples,
        batchSize
      );
    },
    async dispose() {
      await releaseSessions(sessionState?.session);
      sessionState = null;
    }
  };
}

export async function createRecModelSession(
  ort: OrtModule,
  modelBytes: Uint8Array,
  backend: string,
  webgpuState: WebGpuState
): Promise<SessionState> {
  const providerCandidates = getProviderCandidates(backend, webgpuState);
  return withTimeout(
    createSession(ort, modelBytes, providerCandidates),
    60000,
    "Recognition model"
  );
}

function preprocessRec(
  context: { cv: OpenCv; config: RecModelConfig },
  cropMat: Mat,
  inputIndex: number
): RecSample {
  const { cv, config } = context;
  const [channels, targetH, baseW] = config.imageShape;
  const srcW = cropMat.cols;
  const srcH = cropMat.rows;
  if (channels !== 3) {
    throw new Error(`Unexpected recognition channels: ${String(channels)}`);
  }
  const ratio = srcW / Math.max(1, srcH);
  const maxWhRatio = Math.max(baseW / Math.max(1, targetH), ratio);
  const recW = clamp(Math.trunc(targetH * maxWhRatio), 1, MAX_REC_WIDTH);
  const resizedW = Math.min(recW, Math.ceil(targetH * ratio));
  const resized = new cv.Mat();
  const bgr = new cv.Mat();
  cv.resize(cropMat, resized, new cv.Size(resizedW, targetH), 0, 0, cv.INTER_LINEAR);
  if (resized.channels() === 4) {
    cv.cvtColor(resized, bgr, cv.COLOR_RGBA2BGR);
  } else if (resized.channels() === 1) {
    cv.cvtColor(resized, bgr, cv.COLOR_GRAY2BGR);
  } else {
    resized.copyTo(bgr);
  }
  const resizedChw = toBgrFloatCHWFromBgr(bgr.data, resizedW, targetH, REC_NORMALIZE);
  const chw = new Float32Array(3 * targetH * recW);
  const dstPerChannel = targetH * recW;
  const srcPerChannel = targetH * resizedW;
  for (let channel = 0; channel < 3; channel += 1) {
    for (let row = 0; row < targetH; row += 1) {
      const srcStart = channel * srcPerChannel + row * resizedW;
      const dstStart = channel * dstPerChannel + row * recW;
      chw.set(resizedChw.subarray(srcStart, srcStart + resizedW), dstStart);
    }
  }
  bgr.delete();
  resized.delete();
  return { inputIndex, width: recW, chw };
}

interface RecRunContext {
  ort: OrtModule;
  session: InferenceSession;
  config: RecModelConfig;
  charDict: string[];
}

function createBatchTensor(
  ort: OrtModule,
  samples: RecSample[],
  maxW: number,
  targetH: number
): Tensor {
  const batch = samples.length;
  const out = new Float32Array(batch * 3 * targetH * maxW);
  const dstPerChannel = targetH * maxW;
  for (let index = 0; index < batch; index += 1) {
    const sample = samples[index];
    const srcW = sample.width;
    const srcPerChannel = targetH * srcW;
    for (let channel = 0; channel < 3; channel += 1) {
      const srcBase = channel * srcPerChannel;
      const dstBase = index * (3 * dstPerChannel) + channel * dstPerChannel;
      for (let row = 0; row < targetH; row += 1) {
        const srcStart = srcBase + row * srcW;
        const dstStart = dstBase + row * maxW;
        out.set(sample.chw.subarray(srcStart, srcStart + srcW), dstStart);
      }
    }
  }
  return new ort.Tensor("float32", out, [batch, 3, targetH, maxW]);
}

function decodeCTCSample(
  data: Float32Array,
  offset: number,
  timeSteps: number,
  classes: number,
  charDict: string[]
): { text: string; score: number } {
  let prevIdx = -1;
  let text = "";
  const probs: number[] = [];
  for (let step = 0; step < timeSteps; step += 1) {
    let maxIdx = 0;
    let maxVal = -Infinity;
    const stepOffset = offset + step * classes;
    for (let cls = 0; cls < classes; cls += 1) {
      const value = data[stepOffset + cls];
      if (value > maxVal) {
        maxVal = value;
        maxIdx = cls;
      }
    }
    if (maxIdx > 0 && maxIdx !== prevIdx) {
      const dictIdx = maxIdx - 1;
      if (dictIdx >= 0 && dictIdx < charDict.length) {
        text += charDict[dictIdx];
        probs.push(maxVal);
      }
    }
    prevIdx = maxIdx;
  }
  const score = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0;
  return { text, score };
}

function postprocessRec(
  outputData: Float32Array,
  sampleCount: number,
  timeSteps: number,
  classes: number,
  charDict: string[]
): Array<{ text: string; score: number }> {
  const stride = timeSteps * classes;
  const results: Array<{ text: string; score: number }> = [];
  for (let index = 0; index < sampleCount; index += 1) {
    results.push(decodeCTCSample(outputData, index * stride, timeSteps, classes, charDict));
  }
  return results;
}

async function runRecInference(
  context: RecRunContext,
  samples: RecSample[],
  batchSize: number
): Promise<RecResult[]> {
  const { ort, session, config, charDict } = context;
  const recInputName = session.inputNames[0];
  const targetH = config.imageShape[1];
  const ordered = samples.slice().sort((a, b) => a.width - b.width);
  const decoded: Array<{ inputIndex: number; text: string; score: number }> = [];

  for (const batch of chunkArray(ordered, batchSize)) {
    const maxW = batch.reduce((acc, sample) => Math.max(acc, sample.width), 1);
    const tensor = createBatchTensor(ort, batch, maxW, targetH);
    const outputMap = await session.run({ [recInputName]: tensor });
    const output = outputMap[session.outputNames[0]];
    const dims = output.dims;
    if (dims.length !== 3) {
      throw new Error(`Unexpected rec output dims: [${dims.join(", ")}]`);
    }
    const batchResults = postprocessRec(
      output.data as Float32Array,
      dims[0],
      dims[1],
      dims[2],
      charDict
    );
    for (let index = 0; index < batchResults.length; index += 1) {
      decoded.push({
        inputIndex: batch[index].inputIndex,
        ...batchResults[index]
      });
    }
  }

  decoded.sort((a, b) => a.inputIndex - b.inputIndex);
  return decoded.map(({ text, score }) => ({ text, score }));
}
