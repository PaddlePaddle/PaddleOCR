import { assertStandardModelResources } from "../resources/standard-model.js";
import { createSession, getProviderCandidates, releaseSessions } from "../runtime/ort.js";
import { clamp, withTimeout } from "../utils/common.js";
import {
  getTransformOp,
  parseInferenceConfigText,
  parseScaleValue,
  toBgrFloatCHWFromBgr
} from "./common.js";

export const DEFAULT_REC_MODEL_PARSE_FALLBACKS = Object.freeze({
  imageShape: [3, 48, 320],
  scoreThresh: 0.1,
  normalize: {
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    scale: 1 / 255
  },
  charDict: []
});

export const DEFAULT_REC_RUNTIME_LIMITS = Object.freeze({
  maxBatch: 6,
  maxWidth: 3200
});

export const DEFAULT_REC_MODEL_CONFIG = Object.freeze({
  ...DEFAULT_REC_MODEL_PARSE_FALLBACKS,
  maxBatch: DEFAULT_REC_RUNTIME_LIMITS.maxBatch,
  maxWidth: DEFAULT_REC_RUNTIME_LIMITS.maxWidth
});

export function parseRecModelConfigText(text) {
  const parsed = parseInferenceConfigText(text);
  const resize = getTransformOp(parsed?.PreProcess?.transform_ops, "RecResizeImg");
  const normalize = getTransformOp(parsed?.PreProcess?.transform_ops, "NormalizeImage");
  const postprocess = parsed?.PostProcess || {};
  const baseCharDict = postprocess?.character_dict;
  if (!Array.isArray(baseCharDict) || baseCharDict.length === 0) {
    throw new Error("No valid character_dict found in rec inference.yml");
  }

  return {
    imageShape: resize?.image_shape || DEFAULT_REC_MODEL_PARSE_FALLBACKS.imageShape,
    maxBatch: Number(DEFAULT_REC_RUNTIME_LIMITS.maxBatch),
    maxWidth: Number(DEFAULT_REC_RUNTIME_LIMITS.maxWidth),
    scoreThresh: DEFAULT_REC_MODEL_PARSE_FALLBACKS.scoreThresh,
    normalize: normalize
      ? {
          mean: normalize.mean || DEFAULT_REC_MODEL_PARSE_FALLBACKS.normalize.mean,
          std: normalize.std || DEFAULT_REC_MODEL_PARSE_FALLBACKS.normalize.std,
          scale: parseScaleValue(normalize.scale, DEFAULT_REC_MODEL_PARSE_FALLBACKS.normalize.scale)
        }
      : { ...DEFAULT_REC_MODEL_PARSE_FALLBACKS.normalize },
    charDict: [...baseCharDict, " "]
  };
}

export async function createRecModel({ ort, modelBytes, configText, backend, webgpuState }) {
  assertStandardModelResources("Recognition", {
    model: modelBytes,
    config: configText
  });
  const config = parseRecModelConfigText(configText);
  let sessionState = await createRecModelSession(ort, modelBytes, backend, webgpuState);

  return {
    kind: "rec",
    config,
    charDict: config.charDict,
    get provider() {
      return sessionState?.provider || "";
    },
    prepareSample({ cv, cropMat, poly, originalIndex }) {
      return prepareRecSample({ cv, config }, cropMat, poly, originalIndex);
    },
    async recognize(samples) {
      if (!sessionState?.session) {
        throw new Error("Recognition model session is not initialized.");
      }
      return runRecModel(
        {
          ort,
          session: sessionState.session,
          config,
          charDict: config.charDict
        },
        samples
      );
    },
    async dispose() {
      await releaseSessions(sessionState?.session);
      sessionState = null;
    }
  };
}

export async function createRecModelSession(ort, modelBytes, backend, webgpuState) {
  const providerCandidates = getProviderCandidates(backend, webgpuState);
  return withTimeout(
    createSession(ort, modelBytes, providerCandidates),
    60000,
    "Recognition model"
  );
}

export function prepareRecSample(context, cropMat, poly, originalIndex) {
  const { cv, config } = context;
  const [channels, targetH, baseW] = config.imageShape;
  const maxW = Number(config.maxWidth);
  const srcW = cropMat.cols;
  const srcH = cropMat.rows;
  if (channels !== 3) {
    throw new Error(`Unexpected recognition channels: ${channels}`);
  }
  const ratio = srcW / Math.max(1, srcH);
  const maxWhRatio = Math.max(baseW / Math.max(1, targetH), ratio);
  const recW = clamp(Math.trunc(targetH * maxWhRatio), 1, maxW);
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
  const resizedChw = toBgrFloatCHWFromBgr(bgr.data, resizedW, targetH, config.normalize);
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
  return { originalIndex, poly, width: recW, chw };
}

function createBatchTensor(ort, samples, maxW, targetH) {
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

function decodeCTCSample(data, offset, timeSteps, classes, charDict) {
  let prevIdx = -1;
  let text = "";
  const probs = [];
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

export async function runRecModel(context, samples) {
  const { ort, session, config, charDict } = context;
  const recInputName = session.inputNames[0];
  const batchSize = Number(config.maxBatch);
  const targetH = Number(config.imageShape[1]);
  const ordered = samples.slice().sort((a, b) => a.width - b.width);
  const results = [];

  for (let start = 0; start < ordered.length; start += batchSize) {
    const batch = ordered.slice(start, start + batchSize);
    const maxW = batch.reduce((acc, sample) => Math.max(acc, sample.width), 1);
    const tensor = createBatchTensor(ort, batch, maxW, targetH);
    const outputMap = await session.run({ [recInputName]: tensor });
    const output = outputMap[session.outputNames[0]];
    const dims = output.dims;
    if (dims.length !== 3) {
      throw new Error(`Unexpected rec output dims: [${dims.join(", ")}]`);
    }
    const sampleCount = dims[0];
    const timeSteps = dims[1];
    const classes = dims[2];
    const stride = timeSteps * classes;
    for (let index = 0; index < sampleCount; index += 1) {
      const decoded = decodeCTCSample(output.data, index * stride, timeSteps, classes, charDict);
      results.push({
        originalIndex: batch[index].originalIndex,
        poly: batch[index].poly,
        text: decoded.text,
        score: decoded.score
      });
    }
  }

  return results;
}
