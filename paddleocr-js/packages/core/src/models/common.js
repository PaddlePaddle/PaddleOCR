import yaml from "js-yaml";
import ClipperLib from "clipper-lib";

import { clamp, distance2 } from "../utils/common.js";

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function parseInferenceConfigText(text) {
  const parsed = yaml.load(text);
  return isPlainObject(parsed) ? parsed : {};
}

export function parseScaleValue(rawScale, fallback) {
  if (typeof rawScale === "number") return rawScale;
  if (typeof rawScale !== "string") return fallback;
  const normalized = rawScale.replace(/\s/g, "");
  const direct = Number(normalized);
  if (!Number.isNaN(direct)) return direct;
  const divParts = normalized.split("/");
  if (divParts.length === 2) {
    const numerator = Number(divParts[0].replace(/\.+$/, ""));
    const denominator = Number(divParts[1].replace(/\.+$/, ""));
    if (!Number.isNaN(numerator) && !Number.isNaN(denominator) && denominator !== 0) {
      return numerator / denominator;
    }
  }
  return fallback;
}

export function getTransformOp(transformOps, opName) {
  for (const op of transformOps || []) {
    if (Object.prototype.hasOwnProperty.call(op, opName)) {
      return op[opName];
    }
  }
  return null;
}

function findModelNameInYamlNode(value) {
  if (Array.isArray(value)) {
    for (const item of value) {
      const match = findModelNameInYamlNode(item);
      if (match) return match;
    }
    return null;
  }

  if (!isPlainObject(value)) {
    return null;
  }

  for (const [key, childValue] of Object.entries(value)) {
    if (key === "model_name" && typeof childValue === "string" && childValue.trim()) {
      return childValue;
    }
    const match = findModelNameInYamlNode(childValue);
    if (match) return match;
  }

  return null;
}

export function extractInferenceModelName(configText) {
  const parsed = parseInferenceConfigText(configText);
  const preferredCandidates = [parsed.Global?.model_name, parsed.model_name];
  for (const candidate of preferredCandidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate;
    }
  }

  return findModelNameInYamlNode(parsed);
}

export function toBgrFloatCHWFromBgr(bgr, width, height, normalizeConfig) {
  const data = new Float32Array(3 * width * height);
  const hw = width * height;
  const mean = normalizeConfig.mean;
  const std = normalizeConfig.std;
  const scale = normalizeConfig.scale;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const p = idx * 3;
      const b = bgr[p];
      const g = bgr[p + 1];
      const r = bgr[p + 2];
      data[idx] = (b * scale - mean[0]) / std[0];
      data[idx + hw] = (g * scale - mean[1]) / std[1];
      data[idx + 2 * hw] = (r * scale - mean[2]) / std[2];
    }
  }
  return data;
}

function orderQuad(pts) {
  const points = pts.slice().sort((a, b) => a[0] - b[0]);
  let indexA;
  let indexB;
  let indexC;
  let indexD;
  if (points[1][1] > points[0][1]) {
    indexA = 0;
    indexD = 1;
  } else {
    indexA = 1;
    indexD = 0;
  }
  if (points[3][1] > points[2][1]) {
    indexB = 2;
    indexC = 3;
  } else {
    indexB = 3;
    indexC = 2;
  }
  return [points[indexA], points[indexB], points[indexC], points[indexD]];
}

function polygonArea(poly) {
  let area = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const j = (i + 1) % poly.length;
    area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
  }
  return Math.abs(area) * 0.5;
}

function polygonPerimeter(poly) {
  let peri = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const j = (i + 1) % poly.length;
    peri += distance2(poly[i], poly[j]);
  }
  return peri;
}

function chooseMaxAreaPath(paths) {
  let best = null;
  let bestArea = 0;
  for (const path of paths) {
    if (!path || path.length < 4) continue;
    const poly = path.map((pt) => [pt.X, pt.Y]);
    const area = polygonArea(poly);
    if (area > bestArea) {
      bestArea = area;
      best = path;
    }
  }
  return best;
}

export function unclip(poly, unclipRatio) {
  const area = polygonArea(poly);
  const perimeter = polygonPerimeter(poly);
  if (perimeter <= 0) return null;
  const distance = (area * unclipRatio) / perimeter;
  const path = poly.map((p) => ({ X: Math.trunc(p[0]), Y: Math.trunc(p[1]) }));
  const offset = new ClipperLib.ClipperOffset();
  offset.AddPath(path, ClipperLib.JoinType.jtRound, ClipperLib.EndType.etClosedPolygon);
  const expanded = new ClipperLib.Paths();
  offset.Execute(expanded, distance);
  const best = chooseMaxAreaPath(expanded);
  if (!best) return null;
  return best.map((pt) => [pt.X, pt.Y]);
}

export function getMiniBoxFromPoints(cv, points) {
  const flat = [];
  for (const p of points) flat.push(p[0], p[1]);
  const contour = cv.matFromArray(points.length, 1, cv.CV_32FC2, flat);
  const rect = cv.minAreaRect(contour);
  const vertices = cv.RotatedRect.points(rect);
  const box = [];
  for (let i = 0; i < 4; i += 1) box.push([vertices[i].x, vertices[i].y]);
  contour.delete();
  const ordered = orderQuad(box);
  const side = Math.min(distance2(ordered[0], ordered[1]), distance2(ordered[1], ordered[2]));
  return { box: ordered, side };
}

export function boxScoreFast(cv, predMat, box) {
  const h = predMat.rows;
  const w = predMat.cols;
  let minX = w - 1;
  let maxX = 0;
  let minY = h - 1;
  let maxY = 0;
  for (const p of box) {
    minX = Math.min(minX, p[0]);
    maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]);
    maxY = Math.max(maxY, p[1]);
  }
  minX = clamp(Math.floor(minX), 0, w - 1);
  maxX = clamp(Math.ceil(maxX), 0, w - 1);
  minY = clamp(Math.floor(minY), 0, h - 1);
  maxY = clamp(Math.ceil(maxY), 0, h - 1);
  const rw = Math.max(1, maxX - minX + 1);
  const rh = Math.max(1, maxY - minY + 1);
  const roi = predMat.roi(new cv.Rect(minX, minY, rw, rh));
  const mask = new cv.Mat.zeros(rh, rw, cv.CV_8UC1);
  const shifted = box.map((p) => [Math.trunc(p[0] - minX), Math.trunc(p[1] - minY)]);
  const flat = [];
  for (const p of shifted) flat.push(p[0], p[1]);
  const pts = cv.matFromArray(shifted.length, 1, cv.CV_32SC2, flat);
  const ptsVec = new cv.MatVector();
  ptsVec.push_back(pts);
  cv.fillPoly(mask, ptsVec, new cv.Scalar(1));
  const mean = cv.mean(roi, mask)[0];
  roi.delete();
  mask.delete();
  pts.delete();
  ptsVec.delete();
  return mean;
}
