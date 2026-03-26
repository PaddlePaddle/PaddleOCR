import { ensureServedFromHttp } from "./browser.js";

function imageBitmapToImageData(imageBitmap) {
  if (typeof OffscreenCanvas !== "function") {
    throw new Error("Worker mode requires OffscreenCanvas support in this browser.");
  }
  const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("Failed to create a 2D canvas context in the OCR worker.");
  }
  ctx.drawImage(imageBitmap, 0, 0);
  return ctx.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
}

function imageDataToMat(cv, imageData) {
  return cv.matFromArray(imageData.height, imageData.width, cv.CV_8UC4, imageData.data);
}

export async function sourcePayloadToMat(cv, source) {
  if (typeof cv?.Mat === "function" && source instanceof cv.Mat) {
    return {
      width: source.cols,
      height: source.rows,
      mat: source.clone(),
      dispose() {
        this.mat.delete();
      }
    };
  }

  if (
    source?.kind === "imageBitmap" &&
    typeof ImageBitmap !== "undefined" &&
    source.imageBitmap instanceof ImageBitmap
  ) {
    const imageData = imageBitmapToImageData(source.imageBitmap);
    const mat = imageDataToMat(cv, imageData);
    return {
      width: imageData.width,
      height: imageData.height,
      mat,
      dispose() {
        mat.delete();
        source.imageBitmap.close?.();
      }
    };
  }

  throw new Error("Unsupported worker image source payload.");
}

export { ensureServedFromHttp };
