export function ensureServedFromHttp() {
  if (globalThis.location?.protocol === "file:") {
    throw new Error("PaddleOCR.js requires an HTTP(S) origin so model assets can be fetched.");
  }
}

function hasDomConstructor(name) {
  return typeof globalThis[name] !== "undefined";
}

export async function sourceToImageBitmap(source) {
  if (typeof ImageBitmap !== "undefined" && source instanceof ImageBitmap) return source;
  if (source instanceof Blob) return createImageBitmap(source);
  if (hasDomConstructor("HTMLCanvasElement") && source instanceof HTMLCanvasElement) {
    return createImageBitmap(source);
  }
  if (source instanceof ImageData) {
    const canvas = document.createElement("canvas");
    canvas.width = source.width;
    canvas.height = source.height;
    canvas.getContext("2d").putImageData(source, 0, 0);
    return createImageBitmap(canvas);
  }
  if (hasDomConstructor("HTMLImageElement") && source instanceof HTMLImageElement) {
    return createImageBitmap(source);
  }
  throw new Error("Unsupported image source. Use a Blob, ImageBitmap, ImageData, canvas, or img.");
}

async function sourceToClonedImageBitmap(source) {
  if (typeof ImageBitmap !== "undefined" && source instanceof ImageBitmap) {
    return createImageBitmap(source);
  }
  return sourceToImageBitmap(source);
}

export function bitmapToSourceMat(cv, imageBitmap) {
  const canvas = document.createElement("canvas");
  canvas.width = imageBitmap.width;
  canvas.height = imageBitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(imageBitmap, 0, 0);
  return {
    canvas,
    mat: cv.imread(canvas)
  };
}

export async function sourceToMat(cv, source) {
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

  const imageBitmap = await sourceToImageBitmap(source);
  const sourceImage = bitmapToSourceMat(cv, imageBitmap);
  return {
    width: imageBitmap.width,
    height: imageBitmap.height,
    mat: sourceImage.mat,
    dispose() {
      sourceImage.mat.delete();
      imageBitmap.close?.();
    }
  };
}

export async function sourceToWorkerPayload(source) {
  if (typeof ImageBitmap === "undefined" || typeof createImageBitmap !== "function") {
    throw new Error("Worker mode requires ImageBitmap support in this browser.");
  }
  const imageBitmap = await sourceToClonedImageBitmap(source);
  return {
    payload: {
      kind: "imageBitmap",
      imageBitmap
    },
    transferables: [imageBitmap]
  };
}
