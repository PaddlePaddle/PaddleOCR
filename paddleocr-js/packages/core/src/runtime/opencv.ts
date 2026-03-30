import type { OpenCv, CvModule } from "@techstark/opencv-js";
import cvModule from "@techstark/opencv-js";

let cachedCvPromise: Promise<{ cv: OpenCv }> | null = null;

async function getOpenCv(): Promise<{ cv: OpenCv }> {
  let cv: OpenCv;
  if (cvModule instanceof Promise) {
    cv = await cvModule;
  } else {
    const mod = cvModule as CvModule;
    if (mod.Mat) {
      cv = mod;
    } else {
      await new Promise<void>((resolve) => {
        mod.onRuntimeInitialized = () => resolve();
      });
      cv = mod;
    }
  }
  return { cv };
}

export async function initOpenCvRuntime(): Promise<{ cv: OpenCv }> {
  if (!cachedCvPromise) {
    cachedCvPromise = getOpenCv().catch((error) => {
      cachedCvPromise = null;
      throw error;
    });
  }
  return cachedCvPromise;
}
