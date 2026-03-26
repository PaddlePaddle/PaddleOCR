import cvModule from "@techstark/opencv-js";

let cachedCvPromise = null;

async function getOpenCv() {
  let cv;
  if (cvModule instanceof Promise) {
    cv = await cvModule;
  } else {
    if (cvModule.Mat) {
      cv = cvModule;
    } else {
      await new Promise((resolve) => {
        cvModule.onRuntimeInitialized = () => resolve();
      });
      cv = cvModule;
    }
  }
  return { cv };
}

export async function initOpenCvRuntime() {
  if (!cachedCvPromise) {
    cachedCvPromise = getOpenCv().catch((error) => {
      cachedCvPromise = null;
      throw error;
    });
  }
  return cachedCvPromise;
}
