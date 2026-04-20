/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { OpenCv } from "@techstark/opencv-js";
import cvModule from "@techstark/opencv-js";

let cachedCvPromise: Promise<{ cv: OpenCv }> | null = null;

async function getOpenCv(): Promise<{ cv: OpenCv }> {
  let cv: OpenCv;
  if (cvModule instanceof Promise) {
    cv = await cvModule;
  } else {
    const mod = cvModule as { Mat?: unknown; onRuntimeInitialized?: () => void };
    if (mod.Mat) {
      cv = cvModule as OpenCv;
    } else {
      await new Promise<void>((resolve) => {
        mod.onRuntimeInitialized = () => {
          resolve();
        };
      });
      cv = cvModule as OpenCv;
    }
  }
  return { cv };
}

export async function initOpenCvRuntime(): Promise<{ cv: OpenCv }> {
  if (!cachedCvPromise) {
    cachedCvPromise = getOpenCv().catch((error: unknown) => {
      cachedCvPromise = null;
      throw error;
    });
  }
  return cachedCvPromise;
}
