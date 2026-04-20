/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export type OrtModule = typeof import("onnxruntime-web");

export interface WebGpuState {
  available: boolean;
  reason: string;
}

export interface OrtOptions {
  backend?: "webgpu" | "wasm" | "auto" | (string & {});
  wasmPaths?: string;
  numThreads?: number;
  simd?: boolean;
  proxy?: boolean;
  disableWasmProxy?: boolean;
}

export interface OrtRuntimeResult {
  ort: OrtModule;
  webgpuState: WebGpuState;
  backend: string;
}

export interface SessionState {
  session: import("onnxruntime-web").InferenceSession;
  provider: string;
}

let ortModulePromise: Promise<OrtModule> | null = null;

async function loadOrtModule(): Promise<OrtModule> {
  if (ortModulePromise) {
    return ortModulePromise;
  }
  ortModulePromise = import("onnxruntime-web");
  return ortModulePromise;
}

interface GpuLike {
  requestAdapter(): Promise<unknown>;
}

export async function detectWebGpuAvailability(): Promise<WebGpuState> {
  const gpu = (globalThis.navigator as (Navigator & { gpu?: GpuLike }) | undefined)?.gpu;
  if (!gpu?.requestAdapter) {
    return {
      available: false,
      reason: "navigator.gpu is unavailable in this browser."
    };
  }
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return {
        available: false,
        reason: "The browser did not return a WebGPU adapter."
      };
    }
    return {
      available: true,
      reason: ""
    };
  } catch (err: unknown) {
    return {
      available: false,
      reason: err instanceof Error ? err.message : "Failed to request a WebGPU adapter."
    };
  }
}

export function getProviderCandidates(backend: string, webgpuState: WebGpuState): string[][] {
  if (backend === "webgpu") {
    if (!webgpuState.available) {
      throw new Error(`WebGPU is unavailable: ${webgpuState.reason}`);
    }
    return [["webgpu"]];
  }
  if (backend === "wasm") {
    return [["wasm"]];
  }
  return webgpuState.available ? [["webgpu"], ["wasm"]] : [["wasm"]];
}

function applyOrtEnvironmentOptions(ort: OrtModule, ortOptions: OrtOptions): void {
  const wasmOptions = ort.env.wasm;

  if (ortOptions.wasmPaths !== undefined) {
    wasmOptions.wasmPaths = ortOptions.wasmPaths;
  }
  if (ortOptions.numThreads !== undefined) {
    wasmOptions.numThreads = ortOptions.numThreads;
  }
  if (ortOptions.simd !== undefined) {
    wasmOptions.simd = ortOptions.simd;
  }
  if (ortOptions.proxy !== undefined) {
    wasmOptions.proxy = ortOptions.proxy;
  }
  if (ortOptions.disableWasmProxy) {
    wasmOptions.proxy = false;
  }
}

export async function initOrtRuntime(
  ortOptions: OrtOptions | string = {}
): Promise<OrtRuntimeResult> {
  const backend =
    typeof ortOptions === "string"
      ? ortOptions
      : ortOptions.backend === "webgpu" || ortOptions.backend === "wasm"
        ? ortOptions.backend
        : "auto";
  const webgpuState = await detectWebGpuAvailability();
  const ort = await loadOrtModule();
  if (typeof ortOptions !== "string") {
    applyOrtEnvironmentOptions(ort, ortOptions);
  }
  return {
    ort,
    webgpuState,
    backend
  };
}

export async function createSession(
  ort: OrtModule,
  modelBytes: Uint8Array,
  providerCandidates: string[][]
): Promise<SessionState> {
  let lastErr: unknown = null;
  for (const executionProviders of providerCandidates) {
    try {
      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders,
        graphOptimizationLevel: "all"
      });
      return { session, provider: executionProviders[0] };
    } catch (err: unknown) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("Failed to create ONNX session.");
}

export async function releaseSessions(
  ...sessions: Array<import("onnxruntime-web").InferenceSession | null | undefined>
): Promise<void> {
  await Promise.all(
    sessions.map(async (session) => {
      if (!session?.release) return;
      await session.release();
    })
  );
}
