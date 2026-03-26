let ortModulePromise = null;

async function loadOrtModule() {
  if (ortModulePromise) {
    return ortModulePromise;
  }
  ortModulePromise = import("onnxruntime-web");
  return ortModulePromise;
}

export async function detectWebGpuAvailability() {
  if (!globalThis.navigator?.gpu?.requestAdapter) {
    return {
      available: false,
      reason: "navigator.gpu is unavailable in this browser."
    };
  }
  try {
    const adapter = await globalThis.navigator.gpu.requestAdapter();
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
  } catch (err) {
    return {
      available: false,
      reason: err?.message || "Failed to request a WebGPU adapter."
    };
  }
}

export function getProviderCandidates(backend, webgpuState) {
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

function applyOrtEnvironmentOptions(ort, runtimeOptions) {
  const wasmOptions = ort?.env?.wasm;
  if (!wasmOptions) return;

  if (runtimeOptions.wasmPaths !== undefined) {
    wasmOptions.wasmPaths = runtimeOptions.wasmPaths;
  }
  if (runtimeOptions.numThreads !== undefined) {
    wasmOptions.numThreads = runtimeOptions.numThreads;
  }
  if (runtimeOptions.simd !== undefined) {
    wasmOptions.simd = runtimeOptions.simd;
  }
  if (runtimeOptions.proxy !== undefined) {
    wasmOptions.proxy = runtimeOptions.proxy;
  }
  if (runtimeOptions.disableWasmProxy) {
    wasmOptions.proxy = false;
  }
}

export async function initOrtRuntime(runtimeOptions = {}) {
  const backend =
    typeof runtimeOptions === "string"
      ? runtimeOptions
      : runtimeOptions.backend === "webgpu" || runtimeOptions.backend === "wasm"
        ? runtimeOptions.backend
        : "auto";
  const webgpuState = await detectWebGpuAvailability();
  const ort = await loadOrtModule();
  applyOrtEnvironmentOptions(ort, runtimeOptions);
  return {
    ort,
    webgpuState,
    backend
  };
}

export async function createSession(ort, modelBytes, providerCandidates) {
  let lastErr = null;
  for (const executionProviders of providerCandidates) {
    try {
      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders,
        graphOptimizationLevel: "all"
      });
      return { session, provider: executionProviders[0] };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error("Failed to create ONNX session.");
}

export async function releaseSessions(...sessions) {
  await Promise.all(
    sessions.map(async (session) => {
      if (!session?.release) return;
      await session.release();
    })
  );
}
