import { afterEach, describe, expect, it, vi } from "vitest";

async function loadOrtModule(mockedOrt = { env: { wasm: {} } }) {
  vi.resetModules();
  vi.doMock("onnxruntime-web", () => ({
    __esModule: true,
    ...mockedOrt
  }));
  return import("../src/runtime/ort");
}

afterEach(() => {
  vi.resetModules();
  vi.doUnmock("onnxruntime-web");
  vi.unstubAllGlobals();
});

describe("runtime/ort", () => {
  it("reports when WebGPU is unavailable", async () => {
    vi.stubGlobal("navigator", {});
    const { detectWebGpuAvailability } = await loadOrtModule();

    await expect(detectWebGpuAvailability()).resolves.toEqual({
      available: false,
      reason: "navigator.gpu is unavailable in this browser."
    });
  });

  it("reports WebGPU available when the browser returns an adapter", async () => {
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: vi.fn().mockResolvedValue({})
      }
    });
    const { detectWebGpuAvailability } = await loadOrtModule();
    await expect(detectWebGpuAvailability()).resolves.toEqual({
      available: true,
      reason: ""
    });
  });

  it("reports WebGPU unavailable when the browser returns no adapter", async () => {
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: vi.fn().mockResolvedValue(null)
      }
    });
    const { detectWebGpuAvailability } = await loadOrtModule();
    await expect(detectWebGpuAvailability()).resolves.toEqual({
      available: false,
      reason: "The browser did not return a WebGPU adapter."
    });
  });

  it("reports WebGPU unavailable when adapter lookup throws", async () => {
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: vi.fn().mockRejectedValue(new Error("GPU blocked"))
      }
    });
    const { detectWebGpuAvailability } = await loadOrtModule();
    await expect(detectWebGpuAvailability()).resolves.toEqual({
      available: false,
      reason: "GPU blocked"
    });
  });

  it("selects provider candidates from backend preference and WebGPU availability", async () => {
    const { getProviderCandidates } = await loadOrtModule();

    expect(getProviderCandidates("webgpu", { available: true, reason: "" })).toEqual([["webgpu"]]);
    expect(getProviderCandidates("wasm", { available: false, reason: "n/a" })).toEqual([["wasm"]]);
    expect(getProviderCandidates("auto", { available: true, reason: "" })).toEqual([
      ["webgpu"],
      ["wasm"]
    ]);
    expect(getProviderCandidates("auto", { available: false, reason: "n/a" })).toEqual([["wasm"]]);
    expect(() => getProviderCandidates("webgpu", { available: false, reason: "missing" })).toThrow(
      /WebGPU is unavailable: missing/
    );
  });

  it("initializes ORT runtime and applies wasm environment options", async () => {
    vi.stubGlobal("navigator", {});
    const mockedOrt = {
      env: {
        wasm: {}
      }
    };
    const { initOrtRuntime } = await loadOrtModule(mockedOrt);

    const runtime = await initOrtRuntime({
      wasmPaths: "/wasm/",
      numThreads: 2,
      simd: true,
      proxy: true,
      disableWasmProxy: true
    });

    expect(runtime.backend).toBe("auto");
    expect(runtime.webgpuState.available).toBe(false);
    expect(runtime.ort.env.wasm).toEqual({
      wasmPaths: "/wasm/",
      numThreads: 2,
      simd: true,
      proxy: false
    });
    expect(mockedOrt.env.wasm).toEqual({
      wasmPaths: "/wasm/",
      numThreads: 2,
      simd: true,
      proxy: false
    });
  });

  it("creates sessions by trying providers in order", async () => {
    const create = vi
      .fn()
      .mockRejectedValueOnce(new Error("webgpu failed"))
      .mockResolvedValueOnce({ id: "session" });
    const mockedOrt = {
      InferenceSession: { create }
    };
    const { createSession } = await loadOrtModule({
      env: { wasm: {} },
      InferenceSession: { create }
    });
    const modelBytes = new Uint8Array([1, 2, 3]);

    const result = await createSession(mockedOrt, modelBytes, [["webgpu"], ["wasm"]]);

    expect(create).toHaveBeenNthCalledWith(1, modelBytes, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all"
    });
    expect(create).toHaveBeenNthCalledWith(2, modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });
    expect(result).toEqual({
      session: { id: "session" },
      provider: "wasm"
    });
  });

  it("throws the last session creation error when all providers fail", async () => {
    const create = vi
      .fn()
      .mockRejectedValueOnce(new Error("webgpu failed"))
      .mockRejectedValueOnce(new Error("wasm failed"));
    const mockedOrt = {
      InferenceSession: { create }
    };
    const { createSession } = await loadOrtModule({
      env: { wasm: {} },
      InferenceSession: { create }
    });

    await expect(
      createSession(mockedOrt, new Uint8Array([1]), [["webgpu"], ["wasm"]])
    ).rejects.toThrow("wasm failed");
  });

  it("releases every session that exposes a release method", async () => {
    const releaseA = vi.fn().mockResolvedValue(undefined);
    const releaseB = vi.fn().mockResolvedValue(undefined);
    const { releaseSessions } = await loadOrtModule();

    await releaseSessions({ release: releaseA }, null, { release: releaseB }, {});

    expect(releaseA).toHaveBeenCalledTimes(1);
    expect(releaseB).toHaveBeenCalledTimes(1);
  });
});
