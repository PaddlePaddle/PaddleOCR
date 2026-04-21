import { afterEach, describe, expect, it, vi } from "vitest";

async function loadOpenCvModule(mockedDefault) {
  vi.resetModules();
  vi.doMock("@techstark/opencv-js", () => ({
    __esModule: true,
    default: mockedDefault
  }));
  return import("../src/runtime/opencv");
}

afterEach(() => {
  vi.resetModules();
  vi.doUnmock("@techstark/opencv-js");
});

describe("runtime/opencv", () => {
  it("returns OpenCV immediately when the module already exposes Mat", async () => {
    const cv = { Mat() {} };
    const { initOpenCvRuntime } = await loadOpenCvModule(cv);

    await expect(initOpenCvRuntime()).resolves.toEqual({ cv });
  });

  it("awaits promised OpenCV modules", async () => {
    const cv = { Mat() {} };
    const { initOpenCvRuntime } = await loadOpenCvModule(Promise.resolve(cv));

    await expect(initOpenCvRuntime()).resolves.toEqual({ cv });
  });

  it("awaits callback-based OpenCV initialization", async () => {
    const cv = {};
    Object.defineProperty(cv, "onRuntimeInitialized", {
      set(callback) {
        callback();
      }
    });
    const { initOpenCvRuntime } = await loadOpenCvModule(cv);

    await expect(initOpenCvRuntime()).resolves.toEqual({ cv });
  });

  it("caches successful initialization results", async () => {
    const cv = { Mat() {} };
    const { initOpenCvRuntime } = await loadOpenCvModule(cv);

    const first = await initOpenCvRuntime();
    const second = await initOpenCvRuntime();

    expect(first).toBe(second);
  });

  it("re-exports initOpenCvRuntime from the runtime index", async () => {
    const cv = { Mat() {} };
    const runtimeModule = await loadOpenCvModule(cv);
    const runtimeIndex = await import("../src/runtime/index");

    expect(runtimeIndex.initOpenCvRuntime).toBe(runtimeModule.initOpenCvRuntime);
  });
});
