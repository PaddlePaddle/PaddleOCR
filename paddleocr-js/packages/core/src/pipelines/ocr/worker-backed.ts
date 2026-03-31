import { sourceToWorkerPayload } from "../../platform/browser";
import { createWorkerTransportClient } from "../../worker/client";
import type { WorkerTransportClient, WorkerOptions } from "../../worker/client";
import type { OcrModelConfig, OcrRuntimeParamsInput } from "./runtime-params";
import type { InitializationSummary, OcrResult, OcrPipelineRunnerOptions } from "./core";
import { cloneDefaultOcrConfig } from "./shared";
import type { NormalizedPipelineConfig, PipelineRuntimeDefaults } from "./config";
import type { AssetDescriptor } from "../../resources/registry";

function createDefaultWorker(): Worker {
  if (typeof Worker !== "function") {
    throw new Error("worker mode requires Web Worker support in this environment.");
  }
  return new Worker(new URL("./worker-entry.ts", import.meta.url), {
    type: "module",
  });
}

export class WorkerBackedPaddleOCR {
  private options: OcrPipelineRunnerOptions;
  private runtimeDefaults: PipelineRuntimeDefaults;
  private assets: Record<string, AssetDescriptor>;
  private modelSelection: Record<string, string | null> | null;
  private pipelineConfig: NormalizedPipelineConfig | null;
  private lastInitializationSummary: InitializationSummary | null;
  private modelConfig: OcrModelConfig;
  private transportClient: WorkerTransportClient;
  private initPromise: Promise<InitializationSummary> | null;
  private disposed: boolean;

  constructor(options: OcrPipelineRunnerOptions, transportClient: WorkerTransportClient) {
    this.options = options;
    this.runtimeDefaults = { ...(options.runtimeDefaults || {}) };
    this.assets = options.assets;
    this.modelSelection = options.modelSelection || null;
    this.pipelineConfig = options.pipelineConfig || null;
    this.lastInitializationSummary = null;
    this.modelConfig = cloneDefaultOcrConfig();
    this.transportClient = transportClient;
    this.initPromise = null;
    this.disposed = false;
  }

  ensureActive(): void {
    if (this.disposed) {
      throw new Error("PaddleOCR worker instance has been disposed.");
    }
  }

  async initialize(): Promise<InitializationSummary> {
    this.ensureActive();
    if (this.lastInitializationSummary) {
      return this.lastInitializationSummary;
    }
    if (!this.initPromise) {
      this.initPromise = this.transportClient
        .request("init", {
          options: {
            ...this.options,
            runtime: {
              ...((this.options.runtime || {}) as Record<string, unknown>),
              disableWasmProxy: true,
            },
          },
        })
        .then((rawPayload) => {
          const payload = rawPayload as { summary: InitializationSummary; modelConfig: OcrModelConfig };
          this.lastInitializationSummary = payload.summary;
          this.modelConfig = payload.modelConfig;
          return this.lastInitializationSummary;
        })
        .catch((error: unknown) => {
          this.initPromise = null;
          this.transportClient.dispose();
          throw error;
        });
    }
    return this.initPromise;
  }

  getInitializationSummary(): InitializationSummary | null {
    return this.lastInitializationSummary;
  }

  getModelConfig(): OcrModelConfig {
    return this.modelConfig;
  }

  async predict(source: unknown, params: OcrRuntimeParamsInput = {}): Promise<OcrResult> {
    this.ensureActive();
    await this.initialize();
    const { payload, transferables } = await sourceToWorkerPayload(source as Parameters<typeof sourceToWorkerPayload>[0]);
    return this.transportClient.request(
      "predict",
      {
        source: payload,
        params,
      },
      transferables,
    ) as Promise<OcrResult>;
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    try {
      await this.transportClient.request("dispose", {});
    } catch {
      // Transport disposal is authoritative even if the worker cannot respond.
    }
    this.transportClient.dispose();
  }
}

export function createWorkerBackedPaddleOCR(
  options: OcrPipelineRunnerOptions,
  workerOptions: WorkerOptions = {},
): WorkerBackedPaddleOCR {
  const transportClient = createWorkerTransportClient({
    ...workerOptions,
    createWorker: workerOptions.createWorker || createDefaultWorker,
  });
  return new WorkerBackedPaddleOCR(options, transportClient);
}
