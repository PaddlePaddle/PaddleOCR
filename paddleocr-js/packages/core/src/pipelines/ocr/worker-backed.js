import { sourceToWorkerPayload } from "../../platform/browser.js";
import { createWorkerTransportClient } from "../../worker/client.js";
import { cloneDefaultOcrConfig } from "./shared.js";

function createDefaultWorker() {
  if (typeof Worker !== "function") {
    throw new Error("worker mode requires Web Worker support in this environment.");
  }
  return new Worker(new URL("./worker-entry.js", import.meta.url), {
    type: "module"
  });
}

export class WorkerBackedPaddleOCR {
  constructor(options, transportClient) {
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

  ensureActive() {
    if (this.disposed) {
      throw new Error("PaddleOCR worker instance has been disposed.");
    }
  }

  async initialize() {
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
              ...(this.options.runtime || {}),
              disableWasmProxy: true
            }
          }
        })
        .then((payload) => {
          this.lastInitializationSummary = payload.summary;
          this.modelConfig = payload.modelConfig;
          return this.lastInitializationSummary;
        })
        .catch((error) => {
          this.initPromise = null;
          this.transportClient.dispose();
          throw error;
        });
    }
    return this.initPromise;
  }

  getInitializationSummary() {
    return this.lastInitializationSummary;
  }

  getModelConfig() {
    return this.modelConfig;
  }

  async predict(source, params = {}) {
    this.ensureActive();
    await this.initialize();
    const { payload, transferables } = await sourceToWorkerPayload(source);
    return this.transportClient.request(
      "predict",
      {
        source: payload,
        params
      },
      transferables
    );
  }

  async dispose() {
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

export function createWorkerBackedPaddleOCR(options, workerOptions = {}) {
  const transportClient = createWorkerTransportClient({
    ...workerOptions,
    createWorker: workerOptions.createWorker || createDefaultWorker
  });
  return new WorkerBackedPaddleOCR(options, transportClient);
}
