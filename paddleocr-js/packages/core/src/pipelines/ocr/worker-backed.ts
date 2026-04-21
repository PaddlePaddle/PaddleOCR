/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { sourceToWorkerPayload } from "../../platform/browser";
import { createWorkerTransportClient } from "../../worker/client";
import type { WorkerTransportClient, WorkerOptions } from "../../worker/client";
import type { OcrModelConfig, OcrRuntimeParamsInput } from "./runtime-params";
import type { InitializationSummary, OcrResult, OcrPipelineRunnerOptions } from "./core";
import { cloneDefaultOcrConfig } from "./shared";

declare const __ORT_WASM_CDN_PREFIX__: string | undefined;

function createDefaultWorker(): Worker {
  if (typeof Worker !== "function") {
    throw new Error("worker mode requires Web Worker support in this environment.");
  }
  return new Worker(new URL("./worker-entry.ts", import.meta.url), {
    type: "module"
  });
}

export class WorkerBackedPaddleOCR {
  private options: OcrPipelineRunnerOptions;
  private lastInitializationSummary: InitializationSummary | null;
  private modelConfig: OcrModelConfig;
  private transportClient: WorkerTransportClient;
  private initPromise: Promise<InitializationSummary> | null;
  private disposed: boolean;

  constructor(options: OcrPipelineRunnerOptions, transportClient: WorkerTransportClient) {
    this.options = options;
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
      const ortOpts = (this.options.ortOptions || {}) as Record<string, unknown>;
      if (ortOpts["wasmPaths"] === undefined && typeof __ORT_WASM_CDN_PREFIX__ === "string") {
        console.warn(
          "[PaddleOCR.js] Worker mode: ortOptions.wasmPaths is not set — falling back to CDN (%s). " +
            "For version consistency between main thread and worker, set ortOptions.wasmPaths " +
            "to the path where your bundler outputs the onnxruntime-web WASM files " +
            '(e.g. ortOptions: { wasmPaths: "/assets/" }).',
          __ORT_WASM_CDN_PREFIX__
        );
      }
      const wasmCdnFallback =
        ortOpts["wasmPaths"] === undefined && typeof __ORT_WASM_CDN_PREFIX__ === "string"
          ? { wasmPaths: __ORT_WASM_CDN_PREFIX__ }
          : {};
      this.initPromise = this.transportClient
        .request("init", {
          options: {
            ...this.options,
            ortOptions: {
              ...ortOpts,
              ...wasmCdnFallback,
              disableWasmProxy: true
            }
          }
        })
        .then((rawPayload) => {
          const payload = rawPayload as {
            summary: InitializationSummary;
            modelConfig: OcrModelConfig;
          };
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

  async predict(input: unknown, params: OcrRuntimeParamsInput = {}): Promise<OcrResult[]> {
    this.ensureActive();
    await this.initialize();
    const sources: unknown[] = Array.isArray(input) ? input : [input];
    const payloads: Array<{ payload: unknown; transferables: Transferable[] }> = await Promise.all(
      sources.map((source) =>
        sourceToWorkerPayload(source as Parameters<typeof sourceToWorkerPayload>[0])
      )
    );
    const combinedPayloads = payloads.map((p) => p.payload);
    const combinedTransferables = payloads.flatMap((p) => p.transferables);
    return this.transportClient.request(
      "predict",
      {
        sources: combinedPayloads,
        params
      },
      combinedTransferables
    ) as Promise<OcrResult[]>;
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
  workerOptions: WorkerOptions = {}
): WorkerBackedPaddleOCR {
  const transportClient = createWorkerTransportClient({
    ...workerOptions,
    createWorker: workerOptions.createWorker || createDefaultWorker
  });
  return new WorkerBackedPaddleOCR(options, transportClient);
}
