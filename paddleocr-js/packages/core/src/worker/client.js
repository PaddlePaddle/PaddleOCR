import { createTransportRequest, deserializeError, isTransportResponse } from "./protocol.js";

export class WorkerTransportClient {
  constructor(workerOptions = {}) {
    this.workerOptions = workerOptions;
    this.worker = null;
    this.pending = new Map();
    this.nextRequestId = 1;
    this.disposed = false;
  }

  ensureActive() {
    if (this.disposed) {
      throw new Error("Worker transport client has been disposed.");
    }
  }

  ensureWorker() {
    this.ensureActive();
    if (this.worker) {
      return this.worker;
    }

    const workerFactory = this.workerOptions.createWorker;
    if (typeof workerFactory !== "function") {
      throw new Error("Worker transport client requires a createWorker() factory.");
    }
    const worker = workerFactory();
    worker.onmessage = (event) => {
      const message = event.data;
      if (!isTransportResponse(message)) return;
      const pending = this.pending.get(message.requestId);
      if (!pending) return;
      this.pending.delete(message.requestId);
      if (message.status === "success") {
        pending.resolve(message.payload);
      } else {
        pending.reject(deserializeError(message.error));
      }
    };
    worker.onerror = (event) => {
      const error = new Error(event.message || "OCR worker failed.");
      for (const pending of this.pending.values()) {
        pending.reject(error);
      }
      this.pending.clear();
    };
    this.worker = worker;
    return worker;
  }

  request(type, payload, transferables = []) {
    const worker = this.ensureWorker();
    const requestId = this.nextRequestId;
    this.nextRequestId += 1;

    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      worker.postMessage(createTransportRequest(type, payload, requestId), transferables);
    });
  }

  disposeWorker() {
    if (!this.worker) {
      return;
    }
    this.worker.terminate();
    this.worker = null;
  }

  async dispose() {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    for (const pending of this.pending.values()) {
      pending.reject(new Error("Worker transport client has been disposed."));
    }
    this.pending.clear();
    this.disposeWorker();
  }
}

export function createWorkerTransportClient(workerOptions) {
  return new WorkerTransportClient(workerOptions);
}
