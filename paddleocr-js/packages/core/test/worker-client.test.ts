import { describe, expect, it } from "vitest";

import { WorkerTransportClient, createWorkerTransportClient } from "../src/worker/client";
import { createTransportSuccess } from "../src/worker/protocol";

class MockWorker {
  constructor() {
    this.messages = [];
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
  }

  postMessage(message, transferables) {
    this.messages.push({ message, transferables });
  }

  terminate() {
    this.terminated = true;
  }
}

function createClientWithWorker() {
  const worker = new MockWorker();
  const client = new WorkerTransportClient({
    createWorker: () => worker
  });
  return { worker, client };
}

describe("worker transport client", () => {
  it("creates a worker lazily and resolves request responses", async () => {
    const { worker, client } = createClientWithWorker();

    const responsePromise = client.request("predict", { value: 1 }, ["xfer"]);
    expect(worker.messages).toHaveLength(1);
    expect(worker.messages[0]).toMatchObject({
      message: {
        kind: "worker-transport-request",
        type: "predict",
        payload: { value: 1 },
        requestId: 1
      },
      transferables: ["xfer"]
    });

    worker.onmessage({
      data: createTransportSuccess(1, { ok: true })
    });

    await expect(responsePromise).resolves.toEqual({ ok: true });
  });

  it("rejects when the worker reports an error response", async () => {
    const { worker, client } = createClientWithWorker();

    const responsePromise = client.request("predict", {});
    worker.onmessage({
      data: {
        kind: "worker-transport-response",
        status: "error",
        requestId: 1,
        error: {
          name: "TypeError",
          message: "bad payload",
          stack: "stack"
        }
      }
    });

    await expect(responsePromise).rejects.toMatchObject({
      name: "TypeError",
      message: "bad payload",
      stack: "stack"
    });
  });

  it("ignores messages that are not transport responses", async () => {
    const { worker, client } = createClientWithWorker();

    const responsePromise = client.request("predict", {});
    worker.onmessage({
      data: { kind: "other" }
    });
    worker.onmessage({
      data: createTransportSuccess(1, { ok: true })
    });

    await expect(responsePromise).resolves.toEqual({ ok: true });
  });

  it("rejects all pending requests when the worker errors", async () => {
    const { worker, client } = createClientWithWorker();

    const first = client.request("init", {});
    const second = client.request("predict", {});
    worker.onerror({ message: "worker crashed" });

    await expect(first).rejects.toThrow("worker crashed");
    await expect(second).rejects.toThrow("worker crashed");
  });

  it("rejects requests after disposal and terminates the worker", async () => {
    const worker = new MockWorker();
    const client = createWorkerTransportClient({
      createWorker: () => worker
    });

    const pending = client.request("predict", {});
    client.dispose();

    await expect(pending).rejects.toThrow("Worker transport client has been disposed.");
    expect(worker.terminated).toBe(true);
    expect(() => client.dispose()).not.toThrow();
    expect(() => client.ensureActive()).toThrow(/has been disposed/i);
  });

  it("requires a createWorker factory when the worker is first needed", () => {
    const client = new WorkerTransportClient();

    expect(() => client.ensureWorker()).toThrow(/requires a createWorker\(\) factory/i);
  });

  it("allows disposing before a worker has ever been created", () => {
    const client = new WorkerTransportClient({
      createWorker: () => new MockWorker()
    });

    expect(() => client.disposeWorker()).not.toThrow();
    expect(() => client.dispose()).not.toThrow();
  });
});
