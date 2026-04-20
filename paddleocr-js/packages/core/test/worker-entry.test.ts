import { describe, expect, it, vi } from "vitest";

import { attachWorkerMessageHandler } from "../src/worker/entry";

function createWorkerScope() {
  return {
    postMessage: vi.fn()
  };
}

function transportRequest(overrides) {
  return {
    kind: "worker-transport-request",
    type: "predict",
    payload: {},
    requestId: 4,
    ...overrides
  };
}

describe("worker entry message handler", () => {
  it("ignores messages that do not use the transport protocol", async () => {
    const scope = createWorkerScope();

    attachWorkerMessageHandler(async () => ({}), scope);
    await scope.onmessage({ data: { kind: "other" } });

    expect(scope.postMessage).not.toHaveBeenCalled();
  });

  it("posts success responses for handled messages", async () => {
    const scope = createWorkerScope();

    attachWorkerMessageHandler(async (type, payload) => ({ type, payload }), scope);
    await scope.onmessage({
      data: transportRequest({
        payload: { value: 1 },
        requestId: 3
      })
    });

    expect(scope.postMessage).toHaveBeenCalledWith({
      kind: "worker-transport-response",
      status: "success",
      requestId: 3,
      payload: {
        type: "predict",
        payload: { value: 1 }
      }
    });
  });

  it("posts error responses when the handler throws", async () => {
    const scope = createWorkerScope();

    attachWorkerMessageHandler(async () => {
      throw new Error("boom");
    }, scope);

    await scope.onmessage({
      data: transportRequest()
    });

    expect(scope.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "worker-transport-response",
        status: "error",
        requestId: 4,
        error: expect.objectContaining({
          name: "Error",
          message: "boom"
        })
      })
    );
  });
});
