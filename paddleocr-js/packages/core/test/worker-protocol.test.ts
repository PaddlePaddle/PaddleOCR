import { describe, expect, it } from "vitest";

import {
  createTransportError,
  createTransportRequest,
  createTransportSuccess,
  deserializeError,
  isTransportRequest,
  isTransportResponse,
  serializeError
} from "../src/worker/protocol";

describe("worker transport protocol", () => {
  it("creates transport requests with the expected wire format", () => {
    const request = createTransportRequest("predict", { value: 1 }, 7);

    expect(request).toEqual({
      kind: "worker-transport-request",
      type: "predict",
      payload: { value: 1 },
      requestId: 7
    });
    expect(isTransportRequest(request)).toBe(true);
    expect(isTransportResponse(request)).toBe(false);
  });

  it("creates success responses", () => {
    const response = createTransportSuccess(7, { ok: true });

    expect(response).toEqual({
      kind: "worker-transport-response",
      status: "success",
      requestId: 7,
      payload: { ok: true }
    });
    expect(isTransportResponse(response)).toBe(true);
    expect(isTransportRequest(response)).toBe(false);
  });

  it("creates error responses by serializing the error", () => {
    const response = createTransportError(9, new TypeError("bad payload"));

    expect(response).toMatchObject({
      kind: "worker-transport-response",
      status: "error",
      requestId: 9,
      error: {
        name: "TypeError",
        message: "bad payload"
      }
    });
  });

  it("serializes unknown errors with defaults", () => {
    expect(serializeError(null)).toEqual({
      name: "Error",
      message: "Unknown worker error.",
      stack: ""
    });
  });

  it("deserializes error payloads back into Error instances", () => {
    const error = deserializeError({
      name: "RangeError",
      message: "out of range",
      stack: "stack-trace"
    });

    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("RangeError");
    expect(error.message).toBe("out of range");
    expect(error.stack).toBe("stack-trace");
  });
});
