const REQUEST_KIND = "worker-transport-request";
const RESPONSE_KIND = "worker-transport-response";

export function createTransportRequest(type, payload, requestId) {
  return {
    kind: REQUEST_KIND,
    type,
    payload,
    requestId
  };
}

export function createTransportSuccess(requestId, payload) {
  return {
    kind: RESPONSE_KIND,
    status: "success",
    requestId,
    payload
  };
}

export function createTransportError(requestId, error) {
  return {
    kind: RESPONSE_KIND,
    status: "error",
    requestId,
    error: serializeError(error)
  };
}

export function isTransportRequest(message) {
  return message?.kind === REQUEST_KIND;
}

export function isTransportResponse(message) {
  return message?.kind === RESPONSE_KIND;
}

export function serializeError(error) {
  return {
    name: error?.name || "Error",
    message: error?.message || "Unknown worker error.",
    stack: error?.stack || ""
  };
}

export function deserializeError(error) {
  const normalized = error || {};
  const instance = new Error(normalized.message || "Unknown worker error.");
  instance.name = normalized.name || "Error";
  if (normalized.stack) {
    instance.stack = normalized.stack;
  }
  return instance;
}
