import { createTransportError, createTransportSuccess, isTransportRequest } from "./protocol.js";

export function attachWorkerMessageHandler(handleMessage, workerScope = self) {
  workerScope.onmessage = async (event) => {
    const message = event.data;
    if (!isTransportRequest(message)) {
      return;
    }

    try {
      const payload = await handleMessage(message.type, message.payload || {});
      workerScope.postMessage(createTransportSuccess(message.requestId, payload));
    } catch (error) {
      workerScope.postMessage(createTransportError(message.requestId, error));
    }
  };
}
