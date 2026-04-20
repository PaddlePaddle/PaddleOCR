/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { createTransportError, createTransportSuccess, isTransportRequest } from "./protocol";

export type MessageHandler = (type: string, payload: Record<string, unknown>) => Promise<unknown>;

interface WorkerLikeScope {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown): void;
}

export function attachWorkerMessageHandler(
  handleMessage: MessageHandler,
  workerScope: WorkerLikeScope = self as unknown as WorkerLikeScope
): void {
  workerScope.onmessage = (event: MessageEvent) => {
    const message = event.data as unknown;
    if (!isTransportRequest(message)) {
      return;
    }

    void (async () => {
      try {
        const payload = await handleMessage(
          message.type,
          (message.payload || {}) as Record<string, unknown>
        );
        workerScope.postMessage(createTransportSuccess(message.requestId, payload));
      } catch (error: unknown) {
        workerScope.postMessage(createTransportError(message.requestId, error));
      }
    })();
  };
}
