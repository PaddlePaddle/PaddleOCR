/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const REQUEST_KIND = "worker-transport-request";
const RESPONSE_KIND = "worker-transport-response";

export interface SerializedError {
  name: string;
  message: string;
  stack: string;
}

export interface TransportRequest {
  kind: typeof REQUEST_KIND;
  type: string;
  payload: unknown;
  requestId: number;
}

export interface TransportSuccessResponse {
  kind: typeof RESPONSE_KIND;
  status: "success";
  requestId: number;
  payload: unknown;
}

export interface TransportErrorResponse {
  kind: typeof RESPONSE_KIND;
  status: "error";
  requestId: number;
  error: SerializedError;
}

export type TransportResponse = TransportSuccessResponse | TransportErrorResponse;

export function createTransportRequest(
  type: string,
  payload: unknown,
  requestId: number
): TransportRequest {
  return {
    kind: REQUEST_KIND,
    type,
    payload,
    requestId
  };
}

export function createTransportSuccess(
  requestId: number,
  payload: unknown
): TransportSuccessResponse {
  return {
    kind: RESPONSE_KIND,
    status: "success",
    requestId,
    payload
  };
}

export function createTransportError(requestId: number, error: unknown): TransportErrorResponse {
  return {
    kind: RESPONSE_KIND,
    status: "error",
    requestId,
    error: serializeError(error)
  };
}

export function isTransportRequest(message: unknown): message is TransportRequest {
  return (
    typeof message === "object" &&
    message !== null &&
    "kind" in message &&
    message.kind === REQUEST_KIND
  );
}

export function isTransportResponse(message: unknown): message is TransportResponse {
  return (
    typeof message === "object" &&
    message !== null &&
    "kind" in message &&
    message.kind === RESPONSE_KIND
  );
}

export function serializeError(error: unknown): SerializedError {
  const err = error as Partial<Error> | undefined;
  return {
    name: err?.name || "Error",
    message: err?.message || "Unknown worker error.",
    stack: err?.stack || ""
  };
}

export function deserializeError(error: unknown): Error {
  const normalized = (error || {}) as SerializedError;
  const instance = new Error(normalized.message || "Unknown worker error.");
  instance.name = normalized.name || "Error";
  if (normalized.stack) {
    instance.stack = normalized.stack;
  }
  return instance;
}
