/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export function nowMs(): number {
  return performance.now();
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function distance2(p0: [number, number], p1: [number, number]): number {
  const dx = p0[0] - p1[0];
  const dy = p0[1] - p1[1];
  return Math.sqrt(dx * dx + dy * dy);
}

export function formatMs(value: number): string {
  return `${value.toFixed(1)} ms`;
}

export function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let settled = false;
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error(`${label} timed out after ${String(ms / 1000)}s`));
    }, ms);

    promise
      .then((result) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(result);
      })
      .catch((err: unknown) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors -- propagating upstream rejection
        reject(err);
      });
  });
}

export function resolveRuntimeBatchSize(override: unknown, defaultBatchSize: number): number {
  const rawBatch = override ?? defaultBatchSize;
  const coercedBatch =
    typeof rawBatch === "number"
      ? rawBatch
      : typeof rawBatch === "string"
        ? Number.parseInt(rawBatch, 10)
        : Number.NaN;
  return Math.max(1, Number.isFinite(coercedBatch) ? coercedBatch : 1);
}

export function chunkArray<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    chunks.push(items.slice(i, i + size));
  }
  return chunks;
}

export function deepClone<T>(value: T): T {
  return structuredClone(value);
}
