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
      reject(new Error(`${label} timed out after ${ms / 1000}s`));
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
        reject(err);
      });
  });
}

export function deepClone<T>(value: T): T {
  return structuredClone(value);
}
