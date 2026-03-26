export function nowMs() {
  return performance.now();
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function distance2(p0, p1) {
  const dx = p0[0] - p1[0];
  const dy = p0[1] - p1[1];
  return Math.sqrt(dx * dx + dy * dy);
}

export function formatMs(value) {
  return `${value.toFixed(1)} ms`;
}

export function withTimeout(promise, ms, label) {
  let settled = false;
  return new Promise((resolve, reject) => {
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
      .catch((err) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(err);
      });
  });
}

export function deepClone(value) {
  return structuredClone(value);
}
