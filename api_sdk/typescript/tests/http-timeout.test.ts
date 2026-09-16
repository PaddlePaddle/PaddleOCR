import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { expect, test } from "vitest";
import { PaddleOCRClient, RequestTimeoutError } from "../src/index.js";
import { HttpClient } from "../src/internal/http.js";

test.each([
  { method: "status", status: 200 },
  { method: "status", status: 503 },
  { method: "jsonl", status: 200 },
  { method: "resource", status: 200 },
  { method: "cancel", status: 200 },
])("$method interrupts a delayed HTTP $status body", async ({ method, status }) => {
  const controller = new AbortController();
  const reason = new Error("cancelled by caller");
  const server = createServer((_req, res) => {
    res.writeHead(status, { "Content-Type": "application/json" });
    res.flushHeaders();
    const abortTimer = method === "cancel" ? setTimeout(() => controller.abort(reason), 50) : undefined;
    const timer = setTimeout(() => {
      res.end(JSON.stringify({ data: { state: "running" }, msg: "unavailable" }));
    }, 300);
    res.on("close", () => {
      clearTimeout(timer);
      clearTimeout(abortTimer);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    const requestTimeout = method === "cancel" ? 2000 : 100;
    const client = new PaddleOCRClient({
      token: "test-token",
      baseUrl,
      requestTimeout,
    });
    const http = new HttpClient("test-token", baseUrl, requestTimeout);
    const request = method === "jsonl" ? http.fetchJsonl(baseUrl)
      : method === "resource" ? http.fetchResource(baseUrl)
      : client.getStatus("job-1", { signal: controller.signal });
    if (method === "cancel") {
      await expect(request).rejects.toBe(reason);
    } else {
      await expect(request).rejects.toBeInstanceOf(RequestTimeoutError);
    }
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
});
