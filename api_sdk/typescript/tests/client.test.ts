import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test, vi } from "vitest";
import {
  APIError,
  AuthError,
  FileNotFoundError,
  InvalidRequestError,
  JobFailedError,
  Model,
  PaddleOCRClient,
  PollTimeoutError,
  RequestTimeoutError,
  ResponseFormatError,
  ResultParseError,
} from "../src/index.js";
import type { DocParsingResult, Job, OCRResult } from "../src/index.js";

type FetchCall = {
  url: string;
  init: RequestInit;
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function textResponse(body: string, status = 200): Response {
  return new Response(body, { status });
}

function createClient(fetchImpl: typeof fetch, options: Partial<ConstructorParameters<typeof PaddleOCRClient>[0]> = {}) {
  return new PaddleOCRClient({
    token: "test-token",
    baseUrl: "https://api.example.test",
    requestTimeout: 100,
    pollTimeout: 100,
    fetch: fetchImpl,
    ...options,
  });
}

function captureFetch(responses: Array<Response | Error | Promise<Response>>): { fetch: typeof fetch; calls: FetchCall[] } {
  const calls: FetchCall[] = [];
  const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init: init ?? {} });
    const next = responses.shift();
    if (next instanceof Error) {
      throw next;
    }
    if (!next) {
      throw new Error("No mocked response remaining");
    }
    return next;
  }) as unknown as typeof fetch;
  return { fetch: fetchImpl, calls };
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("PaddleOCRClient public contract", () => {
  test("requires token at construction and reads PADDLEOCR_ACCESS_TOKEN fallback", () => {
    const previous = process.env.PADDLEOCR_ACCESS_TOKEN;
    delete process.env.PADDLEOCR_ACCESS_TOKEN;
    expect(() => new PaddleOCRClient()).toThrow(AuthError);

    process.env.PADDLEOCR_ACCESS_TOKEN = "env-token";
    const { fetch } = captureFetch([jsonResponse({ data: { state: "running" } })]);
    expect(() => new PaddleOCRClient({ baseUrl: "https://api.example.test", fetch })).not.toThrow();

    if (previous === undefined) {
      delete process.env.PADDLEOCR_ACCESS_TOKEN;
    } else {
      process.env.PADDLEOCR_ACCESS_TOKEN = previous;
    }
  });

  test("exposes contract names and omits unpublished draft aliases", () => {
    const { fetch } = captureFetch([]);
    const client = createClient(fetch);
    expect(client.getStatus).toBeTypeOf("function");
    expect(client.parseDocument).toBeTypeOf("function");
    expect(client.submitDocumentParsing).toBeTypeOf("function");
    expect(client.waitOcrResult).toBeTypeOf("function");
    expect(client.waitDocumentParsingResult).toBeTypeOf("function");
    expect(client.saveResource).toBeTypeOf("function");
    expect(client.saveOcrResultResources).toBeTypeOf("function");
    expect(client.saveDocumentParsingResultResources).toBeTypeOf("function");

    expect("getResult" in client).toBe(false);
    expect("waitForResult" in client).toBe(false);
    expect("docParsing" in client).toBe(false);
    expect("submitDocParsing" in client).toBe(false);
  });

  test("submitOcr sends contract body and returns job metadata", async () => {
    const { fetch, calls } = captureFetch([jsonResponse({ data: { jobId: "job-1" } })]);
    const client = createClient(fetch);

    const job = await client.submitOcr({
      fileUrl: "https://files.example.test/invoice.pdf",
      pageRanges: "1-2",
      batchId: "batch-1",
      options: { visualize: true },
    });

    expect(job).toEqual({
      jobId: "job-1",
      model: Model.PPOCRv5,
      task: "ocr",
      pageRanges: "1-2",
      batchId: "batch-1",
    });
    expect(calls[0].url).toBe("https://api.example.test/api/v2/ocr/jobs");
    expect(calls[0].init.method).toBe("POST");
    expect(calls[0].init.headers).toMatchObject({
      Authorization: "Bearer test-token",
      "Content-Type": "application/json",
    });
    expect(JSON.parse(String(calls[0].init.body))).toEqual({
      fileUrl: "https://files.example.test/invoice.pdf",
      model: Model.PPOCRv5,
      optionalPayload: { visualize: true },
      pageRanges: "1-2",
      batchId: "batch-1",
    });
  });

  test("clientPlatform is sent as Client-Platform header on API requests", async () => {
    const { fetch, calls } = captureFetch([jsonResponse({ data: { jobId: "job-1" } })]);
    const client = createClient(fetch, { clientPlatform: "my-app" });

    await client.submitOcr({
      fileUrl: "https://files.example.test/invoice.pdf",
    });

    expect(calls[0].init.headers).toMatchObject({
      "Client-Platform": "my-app",
    });
  });

  test("submitOcr propagates explicit OCR model and defaults to PP-OCRv5", async () => {
    const { fetch, calls } = captureFetch([
      jsonResponse({ data: { jobId: "job-explicit" } }),
      jsonResponse({ data: { jobId: "job-default" } }),
    ]);
    const client = createClient(fetch);

    const explicit = await client.submitOcr({
      model: Model.PPOCRv5,
      fileUrl: "https://files.example.test/explicit.pdf",
    });
    const implicit = await client.submitOcr({
      fileUrl: "https://files.example.test/default.pdf",
    });

    expect(explicit.model).toBe(Model.PPOCRv5);
    expect(implicit.model).toBe(Model.PPOCRv5);
    expect(JSON.parse(String(calls[0].init.body)).model).toBe(Model.PPOCRv5);
    expect(JSON.parse(String(calls[1].init.body)).model).toBe(Model.PPOCRv5);
  });

  test("submitOcr and submitDocumentParsing accept official model name strings", async () => {
    const { fetch, calls } = captureFetch([
      jsonResponse({ data: { jobId: "job-ocr" } }),
      jsonResponse({ data: { jobId: "job-doc" } }),
    ]);
    const client = createClient(fetch);

    const ocrJob = await client.submitOcr({
      model: "PP-OCRv5",
      fileUrl: "https://files.example.test/ocr.pdf",
    });
    const docJob = await client.submitDocumentParsing({
      model: "PaddleOCR-VL-1.6",
      fileUrl: "https://files.example.test/doc.pdf",
    });

    expect(ocrJob.model).toBe("PP-OCRv5");
    expect(docJob.model).toBe("PaddleOCR-VL-1.6");
    expect(JSON.parse(String(calls[0].init.body)).model).toBe("PP-OCRv5");
    expect(JSON.parse(String(calls[1].init.body)).model).toBe("PaddleOCR-VL-1.6");
  });

  test("model helpers classify current OCR and document parsing models", async () => {
    const mod = await import("../src/index.js");

    expect(mod.isOCRModel(Model.PPOCRv5)).toBe(true);
    expect(mod.isOCRModel(Model.PPStructureV3)).toBe(false);
    expect(mod.isOCRModel("future-unknown-model")).toBe(false);
    expect(mod.isDocumentParsingModel(Model.PaddleOCRVL)).toBe(true);
    expect(mod.isDocumentParsingModel(Model.PaddleOCRVL16)).toBe(true);
    expect(mod.isDocumentParsingModel(Model.PPOCRv5)).toBe(false);
  });

  test("submitDocumentParsing defaults to PaddleOCR-VL-1.6", async () => {
    const { fetch, calls } = captureFetch([jsonResponse({ data: { jobId: "job-doc" } })]);
    const client = createClient(fetch);

    const job = await client.submitDocumentParsing({
      fileUrl: "https://files.example.test/doc.pdf",
    });

    expect(job.model).toBe(Model.PaddleOCRVL16);
    expect(JSON.parse(String(calls[0].init.body)).model).toBe(Model.PaddleOCRVL16);
  });

  test("submitOcr rejects non-OCR models before network calls", async () => {
    const { fetch } = captureFetch([]);
    const client = createClient(fetch);

    await expect(
      client.submitOcr({
        model: Model.PPStructureV3,
        fileUrl: "https://files.example.test/not-ocr.pdf",
      }),
    ).rejects.toThrow(InvalidRequestError);
    expect(fetch).not.toHaveBeenCalled();
  });

  test("validates submit requests before network calls", async () => {
    const { fetch } = captureFetch([]);
    const client = createClient(fetch);

    await expect(client.submitOcr({})).rejects.toThrow(InvalidRequestError);
    await expect(
      client.submitOcr({ fileUrl: "https://files.example.test/a.pdf", filePath: "./a.pdf" }),
    ).rejects.toThrow(InvalidRequestError);
    expect(fetch).not.toHaveBeenCalled();
  });

  test("getStatus performs one non-blocking status request", async () => {
    const { fetch, calls } = captureFetch([
      jsonResponse({
        data: {
          state: "running",
          extractProgress: { totalPages: 3, extractedPages: 1 },
        },
      }),
    ]);
    const client = createClient(fetch);

    await expect(client.getStatus("job-1")).resolves.toEqual({
      jobId: "job-1",
      state: "running",
      progress: { totalPages: 3, extractedPages: 1 },
      errorMsg: undefined,
    });
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("https://api.example.test/api/v2/ocr/jobs/job-1");
  });

  test("normalizes multiple trailing baseUrl slashes", async () => {
    const { fetch, calls } = captureFetch([jsonResponse({ data: { state: "running" } })]);
    const client = createClient(fetch, { baseUrl: "https://api.example.test///" });

    await client.getStatus("job-1");

    expect(calls[0].url).toBe("https://api.example.test/api/v2/ocr/jobs/job-1");
  });

  test("typed wait accepts bare jobId, fetches JSONL without Authorization, and parses OCR results", async () => {
    const { fetch, calls } = captureFetch([
      jsonResponse({ data: { state: "done", resultUrl: { jsonUrl: "https://storage.example.test/job-1.jsonl" } } }),
      textResponse(JSON.stringify({ result: { ocrResults: [{ prunedResult: { text: "hello" }, ocrImage: "img.png" }] } })),
    ]);
    const client = createClient(fetch);

    await expect(client.waitOcrResult("job-1")).resolves.toMatchObject({
      jobId: "job-1",
      pages: [{ prunedResult: { text: "hello" }, ocrImageUrl: "img.png" }],
    });
    expect(calls[1].url).toBe("https://storage.example.test/job-1.jsonl");
    expect((calls[1].init.headers as Record<string, string>).Authorization).toBeUndefined();
  });

  test("typed waits reject mismatched Job task before polling", async () => {
    const { fetch } = captureFetch([]);
    const client = createClient(fetch);
    const docJob: Job = { jobId: "job-1", model: Model.PPStructureV3, task: "document_parsing" };

    await expect(client.waitOcrResult(docJob)).rejects.toThrow(InvalidRequestError);
    expect(fetch).not.toHaveBeenCalled();
  });

  test("typed waits reject mismatched Job model before polling", async () => {
    const { fetch } = captureFetch([]);
    const client = createClient(fetch);
    const ocrTaskWithDocumentModel: Job = { jobId: "job-1", model: Model.PPStructureV3, task: "ocr" };
    const documentTaskWithOcrModel: Job = { jobId: "job-2", model: Model.PPOCRv5, task: "document_parsing" };

    await expect(client.waitOcrResult(ocrTaskWithDocumentModel)).rejects.toThrow(InvalidRequestError);
    await expect(client.waitDocumentParsingResult(documentTaskWithOcrModel)).rejects.toThrow(InvalidRequestError);
    expect(fetch).not.toHaveBeenCalled();
  });

  test("polling maps failed, timeout, unknown state, and missing result URL", async () => {
    const failed = createClient(captureFetch([jsonResponse({ data: { state: "failed", errorMsg: "boom" } })]).fetch);
    await expect(failed.waitOcrResult("job-1")).rejects.toThrow(JobFailedError);

    const unknown = createClient(captureFetch([jsonResponse({ data: { state: "paused" } })]).fetch);
    await expect(unknown.waitOcrResult("job-1")).rejects.toThrow(ResponseFormatError);

    const missingResult = createClient(captureFetch([jsonResponse({ data: { state: "done" } })]).fetch);
    await expect(missingResult.waitOcrResult("job-1")).rejects.toThrow(ResponseFormatError);

    vi.useFakeTimers();
    const timeoutFetch = vi.fn(async () => jsonResponse({ data: { state: "running" } })) as unknown as typeof fetch;
    const timeout = createClient(timeoutFetch, { pollTimeout: 1 });
    const promise = expect(timeout.waitOcrResult("job-1")).rejects.toThrow(PollTimeoutError);
    await vi.runAllTimersAsync();
    await promise;
  });

  test("malformed successful responses and malformed JSONL use dedicated errors", async () => {
    const malformedSubmit = createClient(captureFetch([jsonResponse({ data: {} })]).fetch);
    await expect(malformedSubmit.submitOcr({ fileUrl: "https://files.example.test/a.pdf" })).rejects.toThrow(
      ResponseFormatError,
    );

    const malformedJsonl = createClient(
      captureFetch([
        jsonResponse({ data: { state: "done", resultUrl: { jsonUrl: "https://storage.example.test/job.jsonl" } } }),
        textResponse("{not-json}\n"),
      ]).fetch,
    );
    await expect(malformedJsonl.waitOcrResult("job-1")).rejects.toThrow(ResultParseError);
  });

  test("malformed fetched OCR result records use ResultParseError", async () => {
    const client = createClient(
      captureFetch([
        jsonResponse({ data: { state: "done", resultUrl: { jsonUrl: "https://storage.example.test/job.jsonl" } } }),
        textResponse(JSON.stringify({ notResult: {} })),
      ]).fetch,
    );

    await expect(client.waitOcrResult("job-1")).rejects.toThrow(ResultParseError);
  });

  test.each([
    ["missing ocrResults", { result: {} }],
    ["missing prunedResult", { result: { ocrResults: [{ ocrImage: "img.png" }] } }],
  ])("malformed OCR payload with %s uses ResultParseError", async (_name, payload) => {
    const client = createClient(
      captureFetch([
        jsonResponse({ data: { state: "done", resultUrl: { jsonUrl: "https://storage.example.test/job.jsonl" } } }),
        textResponse(JSON.stringify(payload)),
      ]).fetch,
    );

    await expect(client.waitOcrResult("job-1")).rejects.toThrow(ResultParseError);
  });

  test("malformed fetched document result records use ResultParseError", async () => {
    const client = createClient(
      captureFetch([
        jsonResponse({ data: { state: "done", resultUrl: { jsonUrl: "https://storage.example.test/job.jsonl" } } }),
        textResponse(JSON.stringify({ notResult: {} })),
      ]).fetch,
    );

    await expect(client.waitDocumentParsingResult("job-1")).rejects.toThrow(ResultParseError);
  });

  test.each([
    ["missing layoutParsingResults", { result: {} }],
    ["missing markdown", { result: { layoutParsingResults: [{}] } }],
    ["missing markdown.text", { result: { layoutParsingResults: [{ markdown: { images: {} } }] } }],
  ])("malformed document payload with %s uses ResultParseError", async (_name, payload) => {
    const client = createClient(
      captureFetch([
        jsonResponse({ data: { state: "done", resultUrl: { jsonUrl: "https://storage.example.test/job.jsonl" } } }),
        textResponse(JSON.stringify(payload)),
      ]).fetch,
    );

    await expect(client.waitDocumentParsingResult("job-1")).rejects.toThrow(ResultParseError);
  });

  test("HTTP status and timeout failures map to contract errors", async () => {
    await expect(createClient(captureFetch([textResponse("bad auth", 401)]).fetch).getStatus("job-1")).rejects.toThrow(
      AuthError,
    );
    await expect(createClient(captureFetch([textResponse("bad request", 400)]).fetch).getStatus("job-1")).rejects.toThrow(
      InvalidRequestError,
    );
    await expect(createClient(captureFetch([textResponse("server error", 500)]).fetch).getStatus("job-1")).rejects.toThrow(
      APIError,
    );

    const timeoutFetch = vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
    ) as unknown as typeof fetch;
    await expect(createClient(timeoutFetch, { requestTimeout: 1 }).getStatus("job-1")).rejects.toThrow(
      RequestTimeoutError,
    );
  });

  test("submitFile reports missing local files before network calls", async () => {
    const { fetch } = captureFetch([]);
    const client = createClient(fetch);

    await expect(client.submitOcr({ filePath: "/definitely/missing/file.pdf" })).rejects.toThrow(FileNotFoundError);
    expect(fetch).not.toHaveBeenCalled();
  });

  test("saveResource writes URL downloads without auth and requires overwrite opt-in", async () => {
    const dir = await mkdtemp(join(tmpdir(), "paddleocr-api-sdk-"));
    try {
      const { fetch, calls } = captureFetch([textResponse("content"), textResponse("replacement")]);
      const client = createClient(fetch);

      const saved = await client.saveResource("https://storage.example.test/a/b/c.png", dir);
      expect(saved).toBe(join(dir, "c.png"));
      await expect(readFile(join(dir, "c.png"), "utf8")).resolves.toBe("content");
      expect((calls[0].init.headers as Record<string, string>).Authorization).toBeUndefined();

      await expect(client.saveResource("https://storage.example.test/a/b/c.png", dir)).rejects.toThrow(
        InvalidRequestError,
      );

      await client.saveResource("https://storage.example.test/a/b/c.png", dir, { overwrite: true });
      await expect(readFile(join(dir, "c.png"), "utf8")).resolves.toBe("replacement");
      await expect(stat(join(dir, "c.png"))).resolves.toBeTruthy();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("saveOcrResultResources saves OCR result image URLs with stable page filenames", async () => {
    const dir = await mkdtemp(join(tmpdir(), "paddleocr-api-sdk-"));
    try {
      const { fetch, calls } = captureFetch([textResponse("page 1"), textResponse("page 2")]);
      const client = createClient(fetch);
      const result: OCRResult = {
        jobId: "job-ocr",
        pages: [
          { prunedResult: { text: "one" }, ocrImageUrl: "https://storage.example.test/results/page-a.png?sig=opaque" },
          { prunedResult: { text: "two" }, ocrImageUrl: "https://storage.example.test/results/page-b.jpg" },
        ],
      };

      const saved = await client.saveOcrResultResources(result, dir);

      expect(saved).toEqual([join(dir, "ocr-page-1.png"), join(dir, "ocr-page-2.jpg")]);
      await expect(readFile(join(dir, "ocr-page-1.png"), "utf8")).resolves.toBe("page 1");
      await expect(readFile(join(dir, "ocr-page-2.jpg"), "utf8")).resolves.toBe("page 2");
      expect(calls).toHaveLength(2);
      expect(calls.map((call) => (call.init.headers as Record<string, string>).Authorization)).toEqual([
        undefined,
        undefined,
      ]);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("saveDocumentParsingResultResources preserves safe document parsing map keys over opaque URL basenames", async () => {
    const dir = await mkdtemp(join(tmpdir(), "paddleocr-api-sdk-"));
    try {
      const { fetch } = captureFetch([textResponse("markdown image"), textResponse("output image")]);
      const client = createClient(fetch);
      const result: DocParsingResult = {
        jobId: "job-doc",
        pages: [
          {
            markdownText: "![figure](figure 1.png)",
            markdownImages: {
              "figure 1.png": "https://storage.example.test/download?id=markdown-image",
            },
            outputImages: {
              "rendered-page.jpg": "https://storage.example.test/blob?id=output-image",
            },
          },
        ],
      };

      const saved = await client.saveDocumentParsingResultResources(result, dir);

      expect(saved).toEqual([join(dir, "figure 1.png"), join(dir, "rendered-page.jpg")]);
      await expect(readFile(join(dir, "figure 1.png"), "utf8")).resolves.toBe("markdown image");
      await expect(readFile(join(dir, "rendered-page.jpg"), "utf8")).resolves.toBe("output image");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("saveDocumentParsingResultResources sorts document parsing map keys for deterministic output", async () => {
    const dir = await mkdtemp(join(tmpdir(), "paddleocr-api-sdk-"));
    try {
      const { fetch, calls } = captureFetch([textResponse("alpha"), textResponse("bravo"), textResponse("charlie")]);
      const client = createClient(fetch);
      const result: DocParsingResult = {
        jobId: "job-doc",
        pages: [
          {
            markdownText: "",
            markdownImages: {
              "charlie.png": "https://storage.example.test/charlie",
              "alpha.png": "https://storage.example.test/alpha",
              "bravo.png": "https://storage.example.test/bravo",
            },
            outputImages: {},
          },
        ],
      };

      const saved = await client.saveDocumentParsingResultResources(result, dir);

      expect(saved).toEqual([join(dir, "alpha.png"), join(dir, "bravo.png"), join(dir, "charlie.png")]);
      expect(calls.map((call) => call.url)).toEqual([
        "https://storage.example.test/alpha",
        "https://storage.example.test/bravo",
        "https://storage.example.test/charlie",
      ]);
      await expect(readFile(join(dir, "alpha.png"), "utf8")).resolves.toBe("alpha");
      await expect(readFile(join(dir, "bravo.png"), "utf8")).resolves.toBe("bravo");
      await expect(readFile(join(dir, "charlie.png"), "utf8")).resolves.toBe("charlie");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("saveOcrResultResources requires overwrite opt-in for bulk result files", async () => {
    const dir = await mkdtemp(join(tmpdir(), "paddleocr-api-sdk-"));
    try {
      const result: OCRResult = {
        jobId: "job-ocr",
        pages: [{ prunedResult: {}, ocrImageUrl: "https://storage.example.test/page.png" }],
      };

      await writeFile(join(dir, "ocr-page-1.png"), "existing");
      const blocked = captureFetch([textResponse("unexpected")]);
      await expect(createClient(blocked.fetch).saveOcrResultResources(result, dir)).rejects.toThrow(InvalidRequestError);
      expect(blocked.calls).toHaveLength(0);

      const replacement = captureFetch([textResponse("replacement")]);
      await createClient(replacement.fetch).saveOcrResultResources(result, dir, { overwrite: true });
      await expect(readFile(join(dir, "ocr-page-1.png"), "utf8")).resolves.toBe("replacement");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test.each([
    ["empty", ""],
    ["absolute", "/absolute.png"],
    ["parent traversal", "../escape.png"],
    ["traversal segment", ".."],
    ["forward separator", "nested/escape.png"],
    ["backslash separator", "nested\\escape.png"],
  ])("saveDocumentParsingResultResources rejects unsafe document parsing map key: %s", async (_name, unsafeKey) => {
    const dir = await mkdtemp(join(tmpdir(), "paddleocr-api-sdk-"));
    try {
      const { fetch } = captureFetch([]);
      const client = createClient(fetch);
      const result: DocParsingResult = {
        jobId: "job-doc",
        pages: [
          {
            markdownText: "",
            markdownImages: {
              [unsafeKey]: "https://storage.example.test/escape.png",
            },
            outputImages: {},
          },
        ],
      };

      await expect(client.saveDocumentParsingResultResources(result, dir)).rejects.toThrow(InvalidRequestError);
      expect(fetch).not.toHaveBeenCalled();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
