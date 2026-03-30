import { attachWorkerMessageHandler } from "../../worker/entry";
import { sourcePayloadToMat, ensureServedFromHttp } from "../../platform/worker";
import type { OcrPipelineRunnerOptions } from "./core";
import { OcrPipelineRunner } from "./core";

function createPaddleOCRWorkerMessageHandler() {
  let ocr: OcrPipelineRunner | null = null;

  async function handleInit(payload: Record<string, unknown>) {
    await ocr?.dispose();
    ocr = new OcrPipelineRunner({
      ...(payload.options as OcrPipelineRunnerOptions),
      ensureServedFromHttp,
      sourceToMat: sourcePayloadToMat,
    });
    const summary = await ocr.initialize();
    return {
      summary,
      modelConfig: ocr.getModelConfig(),
    };
  }

  async function handlePredict(payload: Record<string, unknown>) {
    if (!ocr) {
      throw new Error("OCR worker is not initialized.");
    }
    return ocr.predict(payload.source, payload.params as Record<string, unknown>);
  }

  async function handleDispose() {
    await ocr?.dispose();
    ocr = null;
    return {};
  }

  return async function handleMessage(type: string, payload: Record<string, unknown>) {
    switch (type) {
      case "init":
        return handleInit(payload);
      case "predict":
        return handlePredict(payload);
      case "dispose":
        return handleDispose();
      default:
        throw new Error(`Unsupported worker request type "${type}".`);
    }
  };
}

attachWorkerMessageHandler(createPaddleOCRWorkerMessageHandler());
