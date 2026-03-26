import { attachWorkerMessageHandler } from "../../worker/entry.js";
import { sourcePayloadToMat, ensureServedFromHttp } from "../../platform/worker.js";
import { OcrPipelineRunner } from "./core.js";

function createPaddleOCRWorkerMessageHandler() {
  let ocr = null;

  async function handleInit(payload) {
    await ocr?.dispose();
    ocr = new OcrPipelineRunner({
      ...payload.options,
      ensureServedFromHttp,
      sourceToMat: sourcePayloadToMat
    });
    const summary = await ocr.initialize();
    return {
      summary,
      modelConfig: ocr.getModelConfig()
    };
  }

  async function handlePredict(payload) {
    if (!ocr) {
      throw new Error("OCR worker is not initialized.");
    }
    return ocr.predict(payload.source, payload.params);
  }

  async function handleDispose() {
    await ocr?.dispose();
    ocr = null;
    return {};
  }

  return async function handleMessage(type, payload) {
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
