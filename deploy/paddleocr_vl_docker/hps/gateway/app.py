#!/usr/bin/env python

import logging

import fastapi
from fastapi.responses import JSONResponse
from paddlex_hps_client import triton_request
from paddlex.inference.serving.infra.models import AIStudioNoResultResponse
from paddlex.inference.serving.infra.utils import generate_log_id
from paddlex.inference.serving.schemas import paddleocr_vl as schema
from tritonclient import grpc as triton_grpc

TRITONSERVER_URL = "paddleocr-vl-tritonserver:8001"

logger = logging.getLogger(__name__)


def _configure_logger(logger: logging.Logger):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


_configure_logger(logger)


def _create_aistudio_output_without_result(error_code: str, error_msg: str) -> dict:
    resp = AIStudioNoResultResponse(
        logId=generate_log_id(), errorCode=error_code, errorMsg=error_msg
    )
    return resp.model_dump()


def _add_primary_operations(app: fastapi.FastAPI) -> None:
    def _create_handler(model_name: str):
        def _handler(request: dict):
            output = triton_request(triton_client, model_name, request)
            if output["errorCode"] != 0:
                output = _create_aistudio_output_without_result(
                    output["errorCode"], output["errorMsg"]
                )
                return JSONResponse(status_code=output.errorCode, content=output)
            return JSONResponse(status_code=200, content=output)

        return _handler

    for operation_name, (endpoint, _, _) in schema.PRIMARY_OPERATIONS.items():
        # TODO: API docs
        app.post(
            endpoint,
            operation_id=operation_name,
        )(
            _create_handler(endpoint[1:]),
        )


app = fastapi.FastAPI()


@app.get(
    "/health",
    operation_id="checkHealth",
)
def check_health():
    if not triton_client.is_server_ready():
        output = _create_aistudio_output_without_result(
            503, "The Triton server is not ready."
        )
        return JSONResponse(status_code=503, content=output)
    return _create_aistudio_output_without_result(0, "Healthy")


_add_primary_operations(app)


# HACK
# https://github.com/encode/starlette/issues/864
class _EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/health") == -1


logging.getLogger("uvicorn.access").addFilter(_EndpointFilter())

triton_client: triton_grpc.InferenceServerClient = triton_grpc.InferenceServerClient(
    TRITONSERVER_URL
)
