#!/usr/bin/env python

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import urllib.request
from contextlib import asynccontextmanager
from typing import Optional

import fastapi
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from paddlex.inference.serving.infra.models import AIStudioNoResultResponse
from paddlex.inference.serving.infra.utils import generate_log_id
from paddlex_hps_client import triton_request_async
from tritonclient.grpc import aio as triton_grpc_aio

TRITON_URL = os.getenv("HPS_TRITON_URL", "paddleocr-vl-tritonserver:8001")
MAX_CONCURRENT_INFERENCE_REQUESTS = int(
    os.getenv("HPS_MAX_CONCURRENT_INFERENCE_REQUESTS", "16")
)
MAX_CONCURRENT_NON_INFERENCE_REQUESTS = int(
    os.getenv("HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS", "64")
)
INFERENCE_TIMEOUT = int(os.getenv("HPS_INFERENCE_TIMEOUT", "600"))
LOG_LEVEL = os.getenv("HPS_LOG_LEVEL", "INFO")
HEALTH_CHECK_TIMEOUT = int(os.getenv("HPS_HEALTH_CHECK_TIMEOUT", "5"))
FILTER_HEALTH_ACCESS_LOG = os.getenv(
    "HPS_FILTER_HEALTH_ACCESS_LOG", "true"
).lower() in (
    "true",
    "1",
    "yes",
)

VLM_URL = os.getenv("HPS_VLM_URL", "http://paddleocr-vlm-server:8080")

TRITON_MODEL_LAYOUT_PARSING = "layout-parsing"
TRITON_MODEL_RESTRUCTURE_PAGES = "restructure-pages"
TRITON_MODELS = (TRITON_MODEL_LAYOUT_PARSING, TRITON_MODEL_RESTRUCTURE_PAGES)


logger = logging.getLogger(__name__)


def _configure_logger(logger: logging.Logger) -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


_configure_logger(logger)


def _create_aistudio_output_without_result(
    error_code: int, error_msg: str, *, log_id: Optional[str] = None
) -> dict:
    """Create a standardized error response in AIStudio format."""
    resp = AIStudioNoResultResponse(
        logId=log_id if log_id is not None else generate_log_id(),
        errorCode=error_code,
        errorMsg=error_msg,
    )
    return resp.model_dump()


@asynccontextmanager
async def _lifespan(app: fastapi.FastAPI):
    """
    Manage application lifecycle:
    - Initialize Triton client and semaphores on startup
    - Clean up resources on shutdown
    """
    logger.info("Initializing gateway...")
    logger.info("Triton URL: %s", TRITON_URL)
    logger.info(
        "Max concurrent inference requests: %d", MAX_CONCURRENT_INFERENCE_REQUESTS
    )
    logger.info(
        "Max concurrent non-inference requests: %d",
        MAX_CONCURRENT_NON_INFERENCE_REQUESTS,
    )
    logger.info("Inference timeout: %ds", INFERENCE_TIMEOUT)

    # Initialize async Triton client
    app.state.triton_client = triton_grpc_aio.InferenceServerClient(
        url=TRITON_URL,
        keepalive_options=triton_grpc_aio.KeepAliveOptions(
            keepalive_timeout_ms=INFERENCE_TIMEOUT * 1000,
        ),
    )

    # Separate semaphores for inference and non-inference operations
    app.state.inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE_REQUESTS)
    app.state.non_inference_semaphore = asyncio.Semaphore(
        MAX_CONCURRENT_NON_INFERENCE_REQUESTS
    )

    logger.info("Gateway initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down gateway...")
    await app.state.triton_client.close()
    logger.info("Gateway shutdown complete")


app = fastapi.FastAPI(
    title="PaddleOCR-VL HPS Gateway",
    description="High Performance Server Gateway for PaddleOCR-VL",
    version="1.0.0",
    lifespan=_lifespan,
)


@app.get("/health", operation_id="checkHealth")
async def health():
    """Liveness check - returns healthy if the gateway process is running."""
    return _create_aistudio_output_without_result(0, "Healthy")


async def _check_vlm_ready() -> bool:
    """Check if the VLM server is ready by querying its health endpoint."""

    def _do_check():
        req = urllib.request.Request(f"{VLM_URL}/health")
        try:
            with urllib.request.urlopen(req, timeout=HEALTH_CHECK_TIMEOUT) as resp:
                return resp.status == 200
        except Exception:
            return False

    return await asyncio.to_thread(_do_check)


@app.get("/health/ready", operation_id="checkReady")
async def ready(request: Request):
    """Readiness check - verifies Triton server, models, and VLM server."""
    try:
        client = request.app.state.triton_client

        # Check Triton server readiness with timeout
        is_server_ready = await asyncio.wait_for(
            client.is_server_ready(),
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        if not is_server_ready:
            return JSONResponse(
                status_code=503,
                content=_create_aistudio_output_without_result(
                    503, "Triton server not ready"
                ),
            )

        # Check if required models are ready
        for model_name in TRITON_MODELS:
            is_model_ready = await asyncio.wait_for(
                client.is_model_ready(model_name),
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            if not is_model_ready:
                return JSONResponse(
                    status_code=503,
                    content=_create_aistudio_output_without_result(
                        503, f"Model '{model_name}' not ready"
                    ),
                )

        # Check VLM server readiness
        vlm_ready = await _check_vlm_ready()
        if not vlm_ready:
            return JSONResponse(
                status_code=503,
                content=_create_aistudio_output_without_result(
                    503, "VLM server not ready"
                ),
            )

        return _create_aistudio_output_without_result(0, "Ready")
    except asyncio.TimeoutError:
        logger.error("Health check timed out after %ds", HEALTH_CHECK_TIMEOUT)
        return JSONResponse(
            status_code=503,
            content=_create_aistudio_output_without_result(
                503, "Health check timed out"
            ),
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return JSONResponse(
            status_code=503,
            content=_create_aistudio_output_without_result(
                503, f"Service unavailable: {e}"
            ),
        )


async def _process_triton_request(
    request: Request,
    body: dict,
    model_name: str,
    semaphore: asyncio.Semaphore,
) -> JSONResponse:
    """Process a request through Triton inference server."""
    request_log_id = body.get("logId", generate_log_id())
    logger.info(
        "Processing %r request %s",
        model_name,
        request_log_id,
    )

    if "logId" in body:
        logger.debug(
            "Using external logId for %r request: %s",
            model_name,
            request_log_id,
        )
    body["logId"] = request_log_id

    client = request.app.state.triton_client

    try:
        async with semaphore:
            output = await triton_request_async(
                client,
                model_name,
                body,
                timeout=INFERENCE_TIMEOUT,
            )
    except asyncio.TimeoutError:
        logger.warning(
            "Timeout processing %r request %s",
            model_name,
            request_log_id,
        )
        return JSONResponse(
            status_code=504,
            content=_create_aistudio_output_without_result(
                504, "Gateway timeout", log_id=request_log_id
            ),
        )
    except triton_grpc_aio.InferenceServerException as e:
        if "Deadline Exceeded" in str(e):
            logger.warning(
                "Triton timeout for %r request %s",
                model_name,
                request_log_id,
            )
            return JSONResponse(
                status_code=504,
                content=_create_aistudio_output_without_result(
                    504, "Gateway timeout", log_id=request_log_id
                ),
            )
        logger.error(
            "Triton error for %r request %s: %s",
            model_name,
            request_log_id,
            e,
        )
        return JSONResponse(
            status_code=500,
            content=_create_aistudio_output_without_result(
                500, "Internal server error", log_id=request_log_id
            ),
        )
    except Exception:
        logger.exception(
            "Unexpected error for %r request %s",
            model_name,
            request_log_id,
        )
        return JSONResponse(
            status_code=500,
            content=_create_aistudio_output_without_result(
                500, "Internal server error", log_id=request_log_id
            ),
        )

    if output.get("errorCode", 0) != 0:
        error_code = output.get("errorCode", 500)
        error_msg = output.get("errorMsg", "Unknown error")
        logger.warning(
            "Triton returned error for %r request %s: %s",
            model_name,
            request_log_id,
            error_msg,
        )
        return JSONResponse(
            status_code=error_code,
            content=_create_aistudio_output_without_result(
                error_code, error_msg, log_id=request_log_id
            ),
        )

    logger.info(
        "Completed %r request %s",
        model_name,
        request_log_id,
    )
    return JSONResponse(status_code=200, content=output)


@app.post(
    "/layout-parsing",
    operation_id="infer",
    summary=f"Invoke {TRITON_MODEL_LAYOUT_PARSING} model",
    response_class=JSONResponse,
)
async def _handle_infer(request: Request, body: dict):
    """Handle layout-parsing inference request."""
    return await _process_triton_request(
        request,
        body,
        TRITON_MODEL_LAYOUT_PARSING,
        request.app.state.inference_semaphore,
    )


@app.post(
    "/restructure-pages",
    operation_id="restructurePages",
    summary=f"Invoke {TRITON_MODEL_RESTRUCTURE_PAGES} model",
    response_class=JSONResponse,
)
async def _handle_restructure_pages(request: Request, body: dict):
    """Handle restructure-pages request (non-inference)."""
    return await _process_triton_request(
        request,
        body,
        TRITON_MODEL_RESTRUCTURE_PAGES,
        request.app.state.non_inference_semaphore,
    )


@app.exception_handler(json.JSONDecodeError)
async def _json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    """Handle invalid JSON in request body."""
    logger.warning("Invalid JSON for %s: %s", request.url.path, exc.msg)
    return JSONResponse(
        status_code=400,
        content=_create_aistudio_output_without_result(400, f"Invalid JSON: {exc.msg}"),
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    error_details = exc.errors()
    # Format error messages for readability
    error_messages = []
    for error in error_details:
        loc = ".".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Unknown error")
        error_messages.append(f"{loc}: {msg}" if loc else msg)
    error_msg = "; ".join(error_messages)

    logger.warning("Validation error for %s: %s", request.url.path, error_msg)
    return JSONResponse(
        status_code=422,
        content=_create_aistudio_output_without_result(422, error_msg),
    )


@app.exception_handler(asyncio.TimeoutError)
async def _timeout_exception_handler(request: Request, exc: asyncio.TimeoutError):
    """Handle timeout errors."""
    logger.warning("Request timed out: %s", request.url.path)
    return JSONResponse(
        status_code=504,
        content=_create_aistudio_output_without_result(504, "Gateway timeout"),
    )


@app.exception_handler(Exception)
async def _general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.exception("Unhandled exception for %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=_create_aistudio_output_without_result(500, "Internal server error"),
    )


class _HealthEndpointFilter(logging.Filter):
    """Filter out health check endpoints from access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/health" not in message


# Apply filter to reduce log noise from health checks
if FILTER_HEALTH_ACCESS_LOG:
    logging.getLogger("uvicorn.access").addFilter(_HealthEndpointFilter())
