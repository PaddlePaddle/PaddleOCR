#!/usr/bin/env python3

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from fastmcp import FastMCP

from .inference import create_inference
from .selection import DEFAULT_MODEL, resolve_model
from .providers import InferenceProvider, provider_choices
from .tasks import create_task


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PaddleOCR MCP server.")

    parser.add_argument(
        "--model",
        default=os.getenv("PADDLEOCR_MCP_MODEL", DEFAULT_MODEL),
        help="Model to run. Env: PADDLEOCR_MCP_MODEL.",
    )
    parser.add_argument(
        "--ppocr_source",
        choices=provider_choices(),
        default=os.getenv("PADDLEOCR_MCP_PPOCR_SOURCE", "local"),
        help="Inference provider. Env: PADDLEOCR_MCP_PPOCR_SOURCE.",
    )

    parser.add_argument(
        "--http",
        action="store_true",
        help="Use Streamable HTTP instead of stdio.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind host (with --http). Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP bind port (with --http). Default: 8000.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    parser.add_argument(
        "--pipeline_config",
        default=os.getenv("PADDLEOCR_MCP_PIPELINE_CONFIG"),
        help="Pipeline config file path (local). Env: PADDLEOCR_MCP_PIPELINE_CONFIG.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("PADDLEOCR_MCP_DEVICE"),
        help="Inference device (local). Env: PADDLEOCR_MCP_DEVICE.",
    )

    parser.add_argument(
        "--aistudio-base-url",
        dest="aistudio_base_url",
        default=os.getenv("PADDLEOCR_MCP_AISTUDIO_BASE_URL"),
        help="AI Studio API base URL (aistudio). Env: PADDLEOCR_MCP_AISTUDIO_BASE_URL.",
    )
    parser.add_argument(
        "--qianfan-base-url",
        dest="qianfan_base_url",
        default=os.getenv("PADDLEOCR_MCP_QIANFAN_BASE_URL")
        or "https://qianfan.baidubce.com/v2/ocr",
        help="Qianfan API base URL (qianfan). Env: PADDLEOCR_MCP_QIANFAN_BASE_URL.",
    )
    parser.add_argument(
        "--self-hosted-base-url",
        dest="self_hosted_base_url",
        default=os.getenv("PADDLEOCR_MCP_SELF_HOSTED_BASE_URL"),
        help="Self-hosted service base URL (self_hosted). Env: PADDLEOCR_MCP_SELF_HOSTED_BASE_URL.",
    )
    parser.add_argument(
        "--aistudio_access_token",
        default=os.getenv("PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN"),
        help="AI Studio access token (aistudio). Env: PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN.",
    )
    parser.add_argument(
        "--qianfan_api_key",
        default=os.getenv("PADDLEOCR_MCP_QIANFAN_API_KEY"),
        help="Qianfan API key (qianfan). Env: PADDLEOCR_MCP_QIANFAN_API_KEY.",
    )
    parser.add_argument(
        "--http-timeout",
        dest="http_timeout",
        type=int,
        default=int(os.getenv("PADDLEOCR_MCP_HTTP_TIMEOUT", "600")),
        help=(
            "HTTP read timeout in seconds for synchronous APIs. "
            "Env: PADDLEOCR_MCP_HTTP_TIMEOUT."
        ),
    )
    parser.add_argument(
        "--aistudio-request-timeout",
        dest="aistudio_request_timeout",
        type=int,
        default=int(os.getenv("PADDLEOCR_MCP_AISTUDIO_REQUEST_TIMEOUT", "120")),
        help=(
            "Per-request HTTP timeout in seconds for AI Studio API calls. "
            "Env: PADDLEOCR_MCP_AISTUDIO_REQUEST_TIMEOUT."
        ),
    )
    parser.add_argument(
        "--aistudio-poll-timeout",
        dest="aistudio_poll_timeout",
        type=int,
        default=int(os.getenv("PADDLEOCR_MCP_AISTUDIO_POLL_TIMEOUT", "600")),
        help=(
            "Total job polling timeout in seconds for AI Studio. "
            "Env: PADDLEOCR_MCP_AISTUDIO_POLL_TIMEOUT."
        ),
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not args.http and (args.host != "127.0.0.1" or args.port != 8000):
        print(
            "Host and port arguments are only valid when using HTTP transport (see: `--http`).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.ppocr_source == InferenceProvider.AISTUDIO.value:
        if not args.aistudio_access_token:
            print("Error: The AI Studio access token is required.", file=sys.stderr)
            print(
                "Please either set `--aistudio_access_token` or set the environment variable "
                "`PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN`.",
                file=sys.stderr,
            )
            sys.exit(2)
    elif args.ppocr_source == InferenceProvider.QIANFAN.value:
        if not args.qianfan_api_key:
            print("Error: The Qianfan API key is required.", file=sys.stderr)
            print(
                "Please either set `--qianfan_api_key` or set the environment variable "
                "`PADDLEOCR_MCP_QIANFAN_API_KEY`.",
                file=sys.stderr,
            )
            sys.exit(2)
    elif args.ppocr_source == InferenceProvider.SELF_HOSTED.value:
        if not args.self_hosted_base_url:
            print(
                "Error: The self-hosted service base URL is required.", file=sys.stderr
            )
            print(
                f"Please set `--self-hosted-base-url` or the environment variable "
                "`PADDLEOCR_MCP_SELF_HOSTED_BASE_URL`.",
                file=sys.stderr,
            )
            sys.exit(2)


def _create_inference_from_args(args: argparse.Namespace, model: str):
    provider = args.ppocr_source

    if provider == InferenceProvider.LOCAL.value:
        return create_inference(
            model=model,
            provider=provider,
            config=args.pipeline_config,
            device=args.device,
        )
    elif provider == InferenceProvider.AISTUDIO.value:
        return create_inference(
            model=model,
            provider=provider,
            token=args.aistudio_access_token,
            base_url=args.aistudio_base_url,
            request_timeout=float(args.aistudio_request_timeout),
            poll_timeout=float(args.aistudio_poll_timeout),
        )
    elif provider == InferenceProvider.QIANFAN.value:
        return create_inference(
            model=model,
            provider=provider,
            base_url=args.qianfan_base_url,
            api_key=args.qianfan_api_key,
            http_timeout=args.http_timeout,
        )
    elif provider == InferenceProvider.SELF_HOSTED.value:
        return create_inference(
            model=model,
            provider=provider,
            base_url=args.self_hosted_base_url,
            http_timeout=args.http_timeout,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def async_main() -> None:
    """Asynchronous main entry point."""
    args = _parse_args()
    _validate_args(args)

    try:
        model = resolve_model(args.model, args.ppocr_source)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    inference = _create_inference_from_args(args, model)

    try:
        await inference.start()

        task = create_task(model, inference)

        server_name = f"PaddleOCR {model} MCP server"
        mcp = FastMCP(
            name=server_name,
            mask_error_details=True,
        )

        task.register_tools(mcp)

        log_level = "INFO" if args.verbose else "WARNING"

        if args.http:
            await mcp.run_async(
                transport="streamable-http",
                host=args.host,
                port=args.port,
                log_level=log_level,
            )
        else:
            await mcp.run_async(log_level=log_level)

    except Exception as e:
        print(f"Failed to start the server: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    finally:
        await inference.stop()


def main() -> None:
    """Main entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
