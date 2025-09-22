#!/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import asyncio
import os
import sys

from fastmcp import FastMCP

from .pipelines import create_pipeline_handler


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PaddleOCR MCP server - Supports local library, AI Studio service, and self-hosted servers."
    )

    parser.add_argument(
        "--pipeline",
        choices=["OCR", "PP-StructureV3"],
        default=os.getenv("PADDLEOCR_MCP_PIPELINE", "OCR"),
        help="Pipeline name.",
    )
    parser.add_argument(
        "--ppocr_source",
        choices=["local", "aistudio", "self_hosted"],
        default=os.getenv("PADDLEOCR_MCP_PPOCR_SOURCE", "local"),
        help="Source of PaddleOCR functionality: local (local library), aistudio (AI Studio service), self_hosted (self-hosted server).",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use HTTP transport instead of STDIO (suitable for remote deployment and multiple clients).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for HTTP mode (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP mode (default: 8000).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for debugging."
    )

    # Local mode configuration
    parser.add_argument(
        "--pipeline_config",
        default=os.getenv("PADDLEOCR_MCP_PIPELINE_CONFIG"),
        help="PaddleOCR pipeline configuration file path (for local mode).",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("PADDLEOCR_MCP_DEVICE"),
        help="Device to run inference on.",
    )

    # Service mode configuration
    parser.add_argument(
        "--server_url",
        default=os.getenv("PADDLEOCR_MCP_SERVER_URL"),
        help="Base URL of the underlying server (required in service mode).",
    )
    parser.add_argument(
        "--aistudio_access_token",
        default=os.getenv("PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN"),
        help="AI Studio access token (required for AI Studio).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("PADDLEOCR_MCP_TIMEOUT", "60")),
        help="HTTP read timeout in seconds for API requests to the underlying server.",
    )

    args = parser.parse_args()
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not args.http and (args.host != "127.0.0.1" or args.port != 8000):
        print(
            "Host and port arguments are only valid when using HTTP transport (see: `--http`).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.ppocr_source in ["aistudio", "self_hosted"]:
        if not args.server_url:
            print("Error: The server base URL is required.", file=sys.stderr)
            print(
                "Please either set `--server_url` or set the environment variable "
                "`PADDLEOCR_MCP_SERVER_URL`.",
                file=sys.stderr,
            )
            sys.exit(2)

        if args.ppocr_source == "aistudio" and not args.aistudio_access_token:
            print("Error: The AI Studio access token is required.", file=sys.stderr)
            print(
                "Please either set `--aistudio_access_token` or set the environment variable "
                "`PADDLEOCR_MCP_AISTUDIO_ACCESS_TOKEN`.",
                file=sys.stderr,
            )
            sys.exit(2)


async def async_main() -> None:
    """Asynchronous main entry point."""
    args = _parse_args()

    _validate_args(args)

    try:
        pipeline_handler = create_pipeline_handler(
            args.pipeline,
            args.ppocr_source,
            pipeline_config=args.pipeline_config,
            device=args.device,
            server_url=args.server_url,
            aistudio_access_token=args.aistudio_access_token,
            timeout=args.timeout,
        )
    except Exception as e:
        print(f"Failed to create the pipeline handler: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        await pipeline_handler.start()

        server_name = f"PaddleOCR {args.pipeline} MCP server"
        mcp = FastMCP(
            name=server_name,
            log_level="INFO" if args.verbose else "WARNING",
            mask_error_details=True,
        )

        pipeline_handler.register_tools(mcp)

        if args.http:
            await mcp.run_async(
                transport="streamable-http",
                host=args.host,
                port=args.port,
            )
        else:
            await mcp.run_async()

    except Exception as e:
        print(f"Failed to start the server: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    finally:
        await pipeline_handler.stop()


def main():
    """Main entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
