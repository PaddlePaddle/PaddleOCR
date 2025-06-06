#!/usr/bin/env python3
"""
PaddleOCR MCP Server - Simplified Main Entry
"""

# 标准库导入
import argparse
import logging
import os
import sys

# 第三方库导入
from fastmcp import FastMCP

# 本地应用导入
from core_ocr import register_tools

# 配置日志输出到stderr，避免干扰MCP的JSON通信
logging.basicConfig(
    level=logging.WARNING,  # 只记录WARNING及以上级别
    format="%(name)s: %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger(__name__)


def main():
    """主函数 - 支持三种模式和多传输协议"""
    parser = argparse.ArgumentParser(
        description="PaddleOCR MCP Server v2.0 - Support local/AI Studio/user service modes"
    )

    # OCR来源配置
    parser.add_argument(
        "--ocr_source",
        choices=["local", "aistudio", "user_service"],
        default=os.getenv("PADDLEOCR_MCP_OCR_SOURCE", "local"),
        help="OCR service source: local (local library), aistudio (AI Studio cloud), user_service (user-configured service)",
    )
    parser.add_argument(
        "--server_url",
        default=os.getenv("PADDLEOCR_MCP_SERVER_URL"),
        help="Server base URL (for AI Studio or user service)",
    )
    parser.add_argument(
        "--api_token",
        default=os.getenv("PADDLEOCR_MCP_API_TOKEN"),
        help="API authentication token (required for AI Studio)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("PADDLEOCR_MCP_TIMEOUT", "30")),
        help="Request timeout in seconds",
    )

    # 传输协议配置 (参考markitdown的简洁实现)
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use HTTP transport instead of STDIO (suitable for remote deployment and multiple clients)",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Use SSE transport instead of STDIO (real-time communication support)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host address for HTTP/SSE mode (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port for HTTP/SSE mode (default: 8000)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for debugging"
    )

    args = parser.parse_args()

    # 设置详细日志级别（仅调试时）
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # 传输协议验证 (参考markitdown的robust验证)
    use_web_transport = args.http or args.sse

    if not use_web_transport and (args.host or args.port):
        parser.error(
            "Host and port arguments are only valid when using HTTP or SSE transport (see: --http or --sse)"
        )
        sys.exit(2)  # 参数错误使用2作为错误码

    # API配置验证（只输出错误信息）
    if args.ocr_source in ["aistudio", "user_service"] and not args.server_url:
        print("Error: API mode requires server URL configuration", file=sys.stderr)
        print("  Set environment variable: PADDLEOCR_MCP_SERVER_URL", file=sys.stderr)
        sys.exit(2)  # 参数错误使用2作为错误码

    if args.ocr_source == "aistudio" and not args.api_token:
        print("Error: AI Studio mode requires API token", file=sys.stderr)
        print("  Set environment variable: PADDLEOCR_MCP_API_TOKEN", file=sys.stderr)
        sys.exit(2)  # 参数错误使用2作为错误码

    # 确定服务名称
    if args.ocr_source == "local":
        server_name = "PaddleOCR Local MCP Server"
    elif args.ocr_source == "user_service":
        service_type = (
            "Layout Analysis" if "layout-parsing" in args.server_url.lower() else "OCR"
        )
        server_name = f"PaddleOCR User Service {service_type} MCP Server"
    else:  # aistudio
        service_type = (
            "Layout Analysis" if "layout-parsing" in args.server_url.lower() else "OCR"
        )
        server_name = f"PaddleOCR AI Studio {service_type} MCP Server"

    # 创建并配置服务器
    mcp = FastMCP(name=server_name)

    try:
        # 注册工具
        if args.ocr_source == "local":
            register_tools(mcp, ocr_source_type="local")
        else:
            # API模式 (aistudio 或 user_service)
            api_kwargs = {"api_url": args.server_url, "timeout": args.timeout}
            if args.api_token:
                api_kwargs["api_token"] = args.api_token

            register_tools(mcp, ocr_source_type=args.ocr_source, **api_kwargs)

        # 启动服务器 (参考markitdown的简洁启动逻辑)
        if use_web_transport:
            # 确定传输协议类型
            transport = "sse" if args.sse else "streamable-http"
            host = args.host if args.host else "127.0.0.1"
            port = args.port if args.port else 8000

            if args.verbose:
                logger.info(f"启动{transport}服务器: http://{host}:{port}")

            mcp.run(
                transport=transport,
                host=host,
                port=port,
                log_level="info" if args.verbose else "warning",
            )
        else:
            mcp.run()

    except Exception as e:
        logger.error(f"启动失败: {e}")
        if args.verbose:
            logger.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()
