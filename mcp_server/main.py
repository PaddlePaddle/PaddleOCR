#!/usr/bin/env python3
"""
PaddleOCR MCP Server - Simplified Main Entry
"""

import argparse
import os
import sys
from fastmcp import FastMCP
from core_ocr import register_tools


def log(message: str):
    """输出日志到stderr，避免干扰MCP的JSON通信"""
    print(message, file=sys.stderr)


def main():
    """主函数 - 支持三种模式"""
    parser = argparse.ArgumentParser(
        description="PaddleOCR MCP Server v2.0 - 支持本地/AI Studio/本地服务三种模式"
    )
    
    parser.add_argument("--engine", choices=["local", "aistudio", "local_service"], 
                       default=os.getenv("PADDLEOCR_ENGINE", "local"),
                       help="OCR引擎类型: local(本地), aistudio(AI Studio), local_service(本地服务)")
    parser.add_argument("--api-url", default=os.getenv("PADDLEOCR_API_URL"),
                       help="API地址 (AI Studio或本地服务)")
    parser.add_argument("--api-token", default=os.getenv("PADDLEOCR_API_TOKEN"),
                       help="API令牌 (仅AI Studio需要)")
    parser.add_argument("--timeout", type=int, 
                       default=int(os.getenv("PADDLEOCR_TIMEOUT", "30")),
                       help="API超时时间")
    
    args = parser.parse_args()
    
    # 参数验证
    if args.engine in ["aistudio", "local_service"] and not args.api_url:
        log("❌ API模式需要配置API地址")
        log("   本地服务示例: --api-url http://10.21.226.181:8080/ocr")
        log("   AI Studio示例: --api-url https://xxx.aistudio-hub.baidu.com/ocr")
        log("   或设置环境变量: PADDLEOCR_API_URL")
        sys.exit(1)
    
    if args.engine == "aistudio" and not args.api_token:
        log("❌ AI Studio模式需要API令牌")
        log("   示例: --api-token your_token")
        log("   或设置环境变量: PADDLEOCR_API_TOKEN") 
        sys.exit(1)
    
    # 确定服务名称和描述
    if args.engine == "local":
        server_name = "PaddleOCR Local MCP Server"
        log("🚀 启动本地PaddleOCR文档处理服务")
    elif args.engine == "local_service":
        service_type = "结构分析" if "layout-parsing" in args.api_url.lower() else "OCR"
        server_name = f"PaddleOCR 本地服务 {service_type} MCP Server"
        log(f"🚀 启动本地服务{service_type}处理服务")
        log(f"   服务地址: {args.api_url}")
        log(f"   无需令牌认证")
    else:  # aistudio
        service_type = "结构分析" if "layout-parsing" in args.api_url.lower() else "OCR"
        server_name = f"PaddleOCR AI Studio {service_type} MCP Server"
        log(f"🚀 启动AI Studio云端{service_type}处理服务")
        log(f"   API地址: {args.api_url}")
    
    # 创建并启动服务器
    mcp = FastMCP(name=server_name)
    
    try:
        # 注册工具
        if args.engine == "local":
            register_tools(mcp, engine_type="local")
        else:
            # API模式 (aistudio 或 local_service)
            api_kwargs = {
                "api_url": args.api_url,
                "timeout": args.timeout
            }
            if args.engine == "aistudio":
                api_kwargs["api_token"] = args.api_token
            
            register_tools(mcp, engine_type=args.engine, **api_kwargs)
        
        log("✅ 工具注册成功")
        log("🔄 正在启动MCP服务器...")
        mcp.run()
        
    except Exception as e:
        log(f"❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
