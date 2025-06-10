"""
PaddleOCR MCP Server Package

A Model Context Protocol (MCP) server that provides OCR and document structure analysis
capabilities using PaddleOCR. Supports local libraries, AI Studio APIs, and user-configured services.
"""

__version__ = "1.0.0"
__author__ = "PaddleOCR MCP"

# 公共接口导出
from .core import register_tools, EngineContainer

__all__ = ["register_tools", "EngineContainer"]
