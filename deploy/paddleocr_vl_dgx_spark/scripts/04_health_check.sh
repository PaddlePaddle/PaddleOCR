#!/usr/bin/env bash
# ============================================================
# 调用说明: bash 04_health_check.sh [服务地址] [测试图片]
#   默认: 服务地址 http://127.0.0.1:8080, 测试图片 /data/paddleocr/test/demo.png
# 脚本逻辑: 三层验证:
#   1) VLM 服务健康检查 (127.0.0.1:8081/health);
#   2) API 服务健康检查 (/health);
#   3) 端到端解析测试: POST /layout-parsing 发送base64编码图片,
#      检查返回200且 result 非空, 保存 Markdown 结果到 logs/e2e_result.md。
# 输入输出: 输入为运行中的服务与测试图片; 输出为终端验证报告与
#   logs/e2e_result.md
# 变更记录:
#   2026-08-11 ① 端到端测试的内嵌 Python 从系统 python3 改为 venv 的 python:
#                 系统 Python 未装 requests, 第三步必报 ModuleNotFoundError;
#                 requests 只装在 /data/paddleocr/venv 中。
#   2026-08-12 ② markdown 提取兼容 dict 结构: 实测 paddleocr 3.7.0 的
#                 /layout-parsing 响应中 page["markdown"] 是 dict(正文在
#                 markdown["text"]), 不是 str; 原 join 报 TypeError:
#                 sequence item 0: expected str instance, dict found。
#                 现按类型取值: dict 取 ["text"], str 直接用, 两者兼容。
# ============================================================
set -e

BASE=/data/paddleocr
HOST="${1:-http://127.0.0.1:8080}"
IMG="${2:-$BASE/test/demo.png}"
LOGS=$BASE/logs
mkdir -p "$LOGS"

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

say "== 1. VLM 服务健康检查 =="
curl -sf http://127.0.0.1:8081/health && say " → VLM 正常" || { say " → VLM 异常!"; exit 1; }

say "== 2. API 服务健康检查 =="
curl -sf "$HOST/health" && say " → API 正常" || { say " → API 异常!"; exit 1; }

say "== 3. 端到端解析测试: $IMG =="
if [ ! -f "$IMG" ]; then
    say "测试图片不存在: $IMG, 跳过端到端测试"
    exit 0
fi
# 用 venv 的 python(变更记录①): 系统 python3 无 requests, 会报 ModuleNotFoundError
"$BASE/venv/bin/python" - "$HOST" "$IMG" "$LOGS/e2e_result.md" << 'PYEOF'
# 用途: 调用 /layout-parsing 接口做端到端验证, 保存markdown结果
import base64, json, sys, requests
host, img_path, out_md = sys.argv[1], sys.argv[2], sys.argv[3]
with open(img_path, "rb") as f:
    payload = {"file": base64.b64encode(f.read()).decode("ascii"), "fileType": 1}
r = requests.post(host + "/layout-parsing", json=payload, timeout=300)
print(f"HTTP {r.status_code}")
assert r.status_code == 200, r.text[:500]
res = r.json()["result"]
pages = res.get("layoutParsingResults", [])
print(f"解析页数: {len(pages)}")
# markdown 字段实测为 dict(正文在 ["text"]), 兼容 str 旧格式(变更记录②)
def _md(p):
    v = p.get("markdown", "")
    return v.get("text", "") if isinstance(v, dict) else (v or "")
md = "\n\n".join(_md(p) for p in pages)
open(out_md, "w", encoding="utf-8").write(md)
print(f"Markdown 已保存: {out_md} ({len(md)} 字符)")
print("---- 前200字预览 ----")
print(md[:200])
PYEOF
say "端到端验证完成"
