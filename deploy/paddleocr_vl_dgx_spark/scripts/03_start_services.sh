#!/usr/bin/env bash
# ============================================================
# 调用说明: bash 03_start_services.sh [start|stop|status]
#   默认 start。前置: 已执行 02_setup_runtime.sh 且验证通过。
# 脚本逻辑: 启动 PaddleOCR-VL 两个原生进程(与docker方案等价的两容器架构):
#   1) VLM 推理服务: paddleocr genai_server (vLLM后端),
#      监听 127.0.0.1:8081, 模型 PaddleOCR-VL-1.6-0.9B
#      (首次启动自动从百度BOS下载权重到 ~/.paddlex);
#   2) 等 VLM /health 就绪(最长25分钟=1500s, 首次需从BOS下载权重+加载, 见变更记录②)后,
#      启动解析API: paddlex --serve, 监听 0.0.0.0:8080,
#      pipeline 配置中 server_url 指向 http://127.0.0.1:8081/v1;
#   stop: 按PID文件杀两个进程; status: 显示进程与健康状态。
# 输入输出: 输入为 /data/paddleocr/venv 环境与 config/pipeline_config_vllm.yaml;
#   输出为 8080 端口的文档解析HTTP服务, 服务日志写 logs/vlm_server.log 与
#   logs/api_server.log, PID 写 logs/*.pid
# 变更记录:
#   2026-08-11 ① venv 确认后 export PATH="$VENV/bin:$PATH"(预防性):
#                 02 首次实测已证明 paddleocr/paddlex 内部子进程按 PATH 找
#                 venv 里的可执行文件(找不到 paddlex 即报错), 启动的服务
#                 进程同样可能在内部再开子进程, 提前把 venv/bin 置于 PATH
#                 首位避免重蹈覆辙。
#   2026-08-12 ② VLM 健康等待从 600s 延长到 1500s(25分钟): 服务器 ~/.paddlex
#                 尚无 PaddleOCR-VL-1.6-0.9B 权重缓存(08-12 14:33 实测), 首次
#                 启动 VLM 需从百度 BOS 下载权重+vllm 加载, 600s 可能不够,
#                 超时则 wait_health 返回1 使脚本提前 exit、API 不会起。
#                 延长窗口确保首次下载场景能等到 /health 就绪。
#   2026-08-12 ③ 为服务进程注入 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
#                 (用户拍板): 首次起 VLM 失败, 根因=triton 3.4.0 aarch64 wheel
#                 自带 ptxas 是 CUDA12.8, 不认识 GB10 的 sm_121/--gpu-name 选项
#                 表无 sm_121/sm_121a, vllm 引擎初始化 JIT rotary kernel 报
#                 "Value 'sm_121a' is not defined for option 'gpu-name'"。
#                 CUDA12.9 ptxas 支持 sm_121a, triton knobs.py:191 读
#                 TRITON_PTXAS_PATH 覆盖自带 ptxas。已实证: 不设复现同错、
#                 设置后真实 triton kernel 在 GB10 编译运行成功(max err 0.0)。
# 运行方式提醒: 本脚本内含 wait_health 最长等待25分钟(首次启动 VLM 需下载
#   模型权重), 直接同步执行会 SSH 超时; 应以后台方式启动并轮询本脚本输出。
# ============================================================
set -e

BASE=/data/paddleocr
VENV=$BASE/venv
CONF=$BASE/config/pipeline_config_vllm.yaml
LOGS=$BASE/logs
API_PORT=8080
VLM_PORT=8081
mkdir -p "$LOGS"

# venv 加入 PATH(变更记录①): 服务进程内部若再开子进程调 venv 里的命令
# (如 paddlex), 依赖 PATH 能找到; 与 02 脚本同一类坑位的预防性修复
export PATH="$VENV/bin:$PATH"

# triton ptxas 修复(变更记录③): GB10=sm_121, vllm 引擎初始化会 JIT 编译 triton
# kernel 到 sm_121a, 但 triton 3.4.0 aarch64 wheel 自带 ptxas 是 CUDA12.8,
# 其 --gpu-name 选项表无 sm_121/sm_121a(只到 sm_120) → 必报
# "Value 'sm_121a' is not defined"。CUDA12.9 的 ptxas(脚本07已装)支持 sm_121a,
# triton knobs.py:191 读 TRITON_PTXAS_PATH 覆盖自带 ptxas 路径(08-12 实测:
# 不设此变量复现同错, 设置后 kernel 在 GB10 编译运行成功)。此处校验并导出。
TRITON_PTXAS=/usr/local/cuda-12.9/bin/ptxas
if [ ! -x "$TRITON_PTXAS" ]; then
    echo "错误: $TRITON_PTXAS 不存在, 请先执行 scripts/07_install_cuda_toolkit.sh" >&2
    exit 1
fi
export TRITON_PTXAS_PATH="$TRITON_PTXAS"

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

wait_health() {  # $1=url $2=超时秒数 $3=名称
    local i=0
    while [ $i -lt "$2" ]; do
        if curl -sf "$1" >/dev/null 2>&1; then
            say "$3 就绪: $1"
            return 0
        fi
        sleep 5; i=$((i+5))
        [ $((i % 60)) -eq 0 ] && say "等待 $3 中... 已等 ${i}s (日志: $LOGS)"
    done
    say "错误: $3 在 $2 秒内未就绪, 请查看日志"
    return 1
}

case "${1:-start}" in
start)
    # ---- VLM 推理服务 ----
    if [ -f "$LOGS/vlm_server.pid" ] && kill -0 "$(cat "$LOGS/vlm_server.pid")" 2>/dev/null; then
        say "VLM 服务已在运行(pid $(cat "$LOGS/vlm_server.pid")), 跳过"
    else
        say "启动 VLM 推理服务 (vLLM后端, 端口$VLM_PORT, 首次会下载模型权重) ..."
        nohup "$VENV/bin/paddleocr" genai_server \
            --model_name PaddleOCR-VL-1.6-0.9B \
            --host 127.0.0.1 --port "$VLM_PORT" \
            --backend vllm \
            > "$LOGS/vlm_server.log" 2>&1 &
        echo $! > "$LOGS/vlm_server.pid"
        say "VLM 进程 pid=$(cat "$LOGS/vlm_server.pid")"
    fi
    wait_health "http://127.0.0.1:$VLM_PORT/health" 1500 "VLM服务" || exit 1

    # ---- 文档解析 API ----
    if [ -f "$LOGS/api_server.pid" ] && kill -0 "$(cat "$LOGS/api_server.pid")" 2>/dev/null; then
        say "API 服务已在运行(pid $(cat "$LOGS/api_server.pid")), 跳过"
    else
        say "启动文档解析 API (paddlex --serve, 端口$API_PORT) ..."
        nohup "$VENV/bin/paddlex" --serve \
            --pipeline "$CONF" \
            --host 0.0.0.0 --port "$API_PORT" \
            > "$LOGS/api_server.log" 2>&1 &
        echo $! > "$LOGS/api_server.pid"
        say "API 进程 pid=$(cat "$LOGS/api_server.pid")"
    fi
    wait_health "http://127.0.0.1:$API_PORT/health" 300 "API服务" || exit 1
    say "全部服务就绪: http://$(hostname -I | awk '{print $1}'):$API_PORT (接口 /layout-parsing)"
    ;;
stop)
    for p in api_server vlm_server; do
        if [ -f "$LOGS/$p.pid" ]; then
            kill "$(cat "$LOGS/$p.pid")" 2>/dev/null && say "已停止 $p" || say "$p 未在运行"
            rm -f "$LOGS/$p.pid"
        fi
    done
    ;;
status)
    for p in api_server:$API_PORT vlm_server:$VLM_PORT; do
        name=${p%%:*}; port=${p##*:}
        pid=$(cat "$LOGS/$name.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            health=$(curl -sf -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" || echo "无响应")
            say "$name: 运行中 pid=$pid 端口=$port health=$health"
        else
            say "$name: 未运行"
        fi
    done
    ;;
*)
    say "用法: $0 [start|stop|status]"; exit 1
    ;;
esac
