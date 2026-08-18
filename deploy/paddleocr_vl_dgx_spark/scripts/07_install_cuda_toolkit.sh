#!/usr/bin/env bash
# ============================================================
# 调用说明: echo <SUDO_PASSWORD> | sudo -S bash 07_install_cuda_toolkit.sh
#   (需 root 权限装 apt 包; 脚本内部不再单独 sudo, 由外层一次性提权)
#   用途: 为 02 脚本的源编步骤安装配套的 CUDA 12.9 工具链(nvcc 12.9)。
# 脚本逻辑: 背景=02 变更记录⑨ —— torch 2.8.0+cu129 的 cpp_extension.py
#   在编译扩展时对 nvcc 与 torch.version.cuda 做版本比较(:506),
#   主版本不同直接 RuntimeError(次版本不同仅 warning), 且无绕过开关:
#     RuntimeError: The detected CUDA version (13.0) mismatches the
#     version that was used to compile PyTorch (12.9)
#   机器原生只有 CUDA 13.0(/usr/local/cuda-13.0, 主版本13≠12被拒)与
#   过老 nvcc 12.0(/usr/bin/nvcc, <12.8 编不了 sm_121), 故装 CUDA 12.9:
#   与 torch.version.cuda=12.9 完全一致(连 warning 都没有), 且 >=12.8
#   可编 sm_121。来源=已配置的 NVIDIA 官方 sbsa apt 源
#   (developer.download.nvidia.com/.../ubuntu2404/sbsa, apt-cache policy
#   实测有 cuda-toolkit-12-9 12.9.2-1)。装完位于 /usr/local/cuda-12.9。
#   步骤:
#   1) /usr/local/cuda-12.9/bin/nvcc 已存在且 release 12.9 → 跳过(幂等);
#   2) apt-get update(容忍失败, 沿用现有索引);
#   3) DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-9
#      (完整 toolkit 约5GB, 磁盘余2.7T 充足);
#   4) nvcc --version 校验 release 12.9。
# 输入输出: 输入为 NVIDIA apt 仓库; 输出为 /usr/local/cuda-12.9 工具链,
#   日志增量写 logs/07_install_cuda_toolkit.log(含进度/时间/输入输出)。
# 变更记录:
#   2026-08-12 ① 初版(handover 4.11.4 预登记的"nvcc13.0+torch cu129 版本
#                 混合"风险点落地: 09:32 flash-attn 源编被 cpp_extension
#                 主版本检查拒绝, 本脚本装配套 12.9 工具链解决)
# ============================================================
set -e
set -o pipefail

BASE=/data/paddleocr
LOGS=$BASE/logs
mkdir -p "$LOGS"
LOG=$LOGS/07_install_cuda_toolkit.log
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "########## 07_install_cuda_toolkit.sh 开始 ##########"

# ---------- 1. 幂等检查 ----------
if /usr/local/cuda-12.9/bin/nvcc --version 2>/dev/null | grep -q 'release 12\.9'; then
    say "nvcc 12.9 已安装: $(/usr/local/cuda-12.9/bin/nvcc --version | grep release), 跳过"
    exit 0
fi

# ---------- 2. 刷新索引(容忍失败) ----------
say "apt-get update (容忍失败, 失败则沿用现有索引) ..."
apt-get update 2>&1 | tail -2 | tee -a "$LOG" || say "警告: apt-get update 失败, 用现有索引继续"

# ---------- 3. 安装完整 CUDA 12.9 工具链 ----------
say "安装 cuda-toolkit-12-9 (约5GB下载, 视网速数分钟~十几分钟) ..."
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-9 2>&1 | tee -a "$LOG" | tail -3

# ---------- 4. 校验 ----------
say "校验 /usr/local/cuda-12.9/bin/nvcc ..."
/usr/local/cuda-12.9/bin/nvcc --version | tee -a "$LOG"
say "########## 07 完成: CUDA 12.9 工具链就绪 (/usr/local/cuda-12.9) ##########"
