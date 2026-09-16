#!/usr/bin/env bash
# ============================================================
# 调用说明: bash 00_install_build_deps.sh        (需要sudo权限)
# 脚本逻辑: 安装源码编译 paddlepaddle-gpu 所需的系统依赖:
#   1) 编译工具链: ninja-build patchelf swig gfortran python3-dev
#      libopenblas-dev liblapack-dev libssl-dev zlib1g-dev unzip
#   2) NVIDIA sbsa 源: libcudnn9-dev-cuda-13(cuDNN) + libnccl-dev(NCCL)
#      —— paddle 的 cmake 强制要求 cuDNN 与 NCCL, 当前服务器均未安装
#   3) 顺带安装 dos2unix(脚本从Windows上传时去CRLF用)
# 输入输出: 输入为系统apt源(已含NVIDIA sbsa源); 输出为安装完成的依赖,
#   日志写入 /data/paddleocr/logs/00_install_build_deps.log
# ============================================================
set -e

BASE=/data/paddleocr
LOG=$BASE/logs/00_install_build_deps.log
mkdir -p "$BASE/logs"

# 远程非交互执行时, 可通过环境变量 SUDO_PASS 传入sudo密码(免交互)
if [ -n "${SUDO_PASS:-}" ]; then
    sudo() { echo "$SUDO_PASS" | command sudo -S "$@"; }
fi

ts() { date '+%F %T'; }
echo "[$(ts)] 开始安装编译依赖" | tee -a "$LOG"

echo "[$(ts)] apt update ..." | tee -a "$LOG"
sudo apt-get update 2>&1 | tail -2 | tee -a "$LOG"

echo "[$(ts)] 安装编译工具链 ..." | tee -a "$LOG"
sudo apt-get install -y --no-install-recommends \
    ninja-build patchelf swig gfortran python3-dev unzip dos2unix \
    libopenblas-dev liblapack-dev libssl-dev zlib1g-dev 2>&1 | tail -3 | tee -a "$LOG"

echo "[$(ts)] 安装 cuDNN9 + NCCL (NVIDIA sbsa 源, CUDA13) ..." | tee -a "$LOG"
sudo apt-get install -y --no-install-recommends \
    libcudnn9-dev-cuda-13 libnccl-dev 2>&1 | tail -3 | tee -a "$LOG"

echo "[$(ts)] 验证关键工具:" | tee -a "$LOG"
for t in cmake ninja nvcc swig; do
    v=$(command -v $t >/dev/null 2>&1 && $t --version 2>/dev/null | head -1 || echo "缺失!")
    echo "  $t → $v" | tee -a "$LOG"
done
ldconfig -p | grep -q libcudnn && echo "  cuDNN → OK" | tee -a "$LOG" || echo "  cuDNN → 未找到!" | tee -a "$LOG"
ldconfig -p | grep -q libnccl  && echo "  NCCL  → OK" | tee -a "$LOG" || echo "  NCCL  → 未找到!" | tee -a "$LOG"

echo "[$(ts)] 依赖安装完成" | tee -a "$LOG"
