#!/usr/bin/env bash
# ============================================================
# 调用说明: 在【宿主机】(Windows, 本机) 执行, 不是在 Spark 上执行:
#   bash 00_offline_prepare.sh
#   前置: git 已配置代理访问 GitHub(宿主机默认已配 127.0.0.1:7890);
#         网络可达 download.pytorch.org / jetson-ai-lab / 清华 PyPI 源。
# 脚本逻辑: 一次性离线下载 PaddleOCR-VL 在 DGX Spark 上编译+安装所需的
#   全部源码与 wheel, 并打包成 tar 供 scp 传到 Spark:
#   1) Paddle 源码: 完整 clone 固定 SHA + 全部子模块(含 flashattn 嵌套 cutlass);
#   2) torch 三件套(cu129 aarch64) + triton(jetson aarch64): 直接 URL 下载;
#   3) flash-attn 2.8.3 sdist(源编用);
#   4) 编译前置 + paddleocr[doc-parser] 依赖 + vllm(均 aarch64 Linux cp312 wheel);
#   5) 打包为 paddle_src.tar.gz 与 offline_wheels.tar.gz 两个 tar。
# 输入输出:
#   输入: 宿主机网络(GitHub 经代理 / download.pytorch.org / jetson-ai-lab / 清华源)
#   输出: ./offline/{paddle_src,wheels,sdist} + paddle_src.tar.gz + offline_wheels.tar.gz
#   ★注意宿主机与 Spark 环境区别★: 宿主机=Windows x86_64/Py3.11,
#      Spark=Linux aarch64/Py3.12, 故下载 wheel 必须指定
#      --platform manylinux2014_aarch64 / manylinux_2_28_aarch64 + --python-version 312
# ============================================================
set -e

# ---------- 固定参数 ----------
PADDLE_COMMIT=212a3f64948c45ba5608389a2e1e4bb453555f20
PY_VER=312                       # Spark 的 Python 3.12
ARCH1=manylinux2014_aarch64      # 大多数二进制包的平台标签(shapely/opencv/vllm 等)
ARCH2=manylinux_2_28_aarch64     # torch/outlines_core 等更严格的标签
INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# 脚本所在目录(spark/scripts/), 离线目录在其上一级 spark/offline/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OFFLINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/offline"
SRC_DIR=$OFFLINE_DIR/paddle_src
WHEELS_DIR=$OFFLINE_DIR/wheels
SDIST_DIR=$OFFLINE_DIR/sdist
mkdir -p "$SRC_DIR" "$WHEELS_DIR" "$SDIST_DIR"

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

# 启用 Windows 长路径(否则 cutlass 子模块含超长文件名会 checkout 失败)
git config --global core.longpaths true 2>/dev/null || true

# ---------- 1. Paddle 源码 + 子模块 ----------
say "== 1. Paddle 源码(固定 SHA $PADDLE_COMMIT) + 子模块 =="
if [ ! -d "$SRC_DIR/Paddle/.git" ]; then
    say "clone Paddle 源码(完整历史, 约 500MB) ..."
    git clone https://github.com/PaddlePaddle/Paddle.git "$SRC_DIR/Paddle"
else
    say "源码已存在, 跳过 clone"
fi
cd "$SRC_DIR/Paddle"
say "checkout 固定 SHA ..."
git checkout "$PADDLE_COMMIT" 2>&1 | tail -1
# pocketfft 官方源 gitlab.mpcdf.mpg.de 有反爬, 改用作者 GitHub 镜像
git config submodule.third_party/pocketfft.url https://github.com/mreineck/pocketfft.git
export GIT_TERMINAL_PROMPT=0
say "拉取顶层子模块(非递归) ..."
git submodule update --init 2>&1 | tail -3
say "拉取 flashattn 嵌套子模块(cutlass, 参与编译) ..."
(cd third_party/flashattn && git submodule update --init --recursive 2>&1 | tail -3) || \
    say "警告: flashattn 嵌套子模块未完整, 请确认 core.longpaths 已启用"

# ---------- 2. torch 三件套 + triton(直接 URL) ----------
say "== 2. torch 三件套 + triton =="
cd "$WHEELS_DIR"
TORCH_IDX=https://download.pytorch.org/whl/cu129
dl() { # $1=URL $2=文件名
    if [ -f "$2" ]; then say "已存在: $2"; else
        say "下载: $2"
        curl -sL --retry 3 -o "$2" "$1"
    fi
}
dl "$TORCH_IDX/torch-2.8.0%2Bcu129-cp312-cp312-manylinux_2_28_aarch64.whl"        torch-2.8.0+cu129-cp312-cp312-manylinux_2_28_aarch64.whl
dl "$TORCH_IDX/torchvision-0.23.0-cp312-cp312-manylinux_2_28_aarch64.whl"        torchvision-0.23.0-cp312-cp312-manylinux_2_28_aarch64.whl
dl "$TORCH_IDX/torchaudio-2.8.0-cp312-cp312-manylinux_2_28_aarch64.whl"          torchaudio-2.8.0-cp312-cp312-manylinux_2_28_aarch64.whl
dl "https://pypi.jetson-ai-lab.io/jp6/cu129/+f/7df/de4216178cb05/triton-3.4.0-cp312-cp312-linux_aarch64.whl" triton-3.4.0-cp312-cp312-linux_aarch64.whl

# ---------- 3. flash-attn 2.8.3 sdist(源编用) ----------
say "== 3. flash-attn 2.8.3 sdist =="
if [ -f "$SDIST_DIR/flash_attn-2.8.3.tar.gz" ]; then
    say "已存在: flash_attn-2.8.3.tar.gz"
else
    say "下载 flash_attn-2.8.3.tar.gz(从清华源解析直链) ..."
    python - <<PYEOF
import urllib.request, re, os
base = 'https://pypi.tuna.tsinghua.edu.cn/simple/flash-attn/'
req = urllib.request.Request(base, headers={'User-Agent': 'Mozilla/5.0'})
data = urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'ignore')
for m in re.finditer(r'href="([^"]*flash_attn-2\.8\.3\.tar\.gz[^"]*)"', data):
    href = m.group(1)
    if '.post1' in href:
        continue
    full = href if href.startswith('http') else 'https://pypi.tuna.tsinghua.edu.cn/' + href.lstrip('../../').split('#')[0]
    fn = os.path.join('$SDIST_DIR', 'flash_attn-2.8.3.tar.gz')
    urllib.request.urlretrieve(full, fn)
    print('已下载:', fn, os.path.getsize(fn), '字节')
    break
PYEOF
fi

# ---------- 4. pip wheel(编译前置 + paddleocr 依赖 + vllm) ----------
say "== 4. 编译前置(纯 Python / aarch64) =="
cd "$WHEELS_DIR"
pip download setuptools wheel packaging -d . -i "$INDEX" 2>&1 | tail -2 || true
pip download ninja -d . --platform "$ARCH1" --python-version $PY_VER --implementation cp --abi "cp$PY_VER" --only-binary=:all: -i "$INDEX" 2>&1 | tail -2 || true

say "== 5. paddleocr[doc-parser] 依赖(manylinux2014 aarch64) =="
pip download "paddleocr[doc-parser]" requests filetype -d . \
    --platform "$ARCH1" --python-version $PY_VER --implementation cp --abi "cp$PY_VER" \
    --only-binary=:all: -i "$INDEX" 2>&1 | tail -3 || true

say "== 6. vllm 0.10.2 本体 + outlines_core(2_28) =="
pip download "vllm==0.10.2" -d . --no-deps \
    --platform "$ARCH1" --python-version $PY_VER --implementation cp --abi "cp$PY_VER" \
    --only-binary=:all: -i "$INDEX" 2>&1 | tail -3 || true
pip download "outlines_core==0.2.11" -d . --no-deps \
    --platform "$ARCH2" --python-version $PY_VER --implementation cp --abi "cp$PY_VER" \
    --only-binary=:all: -i "$INDEX" 2>&1 | tail -3 || true

# ---------- 7. 打包 ----------
say "== 7. 打包 =="
cd "$OFFLINE_DIR"
say "打包 paddle_src.tar.gz (排除子模块 .git 历史, 只留主仓库 .git + 子模块工作目录) ..."
tar czf paddle_src.tar.gz \
    --exclude='paddle_src/Paddle/.git/modules' \
    --exclude='paddle_src/Paddle/third_party/*/.git' \
    paddle_src
say "打包 offline_wheels.tar.gz ..."
tar czf offline_wheels.tar.gz wheels sdist
say "产出:"
ls -lh "$OFFLINE_DIR"/*.tar.gz

say "========== 离线准备完成 =========="
say "下一步(scp 传到 Spark):"
say "  scp -i ~/.ssh/<SPARK_SSH_KEY> offline/paddle_src.tar.gz offline/offline_wheels.tar.gz <SPARK_USER>@<SPARK_IP>:/data/paddleocr/offline/"
