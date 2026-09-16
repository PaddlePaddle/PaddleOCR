#!/usr/bin/env bash
# ============================================================
# 调用说明: bash 01_build_paddle_wheel.sh
#   耗时约 30~40 分钟(20核并行编译), 建议:
#     nohup bash 01_build_paddle_wheel.sh &   然后 tail -f 日志观察
#   可选环境变量:
#     FLASHATTN_OFF=1  编译前注释掉 cmake/third_party.cmake 中 FlashAttention 相关行
#                      (社区实测 DGX Spark 上编译 flashattn 会段错误, 见 Paddle#76215 SmoothieNoIce 配方)
#     FORCE_CLEAN=1    强制清空 build_dir 重新 cmake 配置
# 脚本逻辑: 在 DGX Spark 上从源码编译 paddlepaddle-gpu aarch64 wheel
#   (官方无 aarch64+CUDA wheel, Paddle#76215 官方回复确认暂无支持计划, 必须自行编译):
#   1) 完整 clone Paddle 源码(不能 --depth 1, ExternalProject 需要子模块tag),
#      checkout 社区验证过的 SHA 212a3f64(见 gaozhi-ustc/paddleocr-dgx);
#   1.6) stamp 一致性自检: 检测子模块补丁标记, 缺失则删陈旧 stamp 令 ninja 重放
#        官方补丁链, 并恢复 zlib/zconf.h(见变更记录⑧);
#   2) 建独立编译 venv, 安装 python/requirements.txt;
#   3) cmake 关键参数: WITH_GPU=ON / CUDA_ARCH_BIN=12.1(GB10 sm_121) /
#      WITH_ARM=ON / WITH_AVX/MKL/MKLDNN/TENSORRT/SLEEF=OFF;
#      CUDA flags 仅保留 -DEIGEN_DONT_VECTORIZE=1(见变更记录①);
#   4) ninja -j$(nproc) 编译, 产出 wheel 拷贝到 /data/paddleocr/wheels/。
# 输入输出: 输入为 github.com 的 Paddle 源码(需联网, 服务器已验证可达);
#   输出为 /data/paddleocr/wheels/paddlepaddle_gpu-*-linux_aarch64.whl,
#   编译日志写 /data/paddleocr/logs/01_build_paddle_wheel.log
# 变更记录:
#   2026-08-11 ① 删除 -DCMAKE_CUDA_FLAGS 里的 -U__ARM_NEON: 2026-08-11 实测该 flag
#                 在 nvcc 13.0.88 上破坏 glibc math-vector.h 的 #ifdef __ARM_NEON 分支,
#                 导致 cmake CUDA 编译器识别失败; 仅保留 -DEIGEN_DONT_VECTORIZE=1
#                 (gaozhi-ustc 仓库 CUDA13.1 成功配置与 SmoothieNoIce 配方均未用该 flag);
#              ② 加 set -o pipefail: 原先 cmake/ninja 报错被 `| tee | tail` 管道吞掉,
#                 ninja 在缺 build.ninja 时盲跑;
#              ③ cmake 前检测上次失败残留(有 CMakeCache.txt 但无 build.ninja)自动清空
#                 build_dir, 可用 FORCE_CLEAN=1 强制清空;
#              ④ 新增 FLASHATTN_OFF=1 后备开关(GitHub 调研 Paddle#76215 得到的社区配方);
#              ⑤ pocketfft 子模块源 gitlab.mpcdf.mpg.de 被 Anubis 反爬拦截(返回HTML),
#                 固定改指向 GitHub 镜像 github.com/mreineck/pocketfft(含 pinned commit);
#              ⑥ 子模块更新改为非递归+3次重试+GIT低网速快速断开(LOW_SPEED 30s),
#                 仅递归编译必需的 flashattn 嵌套子模块(实测 --recursive 在 GitHub
#                 网络抖动时挂起12分钟不动, openvino 等可选组件嵌套无需下载);
#              ⑦ 固定 CUDA 工具链为 /usr/local/cuda-13.0(2026-08-11 实测真根因!):
#                 PATH 首位的 /usr/bin/nvcc 来自 Ubuntu nvidia-cuda-toolkit 包,
#                 是 CUDA 12.0(不支持 sm_121, EDG 前端不识别 glibc math-vector.h
#                 的 SVE/NEON 向量类型 → cmake CUDA 编译器识别必失败, 已用
#                 probe.cu 对照复现: 12.0 报5错, 13.0 零错误)。此前归因于
#                 -U__ARM_NEON flag 的分析有误, 删该 flag 后错误依旧。
#                 修复: export CUDA_HOME/PATH + cmake 显式 -DCMAKE_CUDA_COMPILER。
#              ⑧ 新增「stamp 一致性自检」(步骤1.6, 2026-08-11 双进程事故遗留修复):
#                 事故中并发 git checkout 曾把 third_party 子模块源码回滚为未打补丁
#                 状态, 而 ExternalProject 的 stamp 步骤标记残留 → ninja 跳过补丁
#                 步骤, 用未打补丁源码编译报 warpctc "Unsupported gpu architecture
#                 'compute_50'"(官方 cuda 补丁把写死架构表换成 NVCC_FLAGS_EXTRA)
#                 与 gloo 两处 C++ 编译错误(官方 patches/gloo 恰能修复)。
#                 自检逻辑: 逐个检查 warpctc/warprnnt/gloo/eigen3 源码中的补丁标记,
#                 缺失则删对应 patch/configure/build/install stamp, 令 ninja 重放
#                 补丁链(Paddle 自带补丁命令均以 git checkout 重置源码开头, 可安全
#                 重放); 另恢复被事故误删的 zlib/zconf.h(zlib 无补丁步骤不自愈)。
# ============================================================
set -e
set -o pipefail

# ---------- 0. 固定 CUDA 工具链(必须是 13.0, 见变更记录⑦) ----------
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "[错误] $CUDA_HOME/bin/nvcc 不存在, 请先确认 CUDA 13.0 安装位置"; exit 1
fi

BASE=/data/paddleocr
OFFLINE_SRC=$BASE/offline/paddle_src/Paddle   # 离线源码(宿主机 00_offline_prepare.sh 下载后传入)
SRC=$BASE/build/Paddle
BUILD=$BASE/build/build_dir
BUILD_ENV=$BASE/build/paddle_build_env
WHEELS=$BASE/wheels
LOGS=$BASE/logs
PADDLE_COMMIT=212a3f64948c45ba5608389a2e1e4bb453555f20   # 社区在GB10上验证过的SHA

mkdir -p "$WHEELS" "$LOGS" "$BASE/build"
LOG=$LOGS/01_build_paddle_wheel.log
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "CUDA 工具链: $(command -v nvcc) | $(nvcc --version | tail -1)"

# ---------- 1. 源码准备 ----------
# 优先使用离线源码(宿主机已含 .git + 全部子模块), 无则回退在线 clone
if [ -d "$OFFLINE_SRC/.git" ]; then
    say "检测到离线源码, 使用 $OFFLINE_SRC (跳过 clone)"
    SRC="$OFFLINE_SRC"
elif [ ! -d "$SRC/.git" ]; then
    say "clone Paddle 源码(完整历史, 约1~2GB) ..."
    git clone https://github.com/PaddlePaddle/Paddle.git "$SRC" 2>&1 | tail -2 | tee -a "$LOG"
else
    say "源码已存在, 跳过 clone"
fi
cd "$SRC"
say "checkout 固定 SHA $PADDLE_COMMIT ..."
git checkout "$PADDLE_COMMIT" 2>&1 | tail -1 | tee -a "$LOG"
# 离线源码(宿主机 Windows 打包)会引入两个问题, 编译前必须修复:
# 1) autocrlf 把子模块文件行尾转成 CRLF, git 判定文件被修改 → ExternalProject patch
#    步骤 git checkout 冲突 / 补丁未应用(eigen3/cccl 实测 1770+ 文件被标记修改);
# 2) 宿主机 .git/config 里 core.symlinks=false 被带过来, git checkout 不建符号链接
#    (warpctc 的 ctc_entrypoint.cu、cccl 的 11 个链接变普通文件 → nvcc 编译报错)。
if [ "$SRC" = "$OFFLINE_SRC" ]; then
    say "恢复离线子模块干净状态(消除 autocrlf 行尾差异 + 重建符号链接) ..."
    # 先强制 core.symlinks=true(主仓库+所有子模块), 再 git checkout 才能正确重建符号链接
    git config core.symlinks true
    git submodule foreach 'git config core.symlinks true' 2>&1 | tail -2 | tee -a "$LOG"
    git checkout -- . 2>&1 | tail -1 | tee -a "$LOG"
    git submodule foreach 'git checkout -- . 2>/dev/null; git clean -fdq 2>/dev/null' 2>&1 | tail -3 | tee -a "$LOG"
    say "离线子模块恢复完成"
fi
# 离线源码(宿主机打包时已排除子模块 .git 历史, 只保留工作目录): 检测子模块工作目录
# 已 checkout 则跳过 submodule update(否则 git 会因缺 .git 历史而尝试网络 clone 失败)
OFFLINE_SUBS_READY=0
if [ "$SRC" = "$OFFLINE_SRC" ] && [ -f "$SRC/third_party/protobuf/CMakeLists.txt" ] \
    && [ -f "$SRC/third_party/gtest/CMakeLists.txt" ] \
    && [ -d "$SRC/third_party/flashattn/csrc/cutlass" ]; then
    say "离线源码子模块工作目录已就绪, 跳过 submodule update"
    OFFLINE_SUBS_READY=1
fi
if [ "$OFFLINE_SUBS_READY" != "1" ]; then
    # pocketfft 官方源 gitlab.mpcdf.mpg.de 有 Anubis 反爬挑战, git 协议克隆必失败
    # (info/refs 返回 HTML), 改用作者 GitHub 镜像(2026-08-11 实测含 pinned commit ea778e3)
    git config submodule.third_party/pocketfft.url https://github.com/mreineck/pocketfft.git
    # git 网络加固: 传输停滞超30秒即中止(实测递归拉取 GitHub 曾挂起12分钟不动), 便于重试
    export GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=30 GIT_TERMINAL_PROMPT=0
    say "初始化顶层子模块(非递归, 最多重试3次) ..."
    SUB_OK=0
    for i in 1 2 3; do
        if git submodule update --init 2>&1 | tail -2 | tee -a "$LOG"; then
            SUB_OK=1; break
        fi
        say "子模块更新第 $i 次尝试失败, 5秒后重试 ..."
        sleep 5
    done
    if [ "$SUB_OK" != "1" ]; then
        say "错误: 顶层子模块初始化3次均失败, 退出"; exit 1
    fi
    # 仅递归 flashattn 嵌套子模块(其自带 cutlass, 参与编译);
    # openvino/flagcx/gloo/gflags/cutlass 的嵌套属可选组件, 不递归(避免大体积下载与挂起)
    if [ "${FLASHATTN_OFF:-0}" != "1" ]; then
        say "初始化 flashattn 嵌套子模块(自带cutlass) ..."
        (cd "$SRC/third_party/flashattn" && git submodule update --init 2>&1 | tail -2 | tee -a "$LOG") \
            || say "警告: flashattn 嵌套子模块未完整, 若编译报错可改用 FLASHATTN_OFF=1"
    fi
fi

# ---------- 1.5 可选: 禁用 FlashAttention(社区后备配方) ----------
if [ "${FLASHATTN_OFF:-0}" = "1" ]; then
    TP=$SRC/cmake/third_party.cmake
    if grep -qE '^[^#]*(flashattn|WITH_FLASHATTN)' "$TP"; then
        say "FLASHATTN_OFF=1: 注释 $TP 中 FlashAttention 相关行 ..."
        sed -i -E '/^[^#]*(flashattn|WITH_FLASHATTN)/ s/^([[:space:]]*)([^#])/\1# FLASHATTN_OFF \2/I' "$TP"
        grep -n 'FLASHATTN_OFF' "$TP" | tee -a "$LOG" || true
    else
        say "FLASHATTN_OFF=1: 未发现未注释的 flashattn 行(可能已处理), 跳过"
    fi
fi

# ---------- 1.6 stamp 一致性自检(变更记录⑧: 双进程事故遗留修复) ----------
# 背景: 事故中并发 git checkout 曾回滚子模块源码, 但 ExternalProject stamp 残留,
# ninja 误以为补丁已打而跳过 → 用未打补丁源码编译必错(warpctc compute_50 / gloo)。
# 此处检查各子模块源码中的"补丁标记", 缺失则删除对应 patch/configure/build/install
# stamp, 令 ninja 重放补丁链; Paddle 自带补丁命令均以 git checkout 重置开头, 可安全重放。
say "stamp 一致性自检(补丁标记 vs stamp) ..."
heal_stamp() {  # $1=third_party子模块目录名 $2=external名
    local d="$BUILD/third_party/$1/src/extern_$2-stamp"
    say "  → $1: 源码缺补丁标记, 删除陈旧stamp(patch/configure/build/install)令其重放"
    rm -f "$d/extern_$2-patch" "$d/extern_$2-configure" \
          "$d/extern_$2-build" "$d/extern_$2-install"
}
# warpctc/warprnnt: 官方 cuda 补丁把写死的架构表替换为 NVCC_FLAGS_EXTRA(sm_121)
grep -q 'NVCC_FLAGS_EXTRA' "$SRC/third_party/warpctc/CMakeLists.txt" \
    || heal_stamp warpctc warpctc
grep -q 'NVCC_FLAGS_EXTRA' "$SRC/third_party/warprnnt/CMakeLists.txt" \
    || heal_stamp warprnnt warprnnt
# gloo: 官方补丁之一为 device.cc 增加 #include <array>
grep -q '#include <array>' "$SRC/third_party/gloo/gloo/transport/tcp/device.cc" \
    || heal_stamp gloo gloo
# eigen3(header-only): 官方 git apply 补丁改动 TensorRandom.h, 以 git status 为标记
if [ -z "$(git -C "$SRC/third_party/eigen3" status --porcelain -- \
      unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h 2>/dev/null)" ]; then
    heal_stamp eigen3 eigen3
fi
# zlib: 无补丁步骤(不会自愈), 事故曾误删 zconf.h, 直接用 git 恢复
if [ ! -f "$SRC/third_party/zlib/zconf.h" ]; then
    say "  → zlib: zconf.h 缺失, git checkout 恢复"
    git -C "$SRC/third_party/zlib" checkout -- .
fi
say "stamp 一致性自检完成"

# ---------- 2. 编译 venv ----------
if [ ! -d "$BUILD_ENV" ]; then
    say "创建编译 venv: $BUILD_ENV"
    python3 -m venv "$BUILD_ENV"
fi
source "$BUILD_ENV/bin/activate"
say "安装 python 构建依赖(requirements.txt + cython/numpy/protobuf) ..."
pip install -q -U pip setuptools wheel 2>&1 | tail -1
pip install -q cython numpy protobuf 2>&1 | tail -1
pip install -q -r "$SRC/python/requirements.txt" 2>&1 | tail -1

# ---------- 3. cmake 配置 ----------
if [ "${FORCE_CLEAN:-0}" = "1" ]; then
    say "FORCE_CLEAN=1: 强制清空旧 build_dir ..."
    rm -rf "$BUILD"
elif [ -f "$BUILD/CMakeCache.txt" ] && [ ! -f "$BUILD/build.ninja" ]; then
    say "检测到上次失败的 cmake 缓存(有 CMakeCache.txt 无 build.ninja), 清空 build_dir ..."
    rm -rf "$BUILD"
fi
mkdir -p "$BUILD"
cd "$BUILD"
say "cmake 配置 (GB10: CUDA_ARCH_BIN=12.1, ARM64, nvcc=$CUDA_HOME/bin/nvcc, CUDA flags 仅 EIGEN_DONT_VECTORIZE) ..."
cmake "$SRC" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DWITH_GPU=ON \
    -DWITH_TESTING=OFF \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="12.1" \
    -DWITH_ARM=ON \
    -DWITH_AVX=OFF \
    -DWITH_MKL=OFF \
    -DWITH_MKLDNN=OFF \
    -DWITH_TENSORRT=OFF \
    -DWITH_SLEEF=OFF \
    -DCMAKE_CUDA_FLAGS="-DEIGEN_DONT_VECTORIZE=1" \
    -DPYTHON_EXECUTABLE="$BUILD_ENV/bin/python3" \
    2>&1 | tee -a "$LOG" | tail -5

# ---------- 4. 编译 ----------
if [ ! -f "$BUILD/build.ninja" ]; then
    say "错误: cmake 配置失败, build.ninja 不存在, 请查看上方日志!"
    exit 1
fi
say "开始编译 ninja -j$(nproc) (预计30~40分钟) ..."
ninja -j"$(nproc)" 2>&1 | tee -a "$LOG" | tail -5
say "编译完成, 打包 wheel ..."
ninja python_dist 2>/dev/null || true    # 部分版本目标名不同, 失败则用已有产物

# ---------- 5. 收集 wheel ----------
WHL=$(ls "$BUILD"/python/dist/paddlepaddle*gpu*.whl 2>/dev/null | head -1)
if [ -z "$WHL" ]; then
    say "错误: 未找到产出的 wheel! 请检查 $LOG"
    exit 1
fi
cp -v "$WHL" "$WHEELS/" | tee -a "$LOG"
say "完成: $(ls -lh "$WHEELS"/*.whl | awk '{print $NF, $5}')"
