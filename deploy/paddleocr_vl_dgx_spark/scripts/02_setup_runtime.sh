#!/usr/bin/env bash
# ============================================================
# 调用说明: bash 02_setup_runtime.sh
#   前置: 已执行 01_build_paddle_wheel.sh, /data/paddleocr/wheels/ 下有
#         paddlepaddle_gpu-*-linux_aarch64.whl
#   耗时: 首跑全程约 2 小时(其中 CUDA torch 下载约7~15分钟 + 源编
#         flash-attn 约1~2小时); 包齐后重跑约10分钟
#         (torch/CUDA检查通过即跳过下载, 编译步骤被 pip 判为 already satisfied)。
# 脚本逻辑: 建立 PaddleOCR-VL 运行时环境(路线①=单venv源编补齐, 用户2026-08-11拍板,
#   背景见 handover.md 4.9/4.10 节):
#   1) 创建 /data/paddleocr/venv (Python3.12);
#   2) pip install "paddleocr[doc-parser]" —— 会自动带入 CPU 版 paddlepaddle;
#   3) 用自编译的 GPU wheel 覆盖安装(已装同版本则跳过, 见变更记录⑥b;
#      paddleocr/paddlex 均不声明 paddlepaddle 依赖, wheel 自身依赖
#      protobuf/opt_einsum/networkx/httpx 等必须由 pip 按 METADATA 补齐);
#   4) ★从一开始就装 CUDA 版 torch 三件套(变更记录⑦)★:
#      PyPI 对 aarch64 只有 CPU 版 torch(2.8.0+cpu), CUDA 版 2.8.0 aarch64
#      仅官方 cu129 索引有(cu126/cu128 均无, 脚本06实测), 故按直接 URL 安装
#      torch-2.8.0+cu129(3.4GB自包含CUDA) / torchvision-0.23.0 / torchaudio-2.8.0;
#      torch 已是 CUDA 版(torch.version.cuda 非空)则跳过; CPU 版残留先卸载。
#      ★装 torch 前先预装 triton 3.4.0 aarch64(变更记录⑧): torch METADATA 钉
#      triton==3.4.0, PyPI 该版本无 aarch64 wheel, 需从 jetson-ai-lab cu129
#      索引直接 URL 安装, 否则 pip 整体依赖解析失败;
#      装完立即做 CUDA 冒烟测试(is_available + GPU matmul), 不过则中止,
#      不浪费后续数小时编译;
#   5) ★源编 flash-attn 2.8.3(路线①核心, 已去除 xformers)★:
#      PyPI 无 flash-attn 适配 aarch64 的 wheel, 只能源编。配方为 DGX Spark/sm_121
#      社区验证参数:
#        TORCH_CUDA_ARCH_LIST="12.1a" MAX_JOBS=4 pip install ... --no-build-isolation
#      flash-attn 版本=2.8.3(变更记录⑩: paddlex_cli.py:386 对 cap>=12.0 强制
#      pin 该版, 原 2.7.4.post1 装完仍过不了 install_genai_server_deps)。
#      ★xformers 已彻底去除★(handover 4.12.3 实证运行时无调用、4.13 拍板):
#      不再源编, 由第 5.5 步打补丁去除 paddlex 的两处纸面引用。
#      编译前固定 CUDA_HOME=/usr/local/cuda-12.9(需先跑脚本07安装; CUDA13.0
#      被 torch cpp_extension 主版本检查拒绝, /usr/bin/nvcc=12.0 编不了
#      sm_121, 见变更记录⑨), 并补装编译前置 setuptools/wheel/ninja/packaging;
#   5.5) ★补丁去除 paddlex 对 xformers 的纸面引用(原脚本08, 已内联)★:
#      deps.py:283 双包检查 → 只查 flash-attn; paddlex_cli.py:378 注释 xformers
#      安装命令。必须在 install_genai_server_deps 之前打, 否则它内部仍会装 xformers;
#   6) 跑 paddleocr install_genai_server_deps vllm 装 vLLM 及其余依赖
#      (补丁后 paddlex_cli.py 不再装 xformers; torch==2.8.0 按 PEP440 匹配
#      2.8.0+cu129, 不会重碰 torch; deps.py:283 只查 flash-attn 应 satisfied);
#      随后补装 filetype>=1.2(变更记录⑪: serving 依赖组实测漏装该包,
#      导致 API 8080 启动报 serving plugin 不可用);
#   7) 六项验收: paddle run_check / import paddleocr / import vllm /
#      import xformers+flash_attn / paddlex genai vllm-server 插件可用性 /
#      paddlex serving 插件可用性(变更记录⑪)。
# 输入输出: 输入为 wheels/ 下的 wheel、download.pytorch.org/whl/cu129/ 的
#   torch 三件套(服务器实测可达约8.5MB/s)、PyPI 镜像(默认清华 TUNA,
#   见变更记录⑥); 输出为可用的 /data/paddleocr/venv,
#   日志增量写 logs/02_setup_runtime.log
# 变更记录:
#   2026-08-11 ① 覆盖安装去掉 --no-deps: 首次实测后 import paddle 报缺 google
#                 (protobuf), 根因是 paddleocr/paddlex 都不声明 paddlepaddle
#                 依赖(软依赖, 运行时 import), 而 --no-deps 又跳过了 wheel 自身
#                 METADATA 里的 protobuf/opt_einsum==3.3.0/networkx/httpx 等;
#              ② venv 建立后 export PATH="$VENV/bin:$PATH": 首次实测
#                 install_genai_server_deps vllm 报 FileNotFoundError 'paddlex',
#                 根因是 paddleocr 内部子进程按 PATH 找 paddlex 可执行文件,
#                 venv 未激活/未入 PATH 时找不到;
#              ③ 加 set -o pipefail: 原先各安装步骤 `| tail` 管道会吞掉 pip
#                 非零退出码, 导致失败被掩盖、脚本谎报"就绪"。
#              ④ 路线①(用户拍板): 插入 flash-attn/xformers 源编步骤:
#                 a. xformers 版本从原计划 0.0.32.post1 改为 0.0.32.post2:
#                    经 GitHub 读两 tag 的 requirements.txt(内容相同, torch>=2.8)
#                    与 setup.py(install_requires 即该文件), post2 是同系列更晚
#                    bugfix 且 torch 约束一致, 取更新者;
#                 b. 编译前 pip install setuptools wheel ninja packaging:
#                    xformers 0.0.32.post2 构建要求 setuptools>=64 + torch>=2.7,
#                    Python3.12 venv 默认不带 setuptools;
#                 c. 固定 CUDA_HOME=/usr/local/cuda-13.0 且 PATH 前置并 nvcc -V
#                    校验(与脚本01同一陷阱: /usr/bin/nvcc=CUDA12.0 编不了 sm_121);
#                 d. 首跑装 torch 的 install_genai_server_deps 改为容忍失败且
#                    torch 已在则跳过(重跑场景避免白等 xformers 0.0.35 源编失败);
#                 e. flash-attn 取 2.7.4.post1: NVIDIA NGC 容器在 GB10 预装同款,
#                    社区 Spark 有验证配方(flash-attention#1969、natolambert/
#                    dgx-spark-setup); xformers 内置 FA2/FA3 扩展对 12.1a 会自动
#                    跳过(源码逻辑确认), 不会编译不支持的组件。
#              ⑤ 验收从三项扩为五项: 增加 xformers/flash_attn 可导入与
#                 paddlex is_genai_engine_plugin_available('vllm-server') 检查。
#              ⑥ PyPI 境外网络闪断的修复(2026-08-11 16:58 实测退出后, 用户批准):
#                 现象=步骤3覆盖安装报 "No matching distribution found for httpx"
#                 (from versions: none), 根因双层:
#                 a. 境外 pypi.org/files.pythonhosted.org 当时不可达(探测4次全000,
#                    同期清华TUNA/阿里云/hf-mirror 均200) → 脚本头 export
#                    PIP_INDEX_URL=清华TUNA(全量镜像, 含 flash-attn/xformers 源码包;
#                    环境变量继承, install_genai_server_deps 的子进程 pip 同样生效);
#                 b. --force-reinstall 会对 wheel 全部依赖重查 PyPI(本可不查):
#                    venv 内 paddlepaddle-gpu 与 wheel 同版本, 覆盖安装本身不必要
#                    → 改为"已装同版本则跳过";
#                 c. pip 升级改为容忍失败(网络异常时回退现有 pip 继续)。
#   2026-08-12 ⑦ torch 从一开始就装 CUDA+aarch64 版(用户拍板, 替换原步骤4的
#                 "install_genai_server_deps 装 torch 并容忍失败"方案):
#                 现象=08-11 源编 flash-attn 报 "nvcc was not found"/"CUDA_HOME
#                 is not set", 且打印 torch.__version__=2.8.0+cpu。根因=PyPI
#                 对 aarch64 只发 CPU 版 torch(官方 Dockerfile 裸装 torch==2.8.0
#                 拿到 CUDA 版仅因 x86 PyPI 默认 CUDA 版), CPU torch 的
#                 cpp_extension 即使设 CUDA_HOME 也解析为 None → CUDA 扩展编不了,
#                 vllm 也无法推理。脚本06实证来源: cu126/cu128 索引均无
#                 torch 2.8.0 aarch64 wheel(cu128 仅 2.7.0/2.7.1/2.9.0/2.9.1/
#                 2.10/2.11); cu129 索引三件套齐全(torch-2.8.0+cu129 3.41GB
#                 自包含CUDA构建/torchvision-0.23.0/torchaudio-2.8.0,
#                 均 cp312 manylinux_2_28_aarch64), HEAD 全部200、实测约8.5MB/s。
#                 PEP440: vllm/paddlex 的 torch==2.8.0 pin 匹配 2.8.0+cu129
#                 (本地版本被忽略); vllm 的 xformers==0.0.32.post1 声明仅限
#                 x86_64, aarch64 不触发。装 torch 后新增 CUDA 冒烟测试
#                 (is_available+matmul)作前置门禁, 不过立即中止不进入编译。
#   2026-08-12 ⑧ 预装 triton 3.4.0 aarch64, 修复 torch 三件套依赖解析失败:
#                 现象=08-12 09:12 重跑, 步骤4 torch 3.4GB 下载完成后 pip 整体
#                 依赖解析失败回滚: ERROR Could not find a version that satisfies
#                 the requirement triton==3.4.0; platform_system == "Linux"
#                 (from torch) (from versions: 3.5.0, 3.5.1, 3.6.0, 3.7.0, 3.7.1)。
#                 根因=torch 2.8.0+cu129 METADATA 钉 triton==3.4.0(Linux 限定、
#                 无 machine 限定符), 而 PyPI 的 triton 3.4.0 只有 x86_64 wheel
#                 (aarch64 wheel 从 3.5.0 起才有), pypi.nvidia.com/triton 404。
#                 解法=装 torch 前先从 jetson-ai-lab 索引(jp6/cu129, JetPack6.2
#                 =CUDA12.9+Py3.12+torch2.8 官方配套)预装
#                 triton-3.4.0-cp312-cp312-linux_aarch64.whl(HEAD 200,
#                 266999596 字节, sha256 嵌入 URL 片段由 pip 原生校验);
#                 triton 已装 3.4.0 则跳过。旁证: vllm/paddlex METADATA 均不声明
#                 triton(仅 torch 钉), 但 vllm 代码 11 处 import triton
#                 (rotary/MoE/量化等路径), 必须真装而非 --no-deps 绕过。
#                 09:12 重跑实测成功: triton 装好、torch 三件套依赖解析通过
#                 (3.4GB wheel 命中 pip 缓存, 18秒装完)。
#   2026-08-12 ⑨ 编译工具链从 CUDA 13.0 换为 12.9(cpp_extension 主版本检查):
#                 现象=08-12 09:30 CUDA 冒烟测试通过(GB10/sm_121 matmul OK,
#                 cu129 torch 第一道关已过), 但 09:32 源编 flash-attn 报
#                 RuntimeError: The detected CUDA version (13.0) mismatches
#                 the version that was used to compile PyTorch (12.9)。
#                 根因=torch cpp_extension.py:506 对 nvcc 与 torch.version.cuda
#                 做版本比较: 主版本不同直接 raise(次版本不同仅 warning),
#                 无绕过开关; 机器只有 CUDA 13.0(主版本13≠12被拒)与
#                 /usr/bin/nvcc 12.0(<12.8 编不了 sm_121)。
#                 解法=新脚本 07_install_cuda_toolkit.sh 从已配置的 NVIDIA 官方
#                 sbsa apt 源装 cuda-toolkit-12-9(12.9.2-1, 约5GB, 装至
#                 /usr/local/cuda-12.9); 本脚本 CUDA_HOME 改指 12.9 并校验
#                 nvcc 为 12.x, 否则报错指向脚本07。12.9 与 torch.version.cuda
#                 =12.9 完全一致(连 warning 都无)且 >=12.8 可编 sm_121。
#                 注: 冒烟测试已通过证明 cu129 torch 运行时在 GB10 完全正常,
#                 不换 torch(不冒 cu130 wheel 缺 sm_121 kernel 的未知风险)。
#   2026-08-12 ⑩ flash-attn 版本 2.7.4.post1 → 2.8.3(用户拍板, 失败5对策A):
#                 现象=02 第五跑双包源编成功后, 收尾 install_genai_server_deps
#                 失败(失败5): paddlex_cli.py:386 对算力 cap>=(12,0) 强制装
#                 flash-attn==2.8.3(官方 x86 sm120 镜像配套版, 无 aarch64 wheel),
#                 pip 回退 sdist 源编时默认 build isolation 环境无 torch 而败
#                 (ModuleNotFoundError: No module named 'torch')。
#                 对策=同一配方源编 2.8.3 覆盖 2.7.4.post1: sdist 已下载核实
#                 (/tmp/fa_src/flash_attn-2.8.3.tar.gz, 8.4MB), 其 setup.py:70
#                 与 2.7.4.post1 相同——读 FLASH_ATTN_CUDA_ARCHS 环境变量、默认
#                 "80;90;100;120" 含 sm_120(4.11.6 已实证 sm_120 产物可在 sm_121
#                 运行) → 沿用现有配方(CUDA_HOME=/usr/local/cuda-12.9、MAX_JOBS=4、
#                 --no-build-isolation、TORCH_CUDA_ARCH_LIST 对 flash-attn 本就
#                 不生效, 无需设 12.1a)。重跑时其余步骤全幂等跳过, 仅 flash-attn
#                 重编约1小时+; xformers 已装 0.0.32.post2 判 already satisfied
#                 直接跳过(不重编)。
#   2026-08-12 ⑪ 补装 filetype>=1.2, 验收五项→六项(+serving 插件, 用户批准):
#                 现象=08-12 14:57 VLM(8081)就绪后 API(8080)启动即退:
#                 DependencyError: The serving plugin is not available。
#                 根因=paddlex EXTRAS["serving"] 组9包逐个实测, 唯独缺
#                 filetype(要求>=1.2; fastapi/uvicorn/starlette/opencv-
#                 contrib-python/pypdfium2/yarl/aiohttp/bce-python-sdk 8包
#                 全在位); install_genai_server_deps 的依赖面不覆盖该包。
#                 filetype 为纯 Python 小包(py2.py3-none-any, 19kB, 无平台
#                 限定/无编译依赖), pip 补装安全; 装后 is_serving_plugin_available
#                 =True, API 启动成功(15:01), 04 端到端通过(HTTP 200, demo.png
#                 标题/正文/表格全识别)。固化为步骤6补装+验收6/6, 防重建再漏。
# ============================================================
set -e
set -o pipefail

BASE=/data/paddleocr
VENV=$BASE/venv
LOGS=$BASE/logs
mkdir -p "$LOGS"
LOG=$LOGS/02_setup_runtime.log
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# pip 源统一走清华 TUNA(变更记录⑥a): 2026-08-11 实测境外 pypi.org 闪断不可达;
# 用环境变量形式导出, install_genai_server_deps 内部子进程的 pip 也会继承
# (步骤4的 torch 三件套走 download.pytorch.org 直接 URL, 不受此索引影响)
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 离线包(宿主机 00_offline_prepare.sh 下载后传入 /data/paddleocr/offline/):
# 若存在则让 pip 本地优先(--find-links), 缺失的仍从清华源在线兜底;
# torch/triton/flash-attn 在下方步骤中直接改读本地 wheel/sdist 文件。
OFFLINE_DIR=$BASE/offline
OFFLINE_WHEELS=$OFFLINE_DIR/wheels
OFFLINE_SDIST=$OFFLINE_DIR/sdist
if [ -d "$OFFLINE_WHEELS" ]; then
    export PIP_FIND_LINKS="$OFFLINE_WHEELS"
    say "检测到离线 wheel 目录 $OFFLINE_WHEELS (pip 本地优先, 缺失在线兜底)"
else
    say "未检测到离线 wheel 目录, 全部走清华源在线安装"
fi

# 源编配方(路线①, DGX Spark/sm_121 社区验证参数, 见脚本头注释)
export TORCH_CUDA_ARCH_LIST="12.1a"   # GB10=sm_121, 带 a 后缀(flash-attention#1969)
export MAX_JOBS=4                     # 防128GB统一内存OOM(natolambert/dgx-spark-setup)
# ★强制 flash-attn 本地源码编译★: 2.8.3 的 setup.py 会先尝试从 GitHub releases 下载
# 预编译 wheel, Spark 上 GitHub 被墙时 urlretrieve 抛连接重置(非 HTTPError/URLError),
# 不被其 except 捕获 → 构建直接失败不回落源码编译; 置 TRUE 跳过下载直接本地编译。
export FLASH_ATTENTION_FORCE_BUILD=TRUE
FLASH_ATTN_VER=2.8.3        # 变更记录⑩: 2.7.4.post1→2.8.3(paddlex 对 cap>=12.0 强制 pin)

WHL=$(ls "$BASE"/wheels/paddlepaddle*gpu*.whl 2>/dev/null | head -1)
if [ -z "$WHL" ]; then
    say "错误: wheels/ 下没有 paddle GPU wheel, 请先执行 01_build_paddle_wheel.sh"
    exit 1
fi

# ---------- 1. venv ----------
if [ ! -d "$VENV" ]; then
    say "创建运行时 venv: $VENV"
    python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
# venv 加入 PATH(变更记录②): paddleocr install_genai_server_deps 内部按 PATH
# 找 paddlex 可执行文件, 不激活 venv/不入 PATH 会报 FileNotFoundError 'paddlex'
export PATH="$VENV/bin:$PATH"
say "升级 pip (变更记录⑥c: 容忍失败, 网络异常时回退现有 pip 继续) ..."
$PIP install -q -U pip 2>&1 | tail -1 | tee -a "$LOG" || say "警告: pip 升级失败, 用现有 pip 版本继续"

# ---------- 2. paddleocr ----------
say "安装 paddleocr[doc-parser] (会带入CPU版paddle, 稍后覆盖) ..."
$PIP install "paddleocr[doc-parser]" requests 2>&1 | tail -2 | tee -a "$LOG"

# ---------- 3. 覆盖安装 GPU wheel(变更记录⑥b: 已装同版本则跳过) ----------
# 不带 --no-deps(变更记录①): wheel 自身依赖(protobuf/opt_einsum/networkx/httpx
# 等)必须由 pip 按 METADATA 补齐, paddleocr/paddlex 均不声明 paddlepaddle 依赖
WHL_VER=$(basename "$WHL" | cut -d- -f2)
CUR_VER=$($PIP show paddlepaddle-gpu 2>/dev/null | awk '/^Version:/{print $2}')
if [ "$CUR_VER" = "$WHL_VER" ]; then
    say "paddlepaddle-gpu $CUR_VER 与 wheel 版本一致, 跳过覆盖安装(避免 PyPI 依赖重查)"
else
    say "覆盖安装自编译 GPU wheel: $(basename "$WHL") (当前已装: ${CUR_VER:-无})"
    $PIP install --force-reinstall "$WHL" 2>&1 | tail -2 | tee -a "$LOG"
fi

# ---------- 4. CUDA 版 torch 三件套(变更记录⑦) + triton 预装(变更记录⑧) ----------
# PyPI aarch64 的 torch 是 CPU 版(2.8.0+cpu), CUDA 版 2.8.0 aarch64 仅在官方
# cu129 索引存在(脚本06实证), 按直接 URL 安装; torch 已是 CUDA 版则跳过
TORCH_IDX=https://download.pytorch.org/whl/cu129
if $PY -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
    say "torch 已是 CUDA 版: $($PY -c 'import torch; print(torch.__version__, torch.version.cuda)'), 跳过安装"
else
    say "卸载 CPU 版 torch 三件套残留(若有) ..."
    $PIP uninstall -y torch torchvision torchaudio 2>&1 | tail -3 | tee -a "$LOG" || true
    # ---- 4.1 预装 triton 3.4.0 aarch64(变更记录⑧, torch 硬依赖的前置件) ----
    # torch 2.8.0+cu129 METADATA 钉 triton==3.4.0(Linux限定、无machine限定符),
    # 而 PyPI 的 triton 3.4.0 仅 x86_64 wheel(aarch64 自 3.5.0 起才有),
    # pypi.nvidia.com/triton 404; 仅 jetson-ai-lab 索引(jp6/cu129,
    # JetPack6.2=CUDA12.9+Py3.12+torch2.8 官方配套)有 cp312 linux_aarch64 版。
    # 不预装则 pip 在依赖解析阶段整体失败回滚(08-12 09:16 实测, 见变更记录⑧)。
    if $PY -c "import triton,sys; sys.exit(0 if triton.__version__=='3.4.0' else 1)" 2>/dev/null; then
        say "triton 3.4.0 aarch64 已安装, 跳过"
    else
        # 本地 wheel 优先(离线包), 缺失则用 jetson-ai-lab 直接 URL
        TRITON_WHL=$(ls "$OFFLINE_WHEELS"/triton-3.4.0-*.whl 2>/dev/null | head -1)
        if [ -z "$TRITON_WHL" ]; then
            TRITON_WHL="https://pypi.jetson-ai-lab.io/jp6/cu129/+f/7df/de4216178cb05/triton-3.4.0-cp312-cp312-linux_aarch64.whl#sha256=7dfde4216178cb05f11a1c030bdd94de77f83fd70432eb4390cf68c9a1194ccc"
        fi
        say "安装 triton 3.4.0 aarch64 (${TRITON_WHL##*/}) ..."
        $PIP install "$TRITON_WHL" 2>&1 | tee -a "$LOG" | tail -5
        say "triton 安装完成: $($PY -c 'import triton; print(triton.__version__)')"
    fi
    # 本地 wheel 优先(离线包), 缺失则用 cu129 索引直接 URL
    TORCH_WHL=$(ls "$OFFLINE_WHEELS"/torch-2.8.0+cu129-*.whl 2>/dev/null | head -1)
    TORCHVISION_WHL=$(ls "$OFFLINE_WHEELS"/torchvision-0.23.0-*.whl 2>/dev/null | head -1)
    TORCHAUDIO_WHL=$(ls "$OFFLINE_WHEELS"/torchaudio-2.8.0-*.whl 2>/dev/null | head -1)
    [ -z "$TORCH_WHL" ] && TORCH_WHL="$TORCH_IDX/torch-2.8.0%2Bcu129-cp312-cp312-manylinux_2_28_aarch64.whl"
    [ -z "$TORCHVISION_WHL" ] && TORCHVISION_WHL="$TORCH_IDX/torchvision-0.23.0-cp312-cp312-manylinux_2_28_aarch64.whl"
    [ -z "$TORCHAUDIO_WHL" ] && TORCHAUDIO_WHL="$TORCH_IDX/torchaudio-2.8.0-cp312-cp312-manylinux_2_28_aarch64.whl"
    say "安装 CUDA 版 torch 三件套 (cu129 aarch64) ..."
    $PIP install \
        "$TORCH_WHL" \
        "$TORCHVISION_WHL" \
        "$TORCHAUDIO_WHL" \
        2>&1 | tee -a "$LOG" | tail -5
fi
# CUDA 冒烟测试(变更记录⑦): 源编前置门禁, 不过立即中止不浪费数小时编译
say "CUDA 冒烟测试: torch.version.cuda / is_available / GPU matmul ..."
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch
print('torch', torch.__version__, '| torch.version.cuda:', torch.version.cuda)
assert torch.version.cuda, 'FAIL: torch 不是 CUDA 版'
assert torch.cuda.is_available(), 'FAIL: torch.cuda.is_available()=False'
print('device:', torch.cuda.get_device_name(0), '| capability:', torch.cuda.get_device_capability(0))
x = torch.randn(512, 512, device='cuda')
y = (x @ x).sum().item()
print('matmul OK, sum =', y)
PYEOF
say "CUDA 冒烟测试通过, torch 三件套就绪"

# ---------- 5. 源编 flash-attn(路线①核心; 已去除 xformers, 运行时无调用) ----------
# 固定 CUDA 工具链(变更记录⑨): 必须 nvcc 12.x(与 torch cu129 主版本一致),
# 且 >=12.8 才能编 sm_121 → 用脚本07装好的 /usr/local/cuda-12.9;
# CUDA 13.0 会被 cpp_extension.py:506 主版本检查 RuntimeError 拒绝
export CUDA_HOME=/usr/local/cuda-12.9
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    say "错误: $CUDA_HOME/bin/nvcc 不存在, 请先执行 scripts/07_install_cuda_toolkit.sh 安装 CUDA 12.9 工具链"
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
NVCC_VER=$(nvcc --version | grep -o 'release [0-9]*\.[0-9]*')
case "$NVCC_VER" in
    "release 12."*) : ;;
    *) say "错误: nvcc 版本 $NVCC_VER 不是 12.x, torch cu129 的 cpp_extension 会拒绝(主版本须一致), 请用脚本07装 CUDA 12.9"; exit 1;;
esac
say "CUDA 工具链: $(command -v nvcc) | $(nvcc --version | grep release | tr -s ' ')"

say "安装编译前置 setuptools/wheel/ninja/packaging (Python3.12 venv 无自带 setuptools) ..."
$PIP install -q setuptools wheel ninja packaging 2>&1 | tail -1 | tee -a "$LOG"

# 本地 sdist 优先(离线包), 缺失则 pip 从清华源拉 sdist 源编
FA_SDIST=$(ls "$OFFLINE_SDIST"/flash_attn-$FLASH_ATTN_VER.tar.gz 2>/dev/null | head -1)
if [ -n "$FA_SDIST" ]; then
    say "源编 flash-attn==$FLASH_ATTN_VER 开始(本地 sdist ${FA_SDIST##*/}, 预计1~2小时, MAX_JOBS=$MAX_JOBS) ..."
    $PIP install "$FA_SDIST" --no-build-isolation 2>&1 | tee -a "$LOG"
else
    say "源编 flash-attn==$FLASH_ATTN_VER 开始(清华源拉 sdist, 预计1~2小时) ..."
    $PIP install "flash-attn==$FLASH_ATTN_VER" --no-build-isolation 2>&1 | tee -a "$LOG"
fi
say "源编 flash-attn==$FLASH_ATTN_VER 完成"

# ---------- 5.5 补丁: 去除 paddlex 对 xformers 的纸面引用(原脚本08, 现内联) ----------
# 背景(handover 4.12.3 实证): xformers 在 PaddleOCR-VL/genai vllm-server 路径无任何
# 运行时调用(ViT 走 flash-attn、主 attention 走 vllm 自有后端), 只剩两处纸面引用:
#   deps.py:283 双包导入检查、paddlex_cli.py:378 的 xformers 安装命令。
# 必须在 install_genai_server_deps 之前打补丁, 否则该命令内部仍会尝试装 xformers
# (xformers 无 aarch64 wheel, 会失败)。补丁后依赖链收敛为只依赖 flash-attn。
say "补丁 paddlex 去除 xformers 纸面引用 (deps.py 检查 + paddlex_cli.py 安装) ..."
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import os, sys
P = os.path.dirname(__import__('paddlex').__file__)
deps_path = os.path.join(P, 'utils', 'deps.py')
cli_path = os.path.join(P, 'paddlex_cli.py')
for f in (deps_path, cli_path):
    assert os.path.isfile(f), f'文件不存在 {f}'
    bak = f + '.bak'
    if not os.path.isfile(bak):
        open(bak, 'w', encoding='utf-8').write(open(f, encoding='utf-8').read())
# 补丁1: deps.py 双包检查 -> 只查 flash-attn
s = open(deps_path, encoding='utf-8').read()
old1 = 'return is_dep_available("xformers") and is_dep_available("flash-attn")'
new1 = 'return is_dep_available("flash-attn")'
if old1 in s:
    open(deps_path, 'w', encoding='utf-8').write(s.replace(old1, new1))
    print('[补丁1 deps.py] 已应用: 双包导入检查改为仅 flash-attn')
elif new1 in s:
    print('[补丁1 deps.py] 跳过: 已是仅 flash-attn(幂等)')
else:
    print('[补丁1 deps.py] 异常: 目标表达式未找到, paddlex 版本可能变动'); sys.exit(1)
# 补丁2: paddlex_cli.py 注释 xformers 安装命令
s = open(cli_path, encoding='utf-8').read()
old2 = 'install_packages(["xformers"], constraints="required")'
new2 = '# PATCHED(no-xformers): ' + old2
if new2 in s:
    print('[补丁2 paddlex_cli.py] 跳过: 已注释 xformers 安装(幂等)')
elif old2 in s:
    open(cli_path, 'w', encoding='utf-8').write(s.replace(old2, new2))
    print('[补丁2 paddlex_cli.py] 已应用: 已注释 xformers 安装命令')
else:
    print('[补丁2 paddlex_cli.py] 异常: 目标表达式未找到, paddlex 版本可能变动'); sys.exit(1)
print('补丁完成')
PYEOF
say "paddlex 补丁完成(去 xformers), 依赖链收敛为只依赖 flash-attn"

# ---------- 6. 跑 install_genai_server_deps 装 vLLM 及其余依赖 ----------
# (补丁后 paddlex_cli.py 不再装 xformers, deps.py 只查 flash-attn;
#  torch==2.8.0 匹配 2.8.0+cu129 不会重碰 torch, 应全部 satisfied)
say "执行 install_genai_server_deps vllm (装 vLLM 及其余依赖, 应全部满足) ..."
"$VENV/bin/paddleocr" install_genai_server_deps vllm 2>&1 | tee -a "$LOG" | tail -5

# 变更记录⑪: serving 依赖组实测漏装 filetype(唯一缺失项), API 8080 启动会报
# "The serving plugin is not available"; 此处显式补装(幂等, 已装则 satisfied)
say "补装 filetype>=1.2 (serving 依赖组唯一漏装项, 变更记录⑪) ..."
$PIP install "filetype>=1.2" 2>&1 | tee -a "$LOG" | tail -2

# ---------- 7. 六项验收(变更记录⑤: 五项; 变更记录⑪: +serving 插件) ----------
say "验收 1/6: paddle GPU run_check ..."
$PY -c "import paddle; print('paddle', paddle.__version__); paddle.utils.run_check()" 2>&1 | tail -6 | tee -a "$LOG"
say "验收 2/6: import paddleocr ..."
$PY -c "import paddleocr; print('paddleocr', paddleocr.__version__)" 2>&1 | tail -1 | tee -a "$LOG"
say "验收 3/6: import vllm ..."
$PY -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1 | tee -a "$LOG"
say "验收 4/6: import flash_attn ..."
$PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>&1 | tail -1 | tee -a "$LOG"
say "验收 5/6: paddlex genai vllm-server 插件可用性 ..."
$PY -c "from paddlex.utils.deps import is_genai_engine_plugin_available as f; assert f('vllm-server'), 'FAIL'; print('genai vllm-server plugin OK')" 2>&1 | tee -a "$LOG"
say "验收 6/6: paddlex serving 插件可用性 (变更记录⑪) ..."
$PY -c "from paddlex.utils.deps import is_serving_plugin_available as f; assert f(), 'FAIL'; print('serving plugin OK')" 2>&1 | tee -a "$LOG"
say "运行时环境就绪: $VENV (02 全部完成)"
