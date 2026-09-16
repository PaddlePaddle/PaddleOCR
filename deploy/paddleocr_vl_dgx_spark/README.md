# PaddleOCR-VL 在 NVIDIA DGX Spark (GB10) 上的部署方案


> 编写日期：2026-08-14

## 标识符约定

文档与脚本中的 `<...>` 为占位符，使用前请替换为实际环境值；服务器工作目录统一为 `/data/paddleocr`。

| 标识符 | 含义 |
|---|---|
| `<SPARK_IP>` | Spark 服务器 IP 地址 |
| `<SPARK_USER>` | Spark 服务器登录用户名 |
| `<SPARK_HOSTNAME>` | Spark 服务器主机名 |
| `<SPARK_SSH_KEY>` | SSH 私钥文件名（位于 `~/.ssh/`） |
| `<SUDO_PASSWORD>` | sudo 密码 |

---

## 1. 为什么选择这种方案

NVIDIA DGX Spark（GB10）是 **aarch64 架构 + Blackwell sm_121 算力 + CUDA 13.0** 的机器，与常见 x86_64 GPU 服务器有三点本质差异，导致官方标准路径全部失效：

| 官方路径 | 失效原因 |
|---|---|
| 官方 Docker 镜像 | paddleocr-vl 镜像仅 x86_64，ARM64 无法运行 |
| `paddlepaddle-gpu` pip wheel | 官方无 aarch64+CUDA 版本，且明确暂无支持计划（Paddle#76215） |
| xformers | 已确认运行时无调用、非必须依赖，本方案不安装 |

因此本方案采用 **「宿主机离线下载 → 传输 Spark → 源码编译 + pip 离线安装」的原生安装包方式**：

- **paddlepaddle-gpu**：源码编译出 aarch64 wheel（社区验证的固定 SHA 与参数）
- **torch**：官方 cu129 索引的 CUDA 版 aarch64 wheel（PyPI 对 aarch64 只发 CPU 版）
- **flash-attn**：无 aarch64 wheel，源码编译（**xformers 已确认运行时无调用，不再下载/编译/安装，并从源码中去除**）
- **paddleocr / paddlex / vLLM**：纯 Python 或官方 aarch64 wheel，pip 安装

架构与官方 Docker 方案**完全等价（两进程）**，只是用两个原生进程替代两个容器。

## 2. 架构

```
 客户端 POST /layout-parsing :8080
   │
   ├─► [进程2] paddlex --serve (PaddleOCR-VL-1.6 流水线, PP-DocLayoutV3 版面检测)
   │         │
   │         └─► [进程1] paddleocr genai_server (vLLM 后端, PaddleOCR-VL-1.6-0.9B)
   │                     仅监听 127.0.0.1:8081
```

## 3. 部署总览（完整流程）

```
阶段A(宿主机/Windows)  阶段B(传输)        阶段C(Spark/Linux aarch64)          阶段D(启动)
00_offline_prepare.sh ──scp──► 解压 ──► 00依赖 → 01编paddle → 07CUDA12.9 → 02运行时 → 03起服务 → 04验证
```

| 阶段 | 位置 | 动作 | 脚本 |
|---|---|---|---|
| A 离线准备 | 宿主机(Windows) | 下载源码+wheel+源编包，打包 | `00_offline_prepare.sh` |
| B 传输 | 宿主机→Spark | scp 两个 tar 到 `/data/paddleocr/offline/` | `scp` |
| C 编译安装 | Spark | 装依赖 → 编 paddle → 装运行时 | `00`/`01`/`07`/`02` |
| D 启动验证 | Spark | 起两进程 → 端到端验证 | `03`/`04` |

## 4. 脚本清单

| 脚本 | 运行位置 | 作用 | 耗时 |
|---|---|---|---|
| `00_offline_prepare.sh` | 宿主机(Windows) | 离线下载 Paddle 源码(固定SHA+子模块)、torch 三件套、triton、flash-attn sdist、paddleocr/vllm 依赖 wheel，打包成 tar | 依网速(下载约 5GB) |
| `00_install_build_deps.sh` | Spark | 装源码编译 paddle 所需系统依赖（编译工具链 + cuDNN9/NCCL），需 sudo | 约 2 分钟 |
| `01_build_paddle_wheel.sh` | Spark | 源码编译 paddlepaddle-gpu aarch64 wheel（优先用离线源码） | 30~40 分钟 |
| `07_install_cuda_toolkit.sh` | Spark | 装 CUDA 12.9 工具链（供 flash-attn 源编，与 torch cu129 主版本一致），需 sudo | 5~15 分钟 |
| `02_setup_runtime.sh` | Spark | 建 venv、装 paddleocr、覆盖 GPU wheel、装 torch、源编 flash-attn、装 vLLM、六项验收（离线 wheel 优先） | 约 2 小时 |
| `03_start_services.sh` | Spark | 启动 VLM(8081)+API(8080) 两进程，`start/stop/status` | 首次约 2 分钟 |
| `04_health_check.sh` | Spark | 三层验证：VLM health / API health / 端到端解析 | 秒级 |

## 5. 详细步骤

### 阶段 A：宿主机离线准备（Windows）

> 前置：宿主机 git 已配代理访问 GitHub（`git config --global http.proxy http://127.0.0.1:7890`），且能访问 download.pytorch.org / jetson-ai-lab / 清华 PyPI 源。

```bash
cd deploy/paddleocr_vl_dgx_spark/scripts
bash 00_offline_prepare.sh
# 产出 offline/paddle_src.tar.gz 与 offline/offline_wheels.tar.gz
```

### 阶段 B：传输到 Spark

```bash
scp -i ~/.ssh/<SPARK_SSH_KEY> \
    offline/paddle_src.tar.gz offline/offline_wheels.tar.gz \
    <SPARK_USER>@<SPARK_IP>:/data/paddleocr/offline/
# Spark 上解压
ssh -i ~/.ssh/<SPARK_SSH_KEY> <SPARK_USER>@<SPARK_IP> \
  'cd /data/paddleocr/offline && tar xzf paddle_src.tar.gz && tar xzf offline_wheels.tar.gz'
```

### 阶段 C：Spark 编译安装

> 脚本上传：`scp -i ~/.ssh/<SPARK_SSH_KEY> -r deploy/paddleocr_vl_dgx_spark/scripts <SPARK_USER>@<SPARK_IP>:/data/paddleocr/`

```bash
# 步骤 0：系统依赖（sudo）
bash scripts/00_install_build_deps.sh

# 步骤 1：编译 paddle GPU wheel（30~40 分钟，建议 nohup 后台跑）
nohup bash scripts/01_build_paddle_wheel.sh &   # tail -f logs/01_build_paddle_wheel.log

# 步骤 2：CUDA 12.9 工具链（sudo，幂等）
echo <SUDO_PASSWORD> | sudo -S bash scripts/07_install_cuda_toolkit.sh

# 步骤 3：运行时环境（约 2 小时，含 flash-attn 源编）
nohup bash scripts/02_setup_runtime.sh &        # tail -f logs/02_setup_runtime.log
```

### 阶段 D：启动与验证

```bash
bash scripts/03_start_services.sh start    # 先起 VLM 8081 等就绪，再起 API 8080
bash scripts/04_health_check.sh            # 三层验证
```

服务地址 `http://<SPARK_IP>:8080`（接口 `/layout-parsing`）。

## 6. 成品离线镜像（可选，跳过阶段 A~C）

完整跑完上述流程后的整机环境已打包为离线镜像，下载恢复后可直接进入阶段 D（起服务+验证），无需编译、全程不联网。

- **内含**：`/data/paddleocr` 项目目录（含全部 Python 依赖的 venv、部署脚本、离线 wheel）、模型权重（`~/.paddlex`，PaddleOCR-VL-1.6 等，约 2.0GB）、`/usr/local/cuda-12.9` 工具链、6 个系统运行时 deb、恢复脚本 `restore.sh`
- **软件栈**：paddleocr 3.7.0 / paddlex 3.7.2（已去除 xformers 依赖）/ paddlepaddle-gpu 3.4.0.dev（aarch64 sm_121）/ vllm 0.10.2 / torch 2.8.0+cu129 / triton 3.4.0 / flash-attn 2.8.3
- **目标机要求**：DGX Spark（GB10 / aarch64 / sm_121，DGX OS Ubuntu 24.04 基线），磁盘空闲 ≥ 40GB（安装后占约 21GB）
- **sha256**：`d9428a17d4005899727acbf8d2d0781c68e5cb7e45a6912ef65b7aa8a887099b`

通过网盘分享的文件：paddleocr-image-20260812.tar.zst
链接: https://pan.baidu.com/s/1h3Glbus3Qjk6UlexB0magw?pwd=u97w 提取码: u97w 
--来自百度网盘超级会员v7的分享

恢复步骤（Spark 上执行，约 10~20 分钟）：

```bash
# 校验完整性并解压（解压出 paddleocr-image/ 目录）
sha256sum paddleocr-image-20260812.tar.zst    # 与上面 sha256 比对
tar --zstd -xf paddleocr-image-20260812.tar.zst
cd paddleocr-image

# 一键恢复（--config 指向外置配置文件目录，即本方案 config/，含 pipeline_config_vllm.yaml）
sudo bash restore.sh --config /path/to/config
#   无人值守可加 --yes；安装到其他盘可加 --target /data2/paddleocr
#   目标机已有 cuda-12.9 / 六个 deb 时可加 --skip-cuda / --skip-debs

# 恢复完成，直接进入阶段 D
bash /data/paddleocr/scripts/03_start_services.sh start
bash /data/paddleocr/scripts/04_health_check.sh
```

## 7. 目录结构（服务器侧）

```
/data/paddleocr/
├── scripts/          # 部署脚本(00离线准备/00依赖/01编paddle/07CUDA12.9/02运行时/03起服务/04验证)
├── config/           # pipeline_config_vllm.yaml
├── offline/          # 离线包(宿主机传入: paddle_src/ wheels/ sdist/)
├── logs/             # 编译/服务日志
├── wheels/           # 编译产出的 paddlepaddle_gpu wheel
├── build/            # Paddle 源码与编译中间产物(约 20GB, 稳定后可删)
└── venv/             # 运行时虚拟环境
```

## 8. 常用运维命令

```bash
bash scripts/03_start_services.sh stop      # 停服务
bash scripts/03_start_services.sh status    # 查状态
bash scripts/03_start_services.sh start     # 起服务(幂等)
bash scripts/04_health_check.sh             # 端到端验证
```
