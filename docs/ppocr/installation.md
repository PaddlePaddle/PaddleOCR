---
comments: true
---

# 安装

# 1. 安装飞桨框架

请参考 [飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 安装 `3.0` 及以上版本的飞桨框架。**推荐使用飞桨官方 Docker 镜像。**

# 2. 安装 PaddleOCR

如果只希望使用 PaddleOCR 的推理功能，请参考 [安装推理包](#21-安装推理包)；如果希望进行模型训练、导出等，请参考 [安装训练依赖](#22-安装训练依赖)。在同一环境中安装推理包和训练依赖是允许的，无需进行环境隔离。

## 2.1 安装推理包

从 PyPI 安装最新版本 PaddleOCR 推理包：

```bash
python -m pip install paddleocr
```

或者从源码安装（默认为开发分支）：

```bash
python -m pip install "git+https://github.com/PaddlePaddle/PaddleOCR.git"
```

## 2.2 安装训练依赖

要进行模型训练、导出等，需要首先将仓库克隆到本地：

```bash
# 推荐方式
git clone https://github.com/PaddlePaddle/PaddleOCR

# （可选）切换到指定分支
git checkout release/3.0

# 如果因为网络问题无法克隆成功，也可选择使用码云上的仓库：
git clone https://gitee.com/paddlepaddle/PaddleOCR

# 注：码云托管代码可能无法实时同步本 GitHub 项目更新，存在3~5天延时，请优先使用推荐方式。
```

执行如下命令安装依赖：

```bash
python -m pip install -r requirements.txt
```
