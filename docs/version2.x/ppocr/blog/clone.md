---
comments: true
---

# 项目克隆

## 1. 克隆PaddleOCR repo代码

```bash linenums="1"
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR
```

如果因为网络问题无法pull成功，也可选择使用码云上的托管：

```bash linenums="1"
git clone --depth 1 https://gitee.com/paddlepaddle/PaddleOCR
```

注：码云托管代码可能无法实时同步本github项目更新，存在3~5天延时，请优先使用推荐方式。

浅克隆只下载当前分支的最新版本。若需指定分支，在克隆时增加 `--branch <分支名>`；若需完整提交历史，进入仓库目录后执行 `git fetch --unshallow`。

## 2. 安装第三方库

```bash linenums="1"
cd PaddleOCR
pip3 install -r requirements.txt
```
