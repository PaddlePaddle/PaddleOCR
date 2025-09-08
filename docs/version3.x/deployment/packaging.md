---
comments: true
---

# 打包PaddleOCR项目

本说明适用于通过PyInstaller打包PaddleOCR项目。

> 由于Nuikta的打包原理与PaddleOCR不适配，当前暂不支持通过Nuikta进行打包。

## 准备环境

- **根据[PaddleOCR安装文档](../installation.md)完成PaddleOCR安装**
- **安装PyInstaller**

安装PyInstaller：

```bash
pip install pyinstaller
```

> 请确认当前准备环境中安装有待打包的Python脚本所需的全部依赖，以避免缺少依赖导致打包后的可执行程序出现异常。

## 执行打包脚本

将下方Python脚本拷贝后存成`py`文件，文件名可以为`package.py`。

```python
import paddlex
import importlib.metadata
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, help='Your file name, e.g. main.py.')
parser.add_argument('--nvidia', action='store_true', help='Include NVIDIA CUDA and cuDNN dependencies.')

args = parser.parse_args()

main_file = args.file

user_deps = [dist.metadata["Name"] for dist in importlib.metadata.distributions()]
deps_all = list(paddlex.utils.deps.DEP_SPECS.keys())
deps_need = [dep for dep in user_deps if dep in deps_all]

cmd = [
    "pyinstaller", main_file,
    "--collect-data", "paddlex",
    "--collect-binaries", "paddle"
]

if args.nvidia:
    cmd += ["--collect-binaries", "nvidia"]

for dep in deps_need:
    cmd += ["--copy-metadata", dep]

print("PyInstaller command:", " ".join(cmd))

try:
    result = subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print("Installation failed:", e)
    sys.exit(1)
```


**打包脚本支持的参数如下：**

| 参数         | 是否必需 | 说明                                                                                                               |
|--------------|------------------------------------------------------------------------------------------------------------------------------|---------|
| --file   | 是     | 你的待打包文件名（如`main.py`）。
| --nvidia     | 否     | 将NVIDIA的CUDA、cuDNN相关依赖库一同打包到可执行文件的同级目录中。如果系统环境变量路径已包含CUDA、cuDNN相关依赖库或者不需要使用CUDA、cuDNN相关依赖库，则无需开启。

**打包脚本调用示例如下：**

```bash
python package.py --file main.py
# 将NVIDIA的CUDA、cuDNN相关依赖库打包至可执行文件的同级目录中。
python package.py --file main.py --nvidia
```

**运行结果**

- 安转脚本将执行类似如下命令：

    `pyinstaller main.py --collect-data paddlex --collect-binaries paddle [--copy-metadata xxx …]`，其中`--copy-metadata xxx`会根据当前环境已安装的PaddleOCR需要的依赖动态添加包的元信息。

- 可执行文件和相关依赖库将生成到`dist`文件夹中。

## 附录

**以上打包流程在如下环境中测试：**

- 操作系统：**Win 11**
- Python：**3.10.18**
- PaddlePaddle：**3.0.0**
- PaddleX：**3.1.3**
- PaddleOCR：**3.1.0**
- PyInstaller：**6.14.2**

**常见问题**

- 在运行可执行文件时，出现报错信息 `RuntimeError: xxx requires additional dependencies`，说明当前打包环境缺少相关依赖，请确认已按照准备环境部分说明正确安装环境。
- 在运行可执行文件时，出现报错信息提示CUDA、cuDNN相关动态链接库找不到，请检查系统环境变量中是否正确添加NVIDIA的CUDA、cuDNN相关依赖库路径或者考虑在运行打包脚本时添加 `--nvidia`，将CUDA、cuDNN相关依赖库打包进可执行文件的同级目录中。
