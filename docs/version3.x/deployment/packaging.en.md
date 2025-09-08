---
comments: true
---

# Package PaddleOCR Projects

This guide applies to packaging PaddleOCR projects using PyInstaller.

> Since Nuikta's packaging principle is incompatible with PaddleOCR, packaging with Nuikta is currently not supported.

## Preparing the Environment

- **Complete the PaddleOCR installation according to the [PaddleOCR Installation Documentation](../installation.en.md).**
- **Install PyInstaller.**

Install PyInstaller:

```bash
pip install pyinstaller
```

> Ensure that all dependencies required by the Python script to be packaged are installed in the current environment to prevent anomalies in the packaged executable due to missing dependencies.

## Running the Packaging Script

Copy the Python script below and save it as a `py` file, for example, `package.py`.

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


**Supported Script Parameters:**

| Parameter         | Required | Description                                                                                                               |
|--------------|------------------------------------------------------------------------------------------------------------------------------|---------|
| --file   | Yes     | The name of the file to be packaged (e.g., `main.py`).
| --nvidia     | No     | Packages NVIDIA CUDA and cuDNN related dependencies into the same directory as the executable. If the system environment paths already include CUDA and cuDNN dependencies or if CUDA and cuDNN dependencies are not required, this option can be omitted.

**Example Usage of the Packaging Script:**

```bash
python package.py --file main.py
# Packages NVIDIA CUDA and cuDNN related dependencies into the same directory as the executable.
python package.py --file main.py --nvidia
```

**Execution Result**

- The packaging script will execute a command similar to:

    `pyinstaller main.py --collect-data paddlex --collect-binaries paddle [--copy-metadata xxx …]`, where `--copy-metadata xxx` dynamically adds package metadata based on the dependencies required by PaddleOCR installed in the current environment.

- The executable file and related dependency libraries will be generated in the `dist` folder.

## Appendix

**The above packaging process was tested in the following environment:**

- Operating System: **Win 11**
- Python: **3.10.18**
- PaddlePaddle: **3.0.0**
- PaddleX: **3.1.3**
- PaddleOCR：**3.1.0**
- PyInstaller: **6.14.2**

**Common Issues**

- When running the executable, if you encounter an error message like `RuntimeError: xxx requires additional dependencies`, it indicates that the current packaging environment lacks the necessary dependencies. Please ensure that the environment is set up correctly as described in the Preparations section.
- When running the executable, if an error message indicates that CUDA or cuDNN related dynamic link libraries cannot be found, please check whether the system environment variables correctly include the paths to NVIDIA CUDA and cuDNN dependencies, or consider adding `--nvidia` when running the packaging script to include CUDA and cuDNN dependencies in the same directory as the executable.
