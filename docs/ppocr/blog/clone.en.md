---
comments: true
---

# Project Clone

## 1. Clone PaddleOCR

```bash linenums="1"
# Recommend
git clone https://github.com/PaddlePaddle/PaddleOCR

# If you cannot pull successfully due to network problems, you can switch to the mirror hosted on Gitee:

git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The mirror on Gitee may not keep in synchronization with the latest project on GitHub. There might be a delay of 3-5 days. Please try GitHub at first.
```

## 2. Install third-party libraries

```bash linenums="1"
cd PaddleOCR
pip3 install -r requirements.txt
```

If you getting this error `OSError: [WinError 126] The specified module could not be found` when you install shapely on windows.

Please try to download Shapely whl file from [http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely).

Reference: [Solve shapely installation on windows](https://stackoverflow.com/questions/44398265/install-shapely-oserror-winerror-126-the-specified-module-could-not-be-found)
