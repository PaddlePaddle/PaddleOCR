# 日志

本文档主要介绍如何配置 PaddleOCR 推理包的日志系统。需要注意的是，PaddleOCR 推理包与训练脚本使用的是不同的日志系统，本文档不涉及训练脚本所使用的日志系统的配置方法。

PaddleOCR 构建了一个基于 Python [`logging` 标准库](https://docs.python.org/zh-cn/3/library/logging.html#module-logging) 的集中式日志系统。换言之，PaddleOCR 使用唯一的日志记录器（logger），可通过 `paddleocr.logger` 访问和配置。

默认情况下，PaddleOCR 的日志级别设为 `ERROR`，这意味着仅当日志级别为 `ERROR` 或更高（如 `CRITICAL`）时，日志信息才会输出。PaddleOCR 同时为该日志记录器配置了一个 `StreamHandler`，将日志输出到标准错误流，并将记录器的 `propagate` 属性设为 `False`，以避免日志信息传递到其父记录器。

若希望禁止 PaddleOCR 对日志系统的自动配置行为，可将环境变量 `DISABLE_AUTO_LOGGING_CONFIG` 设为 `1`。此时，PaddleOCR 将不会对日志记录器进行任何额外配置。

如需更灵活地定制日志行为，可参考 `logging` 标准库的相关文档。以下是一个将日志写入文件的示例：

```python
import logging
from paddleocr import logger

# 将日志写入文件 `paddleocr.log`
fh = logging.FileHandler("paddleocr.log")
logger.addHandler(fh)
```

请注意，PaddleOCR 依赖的其他库（如 [PaddleX](./paddleocr_and_paddlex.md)）拥有各自独立的日志系统，以上配置不会影响这些库的日志输出。
