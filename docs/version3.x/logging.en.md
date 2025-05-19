# Logging

This document mainly introduces how to configure the logging system for the PaddleOCR inference package. It's important to note that PaddleOCR's inference package uses a different logging system than the training scripts, and this document does not cover the configuration of the logging system used in the training scripts.

PaddleOCR has built a centralized logging system based on Python's [`logging` standard library](https://docs.python.org/3/library/logging.html#module-logging). In other words, PaddleOCR uses a single logger, which can be accessed and configured via `paddleocr.logger`.

By default, the logging level in PaddleOCR is set to `ERROR`, meaning that log messages will only be output if their level is `ERROR` or higher (e.g., `CRITICAL`). PaddleOCR also configures a `StreamHandler` for this logger, which outputs logs to the standard error stream, and sets the logger's `propagate` attribute to `False` to prevent log messages from being passed to its parent logger.

If you wish to disable PaddleOCR's automatic logging configuration behavior, you can set the environment variable `DISABLE_AUTO_LOGGING_CONFIG` to `1`. In this case, PaddleOCR will not perform any additional configuration of the logger.

For more flexible customization of logging behavior, refer to the relevant documentation of the `logging` standard library. Below is an example of writing logs to a file:

```python
import logging
from paddleocr import logger

# Write logs to the file `paddleocr.log`
fh = logging.FileHandler("paddleocr.log")
logger.addHandler(fh)
```

Please note that other libraries that PaddleOCR depends on (such as [PaddleX](./paddleocr_and_paddlex.en.md)) have their own independent logging systems, and the above configuration will not affect the log output of these libraries.
