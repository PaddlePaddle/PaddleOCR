from .wandb_logger import WandbLogger


class Loggers(object):
    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def log_metrics(self, metrics, prefix=None, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, prefix=prefix, step=step)

    def log_model(self, is_best, prefix, metadata=None):
        for logger in self.loggers:
            logger.log_model(is_best=is_best, prefix=prefix, metadata=metadata)

    def close(self):
        for logger in self.loggers:
            logger.close()
