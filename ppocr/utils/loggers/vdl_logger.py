from .base_logger import BaseLogger


class VDLLogger(BaseLogger):
    def __init__(self, save_dir):
        super().__init__(save_dir)
        # check the installation of visualdl, if not installed, raise an error
        try:
            from visualdl import LogWriter
        except ImportError:
            raise ImportError(
                "Please install visualdl via `pip install visualdl` or "
                "`pip install paddleocr[visualdl]`."
            )
        self.vdl_writer = LogWriter(logdir=save_dir)

    def log_metrics(self, metrics, prefix=None, step=None):
        if not prefix:
            prefix = ""
        updated_metrics = {prefix + "/" + k: v for k, v in metrics.items()}

        for k, v in updated_metrics.items():
            self.vdl_writer.add_scalar(k, v, step)
    
    def log_model(self, is_best, prefix, metadata=None):
        pass
    
    def close(self):
        self.vdl_writer.close() 