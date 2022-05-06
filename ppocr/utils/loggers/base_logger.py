import os
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @abstractmethod
    def log_metrics(self, metrics, prefix=None):
        pass

    @abstractmethod
    def close(self):
        pass