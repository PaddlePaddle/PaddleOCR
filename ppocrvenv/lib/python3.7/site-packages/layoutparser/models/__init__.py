from . import catalog as _UNUSED
# A trick learned from
# https://github.com/facebookresearch/detectron2/blob/62cf3a2b6840734d2717abdf96e2dd57ed6612a6/detectron2/checkpoint/__init__.py#L6
from .layoutmodel import Detectron2LayoutModel, PaddleDetectionLayoutModel
