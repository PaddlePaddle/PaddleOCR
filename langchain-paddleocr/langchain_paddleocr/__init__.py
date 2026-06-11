from .document_loaders import PaddleOCRLoader

__all__ = ["PaddleOCRLoader", "PaddleOCRVLLoader"]


def __getattr__(name: str) -> object:
    if name == "PaddleOCRVLLoader":
        from .document_loaders import PaddleOCRVLLoader

        return PaddleOCRVLLoader
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
