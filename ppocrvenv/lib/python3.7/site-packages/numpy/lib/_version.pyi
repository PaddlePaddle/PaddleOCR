from typing import Union, List

__all__: List[str]

class NumpyVersion:
    vstring: str
    version: str
    major: int
    minor: int
    bugfix: int
    pre_release: str
    is_devversion: bool
    def __init__(self, vstring: str) -> None: ...
    def __lt__(self, other: Union[str, NumpyVersion]) -> bool: ...
    def __le__(self, other: Union[str, NumpyVersion]) -> bool: ...
    def __eq__(self, other: Union[str, NumpyVersion]) -> bool: ...  # type: ignore[override]
    def __ne__(self, other: Union[str, NumpyVersion]) -> bool: ...  # type: ignore[override]
    def __gt__(self, other: Union[str, NumpyVersion]) -> bool: ...
    def __ge__(self, other: Union[str, NumpyVersion]) -> bool: ...
