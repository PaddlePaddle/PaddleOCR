import sys
from typing import Any, Union

from numpy import dtype, ndarray, uint32
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy.typing import _ArrayLikeInt_co

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

class _MT19937Internal(TypedDict):
    key: ndarray[Any, dtype[uint32]]
    pos: int

class _MT19937State(TypedDict):
    bit_generator: str
    state: _MT19937Internal

class MT19937(BitGenerator):
    def __init__(self, seed: Union[None, _ArrayLikeInt_co, SeedSequence] = ...) -> None: ...
    def _legacy_seeding(self, seed: _ArrayLikeInt_co) -> None: ...
    def jumped(self, jumps: int = ...) -> MT19937: ...
    @property
    def state(self) -> _MT19937State: ...
    @state.setter
    def state(self, value: _MT19937State) -> None: ...
