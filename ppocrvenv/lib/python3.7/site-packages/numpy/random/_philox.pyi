import sys
from typing import Any, Union

from numpy import dtype, ndarray, uint64
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy.typing import _ArrayLikeInt_co

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

class _PhiloxInternal(TypedDict):
    counter: ndarray[Any, dtype[uint64]]
    key: ndarray[Any, dtype[uint64]]

class _PhiloxState(TypedDict):
    bit_generator: str
    state: _PhiloxInternal
    buffer: ndarray[Any, dtype[uint64]]
    buffer_pos: int
    has_uint32: int
    uinteger: int

class Philox(BitGenerator):
    def __init__(
        self,
        seed: Union[None, _ArrayLikeInt_co, SeedSequence] = ...,
        counter: Union[None, _ArrayLikeInt_co] = ...,
        key: Union[None, _ArrayLikeInt_co] = ...,
    ) -> None: ...
    @property
    def state(
        self,
    ) -> _PhiloxState: ...
    @state.setter
    def state(
        self,
        value: _PhiloxState,
    ) -> None: ...
    def jumped(self, jumps: int = ...) -> Philox: ...
    def advance(self, delta: int) -> Philox: ...
