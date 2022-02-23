from typing import Callable, Tuple, Union, Sequence, Any, Optional

__all__ = ['StrongRandom', 'getrandbits', 'randrange', 'randint', 'choice', 'shuffle', 'sample']

class StrongRandom(object):
    def __init__(self, rng: Optional[Any]=None, randfunc: Optional[Callable]=None) -> None: ... #  TODO What is rng?
    def getrandbits(self, k: int) -> int: ...
    def randrange(self, start: int, stop: int = ..., step: int = ...) -> int: ...
    def randint(self, a: int, b: int) -> int: ...
    def choice(self, seq: Sequence) -> object: ...
    def shuffle(self, x: Sequence) -> None: ...
    def sample(self, population: Sequence, k: int) -> list: ...

_r = StrongRandom()
getrandbits = _r.getrandbits
randrange = _r.randrange
randint = _r.randint
choice = _r.choice
shuffle = _r.shuffle
sample = _r.sample
