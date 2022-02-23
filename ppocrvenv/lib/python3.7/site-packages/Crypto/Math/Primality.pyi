from typing import Callable, Optional, Union, Set

PrimeResult = int

COMPOSITE: PrimeResult
PROBABLY_PRIME: PrimeResult

def miller_rabin_test(candidate: int, iterations: int, randfunc: Optional[Callable[[int],bytes]]=None) -> PrimeResult: ...
def lucas_test(candidate: int) -> PrimeResult: ...
_sieve_base: Set[int]
def test_probable_prime(candidate: int, randfunc: Optional[Callable[[int],bytes]]=None) -> PrimeResult: ...
def generate_probable_prime(*,
                            exact_bits: int = ...,
                            randfunc: Callable[[int],bytes] = ...,
                            prime_filter: Callable[[int],bool] = ...) -> int: ...
def generate_probable_safe_prime(*,
                            exact_bits: int = ...,
                            randfunc: Callable[[int],bytes] = ...) -> int: ...
