from typing import (
    Any,
    Hashable,
    Literal,
)

import numpy as np

def unique_label_indices(
    labels: np.ndarray,  # const int64_t[:]
) -> np.ndarray: ...

class Factorizer:
    count: int
    def __init__(self, size_hint: int): ...
    def get_count(self) -> int: ...

class ObjectFactorizer(Factorizer):
    table: PyObjectHashTable
    uniques: ObjectVector
    def factorize(
        self,
        values: np.ndarray,  # ndarray[object]
        sort: bool = ...,
        na_sentinel=...,
        na_value=...,
    ) -> np.ndarray: ...  # np.ndarray[intp]

class Int64Factorizer(Factorizer):
    table: Int64HashTable
    uniques: Int64Vector
    def factorize(
        self,
        values: np.ndarray,  # const int64_t[:]
        sort: bool = ...,
        na_sentinel=...,
        na_value=...,
    ) -> np.ndarray: ...  # np.ndarray[intp]

class Int64Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.int64]

class Int32Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.int32]

class Int16Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.int16]

class Int8Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.int8]

class UInt64Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.uint64]

class UInt32Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.uint32]

class UInt16Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.uint16]

class UInt8Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.uint8]

class Float64Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.float64]

class Float32Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.float32]

class Complex128Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.complex128]

class Complex64Vector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[np.complex64]

class StringVector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[object]

class ObjectVector:
    def __init__(self): ...
    def __len__(self) -> int: ...
    def to_array(self) -> np.ndarray: ...  # np.ndarray[object]

class HashTable:
    # NB: The base HashTable class does _not_ actually have these methods;
    #  we are putting the here for the sake of mypy to avoid
    #  reproducing them in each subclass below.
    def __init__(self, size_hint: int = ...): ...
    def __len__(self) -> int: ...
    def __contains__(self, key: Hashable) -> bool: ...
    def sizeof(self, deep: bool = ...) -> int: ...
    def get_state(self) -> dict[str, int]: ...
    # TODO: `item` type is subclass-specific
    def get_item(self, item): ...  # TODO: return type?
    def set_item(self, item) -> None: ...
    # FIXME: we don't actually have this for StringHashTable or ObjectHashTable?
    def map(
        self,
        keys: np.ndarray,  # np.ndarray[subclass-specific]
        values: np.ndarray,  # const int64_t[:]
    ) -> None: ...
    def map_locations(
        self,
        values: np.ndarray,  # np.ndarray[subclass-specific]
    ) -> None: ...
    def lookup(
        self,
        values: np.ndarray,  # np.ndarray[subclass-specific]
    ) -> np.ndarray: ...  # np.ndarray[np.intp]
    def get_labels(
        self,
        values: np.ndarray,  # np.ndarray[subclass-specific]
        uniques,  # SubclassTypeVector
        count_prior: int = ...,
        na_sentinel: int = ...,
        na_value: object = ...,
    ) -> np.ndarray: ...  # np.ndarray[intp_t]
    def unique(
        self,
        values: np.ndarray,  # np.ndarray[subclass-specific]
        return_inverse: bool = ...,
    ) -> tuple[
        np.ndarray,  # np.ndarray[subclass-specific]
        np.ndarray,  # np.ndarray[np.intp],
    ] | np.ndarray: ...  # np.ndarray[subclass-specific]
    def _unique(
        self,
        values: np.ndarray,  # np.ndarray[subclass-specific]
        uniques,  # FooVector
        count_prior: int = ...,
        na_sentinel: int = ...,
        na_value: object = ...,
        ignore_na: bool = ...,
        return_inverse: bool = ...,
    ) -> tuple[
        np.ndarray,  # np.ndarray[subclass-specific]
        np.ndarray,  # np.ndarray[np.intp],
    ] | np.ndarray: ...  # np.ndarray[subclass-specific]
    def factorize(
        self,
        values: np.ndarray,  # np.ndarray[subclass-specific]
        na_sentinel: int = ...,
        na_value: object = ...,
        mask=...,
    ) -> tuple[
        np.ndarray,  # np.ndarray[subclass-specific]
        np.ndarray,  # np.ndarray[np.intp],
    ]: ...

class Complex128HashTable(HashTable): ...
class Complex64HashTable(HashTable): ...
class Float64HashTable(HashTable): ...
class Float32HashTable(HashTable): ...

class Int64HashTable(HashTable):
    # Only Int64HashTable has get_labels_groupby
    def get_labels_groupby(
        self,
        values: np.ndarray,  # const int64_t[:]
    ) -> tuple[
        np.ndarray,  # np.ndarray[np.intp]
        np.ndarray,  # np.ndarray[np.int64]
    ]: ...

class Int32HashTable(HashTable): ...
class Int16HashTable(HashTable): ...
class Int8HashTable(HashTable): ...
class UInt64HashTable(HashTable): ...
class UInt32HashTable(HashTable): ...
class UInt16HashTable(HashTable): ...
class UInt8HashTable(HashTable): ...
class StringHashTable(HashTable): ...
class PyObjectHashTable(HashTable): ...

def duplicated_int64(
    values: np.ndarray,  # const int64_t[:] values
    keep: Literal["last", "first", False] = ...,
) -> np.ndarray: ...  # np.ndarray[bool]

# TODO: Is it actually bool or is it uint8?

def mode_int64(
    values: np.ndarray,  # const int64_t[:] values
    dropna: bool,
) -> np.ndarray: ...  # np.ndarray[np.int64]
def value_count_int64(
    values: np.ndarray,  # const int64_t[:]
    dropna: bool,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.int64]  # np.ndarray[np.int64]
def duplicated(
    values: np.ndarray,
    keep: Literal["last", "first", False] = ...,
) -> np.ndarray: ...  # np.ndarray[bool]
def mode(values: np.ndarray, dropna: bool) -> np.ndarray: ...
def value_count(
    values: np.ndarray,
    dropna: bool,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.int64]

# arr and values should have same dtype
def ismember(
    arr: np.ndarray,
    values: np.ndarray,
) -> np.ndarray: ...  # np.ndarray[bool]
def object_hash(obj) -> int: ...
def objects_are_equal(a, b) -> bool: ...
