from typing import Mapping, List, Any

from numpy import (
    DataSource as DataSource,
)

from numpy.core.multiarray import (
    packbits as packbits,
    unpackbits as unpackbits,
)

__all__: List[str]

def loads(*args, **kwargs): ...

class BagObj:
    def __init__(self, obj): ...
    def __getattribute__(self, key): ...
    def __dir__(self): ...

def zipfile_factory(file, *args, **kwargs): ...

class NpzFile(Mapping[Any, Any]):
    zip: Any
    fid: Any
    files: Any
    allow_pickle: Any
    pickle_kwargs: Any
    f: Any
    def __init__(self, fid, own_fid=..., allow_pickle=..., pickle_kwargs=...): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback): ...
    def close(self): ...
    def __del__(self): ...
    def __iter__(self): ...
    def __len__(self): ...
    def __getitem__(self, key): ...
    def iteritems(self): ...
    def iterkeys(self): ...

def load(file, mmap_mode=..., allow_pickle=..., fix_imports=..., encoding=...): ...
def save(file, arr, allow_pickle=..., fix_imports=...): ...
def savez(file, *args, **kwds): ...
def savez_compressed(file, *args, **kwds): ...
def loadtxt(
    fname,
    dtype=...,
    comments=...,
    delimiter=...,
    converters=...,
    skiprows=...,
    usecols=...,
    unpack=...,
    ndmin=...,
    encoding=...,
    max_rows=...,
    *,
    like=...,
): ...
def savetxt(
    fname,
    X,
    fmt=...,
    delimiter=...,
    newline=...,
    header=...,
    footer=...,
    comments=...,
    encoding=...,
): ...
def fromregex(file, regexp, dtype, encoding=...): ...
def genfromtxt(
    fname,
    dtype=...,
    comments=...,
    delimiter=...,
    skip_header=...,
    skip_footer=...,
    converters=...,
    missing_values=...,
    filling_values=...,
    usecols=...,
    names=...,
    excludelist=...,
    deletechars=...,
    replace_space=...,
    autostrip=...,
    case_sensitive=...,
    defaultfmt=...,
    unpack=...,
    usemask=...,
    loose=...,
    invalid_raise=...,
    max_rows=...,
    encoding=...,
    *,
    like=...,
): ...
def recfromtxt(fname, **kwargs): ...
def recfromcsv(fname, **kwargs): ...

# NOTE: Deprecated
# def ndfromtxt(fname, **kwargs): ...
# def mafromtxt(fname, **kwargs): ...
