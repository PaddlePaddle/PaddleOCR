from typing import Any, List

from numpy.f2py import (
    f2py_testing as f2py_testing,
)

__all__: List[str]

def run_main(comline_list): ...
def compile(
    source,
    modulename=...,
    extra_args=...,
    verbose=...,
    source_fn=...,
    extension=...,
    full_output=...,
): ...
