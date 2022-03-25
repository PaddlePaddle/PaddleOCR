from typing import Type

from . import _CompareWithType

def cmp_using(
    eq: Optional[_CompareWithType],
    lt: Optional[_CompareWithType],
    le: Optional[_CompareWithType],
    gt: Optional[_CompareWithType],
    ge: Optional[_CompareWithType],
    require_same_type: bool,
    class_name: str,
) -> Type: ...
