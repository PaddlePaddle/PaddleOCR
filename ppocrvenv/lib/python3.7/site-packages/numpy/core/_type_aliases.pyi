import sys
from typing import Dict, Union, Type, List

from numpy import generic, signedinteger, unsignedinteger, floating, complexfloating

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

class _SCTypes(TypedDict):
    int: List[Type[signedinteger]]
    uint: List[Type[unsignedinteger]]
    float: List[Type[floating]]
    complex: List[Type[complexfloating]]
    others: List[type]

sctypeDict: Dict[Union[int, str], Type[generic]]
sctypes: _SCTypes
