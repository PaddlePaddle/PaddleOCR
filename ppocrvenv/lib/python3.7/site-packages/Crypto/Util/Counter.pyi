from typing import Optional, Union, Dict

def new(nbits: int, prefix: Optional[bytes]=..., suffix: Optional[bytes]=..., initial_value: Optional[int]=1,
        little_endian: Optional[bool]=False, allow_wraparound: Optional[bool]=False) -> \
        Dict[str, Union[int, bytes, bool]]: ...
