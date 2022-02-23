import numpy as np

def array_strptime(
    values: np.ndarray,  # np.ndarray[object]
    fmt: str | None,
    exact: bool = True,
    errors: str = "raise",
) -> tuple[np.ndarray, np.ndarray]: ...

# first  ndarray is M8[ns], second is object ndarray of tzinfo | None
