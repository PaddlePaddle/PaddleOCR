import pytest

pytestmark = [
    # 2021-02-01 needed until numba updates their usage
    pytest.mark.filterwarnings(
        r"ignore:`np\.int` is a deprecated alias:DeprecationWarning"
    ),
]
