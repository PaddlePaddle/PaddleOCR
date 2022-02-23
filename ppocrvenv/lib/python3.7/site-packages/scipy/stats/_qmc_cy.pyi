import numpy as np
from scipy._lib._util import DecimalNumber


def _cy_wrapper_centered_discrepancy(
        sample: np.ndarray, 
        iterative: bool, 
        workers: int,
) -> float: ...


def _cy_wrapper_wrap_around_discrepancy(
        sample: np.ndarray,
        iterative: bool, 
        workers: int,
) -> float: ...


def _cy_wrapper_mixture_discrepancy(
        sample: np.ndarray,
        iterative: bool, 
        workers: int,
) -> float: ...


def _cy_wrapper_l2_star_discrepancy(
        sample: np.ndarray,
        iterative: bool,
        workers: int,
) -> float: ...


def _cy_wrapper_update_discrepancy(
        x_new_view: np.ndarray,
        sample_view: np.ndarray,
        initial_disc: DecimalNumber,
) -> float: ...
