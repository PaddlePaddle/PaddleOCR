"""The :mod:`sklearn.inspection` module includes tools for model inspection."""


from ._permutation_importance import permutation_importance

from ._partial_dependence import partial_dependence
from ._plot.partial_dependence import plot_partial_dependence
from ._plot.partial_dependence import PartialDependenceDisplay


__all__ = [
    "partial_dependence",
    "plot_partial_dependence",
    "permutation_importance",
    "PartialDependenceDisplay",
]
