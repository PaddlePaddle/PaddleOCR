import warnings

from pandas.core.indexes.api import (  # noqa:F401
    CategoricalIndex,
    DatetimeIndex,
    Float64Index,
    Index,
    Int64Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    NumericIndex,
    PeriodIndex,
    RangeIndex,
    TimedeltaIndex,
    UInt64Index,
    _new_Index,
    ensure_index,
    ensure_index_from_sequences,
    get_objs_combined_axis,
)
from pandas.core.indexes.multi import sparsify_labels  # noqa:F401

# GH#30193
warnings.warn(
    "pandas.core.index is deprecated and will be removed in a future version.  "
    "The public classes are available in the top-level namespace.",
    FutureWarning,
    stacklevel=2,
)
