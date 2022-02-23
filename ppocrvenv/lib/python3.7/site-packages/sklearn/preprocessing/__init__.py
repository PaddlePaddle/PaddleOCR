"""
The :mod:`sklearn.preprocessing` module includes scaling, centering,
normalization, binarization methods.
"""

from ._function_transformer import FunctionTransformer

from ._data import Binarizer
from ._data import KernelCenterer
from ._data import MinMaxScaler
from ._data import MaxAbsScaler
from ._data import Normalizer
from ._data import RobustScaler
from ._data import StandardScaler
from ._data import QuantileTransformer
from ._data import add_dummy_feature
from ._data import binarize
from ._data import normalize
from ._data import scale
from ._data import robust_scale
from ._data import maxabs_scale
from ._data import minmax_scale
from ._data import quantile_transform
from ._data import power_transform
from ._data import PowerTransformer

from ._encoders import OneHotEncoder
from ._encoders import OrdinalEncoder

from ._label import label_binarize
from ._label import LabelBinarizer
from ._label import LabelEncoder
from ._label import MultiLabelBinarizer

from ._discretization import KBinsDiscretizer

from ._polynomial import PolynomialFeatures
from ._polynomial import SplineTransformer


__all__ = [
    "Binarizer",
    "FunctionTransformer",
    "KBinsDiscretizer",
    "KernelCenterer",
    "LabelBinarizer",
    "LabelEncoder",
    "MultiLabelBinarizer",
    "MinMaxScaler",
    "MaxAbsScaler",
    "QuantileTransformer",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PowerTransformer",
    "RobustScaler",
    "SplineTransformer",
    "StandardScaler",
    "add_dummy_feature",
    "PolynomialFeatures",
    "binarize",
    "normalize",
    "scale",
    "robust_scale",
    "maxabs_scale",
    "minmax_scale",
    "label_binarize",
    "quantile_transform",
    "power_transform",
]
