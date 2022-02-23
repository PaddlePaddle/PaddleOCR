# Authors: Andreas Mueller <amueller@ais.uni-bonn.de>
#          Joris Van den Bossche <jorisvandenbossche@gmail.com>
# License: BSD 3 clause

import warnings
import numpy as np
from scipy import sparse
import numbers

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array, is_scalar_nan
from ..utils.deprecation import deprecated
from ..utils.validation import check_is_fitted
from ..utils.validation import _check_feature_names_in
from ..utils._mask import _get_mask

from ..utils._encode import _encode, _check_unknown, _unique


__all__ = ["OneHotEncoder", "OrdinalEncoder"]


class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """

    def _check_X(self, X, force_all_finite=True):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, e.g. for the `categories_` attribute.

        """
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            # if not a dataframe, do normal check_array validation
            X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
            if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, force_all_finite=force_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            # pandas dataframe, do validation later column by column, in order
            # to keep the dtype information to be used in the encoder.
            needs_validation = force_all_finite

        n_samples, n_features = X.shape
        X_columns = []

        for i in range(n_features):
            Xi = self._get_feature(X, feature_idx=i)
            Xi = check_array(
                Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation
            )
            X_columns.append(Xi)

        return X_columns, n_samples, n_features

    def _get_feature(self, X, feature_idx):
        if hasattr(X, "iloc"):
            # pandas dataframes
            return X.iloc[:, feature_idx]
        # numpy arrays, sparse arrays
        return X[:, feature_idx]

    def _fit(self, X, handle_unknown="error", force_all_finite=True):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )
        self.n_features_in_ = n_features

        if self.categories != "auto":
            if len(self.categories) != n_features:
                raise ValueError(
                    "Shape mismatch: if categories is an array,"
                    " it has to be of shape (n_features,)."
                )

        self.categories_ = []

        for i in range(n_features):
            Xi = X_list[i]
            if self.categories == "auto":
                cats = _unique(Xi)
            else:
                cats = np.array(self.categories[i], dtype=Xi.dtype)
                if Xi.dtype.kind not in "OUS":
                    sorted_cats = np.sort(cats)
                    error_msg = (
                        "Unsorted categories are not supported for numerical categories"
                    )
                    # if there are nans, nan should be the last element
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]) or (
                        np.isnan(sorted_cats[-1]) and not np.isnan(sorted_cats[-1])
                    ):
                        raise ValueError(error_msg)

                if handle_unknown == "error":
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)
            self.categories_.append(cats)

    def _transform(
        self, X, handle_unknown="error", force_all_finite=True, warn_on_unknown=False
    ):
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )

        X_int = np.zeros((n_samples, n_features), dtype=int)
        X_mask = np.ones((n_samples, n_features), dtype=bool)

        columns_with_unknown = []
        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)

            if not np.all(valid_mask):
                if handle_unknown == "error":
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    if warn_on_unknown:
                        columns_with_unknown.append(i)
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    # cast Xi into the largest string type necessary
                    # to handle different lengths of numpy strings
                    if (
                        self.categories_[i].dtype.kind in ("U", "S")
                        and self.categories_[i].itemsize > Xi.itemsize
                    ):
                        Xi = Xi.astype(self.categories_[i].dtype)
                    elif self.categories_[i].dtype.kind == "O" and Xi.dtype.kind == "U":
                        # categories are objects and Xi are numpy strings.
                        # Cast Xi to an object dtype to prevent truncation
                        # when setting invalid values.
                        Xi = Xi.astype("O")
                    else:
                        Xi = Xi.copy()

                    Xi[~valid_mask] = self.categories_[i][0]
            # We use check_unknown=False, since _check_unknown was
            # already called above.
            X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)
        if columns_with_unknown:
            warnings.warn(
                "Found unknown categories in columns "
                f"{columns_with_unknown} during transform. These "
                "unknown categories will be encoded as all zeros",
                UserWarning,
            )

        return X_int, X_mask

    def _more_tags(self):
        return {"X_types": ["categorical"]}


class OneHotEncoder(_BaseEncoder):
    """
    Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse``
    parameter)

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

        .. versionadded:: 0.20

    drop : {'first', 'if_binary'} or an array-like of shape (n_features,), \
            default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into a neural network or an unregularized regression.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - 'if_binary' : drop the first category in each feature with two
          categories. Features with 1 or more than 2 categories are
          left intact.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped.

        .. versionadded:: 0.21
           The parameter `drop` was added in 0.21.

        .. versionchanged:: 0.23
           The option `drop='if_binary'` was added in 0.23.

    sparse : bool, default=True
        Will return sparse matrix if set True else will return an array.

    dtype : number type, default=float
        Desired dtype of output.

    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).

    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
          to be dropped for each feature.
        - ``drop_idx_[i] = None`` if no category is to be dropped from the
          feature with index ``i``, e.g. when `drop='if_binary'` and the
          feature isn't binary.
        - ``drop_idx_ = None`` if all the transformed features will be
          retained.

        .. versionchanged:: 0.23
           Added the possibility to contain `None` values.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    OrdinalEncoder : Performs an ordinal (integer)
      encoding of the categorical features.
    sklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : Performs an approximate one-hot
      encoding of dictionary items or strings.
    LabelBinarizer : Binarizes labels in a one-vs-all
      fashion.
    MultiLabelBinarizer : Transforms between iterable of
      iterables and a multilabel format, e.g. a (samples x classes) binary
      matrix indicating the presence of a class label.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder

    One can discard categories not seen during `fit`:

    >>> enc = OneHotEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OneHotEncoder(handle_unknown='ignore')
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)
    >>> enc.get_feature_names_out(['gender', 'group'])
    array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)

    One can always drop the first column for each feature:

    >>> drop_enc = OneHotEncoder(drop='first').fit(X)
    >>> drop_enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
    array([[0., 0., 0.],
           [1., 1., 0.]])

    Or drop a column for feature only having 2 categories:

    >>> drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
    >>> drop_binary_enc.transform([['Female', 1], ['Male', 2]]).toarray()
    array([[0., 1., 0., 0.],
           [1., 0., 1., 0.]])
    """

    def __init__(
        self,
        *,
        categories="auto",
        drop=None,
        sparse=True,
        dtype=np.float64,
        handle_unknown="error",
    ):
        self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop

    def _validate_keywords(self):
        if self.handle_unknown not in ("error", "ignore"):
            msg = (
                "handle_unknown should be either 'error' or 'ignore', got {0}.".format(
                    self.handle_unknown
                )
            )
            raise ValueError(msg)

    def _compute_drop_idx(self):
        if self.drop is None:
            return None
        elif isinstance(self.drop, str):
            if self.drop == "first":
                return np.zeros(len(self.categories_), dtype=object)
            elif self.drop == "if_binary":
                return np.array(
                    [0 if len(cats) == 2 else None for cats in self.categories_],
                    dtype=object,
                )
            else:
                msg = (
                    "Wrong input for parameter `drop`. Expected "
                    "'first', 'if_binary', None or array of objects, got {}"
                )
                raise ValueError(msg.format(type(self.drop)))

        else:
            try:
                drop_array = np.asarray(self.drop, dtype=object)
                droplen = len(drop_array)
            except (ValueError, TypeError):
                msg = (
                    "Wrong input for parameter `drop`. Expected "
                    "'first', 'if_binary', None or array of objects, got {}"
                )
                raise ValueError(msg.format(type(drop_array)))
            if droplen != len(self.categories_):
                msg = (
                    "`drop` should have length equal to the number "
                    "of features ({}), got {}"
                )
                raise ValueError(msg.format(len(self.categories_), droplen))
            missing_drops = []
            drop_indices = []
            for col_idx, (val, cat_list) in enumerate(
                zip(drop_array, self.categories_)
            ):
                if not is_scalar_nan(val):
                    drop_idx = np.where(cat_list == val)[0]
                    if drop_idx.size:  # found drop idx
                        drop_indices.append(drop_idx[0])
                    else:
                        missing_drops.append((col_idx, val))
                    continue

                # val is nan, find nan in categories manually
                for cat_idx, cat in enumerate(cat_list):
                    if is_scalar_nan(cat):
                        drop_indices.append(cat_idx)
                        break
                else:  # loop did not break thus drop is missing
                    missing_drops.append((col_idx, val))

            if any(missing_drops):
                msg = (
                    "The following categories were supposed to be "
                    "dropped, but were not found in the training "
                    "data.\n{}".format(
                        "\n".join(
                            [
                                "Category: {}, Feature: {}".format(c, v)
                                for c, v in missing_drops
                            ]
                        )
                    )
                )
                raise ValueError(msg)
            return np.array(drop_indices, dtype=object)

    def fit(self, X, y=None):
        """
        Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
            Fitted encoder.
        """
        self._validate_keywords()
        self._fit(X, handle_unknown=self.handle_unknown, force_all_finite="allow-nan")
        self.drop_idx_ = self._compute_drop_idx()
        return self

    def fit_transform(self, X, y=None):
        """
        Fit OneHotEncoder to X, then transform X.

        Equivalent to fit(X).transform(X) but more convenient.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. If `sparse=True`, a sparse matrix will be
            returned.
        """
        self._validate_keywords()
        return super().fit_transform(X, y)

    def transform(self, X):
        """
        Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. If `sparse=True`, a sparse matrix will be
            returned.
        """
        check_is_fitted(self)
        # validation of X happens in _check_X called by _transform
        warn_on_unknown = self.handle_unknown == "ignore" and self.drop is not None
        X_int, X_mask = self._transform(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite="allow-nan",
            warn_on_unknown=warn_on_unknown,
        )

        n_samples, n_features = X_int.shape

        if self.drop_idx_ is not None:
            to_drop = self.drop_idx_.copy()
            # We remove all the dropped categories from mask, and decrement all
            # categories that occur after them to avoid an empty column.
            keep_cells = X_int != to_drop
            n_values = []
            for i, cats in enumerate(self.categories_):
                n_cats = len(cats)

                # drop='if_binary' but feature isn't binary
                if to_drop[i] is None:
                    # set to cardinality to not drop from X_int
                    to_drop[i] = n_cats
                    n_values.append(n_cats)
                else:  # dropped
                    n_values.append(n_cats - 1)

            to_drop = to_drop.reshape(1, -1)
            X_int[X_int > to_drop] -= 1
            X_mask &= keep_cells
        else:
            n_values = [len(cats) for cats in self.categories_]

        mask = X_mask.ravel()
        feature_indices = np.cumsum([0] + n_values)
        indices = (X_int + feature_indices[:-1]).ravel()[mask]

        indptr = np.empty(n_samples + 1, dtype=int)
        indptr[0] = 0
        np.sum(X_mask, axis=1, out=indptr[1:], dtype=indptr.dtype)
        np.cumsum(indptr[1:], out=indptr[1:])
        data = np.ones(indptr[-1])

        out = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(n_samples, feature_indices[-1]),
            dtype=self.dtype,
        )
        if not self.sparse:
            return out.toarray()
        else:
            return out

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        When unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category. If the
        feature with the unknown category has a dropped category, the dropped
        category will be its inverse.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")

        n_samples, _ = X.shape
        n_features = len(self.categories_)
        if self.drop_idx_ is None:
            n_transformed_features = sum(len(cats) for cats in self.categories_)
        else:
            n_transformed_features = sum(
                len(cats) - 1 if to_drop is not None else len(cats)
                for cats, to_drop in zip(self.categories_, self.drop_idx_)
            )

        # validate shape of passed X
        msg = (
            "Shape of the passed X data is not correct. Expected {0} columns, got {1}."
        )
        if X.shape[1] != n_transformed_features:
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        j = 0
        found_unknown = {}

        for i in range(n_features):
            if self.drop_idx_ is None or self.drop_idx_[i] is None:
                cats = self.categories_[i]
            else:
                cats = np.delete(self.categories_[i], self.drop_idx_[i])
            n_categories = len(cats)

            # Only happens if there was a column with a unique
            # category. In this case we just fill the column with this
            # unique category value.
            if n_categories == 0:
                X_tr[:, i] = self.categories_[i][self.drop_idx_[i]]
                j += n_categories
                continue
            sub = X[:, j : j + n_categories]
            # for sparse X argmax returns 2D matrix, ensure 1D array
            labels = np.asarray(sub.argmax(axis=1)).flatten()
            X_tr[:, i] = cats[labels]
            if self.handle_unknown == "ignore":
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                # ignored unknown categories: we have a row of all zero
                if unknown.any():
                    # if categories were dropped then unknown categories will
                    # be mapped to the dropped category
                    if self.drop_idx_ is None or self.drop_idx_[i] is None:
                        found_unknown[i] = unknown
                    else:
                        X_tr[unknown, i] = self.categories_[i][self.drop_idx_[i]]
            else:
                dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                if dropped.any():
                    if self.drop_idx_ is None:
                        all_zero_samples = np.flatnonzero(dropped)
                        raise ValueError(
                            f"Samples {all_zero_samples} can not be inverted "
                            "when drop=None and handle_unknown='error' "
                            "because they contain all zeros"
                        )
                    # we can safely assume that all of the nulls in each column
                    # are the dropped value
                    X_tr[dropped, i] = self.categories_[i][self.drop_idx_[i]]

            j += n_categories

        # if ignored are found: potentially need to upcast result to
        # insert None values
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)

            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None

        return X_tr

    @deprecated(
        "get_feature_names is deprecated in 1.0 and will be removed "
        "in 1.2. Please use get_feature_names_out instead."
    )
    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of str of shape (n_features,)
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : ndarray of shape (n_output_features,)
            Array of feature names.
        """
        check_is_fitted(self)
        cats = self.categories_
        if input_features is None:
            input_features = ["x%d" % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError(
                "input_features should have length equal to number of "
                "features ({}), got {}".format(
                    len(self.categories_), len(input_features)
                )
            )

        feature_names = []
        for i in range(len(cats)):
            names = [input_features[i] + "_" + str(t) for t in cats[i]]
            if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
                names.pop(self.drop_idx_[i])
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then names are generated: `[x0, x1, ..., x(n_features_in_)]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self)
        cats = self.categories_
        input_features = _check_feature_names_in(self, input_features)

        feature_names = []
        for i in range(len(cats)):
            names = [input_features[i] + "_" + str(t) for t in cats[i]]
            if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
                names.pop(self.drop_idx_[i])
            feature_names.extend(names)
        return np.asarray(feature_names, dtype=object)


class OrdinalEncoder(_BaseEncoder):
    """
    Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error' an error will be raised in case an unknown
        categorical feature is present during transform. When set to
        'use_encoded_value', the encoded value of unknown categories will be
        set to the value given for the parameter `unknown_value`. In
        :meth:`inverse_transform`, an unknown category will be denoted as None.

        .. versionadded:: 0.24

    unknown_value : int or np.nan, default=None
        When the parameter handle_unknown is set to 'use_encoded_value', this
        parameter is required and will set the encoded value of unknown
        categories. It has to be distinct from the values used to encode any of
        the categories in `fit`. If set to np.nan, the `dtype` parameter must
        be a float dtype.

        .. versionadded:: 0.24

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during ``fit`` (in order of
        the features in X and corresponding with the output of ``transform``).
        This does not include categories that weren't seen during ``fit``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    OneHotEncoder : Performs a one-hot encoding of categorical features.
    LabelEncoder : Encodes target labels with values between 0 and
        ``n_classes-1``.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to an ordinal encoding.

    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> enc = OrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OrdinalEncoder()
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 3], ['Male', 1]])
    array([[0., 2.],
           [1., 0.]])

    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['Male', 1],
           ['Female', 2]], dtype=object)
    """

    def __init__(
        self,
        *,
        categories="auto",
        dtype=np.float64,
        handle_unknown="error",
        unknown_value=None,
    ):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        """
        Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        handle_unknown_strategies = ("error", "use_encoded_value")
        if self.handle_unknown not in handle_unknown_strategies:
            raise ValueError(
                "handle_unknown should be either 'error' or "
                f"'use_encoded_value', got {self.handle_unknown}."
            )

        if self.handle_unknown == "use_encoded_value":
            if is_scalar_nan(self.unknown_value):
                if np.dtype(self.dtype).kind != "f":
                    raise ValueError(
                        "When unknown_value is np.nan, the dtype "
                        "parameter should be "
                        f"a float dtype. Got {self.dtype}."
                    )
            elif not isinstance(self.unknown_value, numbers.Integral):
                raise TypeError(
                    "unknown_value should be an integer or "
                    "np.nan when "
                    "handle_unknown is 'use_encoded_value', "
                    f"got {self.unknown_value}."
                )
        elif self.unknown_value is not None:
            raise TypeError(
                "unknown_value should only be set when "
                "handle_unknown is 'use_encoded_value', "
                f"got {self.unknown_value}."
            )

        # `_fit` will only raise an error when `self.handle_unknown="error"`
        self._fit(X, handle_unknown=self.handle_unknown, force_all_finite="allow-nan")

        if self.handle_unknown == "use_encoded_value":
            for feature_cats in self.categories_:
                if 0 <= self.unknown_value < len(feature_cats):
                    raise ValueError(
                        "The used value for unknown_value "
                        f"{self.unknown_value} is one of the "
                        "values already used for encoding the "
                        "seen categories."
                    )

        # stores the missing indices per category
        self._missing_indices = {}
        for cat_idx, categories_for_idx in enumerate(self.categories_):
            for i, cat in enumerate(categories_for_idx):
                if is_scalar_nan(cat):
                    self._missing_indices[cat_idx] = i
                    continue

        if np.dtype(self.dtype).kind != "f" and self._missing_indices:
            raise ValueError(
                "There are missing values in features "
                f"{list(self._missing_indices)}. For OrdinalEncoder to "
                "passthrough missing values, the dtype parameter must be a "
                "float"
            )

        return self

    def transform(self, X):
        """
        Transform X to ordinal codes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        X_int, X_mask = self._transform(
            X, handle_unknown=self.handle_unknown, force_all_finite="allow-nan"
        )
        X_trans = X_int.astype(self.dtype, copy=False)

        for cat_idx, missing_idx in self._missing_indices.items():
            X_missing_mask = X_int[:, cat_idx] == missing_idx
            X_trans[X_missing_mask, cat_idx] = np.nan

        # create separate category for unknown values
        if self.handle_unknown == "use_encoded_value":
            X_trans[~X_mask] = self.unknown_value
        return X_trans

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, force_all_finite="allow-nan")

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        # validate shape of passed X
        msg = (
            "Shape of the passed X data is not correct. Expected {0} columns, got {1}."
        )
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        found_unknown = {}

        for i in range(n_features):
            labels = X[:, i].astype("int64", copy=False)

            # replace values of X[:, i] that were nan with actual indices
            if i in self._missing_indices:
                X_i_mask = _get_mask(X[:, i], np.nan)
                labels[X_i_mask] = self._missing_indices[i]

            if self.handle_unknown == "use_encoded_value":
                unknown_labels = labels == self.unknown_value
                X_tr[:, i] = self.categories_[i][np.where(unknown_labels, 0, labels)]
                found_unknown[i] = unknown_labels
            else:
                X_tr[:, i] = self.categories_[i][labels]

        # insert None values for unknown values
        if found_unknown:
            X_tr = X_tr.astype(object, copy=False)

            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None

        return X_tr
