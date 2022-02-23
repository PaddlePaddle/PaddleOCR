# Authors: Lars Buitinck
#          Dan Blanchard <dblanchard@ets.org>
# License: BSD 3 clause

from array import array
from collections.abc import Mapping, Iterable
from operator import itemgetter
from numbers import Number

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array, tosequence
from ..utils.deprecation import deprecated


def _tosequence(X):
    """Turn X into a sequence or ndarray, avoiding a copy if possible."""
    if isinstance(X, Mapping):  # single sample
        return [X]
    else:
        return tosequence(X)


class DictVectorizer(TransformerMixin, BaseEstimator):
    """Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    If a feature value is a sequence or set of strings, this transformer
    will iterate over the values and will count the occurrences of each string
    value.

    However, note that this transformer will only do a binary one-hot encoding
    when feature values are of type string. If categorical features are
    represented as numeric values such as int or iterables of strings, the
    DictVectorizer can be followed by
    :class:`~sklearn.preprocessing.OneHotEncoder` to complete
    binary one-hot encoding.

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Read more in the :ref:`User Guide <dict_feature_extraction>`.

    Parameters
    ----------
    dtype : dtype, default=np.float64
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator : str, default="="
        Separator string used when constructing new features for one-hot
        coding.
    sparse : bool, default=True
        Whether transform should produce scipy.sparse matrices.
    sort : bool, default=True
        Whether ``feature_names_`` and ``vocabulary_`` should be
        sorted when fitting.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    See Also
    --------
    FeatureHasher : Performs vectorization using only a hash function.
    sklearn.preprocessing.OrdinalEncoder : Handles nominal/categorical
        features encoded as columns of arbitrary data types.

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[2., 0., 1.],
           [0., 1., 3.]])
    >>> v.inverse_transform(X) == [{'bar': 2.0, 'foo': 1.0},
    ...                            {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[0., 0., 4.]])
    """

    def __init__(self, *, dtype=np.float64, separator="=", sparse=True, sort=True):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = sort

    def _add_iterable_element(
        self,
        f,
        v,
        feature_names,
        vocab,
        *,
        fitting=True,
        transforming=False,
        indices=None,
        values=None,
    ):
        """Add feature names for iterable of strings"""
        for vv in v:
            if isinstance(vv, str):
                feature_name = "%s%s%s" % (f, self.separator, vv)
                vv = 1
            else:
                raise TypeError(
                    f"Unsupported type {type(vv)} in iterable "
                    "value. Only iterables of string are "
                    "supported."
                )
            if fitting and feature_name not in vocab:
                vocab[feature_name] = len(feature_names)
                feature_names.append(feature_name)

            if transforming and feature_name in vocab:
                indices.append(vocab[feature_name])
                values.append(self.dtype(vv))

    def fit(self, X, y=None):
        """Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

            .. versionchanged:: 0.24
               Accepts multiple string values for one categorical feature.

        y : (ignored)
            Ignored parameter.

        Returns
        -------
        self : object
            DictVectorizer class instance.
        """
        feature_names = []
        vocab = {}

        for x in X:
            for f, v in x.items():
                if isinstance(v, str):
                    feature_name = "%s%s%s" % (f, self.separator, v)
                    v = 1
                elif isinstance(v, Number) or (v is None):
                    feature_name = f
                elif isinstance(v, Mapping):
                    raise TypeError(
                        f"Unsupported value type {type(v)} "
                        f"for {f}: {v}.\n"
                        "Mapping objects are not supported."
                    )
                elif isinstance(v, Iterable):
                    feature_name = None
                    self._add_iterable_element(f, v, feature_names, vocab)

                if feature_name is not None:
                    if feature_name not in vocab:
                        vocab[feature_name] = len(feature_names)
                        feature_names.append(feature_name)

        if self.sort:
            feature_names.sort()
            vocab = {f: i for i, f in enumerate(feature_names)}

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab

        return self

    def _transform(self, X, fitting):
        # Sanity check: Python's array has no way of explicitly requesting the
        # signed 32-bit integers that scipy.sparse needs, so we use the next
        # best thing: typecode "i" (int). However, if that gives larger or
        # smaller integers than 32-bit ones, np.frombuffer screws up.
        assert array("i").itemsize == 4, (
            "sizeof(int) != 4 on your platform; please report this at"
            " https://github.com/scikit-learn/scikit-learn/issues and"
            " include the output from platform.platform() in your bug report"
        )

        dtype = self.dtype
        if fitting:
            feature_names = []
            vocab = {}
        else:
            feature_names = self.feature_names_
            vocab = self.vocabulary_

        transforming = True

        # Process everything as sparse regardless of setting
        X = [X] if isinstance(X, Mapping) else X

        indices = array("i")
        indptr = [0]
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        # collect all the possible feature names and build sparse matrix at
        # same time
        for x in X:
            for f, v in x.items():
                if isinstance(v, str):
                    feature_name = "%s%s%s" % (f, self.separator, v)
                    v = 1
                elif isinstance(v, Number) or (v is None):
                    feature_name = f
                elif not isinstance(v, Mapping) and isinstance(v, Iterable):
                    feature_name = None
                    self._add_iterable_element(
                        f,
                        v,
                        feature_names,
                        vocab,
                        fitting=fitting,
                        transforming=transforming,
                        indices=indices,
                        values=values,
                    )
                else:
                    raise TypeError(
                        f"Unsupported value Type {type(v)} "
                        f"for {f}: {v}.\n"
                        f"{type(v)} objects are not supported."
                    )

                if feature_name is not None:
                    if fitting and feature_name not in vocab:
                        vocab[feature_name] = len(feature_names)
                        feature_names.append(feature_name)

                    if feature_name in vocab:
                        indices.append(vocab[feature_name])
                        values.append(self.dtype(v))

            indptr.append(len(indices))

        if len(indptr) == 1:
            raise ValueError("Sample sequence X is empty.")

        indices = np.frombuffer(indices, dtype=np.intc)
        shape = (len(indptr) - 1, len(vocab))

        result_matrix = sp.csr_matrix(
            (values, indices, indptr), shape=shape, dtype=dtype
        )

        # Sort everything if asked
        if fitting and self.sort:
            feature_names.sort()
            map_index = np.empty(len(feature_names), dtype=np.int32)
            for new_val, f in enumerate(feature_names):
                map_index[new_val] = vocab[f]
                vocab[f] = new_val
            result_matrix = result_matrix[:, map_index]

        if self.sparse:
            result_matrix.sort_indices()
        else:
            result_matrix = result_matrix.toarray()

        if fitting:
            self.feature_names_ = feature_names
            self.vocabulary_ = vocab

        return result_matrix

    def fit_transform(self, X, y=None):
        """Learn a list of feature name -> indices mappings and transform X.

        Like fit(X) followed by transform(X), but does not require
        materializing X in memory.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

            .. versionchanged:: 0.24
               Accepts multiple string values for one categorical feature.

        y : (ignored)
            Ignored parameter.

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        return self._transform(X, fitting=True)

    def inverse_transform(self, X, dict_type=dict):
        """Transform array or sparse matrix X back to feature mappings.

        X must have been produced by this DictVectorizer's transform or
        fit_transform method; it may only have passed through transformers
        that preserve the number of features and their order.

        In the case of one-hot/one-of-K coding, the constructed feature
        names and values are returned rather than the original ones.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Sample matrix.
        dict_type : type, default=dict
            Constructor for feature mappings. Must conform to the
            collections.Mapping API.

        Returns
        -------
        D : list of dict_type objects of shape (n_samples,)
            Feature mappings for the samples in X.
        """
        # COO matrix is not subscriptable
        X = check_array(X, accept_sparse=["csr", "csc"])
        n_samples = X.shape[0]

        names = self.feature_names_
        dicts = [dict_type() for _ in range(n_samples)]

        if sp.issparse(X):
            for i, j in zip(*X.nonzero()):
                dicts[i][names[j]] = X[i, j]
        else:
            for i, d in enumerate(dicts):
                for j, v in enumerate(X[i, :]):
                    if v != 0:
                        d[names[j]] = X[i, j]

        return dicts

    def transform(self, X):
        """Transform feature->value dicts to array or sparse matrix.

        Named features not encountered during fit or fit_transform will be
        silently ignored.

        Parameters
        ----------
        X : Mapping or iterable over Mappings of shape (n_samples,)
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        return self._transform(X, fitting=False)

    @deprecated(
        "get_feature_names is deprecated in 1.0 and will be removed "
        "in 1.2. Please use get_feature_names_out instead."
    )
    def get_feature_names(self):
        """Return a list of feature names, ordered by their indices.

        If one-of-K coding is applied to categorical features, this will
        include the constructed feature names but not the original ones.

        Returns
        -------
        feature_names_ : list of length (n_features,)
           List containing the feature names (e.g., "f=ham" and "f=spam").
        """
        return self.feature_names_

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if any(not isinstance(name, str) for name in self.feature_names_):
            feature_names = [str(name) for name in self.feature_names_]
        else:
            feature_names = self.feature_names_
        return np.asarray(feature_names, dtype=object)

    def restrict(self, support, indices=False):
        """Restrict the features to those in support using feature selection.

        This function modifies the estimator in-place.

        Parameters
        ----------
        support : array-like
            Boolean mask or list of indices (as returned by the get_support
            member of feature selectors).
        indices : bool, default=False
            Whether support is a list of indices.

        Returns
        -------
        self : object
            DictVectorizer class instance.

        Examples
        --------
        >>> from sklearn.feature_extraction import DictVectorizer
        >>> from sklearn.feature_selection import SelectKBest, chi2
        >>> v = DictVectorizer()
        >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        >>> X = v.fit_transform(D)
        >>> support = SelectKBest(chi2, k=2).fit(X, [0, 1])
        >>> v.get_feature_names_out()
        array(['bar', 'baz', 'foo'], ...)
        >>> v.restrict(support.get_support())
        DictVectorizer()
        >>> v.get_feature_names_out()
        array(['bar', 'foo'], ...)
        """
        if not indices:
            support = np.where(support)[0]

        names = self.feature_names_
        new_vocab = {}
        for i in support:
            new_vocab[names[i]] = len(new_vocab)

        self.vocabulary_ = new_vocab
        self.feature_names_ = [
            f for f, i in sorted(new_vocab.items(), key=itemgetter(1))
        ]

        return self

    def _more_tags(self):
        return {"X_types": ["dict"]}
