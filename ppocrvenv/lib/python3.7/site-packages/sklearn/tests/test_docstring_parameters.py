# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

import inspect
import warnings
import importlib

from pkgutil import walk_packages
from inspect import signature

import numpy as np

import sklearn
from sklearn.utils import IS_PYPY
from sklearn.utils._testing import check_docstring_parameters
from sklearn.utils._testing import _get_func_name
from sklearn.utils._testing import ignore_warnings
from sklearn.utils import all_estimators
from sklearn.utils.estimator_checks import _enforce_estimator_tags_y
from sklearn.utils.estimator_checks import _enforce_estimator_tags_x
from sklearn.utils.estimator_checks import _construct_instance
from sklearn.utils.deprecation import _is_deprecated
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

import pytest


# walk_packages() ignores DeprecationWarnings, now we need to ignore
# FutureWarnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    # mypy error: Module has no attribute "__path__"
    sklearn_path = sklearn.__path__  # type: ignore  # mypy issue #1422
    PUBLIC_MODULES = set(
        [
            pckg[1]
            for pckg in walk_packages(prefix="sklearn.", path=sklearn_path)
            if not ("._" in pckg[1] or ".tests." in pckg[1])
        ]
    )

# functions to ignore args / docstring of
_DOCSTRING_IGNORES = [
    "sklearn.utils.deprecation.load_mlcomp",
    "sklearn.pipeline.make_pipeline",
    "sklearn.pipeline.make_union",
    "sklearn.utils.extmath.safe_sparse_dot",
    "sklearn.utils._joblib",
]

# Methods where y param should be ignored if y=None by default
_METHODS_IGNORE_NONE_Y = [
    "fit",
    "score",
    "fit_predict",
    "fit_transform",
    "partial_fit",
    "predict",
]


# numpydoc 0.8.0's docscrape tool raises because of collections.abc under
# Python 3.7
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipif(IS_PYPY, reason="test segfaults on PyPy")
def test_docstring_parameters():
    # Test module docstring formatting

    # Skip test if numpydoc is not found
    pytest.importorskip(
        "numpydoc", reason="numpydoc is required to test the docstrings"
    )

    # XXX unreached code as of v0.22
    from numpydoc import docscrape

    incorrect = []
    for name in PUBLIC_MODULES:
        if name.endswith(".conftest"):
            # pytest tooling, not part of the scikit-learn API
            continue
        if name == "sklearn.utils.fixes":
            # We cannot always control these docstrings
            continue
        with warnings.catch_warnings(record=True):
            module = importlib.import_module(name)
        classes = inspect.getmembers(module, inspect.isclass)
        # Exclude non-scikit-learn classes
        classes = [cls for cls in classes if cls[1].__module__.startswith("sklearn")]
        for cname, cls in classes:
            this_incorrect = []
            if cname in _DOCSTRING_IGNORES or cname.startswith("_"):
                continue
            if inspect.isabstract(cls):
                continue
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            if len(w):
                raise RuntimeError(
                    "Error for __init__ of %s in %s:\n%s" % (cls, name, w[0])
                )

            cls_init = getattr(cls, "__init__", None)

            if _is_deprecated(cls_init):
                continue
            elif cls_init is not None:
                this_incorrect += check_docstring_parameters(cls.__init__, cdoc)

            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                if _is_deprecated(method):
                    continue
                param_ignore = None
                # Now skip docstring test for y when y is None
                # by default for API reason
                if method_name in _METHODS_IGNORE_NONE_Y:
                    sig = signature(method)
                    if "y" in sig.parameters and sig.parameters["y"].default is None:
                        param_ignore = ["y"]  # ignore y for fit and score
                result = check_docstring_parameters(method, ignore=param_ignore)
                this_incorrect += result

            incorrect += this_incorrect

        functions = inspect.getmembers(module, inspect.isfunction)
        # Exclude imported functions
        functions = [fn for fn in functions if fn[1].__module__ == name]
        for fname, func in functions:
            # Don't test private methods / functions
            if fname.startswith("_"):
                continue
            if fname == "configuration" and name.endswith("setup"):
                continue
            name_ = _get_func_name(func)
            if not any(d in name_ for d in _DOCSTRING_IGNORES) and not _is_deprecated(
                func
            ):
                incorrect += check_docstring_parameters(func)

    msg = "\n".join(incorrect)
    if len(incorrect) > 0:
        raise AssertionError("Docstring Error:\n" + msg)


@ignore_warnings(category=FutureWarning)
def test_tabs():
    # Test that there are no tabs in our source files
    for importer, modname, ispkg in walk_packages(sklearn.__path__, prefix="sklearn."):

        if IS_PYPY and (
            "_svmlight_format_io" in modname
            or "feature_extraction._hashing_fast" in modname
        ):
            continue

        # because we don't import
        mod = importlib.import_module(modname)

        try:
            source = inspect.getsource(mod)
        except IOError:  # user probably should have run "make clean"
            continue
        assert "\t" not in source, (
            '"%s" has tabs, please remove them ',
            "or add it to the ignore list" % modname,
        )


def _construct_searchcv_instance(SearchCV):
    return SearchCV(LogisticRegression(), {"C": [0.1, 1]})


def _construct_compose_pipeline_instance(Estimator):
    # Minimal / degenerate instances: only useful to test the docstrings.
    if Estimator.__name__ == "ColumnTransformer":
        return Estimator(transformers=[("transformer", "passthrough", [0, 1])])
    elif Estimator.__name__ == "Pipeline":
        return Estimator(steps=[("clf", LogisticRegression())])
    elif Estimator.__name__ == "FeatureUnion":
        return Estimator(transformer_list=[("transformer", FunctionTransformer())])


def _construct_sparse_coder(Estimator):
    # XXX: hard-coded assumption that n_features=3
    dictionary = np.array(
        [[0, 1, 0], [-1, -1, 2], [1, 1, 1], [0, 1, 1], [0, 2, 1]],
        dtype=np.float64,
    )
    return Estimator(dictionary=dictionary)


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_fit_docstring_attributes(name, Estimator):
    pytest.importorskip("numpydoc")
    from numpydoc import docscrape

    doc = docscrape.ClassDoc(Estimator)
    attributes = doc["Attributes"]

    if Estimator.__name__ in (
        "HalvingRandomSearchCV",
        "RandomizedSearchCV",
        "HalvingGridSearchCV",
        "GridSearchCV",
    ):
        est = _construct_searchcv_instance(Estimator)
    elif Estimator.__name__ in (
        "ColumnTransformer",
        "Pipeline",
        "FeatureUnion",
    ):
        est = _construct_compose_pipeline_instance(Estimator)
    elif Estimator.__name__ == "SparseCoder":
        est = _construct_sparse_coder(Estimator)
    else:
        est = _construct_instance(Estimator)

    if Estimator.__name__ == "SelectKBest":
        est.set_params(k=2)
    elif Estimator.__name__ == "DummyClassifier":
        est.set_params(strategy="stratified")
    elif Estimator.__name__ == "CCA" or Estimator.__name__.startswith("PLS"):
        # default = 2 is invalid for single target
        est.set_params(n_components=1)
    elif Estimator.__name__ in (
        "GaussianRandomProjection",
        "SparseRandomProjection",
    ):
        # default="auto" raises an error with the shape of `X`
        est.set_params(n_components=2)

    # FIXME: TO BE REMOVED in 1.4 (avoid FutureWarning)
    if Estimator.__name__ in (
        "OrthogonalMatchingPursuit",
        "OrthogonalMatchingPursuitCV",
        "Lars",
        "LarsCV",
        "LassoLars",
        "LassoLarsCV",
        "LassoLarsIC",
    ):
        est.set_params(normalize=False)

    # FIXME: TO BE REMOVED for 1.1 (avoid FutureWarning)
    if Estimator.__name__ == "NMF":
        est.set_params(init="nndsvda")

    # FIXME: TO BE REMOVED for 1.2 (avoid FutureWarning)
    if Estimator.__name__ == "TSNE":
        est.set_params(learning_rate=200.0, init="random")

    # For PLS, TODO remove in 1.1
    skipped_attributes = {"x_scores_", "y_scores_"}

    if Estimator.__name__.endswith("Vectorizer"):
        # Vectorizer require some specific input data
        if Estimator.__name__ in (
            "CountVectorizer",
            "HashingVectorizer",
            "TfidfVectorizer",
        ):
            X = [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        elif Estimator.__name__ == "DictVectorizer":
            X = [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1}]
        y = None
    else:
        X, y = make_classification(
            n_samples=20,
            n_features=3,
            n_redundant=0,
            n_classes=2,
            random_state=2,
        )

        y = _enforce_estimator_tags_y(est, y)
        X = _enforce_estimator_tags_x(est, X)

    if "1dlabels" in est._get_tags()["X_types"]:
        est.fit(y)
    elif "2dlabels" in est._get_tags()["X_types"]:
        est.fit(np.c_[y, y])
    else:
        est.fit(X, y)

    for attr in attributes:
        if attr.name in skipped_attributes:
            continue
        desc = " ".join(attr.desc).lower()
        # As certain attributes are present "only" if a certain parameter is
        # provided, this checks if the word "only" is present in the attribute
        # description, and if not the attribute is required to be present.
        if "only " in desc:
            continue
        # ignore deprecation warnings
        with ignore_warnings(category=FutureWarning):
            assert hasattr(est, attr.name)

    fit_attr = _get_all_fitted_attributes(est)
    fit_attr_names = [attr.name for attr in attributes]
    undocumented_attrs = set(fit_attr).difference(fit_attr_names)
    undocumented_attrs = set(undocumented_attrs).difference(skipped_attributes)
    if undocumented_attrs:
        raise AssertionError(
            f"Undocumented attributes for {Estimator.__name__}: {undocumented_attrs}"
        )


def _get_all_fitted_attributes(estimator):
    "Get all the fitted attributes of an estimator including properties"
    # attributes
    fit_attr = list(estimator.__dict__.keys())

    # properties
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)

        for name in dir(estimator.__class__):
            obj = getattr(estimator.__class__, name)
            if not isinstance(obj, property):
                continue

            # ignore properties that raises an AttributeError and deprecated
            # properties
            try:
                getattr(estimator, name)
            except (AttributeError, FutureWarning):
                continue
            fit_attr.append(name)

    return [k for k in fit_attr if k.endswith("_") and not k.startswith("_")]
