# Author: Gael Varoquaux
# License: BSD 3 clause

import re
import numpy as np
import scipy.sparse as sp
import pytest

import sklearn
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_no_warnings
from sklearn.utils._testing import ignore_warnings

from sklearn.base import BaseEstimator, clone, is_classifier, _is_pairwise
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import KernelPCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

from sklearn.base import TransformerMixin
from sklearn.utils._mocking import MockDataFrame
from sklearn import config_context
import pickle


#############################################################################
# A few test classes
class MyEstimator(BaseEstimator):
    def __init__(self, l1=0, empty=None):
        self.l1 = l1
        self.empty = empty


class K(BaseEstimator):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d


class T(BaseEstimator):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class NaNTag(BaseEstimator):
    def _more_tags(self):
        return {"allow_nan": True}


class NoNaNTag(BaseEstimator):
    def _more_tags(self):
        return {"allow_nan": False}


class OverrideTag(NaNTag):
    def _more_tags(self):
        return {"allow_nan": False}


class DiamondOverwriteTag(NaNTag, NoNaNTag):
    def _more_tags(self):
        return dict()


class InheritDiamondOverwriteTag(DiamondOverwriteTag):
    pass


class ModifyInitParams(BaseEstimator):
    """Deprecated behavior.
    Equal parameters but with a type cast.
    Doesn't fulfill a is a
    """

    def __init__(self, a=np.array([0])):
        self.a = a.copy()


class Buggy(BaseEstimator):
    "A buggy estimator that does not set its parameters right."

    def __init__(self, a=None):
        self.a = 1


class NoEstimator:
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def predict(self, X=None):
        return None


class VargEstimator(BaseEstimator):
    """scikit-learn estimators shouldn't have vargs."""

    def __init__(self, *vargs):
        pass


#############################################################################
# The tests


def test_clone():
    # Tests that clone creates a correct deep copy.
    # We create an estimator, make a copy of its original state
    # (which, in this case, is the current state of the estimator),
    # and check that the obtained copy is a correct deep copy.

    from sklearn.feature_selection import SelectFpr, f_classif

    selector = SelectFpr(f_classif, alpha=0.1)
    new_selector = clone(selector)
    assert selector is not new_selector
    assert selector.get_params() == new_selector.get_params()

    selector = SelectFpr(f_classif, alpha=np.zeros((10, 2)))
    new_selector = clone(selector)
    assert selector is not new_selector


def test_clone_2():
    # Tests that clone doesn't copy everything.
    # We first create an estimator, give it an own attribute, and
    # make a copy of its original state. Then we check that the copy doesn't
    # have the specific attribute we manually added to the initial estimator.

    from sklearn.feature_selection import SelectFpr, f_classif

    selector = SelectFpr(f_classif, alpha=0.1)
    selector.own_attribute = "test"
    new_selector = clone(selector)
    assert not hasattr(new_selector, "own_attribute")


def test_clone_buggy():
    # Check that clone raises an error on buggy estimators.
    buggy = Buggy()
    buggy.a = 2
    with pytest.raises(RuntimeError):
        clone(buggy)

    no_estimator = NoEstimator()
    with pytest.raises(TypeError):
        clone(no_estimator)

    varg_est = VargEstimator()
    with pytest.raises(RuntimeError):
        clone(varg_est)

    est = ModifyInitParams()
    with pytest.raises(RuntimeError):
        clone(est)


def test_clone_empty_array():
    # Regression test for cloning estimators with empty arrays
    clf = MyEstimator(empty=np.array([]))
    clf2 = clone(clf)
    assert_array_equal(clf.empty, clf2.empty)

    clf = MyEstimator(empty=sp.csr_matrix(np.array([[0]])))
    clf2 = clone(clf)
    assert_array_equal(clf.empty.data, clf2.empty.data)


def test_clone_nan():
    # Regression test for cloning estimators with default parameter as np.nan
    clf = MyEstimator(empty=np.nan)
    clf2 = clone(clf)

    assert clf.empty is clf2.empty


def test_clone_sparse_matrices():
    sparse_matrix_classes = [
        getattr(sp, name) for name in dir(sp) if name.endswith("_matrix")
    ]

    for cls in sparse_matrix_classes:
        sparse_matrix = cls(np.eye(5))
        clf = MyEstimator(empty=sparse_matrix)
        clf_cloned = clone(clf)
        assert clf.empty.__class__ is clf_cloned.empty.__class__
        assert_array_equal(clf.empty.toarray(), clf_cloned.empty.toarray())


def test_clone_estimator_types():
    # Check that clone works for parameters that are types rather than
    # instances
    clf = MyEstimator(empty=MyEstimator)
    clf2 = clone(clf)

    assert clf.empty is clf2.empty


def test_clone_class_rather_than_instance():
    # Check that clone raises expected error message when
    # cloning class rather than instance
    msg = "You should provide an instance of scikit-learn estimator"
    with pytest.raises(TypeError, match=msg):
        clone(MyEstimator)


def test_repr():
    # Smoke test the repr of the base estimator.
    my_estimator = MyEstimator()
    repr(my_estimator)
    test = T(K(), K())
    assert repr(test) == "T(a=K(), b=K())"

    some_est = T(a=["long_params"] * 1000)
    assert len(repr(some_est)) == 485


def test_str():
    # Smoke test the str of the base estimator
    my_estimator = MyEstimator()
    str(my_estimator)


def test_get_params():
    test = T(K(), K())

    assert "a__d" in test.get_params(deep=True)
    assert "a__d" not in test.get_params(deep=False)

    test.set_params(a__d=2)
    assert test.a.d == 2

    with pytest.raises(ValueError):
        test.set_params(a__a=2)


def test_is_classifier():
    svc = SVC()
    assert is_classifier(svc)
    assert is_classifier(GridSearchCV(svc, {"C": [0.1, 1]}))
    assert is_classifier(Pipeline([("svc", svc)]))
    assert is_classifier(Pipeline([("svc_cv", GridSearchCV(svc, {"C": [0.1, 1]}))]))


def test_set_params():
    # test nested estimator parameter setting
    clf = Pipeline([("svc", SVC())])

    # non-existing parameter in svc
    with pytest.raises(ValueError):
        clf.set_params(svc__stupid_param=True)

    # non-existing parameter of pipeline
    with pytest.raises(ValueError):
        clf.set_params(svm__stupid_param=True)

    # we don't currently catch if the things in pipeline are estimators
    # bad_pipeline = Pipeline([("bad", NoEstimator())])
    # assert_raises(AttributeError, bad_pipeline.set_params,
    #               bad__stupid_param=True)


def test_set_params_passes_all_parameters():
    # Make sure all parameters are passed together to set_params
    # of nested estimator. Regression test for #9944

    class TestDecisionTree(DecisionTreeClassifier):
        def set_params(self, **kwargs):
            super().set_params(**kwargs)
            # expected_kwargs is in test scope
            assert kwargs == expected_kwargs
            return self

    expected_kwargs = {"max_depth": 5, "min_samples_leaf": 2}
    for est in [
        Pipeline([("estimator", TestDecisionTree())]),
        GridSearchCV(TestDecisionTree(), {}),
    ]:
        est.set_params(estimator__max_depth=5, estimator__min_samples_leaf=2)


def test_set_params_updates_valid_params():
    # Check that set_params tries to set SVC().C, not
    # DecisionTreeClassifier().C
    gscv = GridSearchCV(DecisionTreeClassifier(), {})
    gscv.set_params(estimator=SVC(), estimator__C=42.0)
    assert gscv.estimator.C == 42.0


@pytest.mark.parametrize(
    "tree,dataset",
    [
        (
            DecisionTreeClassifier(max_depth=2, random_state=0),
            datasets.make_classification(random_state=0),
        ),
        (
            DecisionTreeRegressor(max_depth=2, random_state=0),
            datasets.make_regression(random_state=0),
        ),
    ],
)
def test_score_sample_weight(tree, dataset):
    rng = np.random.RandomState(0)
    # check that the score with and without sample weights are different
    X, y = dataset

    tree.fit(X, y)
    # generate random sample weights
    sample_weight = rng.randint(1, 10, size=len(y))
    score_unweighted = tree.score(X, y)
    score_weighted = tree.score(X, y, sample_weight=sample_weight)
    msg = "Unweighted and weighted scores are unexpectedly equal"
    assert score_unweighted != score_weighted, msg


def test_clone_pandas_dataframe():
    class DummyEstimator(TransformerMixin, BaseEstimator):
        """This is a dummy class for generating numerical features

        This feature extractor extracts numerical features from pandas data
        frame.

        Parameters
        ----------

        df: pandas data frame
            The pandas data frame parameter.

        Notes
        -----
        """

        def __init__(self, df=None, scalar_param=1):
            self.df = df
            self.scalar_param = scalar_param

        def fit(self, X, y=None):
            pass

        def transform(self, X):
            pass

    # build and clone estimator
    d = np.arange(10)
    df = MockDataFrame(d)
    e = DummyEstimator(df, scalar_param=1)
    cloned_e = clone(e)

    # the test
    assert (e.df == cloned_e.df).values.all()
    assert e.scalar_param == cloned_e.scalar_param


def test_pickle_version_warning_is_not_raised_with_matching_version():
    iris = datasets.load_iris()
    tree = DecisionTreeClassifier().fit(iris.data, iris.target)
    tree_pickle = pickle.dumps(tree)
    assert b"version" in tree_pickle
    tree_restored = assert_no_warnings(pickle.loads, tree_pickle)

    # test that we can predict with the restored decision tree classifier
    score_of_original = tree.score(iris.data, iris.target)
    score_of_restored = tree_restored.score(iris.data, iris.target)
    assert score_of_original == score_of_restored


class TreeBadVersion(DecisionTreeClassifier):
    def __getstate__(self):
        return dict(self.__dict__.items(), _sklearn_version="something")


pickle_error_message = (
    "Trying to unpickle estimator {estimator} from "
    "version {old_version} when using version "
    "{current_version}. This might "
    "lead to breaking code or invalid results. "
    "Use at your own risk."
)


def test_pickle_version_warning_is_issued_upon_different_version():
    iris = datasets.load_iris()
    tree = TreeBadVersion().fit(iris.data, iris.target)
    tree_pickle_other = pickle.dumps(tree)
    message = pickle_error_message.format(
        estimator="TreeBadVersion",
        old_version="something",
        current_version=sklearn.__version__,
    )
    with pytest.warns(UserWarning, match=message):
        pickle.loads(tree_pickle_other)


class TreeNoVersion(DecisionTreeClassifier):
    def __getstate__(self):
        return self.__dict__


def test_pickle_version_warning_is_issued_when_no_version_info_in_pickle():
    iris = datasets.load_iris()
    # TreeNoVersion has no getstate, like pre-0.18
    tree = TreeNoVersion().fit(iris.data, iris.target)

    tree_pickle_noversion = pickle.dumps(tree)
    assert b"version" not in tree_pickle_noversion
    message = pickle_error_message.format(
        estimator="TreeNoVersion",
        old_version="pre-0.18",
        current_version=sklearn.__version__,
    )
    # check we got the warning about using pre-0.18 pickle
    with pytest.warns(UserWarning, match=message):
        pickle.loads(tree_pickle_noversion)


def test_pickle_version_no_warning_is_issued_with_non_sklearn_estimator():
    iris = datasets.load_iris()
    tree = TreeNoVersion().fit(iris.data, iris.target)
    tree_pickle_noversion = pickle.dumps(tree)
    try:
        module_backup = TreeNoVersion.__module__
        TreeNoVersion.__module__ = "notsklearn"
        assert_no_warnings(pickle.loads, tree_pickle_noversion)
    finally:
        TreeNoVersion.__module__ = module_backup


class DontPickleAttributeMixin:
    def __getstate__(self):
        data = self.__dict__.copy()
        data["_attribute_not_pickled"] = None
        return data

    def __setstate__(self, state):
        state["_restored"] = True
        self.__dict__.update(state)


class MultiInheritanceEstimator(DontPickleAttributeMixin, BaseEstimator):
    def __init__(self, attribute_pickled=5):
        self.attribute_pickled = attribute_pickled
        self._attribute_not_pickled = None


def test_pickling_when_getstate_is_overwritten_by_mixin():
    estimator = MultiInheritanceEstimator()
    estimator._attribute_not_pickled = "this attribute should not be pickled"

    serialized = pickle.dumps(estimator)
    estimator_restored = pickle.loads(serialized)
    assert estimator_restored.attribute_pickled == 5
    assert estimator_restored._attribute_not_pickled is None
    assert estimator_restored._restored


def test_pickling_when_getstate_is_overwritten_by_mixin_outside_of_sklearn():
    try:
        estimator = MultiInheritanceEstimator()
        text = "this attribute should not be pickled"
        estimator._attribute_not_pickled = text
        old_mod = type(estimator).__module__
        type(estimator).__module__ = "notsklearn"

        serialized = estimator.__getstate__()
        assert serialized == {"_attribute_not_pickled": None, "attribute_pickled": 5}

        serialized["attribute_pickled"] = 4
        estimator.__setstate__(serialized)
        assert estimator.attribute_pickled == 4
        assert estimator._restored
    finally:
        type(estimator).__module__ = old_mod


class SingleInheritanceEstimator(BaseEstimator):
    def __init__(self, attribute_pickled=5):
        self.attribute_pickled = attribute_pickled
        self._attribute_not_pickled = None

    def __getstate__(self):
        data = self.__dict__.copy()
        data["_attribute_not_pickled"] = None
        return data


@ignore_warnings(category=(UserWarning))
def test_pickling_works_when_getstate_is_overwritten_in_the_child_class():
    estimator = SingleInheritanceEstimator()
    estimator._attribute_not_pickled = "this attribute should not be pickled"

    serialized = pickle.dumps(estimator)
    estimator_restored = pickle.loads(serialized)
    assert estimator_restored.attribute_pickled == 5
    assert estimator_restored._attribute_not_pickled is None


def test_tag_inheritance():
    # test that changing tags by inheritance is not allowed

    nan_tag_est = NaNTag()
    no_nan_tag_est = NoNaNTag()
    assert nan_tag_est._get_tags()["allow_nan"]
    assert not no_nan_tag_est._get_tags()["allow_nan"]

    redefine_tags_est = OverrideTag()
    assert not redefine_tags_est._get_tags()["allow_nan"]

    diamond_tag_est = DiamondOverwriteTag()
    assert diamond_tag_est._get_tags()["allow_nan"]

    inherit_diamond_tag_est = InheritDiamondOverwriteTag()
    assert inherit_diamond_tag_est._get_tags()["allow_nan"]


def test_raises_on_get_params_non_attribute():
    class MyEstimator(BaseEstimator):
        def __init__(self, param=5):
            pass

        def fit(self, X, y=None):
            return self

    est = MyEstimator()
    msg = "'MyEstimator' object has no attribute 'param'"

    with pytest.raises(AttributeError, match=msg):
        est.get_params()


def test_repr_mimebundle_():
    # Checks the display configuration flag controls the json output
    tree = DecisionTreeClassifier()
    output = tree._repr_mimebundle_()
    assert "text/plain" in output
    assert "text/html" not in output

    with config_context(display="diagram"):
        output = tree._repr_mimebundle_()
        assert "text/plain" in output
        assert "text/html" in output


def test_repr_html_wraps():
    # Checks the display configuration flag controls the html output
    tree = DecisionTreeClassifier()
    msg = "_repr_html_ is only defined when"
    with pytest.raises(AttributeError, match=msg):
        output = tree._repr_html_()

    with config_context(display="diagram"):
        output = tree._repr_html_()
        assert "<style>" in output


# TODO: Remove in 1.1 when the _pairwise attribute is removed
def test_is_pairwise():
    # simple checks for _is_pairwise
    pca = KernelPCA(kernel="precomputed")
    with pytest.warns(None) as record:
        assert _is_pairwise(pca)
    assert not record

    # pairwise attribute that is not consistent with the pairwise tag
    class IncorrectTagPCA(KernelPCA):
        _pairwise = False

    pca = IncorrectTagPCA(kernel="precomputed")
    msg = "_pairwise was deprecated in 0.24 and will be removed in 1.1"
    with pytest.warns(FutureWarning, match=msg):
        assert not _is_pairwise(pca)

    # the _pairwise attribute is present and set to True while pairwise tag is
    # not present
    class TruePairwise(BaseEstimator):
        _pairwise = True

    true_pairwise = TruePairwise()
    with pytest.warns(FutureWarning, match=msg):
        assert _is_pairwise(true_pairwise)

    # pairwise attribute is not defined thus tag is used
    est = BaseEstimator()
    with pytest.warns(None) as record:
        assert not _is_pairwise(est)
    assert not record


def test_n_features_in_validation():
    """Check that `_check_n_features` validates data when reset=False"""
    est = MyEstimator()
    X_train = [[1, 2, 3], [4, 5, 6]]
    est._check_n_features(X_train, reset=True)

    assert est.n_features_in_ == 3

    msg = "X does not contain any features, but MyEstimator is expecting 3 features"
    with pytest.raises(ValueError, match=msg):
        est._check_n_features("invalid X", reset=False)


def test_n_features_in_no_validation():
    """Check that `_check_n_features` does not validate data when
    n_features_in_ is not defined."""
    est = MyEstimator()
    est._check_n_features("invalid X", reset=True)

    assert not hasattr(est, "n_features_in_")

    # does not raise
    est._check_n_features("invalid X", reset=False)


def test_feature_names_in():
    """Check that feature_name_in are recorded by `_validate_data`"""
    pd = pytest.importorskip("pandas")
    iris = datasets.load_iris()
    X_np = iris.data
    df = pd.DataFrame(X_np, columns=iris.feature_names)

    class NoOpTransformer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            self._validate_data(X)
            return self

        def transform(self, X):
            self._validate_data(X, reset=False)
            return X

    # fit on dataframe saves the feature names
    trans = NoOpTransformer().fit(df)
    assert_array_equal(trans.feature_names_in_, df.columns)

    # fit again but on ndarray does not keep the previous feature names (see #21383)
    trans.fit(X_np)
    assert not hasattr(trans, "feature_names_in_")

    trans.fit(df)
    msg = "The feature names should match those that were passed"
    df_bad = pd.DataFrame(X_np, columns=iris.feature_names[::-1])
    with pytest.warns(FutureWarning, match=msg):
        trans.transform(df_bad)

    # warns when fitted on dataframe and transforming a ndarray
    msg = (
        "X does not have valid feature names, but NoOpTransformer was "
        "fitted with feature names"
    )
    with pytest.warns(UserWarning, match=msg):
        trans.transform(X_np)

    # warns when fitted on a ndarray and transforming dataframe
    msg = "X has feature names, but NoOpTransformer was fitted without feature names"
    trans = NoOpTransformer().fit(X_np)
    with pytest.warns(UserWarning, match=msg):
        trans.transform(df)

    # fit on dataframe with all integer feature names works without warning
    df_int_names = pd.DataFrame(X_np)
    trans = NoOpTransformer()
    with pytest.warns(None) as record:
        trans.fit(df_int_names)
    assert not record

    # fit on dataframe with no feature names or all integer feature names
    # -> do not warn on transform
    Xs = [X_np, df_int_names]
    for X in Xs:
        with pytest.warns(None) as record:
            trans.transform(X)
        assert not record

    # TODO: Convert to a error in 1.2
    # fit on dataframe with feature names that are mixed warns:
    df_mixed = pd.DataFrame(X_np, columns=["a", "b", 1, 2])
    trans = NoOpTransformer()
    msg = re.escape(
        "Feature names only support names that are all strings. "
        "Got feature names with dtypes: ['int', 'str']"
    )
    with pytest.warns(FutureWarning, match=msg) as record:
        trans.fit(df_mixed)

    # transform on feature names that are mixed also warns:
    with pytest.warns(FutureWarning, match=msg) as record:
        trans.transform(df_mixed)
