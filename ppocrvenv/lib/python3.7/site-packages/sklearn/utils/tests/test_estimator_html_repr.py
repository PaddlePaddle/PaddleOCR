from contextlib import closing
import html
from io import StringIO

import pytest

from sklearn import config_context
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.utils._estimator_html_repr import _write_label_html
from sklearn.utils._estimator_html_repr import _get_visual_block
from sklearn.utils._estimator_html_repr import estimator_html_repr


@pytest.mark.parametrize("checked", [True, False])
def test_write_label_html(checked):
    # Test checking logic and labeling
    name = "LogisticRegression"
    tool_tip = "hello-world"

    with closing(StringIO()) as out:
        _write_label_html(out, name, tool_tip, checked=checked)
        html_label = out.getvalue()
        assert "LogisticRegression</label>" in html_label
        assert html_label.startswith('<div class="sk-label-container">')
        assert "<pre>hello-world</pre>" in html_label
        if checked:
            assert "checked>" in html_label


@pytest.mark.parametrize("est", ["passthrough", "drop", None])
def test_get_visual_block_single_str_none(est):
    # Test estimators that are represented by strings
    est_html_info = _get_visual_block(est)
    assert est_html_info.kind == "single"
    assert est_html_info.estimators == est
    assert est_html_info.names == str(est)
    assert est_html_info.name_details == str(est)


def test_get_visual_block_single_estimator():
    est = LogisticRegression(C=10.0)
    est_html_info = _get_visual_block(est)
    assert est_html_info.kind == "single"
    assert est_html_info.estimators == est
    assert est_html_info.names == est.__class__.__name__
    assert est_html_info.name_details == str(est)


def test_get_visual_block_pipeline():
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("do_nothing", "passthrough"),
            ("do_nothing_more", None),
            ("classifier", LogisticRegression()),
        ]
    )
    est_html_info = _get_visual_block(pipe)
    assert est_html_info.kind == "serial"
    assert est_html_info.estimators == tuple(step[1] for step in pipe.steps)
    assert est_html_info.names == [
        "imputer: SimpleImputer",
        "do_nothing: passthrough",
        "do_nothing_more: passthrough",
        "classifier: LogisticRegression",
    ]
    assert est_html_info.name_details == [str(est) for _, est in pipe.steps]


def test_get_visual_block_feature_union():
    f_union = FeatureUnion([("pca", PCA()), ("svd", TruncatedSVD())])
    est_html_info = _get_visual_block(f_union)
    assert est_html_info.kind == "parallel"
    assert est_html_info.names == ("pca", "svd")
    assert est_html_info.estimators == tuple(
        trans[1] for trans in f_union.transformer_list
    )
    assert est_html_info.name_details == (None, None)


def test_get_visual_block_voting():
    clf = VotingClassifier(
        [("log_reg", LogisticRegression()), ("mlp", MLPClassifier())]
    )
    est_html_info = _get_visual_block(clf)
    assert est_html_info.kind == "parallel"
    assert est_html_info.estimators == tuple(trans[1] for trans in clf.estimators)
    assert est_html_info.names == ("log_reg", "mlp")
    assert est_html_info.name_details == (None, None)


def test_get_visual_block_column_transformer():
    ct = ColumnTransformer(
        [("pca", PCA(), ["num1", "num2"]), ("svd", TruncatedSVD, [0, 3])]
    )
    est_html_info = _get_visual_block(ct)
    assert est_html_info.kind == "parallel"
    assert est_html_info.estimators == tuple(trans[1] for trans in ct.transformers)
    assert est_html_info.names == ("pca", "svd")
    assert est_html_info.name_details == (["num1", "num2"], [0, 3])


def test_estimator_html_repr_pipeline():
    num_trans = Pipeline(
        steps=[("pass", "passthrough"), ("imputer", SimpleImputer(strategy="median"))]
    )

    cat_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", missing_values="empty")),
            ("one-hot", OneHotEncoder(drop="first")),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", num_trans, ["a", "b", "c", "d", "e"]),
            ("cat", cat_trans, [0, 1, 2, 3]),
        ]
    )

    feat_u = FeatureUnion(
        [
            ("pca", PCA(n_components=1)),
            (
                "tsvd",
                Pipeline(
                    [
                        ("first", TruncatedSVD(n_components=3)),
                        ("select", SelectPercentile()),
                    ]
                ),
            ),
        ]
    )

    clf = VotingClassifier(
        [
            ("lr", LogisticRegression(solver="lbfgs", random_state=1)),
            ("mlp", MLPClassifier(alpha=0.001)),
        ]
    )

    pipe = Pipeline(
        [("preprocessor", preprocess), ("feat_u", feat_u), ("classifier", clf)]
    )
    html_output = estimator_html_repr(pipe)

    # top level estimators show estimator with changes
    assert html.escape(str(pipe)) in html_output
    for _, est in pipe.steps:
        assert (
            '<div class="sk-toggleable__content"><pre>' + html.escape(str(est))
        ) in html_output

    # low level estimators do not show changes
    with config_context(print_changed_only=True):
        assert html.escape(str(num_trans["pass"])) in html_output
        assert "passthrough</label>" in html_output
        assert html.escape(str(num_trans["imputer"])) in html_output

        for _, _, cols in preprocess.transformers:
            assert f"<pre>{html.escape(str(cols))}</pre>" in html_output

        # feature union
        for name, _ in feat_u.transformer_list:
            assert f"<label>{html.escape(name)}</label>" in html_output

        pca = feat_u.transformer_list[0][1]
        assert f"<pre>{html.escape(str(pca))}</pre>" in html_output

        tsvd = feat_u.transformer_list[1][1]
        first = tsvd["first"]
        select = tsvd["select"]
        assert f"<pre>{html.escape(str(first))}</pre>" in html_output
        assert f"<pre>{html.escape(str(select))}</pre>" in html_output

        # voting classifier
        for name, est in clf.estimators:
            assert f"<label>{html.escape(name)}</label>" in html_output
            assert f"<pre>{html.escape(str(est))}</pre>" in html_output


@pytest.mark.parametrize("final_estimator", [None, LinearSVC()])
def test_stacking_classsifer(final_estimator):
    estimators = [
        ("mlp", MLPClassifier(alpha=0.001)),
        ("tree", DecisionTreeClassifier()),
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

    html_output = estimator_html_repr(clf)

    assert html.escape(str(clf)) in html_output
    # If final_estimator's default changes from LogisticRegression
    # this should be updated
    if final_estimator is None:
        assert "LogisticRegression(" in html_output
    else:
        assert final_estimator.__class__.__name__ in html_output


@pytest.mark.parametrize("final_estimator", [None, LinearSVR()])
def test_stacking_regressor(final_estimator):
    reg = StackingRegressor(
        estimators=[("svr", LinearSVR())], final_estimator=final_estimator
    )
    html_output = estimator_html_repr(reg)

    assert html.escape(str(reg.estimators[0][0])) in html_output
    assert "LinearSVR</label>" in html_output
    if final_estimator is None:
        assert "RidgeCV</label>" in html_output
    else:
        assert html.escape(final_estimator.__class__.__name__) in html_output


def test_birch_duck_typing_meta():
    # Test duck typing meta estimators with Birch
    birch = Birch(n_clusters=AgglomerativeClustering(n_clusters=3))
    html_output = estimator_html_repr(birch)

    # inner estimators do not show changes
    with config_context(print_changed_only=True):
        assert f"<pre>{html.escape(str(birch.n_clusters))}" in html_output
        assert "AgglomerativeClustering</label>" in html_output

    # outer estimator contains all changes
    assert f"<pre>{html.escape(str(birch))}" in html_output


def test_ovo_classifier_duck_typing_meta():
    # Test duck typing metaestimators with OVO
    ovo = OneVsOneClassifier(LinearSVC(penalty="l1"))
    html_output = estimator_html_repr(ovo)

    # inner estimators do not show changes
    with config_context(print_changed_only=True):
        assert f"<pre>{html.escape(str(ovo.estimator))}" in html_output
        assert "LinearSVC</label>" in html_output

    # outer estimator
    assert f"<pre>{html.escape(str(ovo))}" in html_output


def test_duck_typing_nested_estimator():
    # Test duck typing metaestimators with GP
    kernel = RationalQuadratic(length_scale=1.0, alpha=0.1)
    gp = GaussianProcessRegressor(kernel=kernel)
    html_output = estimator_html_repr(gp)

    assert f"<pre>{html.escape(str(kernel))}" in html_output
    assert f"<pre>{html.escape(str(gp))}" in html_output


@pytest.mark.parametrize("print_changed_only", [True, False])
def test_one_estimator_print_change_only(print_changed_only):
    pca = PCA(n_components=10)

    with config_context(print_changed_only=print_changed_only):
        pca_repr = html.escape(str(pca))
        html_output = estimator_html_repr(pca)
        assert pca_repr in html_output


def test_fallback_exists():
    """Check that repr fallback is in the HTML."""
    pca = PCA(n_components=10)
    html_output = estimator_html_repr(pca)

    assert (
        f'<div class="sk-text-repr-fallback"><pre>{html.escape(str(pca))}'
        in html_output
    )


def test_show_arrow_pipeline():
    """Show arrow in pipeline for top level in pipeline"""
    pipe = Pipeline([("scale", StandardScaler()), ("log_Reg", LogisticRegression())])

    html_output = estimator_html_repr(pipe)
    assert (
        'class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline'
        in html_output
    )
