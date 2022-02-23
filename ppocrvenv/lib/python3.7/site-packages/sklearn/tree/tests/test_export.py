"""
Testing for export functions of decision trees (sklearn.tree.export).
"""
from re import finditer, search
from textwrap import dedent

from numpy.random import RandomState
import pytest

from sklearn.base import is_classifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_graphviz, plot_tree, export_text
from io import StringIO
from sklearn.exceptions import NotFittedError

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
y2 = [[-1, 1], [-1, 1], [-1, 1], [1, 2], [1, 2], [1, 3]]
w = [1, 1, 1, 0.5, 0.5, 0.5]
y_degraded = [1, 1, 1, 1, 1, 1]


def test_graphviz_toy():
    # Check correctness of export_graphviz
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    clf.fit(X, y)

    # Test export code
    contents1 = export_graphviz(clf, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="X[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with feature_names
    contents1 = export_graphviz(
        clf, feature_names=["feature0", "feature1"], out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="feature0 <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with class_names
    contents1 = export_graphviz(clf, class_names=["yes", "no"], out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="X[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = yes"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\n'
        'class = yes"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\n'
        'class = no"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test plot_options
    contents1 = export_graphviz(
        clf,
        filled=True,
        impurity=False,
        proportion=True,
        special_characters=True,
        rounded=True,
        out_file=None,
        fontname="sans",
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled, rounded", color="black", '
        'fontname="sans"] ;\n'
        'edge [fontname="sans"] ;\n'
        "0 [label=<X<SUB>0</SUB> &le; 0.0<br/>samples = 100.0%<br/>"
        'value = [0.5, 0.5]>, fillcolor="#ffffff"] ;\n'
        "1 [label=<samples = 50.0%<br/>value = [1.0, 0.0]>, "
        'fillcolor="#e58139"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        "2 [label=<samples = 50.0%<br/>value = [0.0, 1.0]>, "
        'fillcolor="#399de5"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test max_depth
    contents1 = export_graphviz(clf, max_depth=0, class_names=True, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="X[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = y[0]"] ;\n'
        '1 [label="(...)"] ;\n'
        "0 -> 1 ;\n"
        '2 [label="(...)"] ;\n'
        "0 -> 2 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test max_depth with plot_options
    contents1 = export_graphviz(
        clf, max_depth=0, filled=True, out_file=None, node_ids=True
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="node #0\\nX[0] <= 0.0\\ngini = 0.5\\n'
        'samples = 6\\nvalue = [3, 3]", fillcolor="#ffffff"] ;\n'
        '1 [label="(...)", fillcolor="#C0C0C0"] ;\n'
        "0 -> 1 ;\n"
        '2 [label="(...)", fillcolor="#C0C0C0"] ;\n'
        "0 -> 2 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test multi-output with weighted samples
    clf = DecisionTreeClassifier(
        max_depth=2, min_samples_split=2, criterion="gini", random_state=2
    )
    clf = clf.fit(X, y2, sample_weight=w)

    contents1 = export_graphviz(clf, filled=True, impurity=False, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="X[0] <= 0.0\\nsamples = 6\\n'
        "value = [[3.0, 1.5, 0.0]\\n"
        '[3.0, 1.0, 0.5]]", fillcolor="#ffffff"] ;\n'
        '1 [label="samples = 3\\nvalue = [[3, 0, 0]\\n'
        '[3, 0, 0]]", fillcolor="#e58139"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="X[0] <= 1.5\\nsamples = 3\\n'
        "value = [[0.0, 1.5, 0.0]\\n"
        '[0.0, 1.0, 0.5]]", fillcolor="#f1bd97"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        '3 [label="samples = 2\\nvalue = [[0, 1, 0]\\n'
        '[0, 1, 0]]", fillcolor="#e58139"] ;\n'
        "2 -> 3 ;\n"
        '4 [label="samples = 1\\nvalue = [[0.0, 0.5, 0.0]\\n'
        '[0.0, 0.0, 0.5]]", fillcolor="#e58139"] ;\n'
        "2 -> 4 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test regression output with plot_options
    clf = DecisionTreeRegressor(
        max_depth=3, min_samples_split=2, criterion="squared_error", random_state=2
    )
    clf.fit(X, y)

    contents1 = export_graphviz(
        clf,
        filled=True,
        leaves_parallel=True,
        out_file=None,
        rotate=True,
        rounded=True,
        fontname="sans",
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled, rounded", color="black", '
        'fontname="sans"] ;\n'
        "graph [ranksep=equally, splines=polyline] ;\n"
        'edge [fontname="sans"] ;\n'
        "rankdir=LR ;\n"
        '0 [label="X[0] <= 0.0\\nsquared_error = 1.0\\nsamples = 6\\n'
        'value = 0.0", fillcolor="#f2c09c"] ;\n'
        '1 [label="squared_error = 0.0\\nsamples = 3\\'
        'nvalue = -1.0", '
        'fillcolor="#ffffff"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=-45, "
        'headlabel="True"] ;\n'
        '2 [label="squared_error = 0.0\\nsamples = 3\\nvalue = 1.0", '
        'fillcolor="#e58139"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=45, "
        'headlabel="False"] ;\n'
        "{rank=same ; 0} ;\n"
        "{rank=same ; 1; 2} ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test classifier with degraded learning set
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y_degraded)

    contents1 = export_graphviz(clf, filled=True, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="gini = 0.0\\nsamples = 6\\nvalue = 6.0", '
        'fillcolor="#ffffff"] ;\n'
        "}"
    )


def test_graphviz_errors():
    # Check for errors of export_graphviz
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)

    # Check not-fitted decision tree error
    out = StringIO()
    with pytest.raises(NotFittedError):
        export_graphviz(clf, out)

    clf.fit(X, y)

    # Check if it errors when length of feature_names
    # mismatches with number of features
    message = "Length of feature_names, 1 does not match number of features, 2"
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a"])

    message = "Length of feature_names, 3 does not match number of features, 2"
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a", "b", "c"])

    # Check error when argument is not an estimator
    message = "is not an estimator instance"
    with pytest.raises(TypeError, match=message):
        export_graphviz(clf.fit(X, y).tree_)

    # Check class_names error
    out = StringIO()
    with pytest.raises(IndexError):
        export_graphviz(clf, out, class_names=[])

    # Check precision error
    out = StringIO()
    with pytest.raises(ValueError, match="should be greater or equal"):
        export_graphviz(clf, out, precision=-1)
    with pytest.raises(ValueError, match="should be an integer"):
        export_graphviz(clf, out, precision="1")


def test_friedman_mse_in_graphviz():
    clf = DecisionTreeRegressor(criterion="friedman_mse", random_state=0)
    clf.fit(X, y)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data)

    clf = GradientBoostingClassifier(n_estimators=2, random_state=0)
    clf.fit(X, y)
    for estimator in clf.estimators_:
        export_graphviz(estimator[0], out_file=dot_data)

    for finding in finditer(r"\[.*?samples.*?\]", dot_data.getvalue()):
        assert "friedman_mse" in finding.group()


def test_precision():

    rng_reg = RandomState(2)
    rng_clf = RandomState(8)
    for X, y, clf in zip(
        (rng_reg.random_sample((5, 2)), rng_clf.random_sample((1000, 4))),
        (rng_reg.random_sample((5,)), rng_clf.randint(2, size=(1000,))),
        (
            DecisionTreeRegressor(
                criterion="friedman_mse", random_state=0, max_depth=1
            ),
            DecisionTreeClassifier(max_depth=1, random_state=0),
        ),
    ):

        clf.fit(X, y)
        for precision in (4, 3):
            dot_data = export_graphviz(
                clf, out_file=None, precision=precision, proportion=True
            )

            # With the current random state, the impurity and the threshold
            # will have the number of precision set in the export_graphviz
            # function. We will check the number of precision with a strict
            # equality. The value reported will have only 2 precision and
            # therefore, only a less equal comparison will be done.

            # check value
            for finding in finditer(r"value = \d+\.\d+", dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) <= precision + 1
            # check impurity
            if is_classifier(clf):
                pattern = r"gini = \d+\.\d+"
            else:
                pattern = r"friedman_mse = \d+\.\d+"

            # check impurity
            for finding in finditer(pattern, dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) == precision + 1
            # check threshold
            for finding in finditer(r"<= \d+\.\d+", dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) == precision + 1


def test_export_text_errors():
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    err_msg = "max_depth bust be >= 0, given -1"
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, max_depth=-1)
    err_msg = "feature_names must contain 2 elements, got 1"
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, feature_names=["a"])
    err_msg = "decimals must be >= 0, given -1"
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, decimals=-1)
    err_msg = "spacing must be > 0, given 0"
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, spacing=0)


def test_export_text():
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: -1
    |--- feature_1 >  0.00
    |   |--- class: 1
    """
    ).lstrip()

    assert export_text(clf) == expected_report
    # testing that leaves at level 1 are not truncated
    assert export_text(clf, max_depth=0) == expected_report
    # testing that the rest of the tree is truncated
    assert export_text(clf, max_depth=10) == expected_report

    expected_report = dedent(
        """
    |--- b <= 0.00
    |   |--- class: -1
    |--- b >  0.00
    |   |--- class: 1
    """
    ).lstrip()
    assert export_text(clf, feature_names=["a", "b"]) == expected_report

    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- weights: [3.00, 0.00] class: -1
    |--- feature_1 >  0.00
    |   |--- weights: [0.00, 3.00] class: 1
    """
    ).lstrip()
    assert export_text(clf, show_weights=True) == expected_report

    expected_report = dedent(
        """
    |- feature_1 <= 0.00
    | |- class: -1
    |- feature_1 >  0.00
    | |- class: 1
    """
    ).lstrip()
    assert export_text(clf, spacing=1) == expected_report

    X_l = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-1, 1]]
    y_l = [-1, -1, -1, 1, 1, 1, 2]
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(X_l, y_l)
    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: -1
    |--- feature_1 >  0.00
    |   |--- truncated branch of depth 2
    """
    ).lstrip()
    assert export_text(clf, max_depth=0) == expected_report

    X_mo = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y_mo = [[-1, -1], [-1, -1], [-1, -1], [1, 1], [1, 1], [1, 1]]

    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X_mo, y_mo)

    expected_report = dedent(
        """
    |--- feature_1 <= 0.0
    |   |--- value: [-1.0, -1.0]
    |--- feature_1 >  0.0
    |   |--- value: [1.0, 1.0]
    """
    ).lstrip()
    assert export_text(reg, decimals=1) == expected_report
    assert export_text(reg, decimals=1, show_weights=True) == expected_report

    X_single = [[-2], [-1], [-1], [1], [1], [2]]
    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X_single, y_mo)

    expected_report = dedent(
        """
    |--- first <= 0.0
    |   |--- value: [-1.0, -1.0]
    |--- first >  0.0
    |   |--- value: [1.0, 1.0]
    """
    ).lstrip()
    assert export_text(reg, decimals=1, feature_names=["first"]) == expected_report
    assert (
        export_text(reg, decimals=1, show_weights=True, feature_names=["first"])
        == expected_report
    )


def test_plot_tree_entropy(pyplot):
    # mostly smoke tests
    # Check correctness of export_graphviz for criterion = entropy
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="entropy", random_state=2
    )
    clf.fit(X, y)

    # Test export code
    feature_names = ["first feat", "sepal_width"]
    nodes = plot_tree(clf, feature_names=feature_names)
    assert len(nodes) == 3
    assert (
        nodes[0].get_text()
        == "first feat <= 0.0\nentropy = 1.0\nsamples = 6\nvalue = [3, 3]"
    )
    assert nodes[1].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert nodes[2].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [0, 3]"


def test_plot_tree_gini(pyplot):
    # mostly smoke tests
    # Check correctness of export_graphviz for criterion = gini
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    clf.fit(X, y)

    # Test export code
    feature_names = ["first feat", "sepal_width"]
    nodes = plot_tree(clf, feature_names=feature_names)
    assert len(nodes) == 3
    assert (
        nodes[0].get_text()
        == "first feat <= 0.0\ngini = 0.5\nsamples = 6\nvalue = [3, 3]"
    )
    assert nodes[1].get_text() == "gini = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert nodes[2].get_text() == "gini = 0.0\nsamples = 3\nvalue = [0, 3]"


def test_not_fitted_tree(pyplot):

    # Testing if not fitted tree throws the correct error
    clf = DecisionTreeRegressor()
    with pytest.raises(NotFittedError):
        plot_tree(clf)
