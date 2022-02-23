import re
from inspect import signature
import pkgutil
import inspect
import importlib
from typing import Optional

import pytest
from sklearn.utils import all_estimators
import sklearn

numpydoc_validation = pytest.importorskip("numpydoc.validate")

FUNCTION_DOCSTRING_IGNORE_LIST = [
    "sklearn.cluster._kmeans.kmeans_plusplus",
    "sklearn.cluster._optics.cluster_optics_xi",
    "sklearn.cluster._optics.compute_optics_graph",
    "sklearn.cluster._spectral.spectral_clustering",
    "sklearn.compose._column_transformer.make_column_transformer",
    "sklearn.covariance._graph_lasso.graphical_lasso",
    "sklearn.covariance._robust_covariance.fast_mcd",
    "sklearn.covariance._shrunk_covariance.ledoit_wolf",
    "sklearn.covariance._shrunk_covariance.ledoit_wolf_shrinkage",
    "sklearn.covariance._shrunk_covariance.shrunk_covariance",
    "sklearn.datasets._base.get_data_home",
    "sklearn.datasets._base.load_boston",
    "sklearn.datasets._base.load_breast_cancer",
    "sklearn.datasets._base.load_digits",
    "sklearn.datasets._base.load_linnerud",
    "sklearn.datasets._base.load_sample_image",
    "sklearn.datasets._base.load_wine",
    "sklearn.datasets._california_housing.fetch_california_housing",
    "sklearn.datasets._covtype.fetch_covtype",
    "sklearn.datasets._kddcup99.fetch_kddcup99",
    "sklearn.datasets._lfw.fetch_lfw_pairs",
    "sklearn.datasets._lfw.fetch_lfw_people",
    "sklearn.datasets._olivetti_faces.fetch_olivetti_faces",
    "sklearn.datasets._openml.fetch_openml",
    "sklearn.datasets._rcv1.fetch_rcv1",
    "sklearn.datasets._samples_generator.make_biclusters",
    "sklearn.datasets._samples_generator.make_blobs",
    "sklearn.datasets._samples_generator.make_checkerboard",
    "sklearn.datasets._samples_generator.make_classification",
    "sklearn.datasets._samples_generator.make_gaussian_quantiles",
    "sklearn.datasets._samples_generator.make_hastie_10_2",
    "sklearn.datasets._samples_generator.make_multilabel_classification",
    "sklearn.datasets._samples_generator.make_regression",
    "sklearn.datasets._samples_generator.make_sparse_coded_signal",
    "sklearn.datasets._samples_generator.make_sparse_spd_matrix",
    "sklearn.datasets._samples_generator.make_spd_matrix",
    "sklearn.datasets._species_distributions.fetch_species_distributions",
    "sklearn.datasets._svmlight_format_io.dump_svmlight_file",
    "sklearn.datasets._svmlight_format_io.load_svmlight_file",
    "sklearn.datasets._svmlight_format_io.load_svmlight_files",
    "sklearn.datasets._twenty_newsgroups.fetch_20newsgroups",
    "sklearn.decomposition._dict_learning.dict_learning",
    "sklearn.decomposition._dict_learning.dict_learning_online",
    "sklearn.decomposition._dict_learning.sparse_encode",
    "sklearn.decomposition._fastica.fastica",
    "sklearn.decomposition._nmf.non_negative_factorization",
    "sklearn.externals._packaging.version.parse",
    "sklearn.feature_extraction.image.extract_patches_2d",
    "sklearn.feature_extraction.image.grid_to_graph",
    "sklearn.feature_extraction.image.img_to_graph",
    "sklearn.feature_extraction.text.strip_accents_ascii",
    "sklearn.feature_extraction.text.strip_accents_unicode",
    "sklearn.feature_extraction.text.strip_tags",
    "sklearn.feature_selection._univariate_selection.chi2",
    "sklearn.feature_selection._univariate_selection.f_oneway",
    "sklearn.feature_selection._univariate_selection.r_regression",
    "sklearn.inspection._partial_dependence.partial_dependence",
    "sklearn.inspection._plot.partial_dependence.plot_partial_dependence",
    "sklearn.isotonic.isotonic_regression",
    "sklearn.linear_model._least_angle.lars_path",
    "sklearn.linear_model._least_angle.lars_path_gram",
    "sklearn.linear_model._omp.orthogonal_mp",
    "sklearn.linear_model._omp.orthogonal_mp_gram",
    "sklearn.linear_model._ridge.ridge_regression",
    "sklearn.manifold._locally_linear.locally_linear_embedding",
    "sklearn.manifold._t_sne.trustworthiness",
    "sklearn.metrics._classification.brier_score_loss",
    "sklearn.metrics._classification.classification_report",
    "sklearn.metrics._classification.cohen_kappa_score",
    "sklearn.metrics._classification.f1_score",
    "sklearn.metrics._classification.fbeta_score",
    "sklearn.metrics._classification.hinge_loss",
    "sklearn.metrics._classification.jaccard_score",
    "sklearn.metrics._classification.log_loss",
    "sklearn.metrics._classification.precision_recall_fscore_support",
    "sklearn.metrics._plot.confusion_matrix.plot_confusion_matrix",
    "sklearn.metrics._plot.det_curve.plot_det_curve",
    "sklearn.metrics._plot.precision_recall_curve.plot_precision_recall_curve",
    "sklearn.metrics._ranking.auc",
    "sklearn.metrics._ranking.average_precision_score",
    "sklearn.metrics._ranking.coverage_error",
    "sklearn.metrics._ranking.dcg_score",
    "sklearn.metrics._ranking.label_ranking_average_precision_score",
    "sklearn.metrics._ranking.label_ranking_loss",
    "sklearn.metrics._ranking.ndcg_score",
    "sklearn.metrics._ranking.precision_recall_curve",
    "sklearn.metrics._ranking.roc_auc_score",
    "sklearn.metrics._ranking.roc_curve",
    "sklearn.metrics._ranking.top_k_accuracy_score",
    "sklearn.metrics._regression.mean_absolute_error",
    "sklearn.metrics._regression.mean_pinball_loss",
    "sklearn.metrics._scorer.make_scorer",
    "sklearn.metrics.cluster._bicluster.consensus_score",
    "sklearn.metrics.cluster._supervised.adjusted_mutual_info_score",
    "sklearn.metrics.cluster._supervised.adjusted_rand_score",
    "sklearn.metrics.cluster._supervised.completeness_score",
    "sklearn.metrics.cluster._supervised.entropy",
    "sklearn.metrics.cluster._supervised.fowlkes_mallows_score",
    "sklearn.metrics.cluster._supervised.homogeneity_completeness_v_measure",
    "sklearn.metrics.cluster._supervised.homogeneity_score",
    "sklearn.metrics.cluster._supervised.mutual_info_score",
    "sklearn.metrics.cluster._supervised.normalized_mutual_info_score",
    "sklearn.metrics.cluster._supervised.pair_confusion_matrix",
    "sklearn.metrics.cluster._supervised.rand_score",
    "sklearn.metrics.cluster._supervised.v_measure_score",
    "sklearn.metrics.pairwise.additive_chi2_kernel",
    "sklearn.metrics.pairwise.check_paired_arrays",
    "sklearn.metrics.pairwise.check_pairwise_arrays",
    "sklearn.metrics.pairwise.chi2_kernel",
    "sklearn.metrics.pairwise.cosine_distances",
    "sklearn.metrics.pairwise.cosine_similarity",
    "sklearn.metrics.pairwise.distance_metrics",
    "sklearn.metrics.pairwise.haversine_distances",
    "sklearn.metrics.pairwise.kernel_metrics",
    "sklearn.metrics.pairwise.laplacian_kernel",
    "sklearn.metrics.pairwise.manhattan_distances",
    "sklearn.metrics.pairwise.nan_euclidean_distances",
    "sklearn.metrics.pairwise.paired_cosine_distances",
    "sklearn.metrics.pairwise.paired_distances",
    "sklearn.metrics.pairwise.paired_euclidean_distances",
    "sklearn.metrics.pairwise.paired_manhattan_distances",
    "sklearn.metrics.pairwise.pairwise_distances_argmin",
    "sklearn.metrics.pairwise.pairwise_distances_argmin_min",
    "sklearn.metrics.pairwise.pairwise_distances_chunked",
    "sklearn.metrics.pairwise.pairwise_kernels",
    "sklearn.metrics.pairwise.polynomial_kernel",
    "sklearn.metrics.pairwise.rbf_kernel",
    "sklearn.metrics.pairwise.sigmoid_kernel",
    "sklearn.model_selection._split.check_cv",
    "sklearn.model_selection._validation.cross_validate",
    "sklearn.model_selection._validation.learning_curve",
    "sklearn.model_selection._validation.permutation_test_score",
    "sklearn.model_selection._validation.validation_curve",
    "sklearn.neighbors._graph.kneighbors_graph",
    "sklearn.neighbors._graph.radius_neighbors_graph",
    "sklearn.pipeline.make_union",
    "sklearn.preprocessing._data.binarize",
    "sklearn.preprocessing._data.maxabs_scale",
    "sklearn.preprocessing._data.normalize",
    "sklearn.preprocessing._data.power_transform",
    "sklearn.preprocessing._data.quantile_transform",
    "sklearn.preprocessing._data.robust_scale",
    "sklearn.preprocessing._data.scale",
    "sklearn.preprocessing._label.label_binarize",
    "sklearn.random_projection.johnson_lindenstrauss_min_dim",
    "sklearn.svm._bounds.l1_min_c",
    "sklearn.tree._export.plot_tree",
    "sklearn.utils.axis0_safe_slice",
    "sklearn.utils.extmath.density",
    "sklearn.utils.extmath.fast_logdet",
    "sklearn.utils.extmath.randomized_svd",
    "sklearn.utils.extmath.safe_sparse_dot",
    "sklearn.utils.extmath.squared_norm",
    "sklearn.utils.extmath.stable_cumsum",
    "sklearn.utils.extmath.svd_flip",
    "sklearn.utils.extmath.weighted_mode",
    "sklearn.utils.fixes.delayed",
    "sklearn.utils.fixes.linspace",
    # To be fixed in upstream issue:
    # https://github.com/joblib/threadpoolctl/issues/108
    "sklearn.utils.fixes.threadpool_info",
    "sklearn.utils.fixes.threadpool_limits",
    "sklearn.utils.gen_batches",
    "sklearn.utils.gen_even_slices",
    "sklearn.utils.get_chunk_n_rows",
    "sklearn.utils.graph.graph_shortest_path",
    "sklearn.utils.graph.single_source_shortest_path_length",
    "sklearn.utils.is_scalar_nan",
    "sklearn.utils.metaestimators.available_if",
    "sklearn.utils.metaestimators.if_delegate_has_method",
    "sklearn.utils.multiclass.check_classification_targets",
    "sklearn.utils.multiclass.class_distribution",
    "sklearn.utils.multiclass.type_of_target",
    "sklearn.utils.multiclass.unique_labels",
    "sklearn.utils.resample",
    "sklearn.utils.safe_mask",
    "sklearn.utils.safe_sqr",
    "sklearn.utils.shuffle",
    "sklearn.utils.sparsefuncs.count_nonzero",
    "sklearn.utils.sparsefuncs.csc_median_axis_0",
    "sklearn.utils.sparsefuncs.incr_mean_variance_axis",
    "sklearn.utils.sparsefuncs.inplace_swap_column",
    "sklearn.utils.sparsefuncs.inplace_swap_row",
    "sklearn.utils.sparsefuncs.inplace_swap_row_csc",
    "sklearn.utils.sparsefuncs.inplace_swap_row_csr",
    "sklearn.utils.sparsefuncs.mean_variance_axis",
    "sklearn.utils.sparsefuncs.min_max_axis",
    "sklearn.utils.tosequence",
    "sklearn.utils.validation.assert_all_finite",
    "sklearn.utils.validation.check_is_fitted",
    "sklearn.utils.validation.check_memory",
    "sklearn.utils.validation.check_random_state",
]
FUNCTION_DOCSTRING_IGNORE_LIST = set(FUNCTION_DOCSTRING_IGNORE_LIST)


def get_all_methods():
    estimators = all_estimators()
    for name, Estimator in estimators:
        if name.startswith("_"):
            # skip private classes
            continue
        methods = []
        for name in dir(Estimator):
            if name.startswith("_"):
                continue
            method_obj = getattr(Estimator, name)
            if hasattr(method_obj, "__call__") or isinstance(method_obj, property):
                methods.append(name)
        methods.append(None)

        for method in sorted(methods, key=lambda x: str(x)):
            yield Estimator, method


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    if not mod.startswith("sklearn.") or mod.endswith("estimator_checks"):
        return False

    return True


def get_all_functions_names():
    """Get all public functions define in the sklearn module"""
    modules_to_ignore = {
        "tests",
        "externals",
        "setup",
        "conftest",
        "experimental",
        "estimator_checks",
    }

    all_functions_names = set()
    for module_finder, module_name, ispkg in pkgutil.walk_packages(
        path=sklearn.__path__, prefix="sklearn."
    ):
        module_parts = module_name.split(".")
        if (
            any(part in modules_to_ignore for part in module_parts)
            or "._" in module_name
        ):
            continue

        module = importlib.import_module(module_name)
        functions = inspect.getmembers(module, _is_checked_function)
        for name, func in functions:
            full_name = f"{func.__module__}.{func.__name__}"
            all_functions_names.add(full_name)

    return sorted(all_functions_names)


def filter_errors(errors, method, Estimator=None):
    """
    Ignore some errors based on the method type.

    These rules are specific for scikit-learn."""
    for code, message in errors:
        # We ignore following error code,
        #  - RT02: The first line of the Returns section
        #    should contain only the type, ..
        #   (as we may need refer to the name of the returned
        #    object)
        #  - GL01: Docstring text (summary) should start in the line
        #    immediately after the opening quotes (not in the same line,
        #    or leaving a blank line in between)
        #  - GL02: If there's a blank line, it should be before the
        #    first line of the Returns section, not after (it allows to have
        #    short docstrings for properties).

        if code in ["RT02", "GL01", "GL02"]:
            continue

        # Ignore PR02: Unknown parameters for properties. We sometimes use
        # properties for ducktyping, i.e. SGDClassifier.predict_proba
        if code == "PR02" and Estimator is not None and method is not None:
            method_obj = getattr(Estimator, method)
            if isinstance(method_obj, property):
                continue

        # Following codes are only taken into account for the
        # top level class docstrings:
        #  - ES01: No extended summary found
        #  - SA01: See Also section not found
        #  - EX01: No examples section found

        if method is not None and code in ["EX01", "SA01", "ES01"]:
            continue
        yield code, message


def repr_errors(res, estimator=None, method: Optional[str] = None) -> str:
    """Pretty print original docstring and the obtained errors

    Parameters
    ----------
    res : dict
        result of numpydoc.validate.validate
    estimator : {estimator, None}
        estimator object or None
    method : str
        if estimator is not None, either the method name or None.

    Returns
    -------
    str
       String representation of the error.
    """
    if method is None:
        if hasattr(estimator, "__init__"):
            method = "__init__"
        elif estimator is None:
            raise ValueError("At least one of estimator, method should be provided")
        else:
            raise NotImplementedError

    if estimator is not None:
        obj = getattr(estimator, method)
        try:
            obj_signature = str(signature(obj))
        except TypeError:
            # In particular we can't parse the signature of properties
            obj_signature = (
                "\nParsing of the method signature failed, "
                "possibly because this is a property."
            )

        obj_name = estimator.__name__ + "." + method
    else:
        obj_signature = ""
        obj_name = method

    msg = "\n\n" + "\n\n".join(
        [
            str(res["file"]),
            obj_name + obj_signature,
            res["docstring"],
            "# Errors",
            "\n".join(
                " - {}: {}".format(code, message) for code, message in res["errors"]
            ),
        ]
    )
    return msg


@pytest.mark.parametrize("function_name", get_all_functions_names())
def test_function_docstring(function_name, request):
    """Check function docstrings using numpydoc."""
    if function_name in FUNCTION_DOCSTRING_IGNORE_LIST:
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )

    res = numpydoc_validation.validate(function_name)

    res["errors"] = list(filter_errors(res["errors"], method="function"))

    if res["errors"]:
        msg = repr_errors(res, method=f"Tested function: {function_name}")

        raise ValueError(msg)


@pytest.mark.parametrize("Estimator, method", get_all_methods())
def test_docstring(Estimator, method, request):
    base_import_path = Estimator.__module__
    import_path = [base_import_path, Estimator.__name__]
    if method is not None:
        import_path.append(method)

    import_path = ".".join(import_path)

    res = numpydoc_validation.validate(import_path)

    res["errors"] = list(filter_errors(res["errors"], method, Estimator=Estimator))

    if res["errors"]:
        msg = repr_errors(res, Estimator, method)

        raise ValueError(msg)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Validate docstring with numpydoc.")
    parser.add_argument("import_path", help="Import path to validate")

    args = parser.parse_args()

    res = numpydoc_validation.validate(args.import_path)

    import_path_sections = args.import_path.split(".")
    # When applied to classes, detect class method. For functions
    # method = None.
    # TODO: this detection can be improved. Currently we assume that we have
    # class # methods if the second path element before last is in camel case.
    if len(import_path_sections) >= 2 and re.match(
        r"(?:[A-Z][a-z]*)+", import_path_sections[-2]
    ):
        method = import_path_sections[-1]
    else:
        method = None

    res["errors"] = list(filter_errors(res["errors"], method))

    if res["errors"]:
        msg = repr_errors(res, method=args.import_path)

        print(msg)
        sys.exit(1)
    else:
        print("All docstring checks passed for {}!".format(args.import_path))
