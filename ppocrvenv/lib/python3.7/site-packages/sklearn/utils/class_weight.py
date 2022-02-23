# Authors: Andreas Mueller
#          Manoj Kumar
# License: BSD 3 clause

import numpy as np


def compute_class_weight(class_weight, *, classes, y):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'balanced' or None
        If 'balanced', class weights will be given by
        ``n_samples / (n_classes * np.bincount(y))``.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        ``np.unique(y_org)`` with ``y_org`` the original class labels.

    y : array-like of shape (n_samples,)
        Array of original class labels per sample.

    Returns
    -------
    class_weight_vect : ndarray of shape (n_classes,)
        Array with class_weight_vect[i] the weight for i-th class.

    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.
    """
    # Import error caused by circular imports.
    from ..preprocessing import LabelEncoder

    if set(y) - set(classes):
        raise ValueError("classes should include all valid labels that can be in y")
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(classes.shape[0], dtype=np.float64, order="C")
    elif class_weight == "balanced":
        # Find the weight of each class as present in y.
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]
    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order="C")
        if not isinstance(class_weight, dict):
            raise ValueError(
                "class_weight must be dict, 'balanced', or None, got: %r" % class_weight
            )
        for c in class_weight:
            i = np.searchsorted(classes, c)
            if i >= len(classes) or classes[i] != c:
                raise ValueError("Class label {} not present.".format(c))
            else:
                weight[i] = class_weight[c]

    return weight


def compute_sample_weight(class_weight, y, *, indices=None):
    """Estimate sample weights by class for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, list of dicts, "balanced", or None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data:
        ``n_samples / (n_classes * np.bincount(y))``.

        For multi-output, the weights of each column of y will be multiplied.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Array of original class labels per sample.

    indices : array-like of shape (n_subsample,), default=None
        Array of indices to be used in a subsample. Can be of length less than
        n_samples in the case of a subsample, or equal to n_samples in the
        case of a bootstrap subsample with repeated indices. If None, the
        sample weight will be calculated over the full sample. Only "balanced"
        is supported for class_weight if this is provided.

    Returns
    -------
    sample_weight_vect : ndarray of shape (n_samples,)
        Array with sample weights as applied to the original y.
    """

    y = np.atleast_1d(y)
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))
    n_outputs = y.shape[1]

    if isinstance(class_weight, str):
        if class_weight not in ["balanced"]:
            raise ValueError(
                'The only valid preset for class_weight is "balanced". Given "%s".'
                % class_weight
            )
    elif indices is not None and not isinstance(class_weight, str):
        raise ValueError(
            'The only valid class_weight for subsampling is "balanced". Given "%s".'
            % class_weight
        )
    elif n_outputs > 1:
        if not hasattr(class_weight, "__iter__") or isinstance(class_weight, dict):
            raise ValueError(
                "For multi-output, class_weight should be a "
                "list of dicts, or a valid string."
            )
        if len(class_weight) != n_outputs:
            raise ValueError(
                "For multi-output, number of elements in "
                "class_weight should match number of outputs."
            )

    expanded_class_weight = []
    for k in range(n_outputs):

        y_full = y[:, k]
        classes_full = np.unique(y_full)
        classes_missing = None

        if class_weight == "balanced" or n_outputs == 1:
            class_weight_k = class_weight
        else:
            class_weight_k = class_weight[k]

        if indices is not None:
            # Get class weights for the subsample, covering all classes in
            # case some labels that were present in the original data are
            # missing from the sample.
            y_subsample = y[indices, k]
            classes_subsample = np.unique(y_subsample)

            weight_k = np.take(
                compute_class_weight(
                    class_weight_k, classes=classes_subsample, y=y_subsample
                ),
                np.searchsorted(classes_subsample, classes_full),
                mode="clip",
            )

            classes_missing = set(classes_full) - set(classes_subsample)
        else:
            weight_k = compute_class_weight(
                class_weight_k, classes=classes_full, y=y_full
            )

        weight_k = weight_k[np.searchsorted(classes_full, y_full)]

        if classes_missing:
            # Make missing classes' weight zero
            weight_k[np.in1d(y_full, list(classes_missing))] = 0.0

        expanded_class_weight.append(weight_k)

    expanded_class_weight = np.prod(expanded_class_weight, axis=0, dtype=np.float64)

    return expanded_class_weight
