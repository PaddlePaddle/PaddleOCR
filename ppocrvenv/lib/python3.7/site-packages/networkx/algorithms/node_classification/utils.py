def _propagate(P, F, B):
    """Propagate labels by one step

    Parameters
    ----------
    P : scipy sparse matrix, shape = [n_samples, n_samples]
        Propagation matrix
    F : numpy array, shape = [n_samples, n_classes]
        Label matrix
    B : numpy array, shape = [n_samples, n_classes]
        Base matrix

    Returns
    ----------
    F_new : array, shape = [n_samples, n_classes]
        Label matrix
    """
    F_new = (P @ F) + B
    return F_new


def _get_label_info(G, label_name):
    """Get and return information of labels from the input graph

    Parameters
    ----------
    G : Network X graph
    label_name : string
        Name of the target label

    Returns
    ----------
    labels : numpy array, shape = [n_labeled_samples, 2]
        Array of pairs of labeled node ID and label ID
    label_dict : numpy array, shape = [n_classes]
        Array of labels
        i-th element contains the label corresponding label ID `i`
    """
    import numpy as np

    labels = []
    label_to_id = {}
    lid = 0
    for i, n in enumerate(G.nodes(data=True)):
        if label_name in n[1]:
            label = n[1][label_name]
            if label not in label_to_id:
                label_to_id[label] = lid
                lid += 1
            labels.append([i, label_to_id[label]])
    labels = np.array(labels)
    label_dict = np.array(
        [label for label, _ in sorted(label_to_id.items(), key=lambda x: x[1])]
    )
    return (labels, label_dict)


def _init_label_matrix(n_samples, n_classes):
    """Create and return zero matrix

    Parameters
    ----------
    n_samples : integer
        The number of nodes (samples) on the input graph
    n_classes : integer
        The number of classes (distinct labels) on the input graph

    Returns
    ----------
    F : numpy array, shape = [n_samples, n_classes]
        Label matrix
    """
    import numpy as np

    F = np.zeros((n_samples, n_classes))
    return F


def _predict(F, label_dict):
    """Predict labels by learnt label matrix

    Parameters
    ----------
    F : numpy array, shape = [n_samples, n_classes]
        Learnt (resulting) label matrix
    label_dict : numpy array, shape = [n_classes]
        Array of labels
        i-th element contains the label corresponding label ID `i`

    Returns
    ----------
    predicted : numpy array, shape = [n_samples]
        Array of predicted labels
    """
    import numpy as np

    predicted_label_ids = np.argmax(F, axis=1)
    predicted = label_dict[predicted_label_ids].tolist()
    return predicted
