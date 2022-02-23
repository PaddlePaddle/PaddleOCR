import gzip
import json
import os
import shutil
import hashlib
from os.path import join
from warnings import warn
from contextlib import closing
from functools import wraps
from typing import Callable, Optional, Dict, Tuple, List, Any, Union
import itertools
from collections.abc import Generator
from collections import OrderedDict
from functools import partial
from tempfile import TemporaryDirectory

from urllib.request import urlopen, Request

import numpy as np
import scipy.sparse

from ..externals import _arff
from ..externals._arff import ArffSparseDataType, ArffContainerType
from . import get_data_home
from urllib.error import HTTPError
from ..utils import Bunch
from ..utils import is_scalar_nan
from ..utils import get_chunk_n_rows
from ..utils import _chunk_generator
from ..utils import check_pandas_support  # noqa

__all__ = ["fetch_openml"]

_OPENML_PREFIX = "https://openml.org/"
_SEARCH_NAME = "api/v1/json/data/list/data_name/{}/limit/2"
_DATA_INFO = "api/v1/json/data/{}"
_DATA_FEATURES = "api/v1/json/data/features/{}"
_DATA_QUALITIES = "api/v1/json/data/qualities/{}"
_DATA_FILE = "data/v1/download/{}"

OpenmlQualitiesType = List[Dict[str, str]]
OpenmlFeaturesType = List[Dict[str, str]]


def _get_local_path(openml_path: str, data_home: str) -> str:
    return os.path.join(data_home, "openml.org", openml_path + ".gz")


def _retry_with_clean_cache(openml_path: str, data_home: Optional[str]) -> Callable:
    """If the first call to the decorated function fails, the local cached
    file is removed, and the function is called again. If ``data_home`` is
    ``None``, then the function is called once.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw):
            if data_home is None:
                return f(*args, **kw)
            try:
                return f(*args, **kw)
            except HTTPError:
                raise
            except Exception:
                warn("Invalid cache, redownloading file", RuntimeWarning)
                local_path = _get_local_path(openml_path, data_home)
                if os.path.exists(local_path):
                    os.unlink(local_path)
                return f(*args, **kw)

        return wrapper

    return decorator


def _open_openml_url(openml_path: str, data_home: Optional[str]):
    """
    Returns a resource from OpenML.org. Caches it to data_home if required.

    Parameters
    ----------
    openml_path : str
        OpenML URL that will be accessed. This will be prefixes with
        _OPENML_PREFIX

    data_home : str
        Directory to which the files will be cached. If None, no caching will
        be applied.

    Returns
    -------
    result : stream
        A stream to the OpenML resource
    """

    def is_gzip_encoded(_fsrc):
        return _fsrc.info().get("Content-Encoding", "") == "gzip"

    req = Request(_OPENML_PREFIX + openml_path)
    req.add_header("Accept-encoding", "gzip")

    if data_home is None:
        fsrc = urlopen(req)
        if is_gzip_encoded(fsrc):
            return gzip.GzipFile(fileobj=fsrc, mode="rb")
        return fsrc

    local_path = _get_local_path(openml_path, data_home)
    dir_name, file_name = os.path.split(local_path)
    if not os.path.exists(local_path):
        os.makedirs(dir_name, exist_ok=True)
        try:
            # Create a tmpdir as a subfolder of dir_name where the final file will
            # be moved to if the download is successful. This guarantees that the
            # renaming operation to the final location is atomic to ensure the
            # concurrence safety of the dataset caching mechanism.
            with TemporaryDirectory(dir=dir_name) as tmpdir:
                with closing(urlopen(req)) as fsrc:
                    opener: Callable
                    if is_gzip_encoded(fsrc):
                        opener = open
                    else:
                        opener = gzip.GzipFile
                    with opener(os.path.join(tmpdir, file_name), "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                shutil.move(fdst.name, local_path)
        except Exception:
            if os.path.exists(local_path):
                os.unlink(local_path)
            raise

    # XXX: First time, decompression will not be necessary (by using fsrc), but
    # it will happen nonetheless
    return gzip.GzipFile(local_path, "rb")


class OpenMLError(ValueError):
    """HTTP 412 is a specific OpenML error code, indicating a generic error"""

    pass


def _get_json_content_from_openml_api(
    url: str, error_message: Optional[str], data_home: Optional[str]
) -> Dict:
    """
    Loads json data from the openml api

    Parameters
    ----------
    url : str
        The URL to load from. Should be an official OpenML endpoint

    error_message : str or None
        The error message to raise if an acceptable OpenML error is thrown
        (acceptable error is, e.g., data id not found. Other errors, like 404's
        will throw the native error message)

    data_home : str or None
        Location to cache the response. None if no cache is required.

    Returns
    -------
    json_data : json
        the json result from the OpenML server if the call was successful.
        An exception otherwise.
    """

    @_retry_with_clean_cache(url, data_home)
    def _load_json():
        with closing(_open_openml_url(url, data_home)) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return _load_json()
    except HTTPError as error:
        # 412 is an OpenML specific error code, indicating a generic error
        # (e.g., data not found)
        if error.code != 412:
            raise error

    # 412 error, not in except for nicer traceback
    raise OpenMLError(error_message)


def _split_sparse_columns(
    arff_data: ArffSparseDataType, include_columns: List
) -> ArffSparseDataType:
    """
    obtains several columns from sparse arff representation. Additionally, the
    column indices are re-labelled, given the columns that are not included.
    (e.g., when including [1, 2, 3], the columns will be relabelled to
    [0, 1, 2])

    Parameters
    ----------
    arff_data : tuple
        A tuple of three lists of equal size; first list indicating the value,
        second the x coordinate and the third the y coordinate.

    include_columns : list
        A list of columns to include.

    Returns
    -------
    arff_data_new : tuple
        Subset of arff data with only the include columns indicated by the
        include_columns argument.
    """
    arff_data_new: ArffSparseDataType = (list(), list(), list())
    reindexed_columns = {
        column_idx: array_idx for array_idx, column_idx in enumerate(include_columns)
    }
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            arff_data_new[0].append(val)
            arff_data_new[1].append(row_idx)
            arff_data_new[2].append(reindexed_columns[col_idx])
    return arff_data_new


def _sparse_data_to_array(
    arff_data: ArffSparseDataType, include_columns: List
) -> np.ndarray:
    # turns the sparse data back into an array (can't use toarray() function,
    # as this does only work on numeric data)
    num_obs = max(arff_data[1]) + 1
    y_shape = (num_obs, len(include_columns))
    reindexed_columns = {
        column_idx: array_idx for array_idx, column_idx in enumerate(include_columns)
    }
    # TODO: improve for efficiency
    y = np.empty(y_shape, dtype=np.float64)
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            y[row_idx, reindexed_columns[col_idx]] = val
    return y


def _convert_arff_data(
    arff: ArffContainerType,
    col_slice_x: List[int],
    col_slice_y: List[int],
    shape: Optional[Tuple] = None,
) -> Tuple:
    """
    converts the arff object into the appropriate matrix type (np.array or
    scipy.sparse.csr_matrix) based on the 'data part' (i.e., in the
    liac-arff dict, the object from the 'data' key)

    Parameters
    ----------
    arff : dict
        As obtained from liac-arff object.

    col_slice_x : list
        The column indices that are sliced from the original array to return
        as X data

    col_slice_y : list
        The column indices that are sliced from the original array to return
        as y data

    Returns
    -------
    X : np.array or scipy.sparse.csr_matrix
    y : np.array
    """
    arff_data = arff["data"]
    if isinstance(arff_data, Generator):
        if shape is None:
            raise ValueError("shape must be provided when arr['data'] is a Generator")
        if shape[0] == -1:
            count = -1
        else:
            count = shape[0] * shape[1]
        data = np.fromiter(
            itertools.chain.from_iterable(arff_data), dtype="float64", count=count
        )
        data = data.reshape(*shape)
        X = data[:, col_slice_x]
        y = data[:, col_slice_y]
        return X, y
    elif isinstance(arff_data, tuple):
        arff_data_X = _split_sparse_columns(arff_data, col_slice_x)
        num_obs = max(arff_data[1]) + 1
        X_shape = (num_obs, len(col_slice_x))
        X = scipy.sparse.coo_matrix(
            (arff_data_X[0], (arff_data_X[1], arff_data_X[2])),
            shape=X_shape,
            dtype=np.float64,
        )
        X = X.tocsr()
        y = _sparse_data_to_array(arff_data, col_slice_y)
        return X, y
    else:
        # This should never happen
        raise ValueError("Unexpected Data Type obtained from arff.")


def _feature_to_dtype(feature: Dict[str, str]):
    """Map feature to dtype for pandas DataFrame"""
    if feature["data_type"] == "string":
        return object
    elif feature["data_type"] == "nominal":
        return "category"
    # only numeric, integer, real are left
    elif feature["number_of_missing_values"] != "0" or feature["data_type"] in [
        "numeric",
        "real",
    ]:
        # cast to floats when there are any missing values
        return np.float64
    elif feature["data_type"] == "integer":
        return np.int64
    raise ValueError("Unsupported feature: {}".format(feature))


def _convert_arff_data_dataframe(
    arff: ArffContainerType, columns: List, features_dict: Dict[str, Any]
) -> Tuple:
    """Convert the ARFF object into a pandas DataFrame.

    Parameters
    ----------
    arff : dict
        As obtained from liac-arff object.

    columns : list
        Columns from dataframe to return.

    features_dict : dict
        Maps feature name to feature info from openml.

    Returns
    -------
    result : tuple
        tuple with the resulting dataframe
    """
    pd = check_pandas_support("fetch_openml with as_frame=True")

    attributes = OrderedDict(arff["attributes"])
    arff_columns = list(attributes)

    if not isinstance(arff["data"], Generator):
        raise ValueError(
            "arff['data'] must be a generator when converting to pd.DataFrame."
        )

    # calculate chunksize
    first_row = next(arff["data"])
    first_df = pd.DataFrame([first_row], columns=arff_columns)

    row_bytes = first_df.memory_usage(deep=True).sum()
    chunksize = get_chunk_n_rows(row_bytes)

    # read arff data with chunks
    columns_to_keep = [col for col in arff_columns if col in columns]
    dfs = []
    dfs.append(first_df[columns_to_keep])
    for data in _chunk_generator(arff["data"], chunksize):
        dfs.append(pd.DataFrame(data, columns=arff_columns)[columns_to_keep])
    df = pd.concat(dfs, ignore_index=True)

    for column in columns_to_keep:
        dtype = _feature_to_dtype(features_dict[column])
        if dtype == "category":
            cats_without_missing = [
                cat
                for cat in attributes[column]
                if cat is not None and not is_scalar_nan(cat)
            ]
            dtype = pd.api.types.CategoricalDtype(cats_without_missing)
        df[column] = df[column].astype(dtype, copy=False)
    return (df,)


def _get_data_info_by_name(
    name: str, version: Union[int, str], data_home: Optional[str]
):
    """
    Utilizes the openml dataset listing api to find a dataset by
    name/version
    OpenML api function:
    https://www.openml.org/api_docs#!/data/get_data_list_data_name_data_name

    Parameters
    ----------
    name : str
        name of the dataset

    version : int or str
        If version is an integer, the exact name/version will be obtained from
        OpenML. If version is a string (value: "active") it will take the first
        version from OpenML that is annotated as active. Any other string
        values except "active" are treated as integer.

    data_home : str or None
        Location to cache the response. None if no cache is required.

    Returns
    -------
    first_dataset : json
        json representation of the first dataset object that adhired to the
        search criteria

    """
    if version == "active":
        # situation in which we return the oldest active version
        url = _SEARCH_NAME.format(name) + "/status/active/"
        error_msg = "No active dataset {} found.".format(name)
        json_data = _get_json_content_from_openml_api(
            url, error_msg, data_home=data_home
        )
        res = json_data["data"]["dataset"]
        if len(res) > 1:
            warn(
                "Multiple active versions of the dataset matching the name"
                " {name} exist. Versions may be fundamentally different, "
                "returning version"
                " {version}.".format(name=name, version=res[0]["version"])
            )
        return res[0]

    # an integer version has been provided
    url = (_SEARCH_NAME + "/data_version/{}").format(name, version)
    try:
        json_data = _get_json_content_from_openml_api(
            url, error_message=None, data_home=data_home
        )
    except OpenMLError:
        # we can do this in 1 function call if OpenML does not require the
        # specification of the dataset status (i.e., return datasets with a
        # given name / version regardless of active, deactivated, etc. )
        # TODO: feature request OpenML.
        url += "/status/deactivated"
        error_msg = "Dataset {} with version {} not found.".format(name, version)
        json_data = _get_json_content_from_openml_api(
            url, error_msg, data_home=data_home
        )

    return json_data["data"]["dataset"][0]


def _get_data_description_by_id(
    data_id: int, data_home: Optional[str]
) -> Dict[str, Any]:
    # OpenML API function: https://www.openml.org/api_docs#!/data/get_data_id
    url = _DATA_INFO.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(
        url, error_message, data_home=data_home
    )
    return json_data["data_set_description"]


def _get_data_features(data_id: int, data_home: Optional[str]) -> OpenmlFeaturesType:
    # OpenML function:
    # https://www.openml.org/api_docs#!/data/get_data_features_id
    url = _DATA_FEATURES.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(
        url, error_message, data_home=data_home
    )
    return json_data["data_features"]["feature"]


def _get_data_qualities(data_id: int, data_home: Optional[str]) -> OpenmlQualitiesType:
    # OpenML API function:
    # https://www.openml.org/api_docs#!/data/get_data_qualities_id
    url = _DATA_QUALITIES.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(
        url, error_message, data_home=data_home
    )
    # the qualities might not be available, but we still try to process
    # the data
    return json_data.get("data_qualities", {}).get("quality", [])


def _get_num_samples(data_qualities: OpenmlQualitiesType) -> int:
    """Get the number of samples from data qualities.

    Parameters
    ----------
    data_qualities : list of dict
        Used to retrieve the number of instances (samples) in the dataset.

    Returns
    -------
    n_samples : int
        The number of samples in the dataset or -1 if data qualities are
        unavailable.
    """
    # If the data qualities are unavailable, we return -1
    default_n_samples = -1

    qualities = {d["name"]: d["value"] for d in data_qualities}
    return int(float(qualities.get("NumberOfInstances", default_n_samples)))


def _load_arff_response(
    url: str,
    data_home: Optional[str],
    return_type,
    encode_nominal: bool,
    parse_arff: Callable[[ArffContainerType], Tuple],
    md5_checksum: str,
) -> Tuple:
    """Load arff data with url and parses arff response with parse_arff"""
    response = _open_openml_url(url, data_home)

    with closing(response):
        # Note that if the data is dense, no reading is done until the data
        # generator is iterated.
        actual_md5_checksum = hashlib.md5()

        def _stream_checksum_generator(response):
            for line in response:
                actual_md5_checksum.update(line)
                yield line.decode("utf-8")

        stream = _stream_checksum_generator(response)

        arff = _arff.load(
            stream, return_type=return_type, encode_nominal=encode_nominal
        )

        parsed_arff = parse_arff(arff)

        # consume remaining stream, if early exited
        for _ in stream:
            pass

        if actual_md5_checksum.hexdigest() != md5_checksum:
            raise ValueError(
                "md5 checksum of local file for "
                + url
                + " does not match description. "
                "Downloaded file could have been modified / "
                "corrupted, clean cache and retry..."
            )

        return parsed_arff


def _download_data_to_bunch(
    url: str,
    sparse: bool,
    data_home: Optional[str],
    *,
    as_frame: bool,
    features_list: List,
    data_columns: List[int],
    target_columns: List,
    shape: Optional[Tuple[int, int]],
    md5_checksum: str,
):
    """Download OpenML ARFF and convert to Bunch of data"""
    # NB: this function is long in order to handle retry for any failure
    #     during the streaming parse of the ARFF.

    # Prepare which columns and data types should be returned for the X and y
    features_dict = {feature["name"]: feature for feature in features_list}

    # XXX: col_slice_y should be all nominal or all numeric
    _verify_target_data_type(features_dict, target_columns)

    col_slice_y = [int(features_dict[col_name]["index"]) for col_name in target_columns]

    col_slice_x = [int(features_dict[col_name]["index"]) for col_name in data_columns]
    for col_idx in col_slice_y:
        feat = features_list[col_idx]
        nr_missing = int(feat["number_of_missing_values"])
        if nr_missing > 0:
            raise ValueError(
                "Target column {} has {} missing values. "
                "Missing values are not supported for target "
                "columns. ".format(feat["name"], nr_missing)
            )

    # Access an ARFF file on the OpenML server. Documentation:
    # https://www.openml.org/api_data_docs#!/data/get_download_id

    if sparse is True:
        return_type = _arff.COO
    else:
        return_type = _arff.DENSE_GEN

    frame = nominal_attributes = None

    parse_arff: Callable
    postprocess: Callable
    if as_frame:
        columns = data_columns + target_columns
        parse_arff = partial(
            _convert_arff_data_dataframe, columns=columns, features_dict=features_dict
        )

        def postprocess(frame):
            X = frame[data_columns]
            if len(target_columns) >= 2:
                y = frame[target_columns]
            elif len(target_columns) == 1:
                y = frame[target_columns[0]]
            else:
                y = None
            return X, y, frame, nominal_attributes

    else:

        def parse_arff(arff):
            X, y = _convert_arff_data(arff, col_slice_x, col_slice_y, shape)
            # nominal attributes is a dict mapping from the attribute name to
            # the possible values. Includes also the target column (which will
            # be popped off below, before it will be packed in the Bunch
            # object)
            nominal_attributes = {
                k: v
                for k, v in arff["attributes"]
                if isinstance(v, list) and k in data_columns + target_columns
            }
            return X, y, nominal_attributes

        def postprocess(X, y, nominal_attributes):
            is_classification = {
                col_name in nominal_attributes for col_name in target_columns
            }
            if not is_classification:
                # No target
                pass
            elif all(is_classification):
                y = np.hstack(
                    [
                        np.take(
                            np.asarray(nominal_attributes.pop(col_name), dtype="O"),
                            y[:, i : i + 1].astype(int, copy=False),
                        )
                        for i, col_name in enumerate(target_columns)
                    ]
                )
            elif any(is_classification):
                raise ValueError(
                    "Mix of nominal and non-nominal targets is not currently supported"
                )

            # reshape y back to 1-D array, if there is only 1 target column;
            # back to None if there are not target columns
            if y.shape[1] == 1:
                y = y.reshape((-1,))
            elif y.shape[1] == 0:
                y = None
            return X, y, frame, nominal_attributes

    out = _retry_with_clean_cache(url, data_home)(_load_arff_response)(
        url,
        data_home,
        return_type=return_type,
        encode_nominal=not as_frame,
        parse_arff=parse_arff,
        md5_checksum=md5_checksum,
    )
    X, y, frame, nominal_attributes = postprocess(*out)

    return Bunch(
        data=X,
        target=y,
        frame=frame,
        categories=nominal_attributes,
        feature_names=data_columns,
        target_names=target_columns,
    )


def _verify_target_data_type(features_dict, target_columns):
    # verifies the data type of the y array in case there are multiple targets
    # (throws an error if these targets do not comply with sklearn support)
    if not isinstance(target_columns, list):
        raise ValueError("target_column should be list, got: %s" % type(target_columns))
    found_types = set()
    for target_column in target_columns:
        if target_column not in features_dict:
            raise KeyError("Could not find target_column={}")
        if features_dict[target_column]["data_type"] == "numeric":
            found_types.add(np.float64)
        else:
            found_types.add(object)

        # note: we compare to a string, not boolean
        if features_dict[target_column]["is_ignore"] == "true":
            warn("target_column={} has flag is_ignore.".format(target_column))
        if features_dict[target_column]["is_row_identifier"] == "true":
            warn("target_column={} has flag is_row_identifier.".format(target_column))
    if len(found_types) > 1:
        raise ValueError(
            "Can only handle homogeneous multi-target datasets, "
            "i.e., all targets are either numeric or "
            "categorical."
        )


def _valid_data_column_names(features_list, target_columns):
    # logic for determining on which columns can be learned. Note that from the
    # OpenML guide follows that columns that have the `is_row_identifier` or
    # `is_ignore` flag, these can not be learned on. Also target columns are
    # excluded.
    valid_data_column_names = []
    for feature in features_list:
        if (
            feature["name"] not in target_columns
            and feature["is_ignore"] != "true"
            and feature["is_row_identifier"] != "true"
        ):
            valid_data_column_names.append(feature["name"])
    return valid_data_column_names


def fetch_openml(
    name: Optional[str] = None,
    *,
    version: Union[str, int] = "active",
    data_id: Optional[int] = None,
    data_home: Optional[str] = None,
    target_column: Optional[Union[str, List]] = "default-target",
    cache: bool = True,
    return_X_y: bool = False,
    as_frame: Union[str, bool] = "auto",
):
    """Fetch dataset from openml by name or dataset id.

    Datasets are uniquely identified by either an integer ID or by a
    combination of name and version (i.e. there might be multiple
    versions of the 'iris' dataset). Please give either name or data_id
    (not both). In case a name is given, a version can also be
    provided.

    Read more in the :ref:`User Guide <openml>`.

    .. versionadded:: 0.20

    .. note:: EXPERIMENTAL

        The API is experimental (particularly the return value structure),
        and might have small backward-incompatible changes without notice
        or warning in future releases.

    Parameters
    ----------
    name : str, default=None
        String identifier of the dataset. Note that OpenML can have multiple
        datasets with the same name.

    version : int or 'active', default='active'
        Version of the dataset. Can only be provided if also ``name`` is given.
        If 'active' the oldest version that's still active is used. Since
        there may be more than one active version of a dataset, and those
        versions may fundamentally be different from one another, setting an
        exact version is highly recommended.

    data_id : int, default=None
        OpenML ID of the dataset. The most specific way of retrieving a
        dataset. If data_id is not given, name (and potential version) are
        used to obtain a dataset.

    data_home : str, default=None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    target_column : str, list or None, default='default-target'
        Specify the column name in the data to use as target. If
        'default-target', the standard target column a stored on the server
        is used. If ``None``, all columns are returned as data and the
        target is ``None``. If list (of strings), all columns with these names
        are returned as multi-target (Note: not all scikit-learn classifiers
        can handle all types of multi-output combinations)

    cache : bool, default=True
        Whether to cache downloaded datasets using joblib.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` objects.

    as_frame : bool or 'auto', default='auto'
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.

        If as_frame is 'auto', the data and target will be converted to
        DataFrame or Series as if as_frame is set to True, unless the dataset
        is stored in sparse format.

        .. versionchanged:: 0.24
           The default value of `as_frame` changed from `False` to `'auto'`
           in 0.24.

    Returns
    -------

    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : np.array, scipy.sparse.csr_matrix of floats, or pandas DataFrame
            The feature matrix. Categorical features are encoded as ordinals.
        target : np.array, pandas Series or DataFrame
            The regression target or classification labels, if applicable.
            Dtype is float if numeric, and object if categorical. If
            ``as_frame`` is True, ``target`` is a pandas object.
        DESCR : str
            The full description of the dataset
        feature_names : list
            The names of the dataset columns
        target_names: list
            The names of the target columns

        .. versionadded:: 0.22

        categories : dict or None
            Maps each categorical feature name to a list of values, such
            that the value encoded as i is ith in the list. If ``as_frame``
            is True, this is None.
        details : dict
            More metadata from OpenML
        frame : pandas DataFrame
            Only present when `as_frame=True`. DataFrame with ``data`` and
            ``target``.

    (data, target) : tuple if ``return_X_y`` is True

        .. note:: EXPERIMENTAL

            This interface is **experimental** and subsequent releases may
            change attributes without notice (although there should only be
            minor changes to ``data`` and ``target``).

        Missing values in the 'data' are represented as NaN's. Missing values
        in 'target' are represented as NaN's (numerical target) or None
        (categorical target)
    """
    if cache is False:
        # no caching will be applied
        data_home = None
    else:
        data_home = get_data_home(data_home=data_home)
        data_home = join(data_home, "openml")

    # check valid function arguments. data_id XOR (name, version) should be
    # provided
    if name is not None:
        # OpenML is case-insensitive, but the caching mechanism is not
        # convert all data names (str) to lower case
        name = name.lower()
        if data_id is not None:
            raise ValueError(
                "Dataset data_id={} and name={} passed, but you can only "
                "specify a numeric data_id or a name, not "
                "both.".format(data_id, name)
            )
        data_info = _get_data_info_by_name(name, version, data_home)
        data_id = data_info["did"]
    elif data_id is not None:
        # from the previous if statement, it is given that name is None
        if version != "active":
            raise ValueError(
                "Dataset data_id={} and version={} passed, but you can only "
                "specify a numeric data_id or a version, not "
                "both.".format(data_id, version)
            )
    else:
        raise ValueError(
            "Neither name nor data_id are provided. Please provide name or data_id."
        )

    data_description = _get_data_description_by_id(data_id, data_home)
    if data_description["status"] != "active":
        warn(
            "Version {} of dataset {} is inactive, meaning that issues have "
            "been found in the dataset. Try using a newer version from "
            "this URL: {}".format(
                data_description["version"],
                data_description["name"],
                data_description["url"],
            )
        )
    if "error" in data_description:
        warn(
            "OpenML registered a problem with the dataset. It might be "
            "unusable. Error: {}".format(data_description["error"])
        )
    if "warning" in data_description:
        warn(
            "OpenML raised a warning on the dataset. It might be "
            "unusable. Warning: {}".format(data_description["warning"])
        )

    return_sparse = False
    if data_description["format"].lower() == "sparse_arff":
        return_sparse = True

    if as_frame == "auto":
        as_frame = not return_sparse

    if as_frame and return_sparse:
        raise ValueError("Cannot return dataframe with sparse data")

    # download data features, meta-info about column types
    features_list = _get_data_features(data_id, data_home)

    if not as_frame:
        for feature in features_list:
            if "true" in (feature["is_ignore"], feature["is_row_identifier"]):
                continue
            if feature["data_type"] == "string":
                raise ValueError(
                    "STRING attributes are not supported for "
                    "array representation. Try as_frame=True"
                )

    if target_column == "default-target":
        # determines the default target based on the data feature results
        # (which is currently more reliable than the data description;
        # see issue: https://github.com/openml/OpenML/issues/768)
        target_columns = [
            feature["name"]
            for feature in features_list
            if feature["is_target"] == "true"
        ]
    elif isinstance(target_column, str):
        # for code-simplicity, make target_column by default a list
        target_columns = [target_column]
    elif target_column is None:
        target_columns = []
    elif isinstance(target_column, list):
        target_columns = target_column
    else:
        raise TypeError(
            "Did not recognize type of target_column"
            "Should be str, list or None. Got: "
            "{}".format(type(target_column))
        )
    data_columns = _valid_data_column_names(features_list, target_columns)

    shape: Optional[Tuple[int, int]]
    # determine arff encoding to return
    if not return_sparse:
        # The shape must include the ignored features to keep the right indexes
        # during the arff data conversion.
        data_qualities = _get_data_qualities(data_id, data_home)
        shape = _get_num_samples(data_qualities), len(features_list)
    else:
        shape = None

    # obtain the data
    url = _DATA_FILE.format(data_description["file_id"])
    bunch = _download_data_to_bunch(
        url,
        return_sparse,
        data_home,
        as_frame=bool(as_frame),
        features_list=features_list,
        shape=shape,
        target_columns=target_columns,
        data_columns=data_columns,
        md5_checksum=data_description["md5_checksum"],
    )

    if return_X_y:
        return bunch.data, bunch.target

    description = "{}\n\nDownloaded from openml.org.".format(
        data_description.pop("description")
    )

    bunch.update(
        DESCR=description,
        details=data_description,
        url="https://www.openml.org/d/{}".format(data_id),
    )

    return bunch
