import inspect
import functools
import sys
import warnings
from collections.abc import Iterable

import numpy as np
import scipy
from numpy.lib import NumpyVersion

from ._warnings import all_warnings, warn


__all__ = ['deprecated', 'get_bound_method_class', 'all_warnings',
           'safe_as_int', 'check_shape_equality', 'check_nD', 'warn',
           'reshape_nd', 'identity', 'slice_at_axis']


class skimage_deprecation(Warning):
    """Create our own deprecation class, since Python >= 2.7
    silences deprecations by default.

    """
    pass


class change_default_value:
    """Decorator for changing the default value of an argument.

    Parameters
    ----------
    arg_name: str
        The name of the argument to be updated.
    new_value: any
        The argument new value.
    changed_version : str
        The package version in which the change will be introduced.
    warning_msg: str
        Optional warning message. If None, a generic warning message
        is used.

    """

    def __init__(self, arg_name, *, new_value, changed_version,
                 warning_msg=None):
        self.arg_name = arg_name
        self.new_value = new_value
        self.warning_msg = warning_msg
        self.changed_version = changed_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        old_value = parameters[self.arg_name].default

        if self.warning_msg is None:
            self.warning_msg = (
                f'The new recommended value for {self.arg_name} is '
                f'{self.new_value}. Until version {self.changed_version}, '
                f'the default {self.arg_name} value is {old_value}. '
                f'From version {self.changed_version}, the {self.arg_name} '
                f'default value will be {self.new_value}. To avoid '
                f'this warning, please explicitly set {self.arg_name} value.')

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            if len(args) < arg_idx + 1 and self.arg_name not in kwargs.keys():
                # warn that arg_name default value changed:
                warnings.warn(self.warning_msg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return fixed_func


class remove_arg:
    """Decorator to remove an argument from function's signature.

    Parameters
    ----------
    arg_name: str
        The name of the argument to be removed.
    changed_version : str
        The package version in which the warning will be replaced by
        an error.
    help_msg: str
        Optional message appended to the generic warning message.

    """

    def __init__(self, arg_name, *, changed_version, help_msg=None):
        self.arg_name = arg_name
        self.help_msg = help_msg
        self.changed_version = changed_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        warning_msg = (
            f'{self.arg_name} argument is deprecated and will be removed '
            f'in version {self.changed_version}. To avoid this warning, '
            f'please do not use the {self.arg_name} argument. Please '
            f'see {func.__name__} documentation for more details.')

        if self.help_msg is not None:
            warning_msg += f' {self.help_msg}'

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            if len(args) > arg_idx or self.arg_name in kwargs.keys():
                # warn that arg_name is deprecated
                warnings.warn(warning_msg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return fixed_func


def docstring_add_deprecated(func, kwarg_mapping, deprecated_version):
    """Add deprecated kwarg(s) to the "Other Params" section of a docstring.

    Parameters
    ---------
    func : function
        The function whose docstring we wish to update.
    kwarg_mapping : dict
        A dict containing {old_arg: new_arg} key/value pairs as used by
        `deprecate_kwarg`.
    deprecated_version : str
        A major.minor version string specifying when old_arg was
        deprecated.

    Returns
    -------
    new_doc : str
        The updated docstring. Returns the original docstring if numpydoc is
        not available.
    """
    if func.__doc__ is None:
        return None
    try:
        from numpydoc.docscrape import FunctionDoc, Parameter
    except ImportError:
        # Return an unmodified docstring if numpydoc is not available.
        return func.__doc__

    Doc = FunctionDoc(func)
    for old_arg, new_arg in kwarg_mapping.items():
        desc = [f'Deprecated in favor of `{new_arg}`.',
                f'',
                f'.. deprecated:: {deprecated_version}']
        Doc['Other Parameters'].append(
            Parameter(name=old_arg,
                      type='DEPRECATED',
                      desc=desc)
        )
    new_docstring = str(Doc)

    # new_docstring will have a header starting with:
    #
    # .. function:: func.__name__
    #
    # and some additional blank lines. We strip these off below.
    split = new_docstring.split('\n')
    no_header = split[1:]
    while not no_header[0].strip():
        no_header.pop(0)

    # Store the initial description before any of the Parameters fields.
    # Usually this is a single line, but the while loop covers any case
    # where it is not.
    descr = no_header.pop(0)
    while no_header[0].strip():
        descr += '\n    ' + no_header.pop(0)
    descr += '\n\n'
    # '\n    ' rather than '\n' here to restore the original indentation.
    final_docstring = descr + '\n    '.join(no_header)
    # strip any extra spaces from ends of lines
    final_docstring = '\n'.join(
        [line.rstrip() for line in final_docstring.split('\n')]
    )
    return final_docstring


class deprecate_kwarg:
    """Decorator ensuring backward compatibility when argument names are
    modified in a function definition.

    Parameters
    ----------
    kwarg_mapping: dict
        Mapping between the function's old argument names and the new
        ones.
    deprecated_version : str
        The package version in which the argument was first deprecated.
    warning_msg: str
        Optional warning message. If None, a generic warning message
        is used.
    removed_version : str
        The package version in which the deprecated argument will be
        removed.

    """

    def __init__(self, kwarg_mapping, deprecated_version, warning_msg=None,
                 removed_version=None):
        self.kwarg_mapping = kwarg_mapping
        if warning_msg is None:
            self.warning_msg = ("`{old_arg}` is a deprecated argument name "
                                "for `{func_name}`. ")
            if removed_version is not None:
                self.warning_msg += (f'It will be removed in '
                                     f'version {removed_version}.')
            self.warning_msg += "Please use `{new_arg}` instead."
        else:
            self.warning_msg = warning_msg

        self.deprecated_version = deprecated_version

    def __call__(self, func):

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            for old_arg, new_arg in self.kwarg_mapping.items():
                if old_arg in kwargs:
                    #  warn that the function interface has changed:
                    warnings.warn(self.warning_msg.format(
                        old_arg=old_arg, func_name=func.__name__,
                        new_arg=new_arg), FutureWarning, stacklevel=2)
                    # Substitute new_arg to old_arg
                    kwargs[new_arg] = kwargs.pop(old_arg)

            # Call the function with the fixed arguments
            return func(*args, **kwargs)

        if func.__doc__ is not None:
            newdoc = docstring_add_deprecated(func, self.kwarg_mapping,
                                              self.deprecated_version)
            fixed_func.__doc__ = newdoc
        return fixed_func


class deprecate_multichannel_kwarg(deprecate_kwarg):
    """Decorator for deprecating multichannel keyword in favor of channel_axis.

    Parameters
    ----------
    removed_version : str
        The package version in which the deprecated argument will be
        removed.

    """

    def __init__(self, removed_version='1.0', multichannel_position=None):
        super().__init__(
            kwarg_mapping={'multichannel': 'channel_axis'},
            deprecated_version='0.19',
            warning_msg=None,
            removed_version=removed_version)
        self.position = multichannel_position

    def __call__(self, func):
        @functools.wraps(func)
        def fixed_func(*args, **kwargs):

            if self.position is not None and len(args) > self.position:
                warning_msg = (
                    "Providing the `multichannel` argument positionally to "
                    "{func_name} is deprecated. Use the `channel_axis` kwarg "
                    "instead."
                )
                warnings.warn(warning_msg.format(func_name=func.__name__),
                              FutureWarning,
                              stacklevel=2)
                if 'channel_axis' in kwargs:
                    raise ValueError(
                        "Cannot provide both a `channel_axis` kwarg and a "
                        "positional `multichannel` value."
                    )
                else:
                    channel_axis = -1 if args[self.position] else None
                    kwargs['channel_axis'] = channel_axis

            if 'multichannel' in kwargs:
                #  warn that the function interface has changed:
                warnings.warn(self.warning_msg.format(
                    old_arg='multichannel', func_name=func.__name__,
                    new_arg='channel_axis'), FutureWarning, stacklevel=2)

                # multichannel = True -> last axis corresponds to channels
                convert = {True: -1, False: None}
                kwargs['channel_axis'] = convert[kwargs.pop('multichannel')]

            # Call the function with the fixed arguments
            return func(*args, **kwargs)

        if func.__doc__ is not None:
            newdoc = docstring_add_deprecated(
                func, {'multichannel': 'channel_axis'}, '0.19')
            fixed_func.__doc__ = newdoc
        return fixed_func


class channel_as_last_axis():
    """Decorator for automatically making channels axis last for all arrays.

    This decorator reorders axes for compatibility with functions that only
    support channels along the last axis. After the function call is complete
    the channels axis is restored back to its original position.

    Parameters
    ----------
    channel_arg_positions : tuple of int, optional
        Positional arguments at the positions specified in this tuple are
        assumed to be multichannel arrays. The default is to assume only the
        first argument to the function is a multichannel array.
    channel_kwarg_names : tuple of str, optional
        A tuple containing the names of any keyword arguments corresponding to
        multichannel arrays.
    multichannel_output : bool, optional
        A boolean that should be True if the output of the function is not a
        multichannel array and False otherwise. This decorator does not
        currently support the general case of functions with multiple outputs
        where some or all are multichannel.

    """
    def __init__(self, channel_arg_positions=(0,), channel_kwarg_names=(),
                 multichannel_output=True):
        self.arg_positions = set(channel_arg_positions)
        self.kwarg_names = set(channel_kwarg_names)
        self.multichannel_output = multichannel_output

    def __call__(self, func):
        @functools.wraps(func)
        def fixed_func(*args, **kwargs):

            channel_axis = kwargs.get('channel_axis', None)

            if channel_axis is None:
                return func(*args, **kwargs)

            # TODO: convert scalars to a tuple in anticipation of eventually
            #       supporting a tuple of channel axes. Right now, only an
            #       integer or a single-element tuple is supported, though.
            if np.isscalar(channel_axis):
                channel_axis = (channel_axis,)
            if len(channel_axis) > 1:
                raise ValueError(
                    "only a single channel axis is currently suported")

            if channel_axis == (-1,) or channel_axis == -1:
                return func(*args, **kwargs)

            if self.arg_positions:
                new_args = []
                for pos, arg in enumerate(args):
                    if pos in self.arg_positions:
                        new_args.append(np.moveaxis(arg, channel_axis[0], -1))
                    else:
                        new_args.append(arg)
                new_args = tuple(new_args)
            else:
                new_args = args

            for name in self.kwarg_names:
                kwargs[name] = np.moveaxis(kwargs[name], channel_axis[0], -1)

            # now that we have moved the channels axis to the last position,
            # change the channel_axis argument to -1
            kwargs["channel_axis"] = -1

            # Call the function with the fixed arguments
            out = func(*new_args, **kwargs)
            if self.multichannel_output:
                out = np.moveaxis(out, -1, channel_axis[0])
            return out

        return fixed_func


class deprecated(object):
    """Decorator to mark deprecated functions with warning.

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    Parameters
    ----------
    alt_func : str
        If given, tell user what function to use instead.
    behavior : {'warn', 'raise'}
        Behavior during call to deprecated function: 'warn' = warn user that
        function is deprecated; 'raise' = raise error.
    removed_version : str
        The package version in which the deprecated function will be removed.
    """

    def __init__(self, alt_func=None, behavior='warn', removed_version=None):
        self.alt_func = alt_func
        self.behavior = behavior
        self.removed_version = removed_version

    def __call__(self, func):

        alt_msg = ''
        if self.alt_func is not None:
            alt_msg = ' Use ``%s`` instead.' % self.alt_func
        rmv_msg = ''
        if self.removed_version is not None:
            rmv_msg = (' and will be removed in version %s' %
                       self.removed_version)

        msg = ('Function ``%s`` is deprecated' % func.__name__ +
               rmv_msg + '.' + alt_msg)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.behavior == 'warn':
                func_code = func.__code__
                warnings.simplefilter('always', skimage_deprecation)
                warnings.warn_explicit(msg,
                                       category=skimage_deprecation,
                                       filename=func_code.co_filename,
                                       lineno=func_code.co_firstlineno + 1)
            elif self.behavior == 'raise':
                raise skimage_deprecation(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = '**Deprecated function**.' + alt_msg
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + '\n\n    ' + wrapped.__doc__

        return wrapped


def get_bound_method_class(m):
    """Return the class for a bound method.

    """
    return m.im_class if sys.version < '3' else m.__self__.__class__


def safe_as_int(val, atol=1e-3):
    """
    Attempt to safely cast values to integer format.

    Parameters
    ----------
    val : scalar or iterable of scalars
        Number or container of numbers which are intended to be interpreted as
        integers, e.g., for indexing purposes, but which may not carry integer
        type.
    atol : float
        Absolute tolerance away from nearest integer to consider values in
        ``val`` functionally integers.

    Returns
    -------
    val_int : NumPy scalar or ndarray of dtype `np.int64`
        Returns the input value(s) coerced to dtype `np.int64` assuming all
        were within ``atol`` of the nearest integer.

    Notes
    -----
    This operation calculates ``val`` modulo 1, which returns the mantissa of
    all values. Then all mantissas greater than 0.5 are subtracted from one.
    Finally, the absolute tolerance from zero is calculated. If it is less
    than ``atol`` for all value(s) in ``val``, they are rounded and returned
    in an integer array. Or, if ``val`` was a scalar, a NumPy scalar type is
    returned.

    If any value(s) are outside the specified tolerance, an informative error
    is raised.

    Examples
    --------
    >>> safe_as_int(7.0)
    7

    >>> safe_as_int([9, 4, 2.9999999999])
    array([9, 4, 3])

    >>> safe_as_int(53.1)
    Traceback (most recent call last):
        ...
    ValueError: Integer argument required but received 53.1, check inputs.

    >>> safe_as_int(53.01, atol=0.01)
    53

    """
    mod = np.asarray(val) % 1                # Extract mantissa

    # Check for and subtract any mod values > 0.5 from 1
    if mod.ndim == 0:                        # Scalar input, cannot be indexed
        if mod > 0.5:
            mod = 1 - mod
    else:                                    # Iterable input, now ndarray
        mod[mod > 0.5] = 1 - mod[mod > 0.5]  # Test on each side of nearest int

    try:
        np.testing.assert_allclose(mod, 0, atol=atol)
    except AssertionError:
        raise ValueError(f'Integer argument required but received '
                         f'{val}, check inputs.')

    return np.round(val).astype(np.int64)


def check_shape_equality(im1, im2):
    """Raise an error if the shape do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return


def slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), Ellipsis)
    """
    return (slice(None),) * axis + (sl,) + (...,)


def reshape_nd(arr, ndim, dim):
    """Reshape a 1D array to have n dimensions, all singletons but one.

    Parameters
    ----------
    arr : array, shape (N,)
        Input array
    ndim : int
        Number of desired dimensions of reshaped array.
    dim : int
        Which dimension/axis will not be singleton-sized.

    Returns
    -------
    arr_reshaped : array, shape ([1, ...], N, [1,...])
        View of `arr` reshaped to the desired shape.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> arr = rng.random(7)
    >>> reshape_nd(arr, 2, 0).shape
    (7, 1)
    >>> reshape_nd(arr, 3, 1).shape
    (1, 7, 1)
    >>> reshape_nd(arr, 4, -1).shape
    (1, 1, 1, 7)
    """
    if arr.ndim != 1:
        raise ValueError("arr must be a 1D array")
    new_shape = [1] * ndim
    new_shape[dim] = -1
    return np.reshape(arr, new_shape)


def check_nD(array, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims and array isn't empty.

    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.

    """
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim]))
        )


def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.

    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes
    -----
    * Input images with `float32` data type are not upcast.

    Returns
    -------
    image : ndarray
        Transformed version of the input.

    """
    if image.dtype == np.float16:
        return image.astype(np.float32)
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        from ..util.dtype import img_as_float
        image = img_as_float(image)
    return image


def _validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)

    """

    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the "
                         "range 0-5.")

    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
             "with bool data type. Please set order to 0 or explicitely "
             "cast input image to another data type.")

    return order


def _to_np_mode(mode):
    """Convert padding modes from `ndi.correlate` to `np.pad`."""
    mode_translation_dict = dict(nearest='edge', reflect='symmetric',
                                 mirror='reflect')
    if mode in mode_translation_dict:
        mode = mode_translation_dict[mode]
    return mode


def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(constant='constant', edge='nearest',
                                 symmetric='reflect', reflect='mirror',
                                 wrap='wrap')
    if mode not in mode_translation_dict:
        raise ValueError(
            (f"Unknown mode: '{mode}', or cannot translate mode. The "
             f"mode should be one of 'constant', 'edge', 'symmetric', "
             f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
             f"more info."))
    return _fix_ndimage_mode(mode_translation_dict[mode])


def _fix_ndimage_mode(mode):
    # SciPy 1.6.0 introduced grid variants of constant and wrap which
    # have less surprising behavior for images. Use these when available
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    if NumpyVersion(scipy.__version__) >= '1.6.0':
        mode = grid_modes.get(mode, mode)
    return mode


new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,      # np.float128 ; doesn't exist on windows
    'G': np.complex128,   # np.complex256 ; doesn't exist on windows
}


def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or Iterable of np.dtype
        The input dtype. If a sequence of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, Iterable) and not isinstance(input_dtype, str):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def identity(image, *args, **kwargs):
    """Returns the first argument unmodified."""
    return image
