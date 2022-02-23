"""Classes and methods to use for parameters of augmenters.

This module contains e.g. classes representing probability
distributions (guassian, poisson etc.), classes representing noise sources
and methods to normalize parameter-related user inputs.

"""
from __future__ import print_function, division, absolute_import
import copy as copy_module
from collections import defaultdict
from abc import ABCMeta, abstractmethod
import tempfile

import numpy as np
import six
import six.moves as sm
import scipy
import scipy.stats
import imageio

from . import imgaug as ia
from . import dtypes as iadt
from . import random as iarandom
from .external.opensimplex import OpenSimplex


def _check_value_range(value, name, value_range):
    if value_range is None:
        return True

    if isinstance(value_range, tuple):
        assert len(value_range) == 2, (
            "If 'value_range' is a tuple, it must contain exactly 2 entries, "
            "got %d." % (len(value_range),))

        if value_range[0] is None and value_range[1] is None:
            return True

        if value_range[0] is None:
            assert value <= value_range[1], (
                "Parameter '%s' is outside of the expected value "
                "range (x <= %.4f)" % (name, value_range[1]))
            return True

        if value_range[1] is None:
            assert value_range[0] <= value, (
                "Parameter '%s' is outside of the expected value "
                "range (%.4f <= x)" % (name, value_range[0]))
            return True

        assert value_range[0] <= value <= value_range[1], (
            "Parameter '%s' is outside of the expected value "
            "range (%.4f <= x <= %.4f)" % (
                name, value_range[0], value_range[1]))

        return True

    if ia.is_callable(value_range):
        value_range(value)
        return True

    raise Exception("Unexpected input for value_range, got %s." % (
        str(value_range),))


# FIXME this uses _check_value_range, which checks for a<=x<=b, but a produced
#       Uniform parameter has value range a<=x<b.
def handle_continuous_param(param, name, value_range=None,
                            tuple_to_uniform=True, list_to_choice=True):
    if ia.is_single_number(param):
        _check_value_range(param, name, value_range)
        return Deterministic(param)

    if tuple_to_uniform and isinstance(param, tuple):
        assert len(param) == 2, (
            "Expected parameter '%s' with type tuple to have exactly two "
            "entries, but got %d." % (name, len(param)))
        assert all([ia.is_single_number(v) for v in param]), (
            "Expected parameter '%s' with type tuple to only contain "
            "numbers, got %s." % (name, [type(v) for v in param],))
        _check_value_range(param[0], name, value_range)
        _check_value_range(param[1], name, value_range)
        return Uniform(param[0], param[1])

    if (list_to_choice and ia.is_iterable(param)
            and not isinstance(param, tuple)):
        assert all([ia.is_single_number(v) for v in param]), (
            "Expected iterable parameter '%s' to only contain numbers, "
            "got %s." % (name, [type(v) for v in param],))
        for param_i in param:
            _check_value_range(param_i, name, value_range)
        return Choice(param)

    if isinstance(param, StochasticParameter):
        return param

    allowed_type = "number"
    list_str = ", list of %s" % (allowed_type,) if list_to_choice else ""
    raise Exception(
        "Expected %s, tuple of two %s%s or StochasticParameter for %s, "
        "got %s." % (
            allowed_type, allowed_type, list_str, name, type(param),))


def handle_discrete_param(param, name, value_range=None, tuple_to_uniform=True,
                          list_to_choice=True, allow_floats=True):
    if (ia.is_single_integer(param)
            or (allow_floats and ia.is_single_float(param))):
        _check_value_range(param, name, value_range)
        return Deterministic(int(param))

    if tuple_to_uniform and isinstance(param, tuple):
        assert len(param) == 2, (
            "Expected parameter '%s' with type tuple to have exactly two "
            "entries, but got %d." % (name, len(param)))
        is_valid_types = all([
            ia.is_single_number(v)
            if allow_floats else ia.is_single_integer(v)
            for v in param])
        assert is_valid_types, (
            "Expected parameter '%s' of type tuple to only contain %s, "
            "got %s." % (
                name,
                "number" if allow_floats else "integer",
                [type(v) for v in param],))

        _check_value_range(param[0], name, value_range)
        _check_value_range(param[1], name, value_range)
        return DiscreteUniform(int(param[0]), int(param[1]))

    if (list_to_choice and ia.is_iterable(param)
            and not isinstance(param, tuple)):
        is_valid_types = all([
            ia.is_single_number(v)
            if allow_floats else ia.is_single_integer(v)
            for v in param])
        assert is_valid_types, (
            "Expected iterable parameter '%s' to only contain %s, "
            "got %s." % (
                name,
                "number" if allow_floats else "integer",
                [type(v) for v in param],))

        for param_i in param:
            _check_value_range(param_i, name, value_range)
        return Choice([int(param_i) for param_i in param])

    if isinstance(param, StochasticParameter):
        return param

    allowed_type = "number" if allow_floats else "int"
    list_str = ", list of %s" % (allowed_type,) if list_to_choice else ""
    raise Exception(
        "Expected %s, tuple of two %s%s or StochasticParameter for %s, "
        "got %s." % (
            allowed_type, allowed_type, list_str, name, type(param),))


# Added in 0.4.0.
def handle_categorical_string_param(param, name, valid_values=None):
    if param == ia.ALL and valid_values is not None:
        return Choice(list(valid_values))

    if ia.is_string(param):
        if valid_values is not None:
            assert param in valid_values, (
                "Expected parameter '%s' to be one of: %s. Got: %s." % (
                    name, ", ".join(list(valid_values)), param))
        return Deterministic(param)

    if isinstance(param, list):
        assert all([ia.is_string(val) for val in param]), (
            "Expected list provided for parameter '%s' to only contain "
            "strings, got types: %s." % (
                name, ", ".join([type(v).__name__ for v in param])))
        if valid_values is not None:
            assert all([val in valid_values for val in param]), (
                "Expected list provided for parameter '%s' to only contain "
                "the following allowed strings: %s. Got strings: %s." % (
                    name, ", ".join(valid_values), ", ".join(param)
                ))
        return Choice(param)

    if isinstance(param, StochasticParameter):
        return param

    raise Exception(
        "Expected parameter '%s' to be%s a string, a list of "
        "strings or StochasticParameter, got %s." % (
            name,
            " imgaug.ALL," if valid_values is not None else "",
            type(param).__name__,))


def handle_discrete_kernel_size_param(param, name, value_range=(1, None),
                                      allow_floats=True):
    if (ia.is_single_integer(param)
            or (allow_floats and ia.is_single_float(param))):
        _check_value_range(param, name, value_range)
        return Deterministic(int(param)), None

    if isinstance(param, tuple):
        assert len(param) == 2, (
            "Expected parameter '%s' with type tuple to have exactly two "
            "entries, but got %d." % (name, len(param)))
        if (all([ia.is_single_integer(param_i) for param_i in param])
                or (allow_floats and all([ia.is_single_float(param_i)
                                          for param_i in param]))):
            _check_value_range(param[0], name, value_range)
            _check_value_range(param[1], name, value_range)
            return DiscreteUniform(int(param[0]), int(param[1])), None

        if all([isinstance(param_i, StochasticParameter)
                for param_i in param]):
            return param[0], param[1]

        handled = (
            handle_discrete_param(
                param[0], "%s[0]" % (name,), value_range,
                allow_floats=allow_floats),
            handle_discrete_param(
                param[1], "%s[1]" % (name,), value_range,
                allow_floats=allow_floats)
        )

        return handled

    if ia.is_iterable(param) and not isinstance(param, tuple):
        is_valid_types = all([
            ia.is_single_number(v)
            if allow_floats else ia.is_single_integer(v)
            for v in param])
        assert is_valid_types, (
            "Expected iterable parameter '%s' to only contain %s, "
            "got %s." % (
                name,
                "number" if allow_floats else "integer",
                [type(v) for v in param],))

        for param_i in param:
            _check_value_range(param_i, name, value_range)
        return Choice([int(param_i) for param_i in param]), None

    if isinstance(param, StochasticParameter):
        return param, None

    raise Exception(
        "Expected int, tuple/list with 2 entries or StochasticParameter. "
        "Got %s." % (type(param),))


def handle_probability_param(param, name, tuple_to_uniform=False,
                             list_to_choice=False):
    eps = 1e-6

    if param in [True, False, 0, 1]:
        return Deterministic(int(param))

    if ia.is_single_number(param):
        assert 0.0 <= param <= 1.0, (
            "Expected probability of parameter '%s' to be in the interval "
            "[0.0, 1.0], got %.4f." % (name, param,))
        if 0.0-eps < param < 0.0+eps or 1.0-eps < param < 1.0+eps:
            return Deterministic(int(np.round(param)))
        return Binomial(param)

    if tuple_to_uniform and isinstance(param, tuple):
        assert all([ia.is_single_number(v) for v in param]), (
            "Expected parameter '%s' of type tuple to only contain numbers, "
            "got %s." % (name, [type(v) for v in param],))
        assert len(param) == 2, (
            "Expected parameter '%s' of type tuple to contain exactly 2 "
            "entries, got %d." % (name, len(param)))
        assert 0 <= param[0] <= 1.0 and 0 <= param[1] <= 1.0, (
            "Expected parameter '%s' of type tuple to contain two "
            "probabilities in the interval [0.0, 1.0]. "
            "Got values %.4f and %.4f." % (name, param[0], param[1]))
        return Binomial(Uniform(param[0], param[1]))

    if list_to_choice and ia.is_iterable(param):
        assert all([ia.is_single_number(v) for v in param]), (
            "Expected iterable parameter '%s' to only contain numbers, "
            "got %s." % (name, [type(v) for v in param],))
        assert all([0 <= p_i <= 1.0 for p_i in param]), (
            "Expected iterable parameter '%s' to only contain probabilities "
            "in the interval [0.0, 1.0], got values %s." % (
                name, ", ".join(["%.4f" % (p_i,) for p_i in param])))
        return Binomial(Choice(param))

    if isinstance(param, StochasticParameter):
        return param

    raise Exception(
        "Expected boolean or number or StochasticParameter for %s, "
        "got %s." % (name, type(param),))


def force_np_float_dtype(val):
    if val.dtype.kind == "f":
        return val
    return val.astype(np.float64)


def both_np_float_if_one_is_float(a, b):
    # pylint: disable=invalid-name
    a_f = a.dtype.type in ia.NP_FLOAT_TYPES
    b_f = b.dtype.type in ia.NP_FLOAT_TYPES
    if a_f and b_f:
        return a, b
    if a_f:
        return a, b.astype(np.float64)
    if b_f:
        return a.astype(np.float64), b
    return a.astype(np.float64), b.astype(np.float64)


def draw_distributions_grid(params, rows=None, cols=None,
                            graph_sizes=(350, 350), sample_sizes=None,
                            titles=None):
    if titles is None:
        titles = [None] * len(params)
    elif titles is False:
        titles = [False] * len(params)

    if sample_sizes is not None:
        images = [
            param_i.draw_distribution_graph(size=size_i, title=title_i)
            for param_i, size_i, title_i in zip(params, sample_sizes, titles)]
    else:
        images = [
            param_i.draw_distribution_graph(title=title_i)
            for param_i, title_i in zip(params, titles)]

    images_rs = ia.imresize_many_images(images, sizes=graph_sizes)
    grid = ia.draw_grid(images_rs, rows=rows, cols=cols)
    return grid


def show_distributions_grid(params, rows=None, cols=None,
                            graph_sizes=(350, 350), sample_sizes=None,
                            titles=None):
    ia.imshow(
        draw_distributions_grid(
            params,
            graph_sizes=graph_sizes,
            sample_sizes=sample_sizes,
            rows=rows,
            cols=cols,
            titles=titles
        )
    )


@six.add_metaclass(ABCMeta)
class StochasticParameter(object):
    """Abstract parent class for all stochastic parameters.

    Stochastic parameters are here all parameters from which values are
    supposed to be sampled. Usually the sampled values are to a degree random.
    E.g. a stochastic parameter may be the uniform distribution over the
    interval ``[-10, 10]``. Samples from that distribution (and therefore the
    stochastic parameter) could be ``5.2``, ``-3.7``, ``-9.7``, ``6.4``, etc.

    """

    def __init__(self):
        pass

    def draw_sample(self, random_state=None):
        """
        Draws a single sample value from this parameter.

        Parameters
        ----------
        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            A seed or random number generator to use during the sampling
            process. If ``None``, the global RNG will be used.
            See also :func:`~imgaug.augmenters.meta.Augmenter.__init__`
            for a similar parameter with more details.

        Returns
        -------
        any
            A single sample value.

        """
        return self.draw_samples(1, random_state=random_state)[0]

    def draw_samples(self, size, random_state=None):
        """Draw one or more samples from the parameter.

        Parameters
        ----------
        size : tuple of int or int
            Number of samples by dimension.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            A seed or random number generator to use during the sampling
            process. If ``None``, the global RNG will be used.
            See also :func:`~imgaug.augmenters.meta.Augmenter.__init__`
            for a similar parameter with more details.

        Returns
        -------
        ndarray
            Sampled values. Usually a numpy ndarray of basically any dtype,
            though not strictly limited to numpy arrays. Its shape is expected
            to match `size`.

        """
        random_state = iarandom.RNG(random_state)
        samples = self._draw_samples(
            size if not ia.is_single_integer(size) else tuple([size]),
            random_state)
        random_state.advance_()
        return samples

    @abstractmethod
    def _draw_samples(self, size, random_state):
        raise NotImplementedError()

    def __add__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Add(self, other)
        raise Exception(
            "Invalid datatypes in: StochasticParameter + %s. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __sub__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Subtract(self, other)
        raise Exception(
            "Invalid datatypes in: StochasticParameter - %s. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __mul__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Multiply(self, other)
        raise Exception(
            "Invalid datatypes in: StochasticParameter * %s. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __pow__(self, other, z=None):
        if z is not None:
            raise NotImplementedError(
                "Modulo power is currently not supported by "
                "StochasticParameter.")
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Power(self, other)
        raise Exception(
            "Invalid datatypes in: StochasticParameter ** %s. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __div__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(self, other)
        raise Exception(
            "Invalid datatypes in: StochasticParameter / %s. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __truediv__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(self, other)
        raise Exception(
            "Invalid datatypes in: StochasticParameter / %s (truediv). "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __floordiv__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Discretize(Divide(self, other))
        raise Exception(
            "Invalid datatypes in: StochasticParameter // %s (floordiv). "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __radd__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Add(other, self)
        raise Exception(
            "Invalid datatypes in: %s + StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __rsub__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Subtract(other, self)
        raise Exception(
            "Invalid datatypes in: %s - StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __rmul__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Multiply(other, self)
        raise Exception(
            "Invalid datatypes in: %s * StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __rpow__(self, other, z=None):
        if z is not None:
            raise NotImplementedError(
                "Modulo power is currently not supported by "
                "StochasticParameter.")
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Power(other, self)
        raise Exception(
            "Invalid datatypes in: %s ** StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __rdiv__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(other, self)
        raise Exception(
            "Invalid datatypes in: %s / StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __rtruediv__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(other, self)
        raise Exception(
            "Invalid datatypes in: %s / StochasticParameter (rtruediv). "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def __rfloordiv__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Discretize(Divide(other, self))
        raise Exception(
            "Invalid datatypes in: StochasticParameter // %s (rfloordiv). "
            "Expected second argument to be number or "
            "StochasticParameter." % (type(other),))

    def copy(self):
        """Create a shallow copy of this parameter.

        Returns
        -------
        imgaug.parameters.StochasticParameter
            Shallow copy.

        """
        return copy_module.copy(self)

    def deepcopy(self):
        """Create a deep copy of this parameter.

        Returns
        -------
        imgaug.parameters.StochasticParameter
            Deep copy.

        """
        return copy_module.deepcopy(self)

    def draw_distribution_graph(self, title=None, size=(1000, 1000), bins=100):
        """Generate an image visualizing the parameter's sample distribution.

        Parameters
        ----------
        title : None or False or str, optional
            Title of the plot. ``None`` is automatically replaced by a title
            derived from ``str(param)``. If set to ``False``, no title will be
            shown.

        size : tuple of int
            Number of points to sample. This is always expected to have at
            least two values. The first defines the number of sampling runs,
            the second (and further) dimensions define the size assigned
            to each :func:`~imgaug.parameters.StochasticParameter.draw_samples`
            call. E.g. ``(10, 20, 15)`` will lead to ``10`` calls of
            ``draw_samples(size=(20, 15))``. The results will be merged to a
            single 1d array.

        bins : int
            Number of bins in the plot histograms.

        Returns
        -------
        data : (H,W,3) ndarray
            Image of the plot.

        """
        # import only when necessary (faster startup; optional dependency;
        # less fragile -- see issue #225)
        import matplotlib.pyplot as plt

        points = []
        for _ in sm.xrange(size[0]):
            points.append(self.draw_samples(size[1:]).flatten())
        points = np.concatenate(points)

        fig = plt.figure()
        fig.add_subplot(111)
        ax = fig.gca()
        heights, bins = np.histogram(points, bins=bins)
        heights = heights / sum(heights)
        ax.bar(bins[:-1], heights,
               width=(max(bins) - min(bins))/len(bins),
               color="blue",
               alpha=0.75)

        if title is None:
            title = str(self)
        if title is not False:
            # split long titles - otherwise matplotlib generates errors
            title_fragments = [title[i:i+50]
                               for i in sm.xrange(0, len(title), 50)]
            ax.set_title("\n".join(title_fragments))
        fig.tight_layout(pad=0)

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            # we don't add bbox_inches='tight' here so that
            # draw_distributions_grid has an easier time combining many plots
            fig.savefig(f.name)
            data = imageio.imread(f)[..., 0:3]

        plt.close()

        return data


class Deterministic(StochasticParameter):
    """Parameter that is a constant value.

    If ``N`` values are sampled from this parameter, it will return ``N`` times
    ``V``, where ``V`` is the constant value.

    Parameters
    ----------
    value : number or str or imgaug.parameters.StochasticParameter
        A constant value to use.
        A string may be provided to generate arrays of strings.
        If this is a StochasticParameter, a single value will be sampled
        from it exactly once and then used as the constant value.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Deterministic(10)
    >>> param.draw_sample()
    10

    Will always sample the value 10.

    """
    def __init__(self, value):
        super(Deterministic, self).__init__()

        if isinstance(value, StochasticParameter):
            self.value = value.draw_sample()
        elif ia.is_single_number(value) or ia.is_string(value):
            self.value = value
        else:
            raise Exception("Expected StochasticParameter object or number or "
                            "string, got %s." % (type(value),))

    def _draw_samples(self, size, random_state):
        kwargs = {}
        if ia.is_single_integer(self.value):
            kwargs = {"dtype": np.int32}
        elif ia.is_single_float(self.value):
            kwargs = {"dtype": np.float32}
        return np.full(size, self.value, **kwargs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if ia.is_single_integer(self.value):
            return "Deterministic(int %d)" % (self.value,)
        if ia.is_single_float(self.value):
            return "Deterministic(float %.8f)" % (self.value,)
        return "Deterministic(%s)" % (str(self.value),)


# TODO replace two-value parameters used in tests with this
class DeterministicList(StochasticParameter):
    """Parameter that repeats elements from a list in the given order.

    E.g. of samples of shape ``(A, B, C)`` are requested, this parameter will
    return the first ``A*B*C`` elements, reshaped to ``(A, B, C)`` from the
    provided list. If the list contains less than ``A*B*C`` elements, it
    will (by default) be tiled until it is long enough (i.e. the sampling
    will start again at the first element, if necessary multiple times).

    Added in 0.4.0.

    Parameters
    ----------
    values : ndarray or iterable of number
        An iterable of values to sample from in the order within the iterable.

    """

    # Added in 0.4.0.
    def __init__(self, values):
        super(DeterministicList, self).__init__()

        assert ia.is_iterable(values), (
            "Expected to get an iterable as input, got type %s." % (
                type(values).__name__,))
        assert len(values) > 0, ("Expected to get at least one value, got "
                                 "zero.")

        if ia.is_np_array(values):
            # this would not be able to handle e.g. [[1, 2], [3]] and output
            # dtype object due to the non-regular shape, hence we have the
            # else block
            self.values = values.flatten()
        else:
            self.values = np.array(list(ia.flatten(values)))
            kind = self.values.dtype.kind

            # limit to 32bit instead of 64bit for efficiency
            if kind == "i":
                self.values = self.values.astype(np.int32)
            elif kind == "f":
                self.values = self.values.astype(np.float32)

    # Added in 0.4.0.
    def _draw_samples(self, size, random_state):
        nb_requested = int(np.prod(size))
        values = self.values
        if nb_requested > self.values.size:
            # we don't use itertools.cycle() here, as that would require
            # running through a loop potentially many times (as `size` can
            # be very large), which would be slow
            multiplier = int(np.ceil(nb_requested / values.size))
            values = np.tile(values, (multiplier,))
        return values[:nb_requested].reshape(size)

    # Added in 0.4.0.
    def __repr__(self):
        return self.__str__()

    # Added in 0.4.0.
    def __str__(self):
        if self.values.dtype.kind == "f":
            values = ["%.4f" % (value,) for value in self.values]
            return "DeterministicList([%s])" % (", ".join(values),)
        return "DeterministicList(%s)" % (str(self.values.tolist()),)


class Choice(StochasticParameter):
    """Parameter that samples value from a list of allowed values.

    Parameters
    ----------
    a : iterable
        List of allowed values.
        Usually expected to be ``int`` s, ``float`` s or ``str`` s.
        May also contain ``StochasticParameter`` s. Each
        ``StochasticParameter`` that is randomly picked will automatically be
        replaced by a sample of itself (or by ``N`` samples if the parameter
        was picked ``N`` times).

    replace : bool, optional
        Whether to perform sampling with or without replacing.

    p : None or iterable of number, optional
        Probabilities of each element in `a`.
        Must have the same length as `a` (if provided).

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Choice([5, 17, 25], p=[0.25, 0.5, 0.25])
    >>> sample = param.draw_sample()
    >>> assert sample in [5, 17, 25]

    Create and sample from a parameter, which will produce with ``50%``
    probability the sample ``17`` and in the other ``50%`` of all cases the
    sample ``5`` or ``25``..

    """
    def __init__(self, a, replace=True, p=None):
        # pylint: disable=invalid-name
        super(Choice, self).__init__()

        assert ia.is_iterable(a), (
            "Expected a to be an iterable (e.g. list), got %s." % (type(a),))
        self.a = a
        self.replace = replace
        if p is not None:
            assert ia.is_iterable(p), (
                "Expected p to be None or an iterable, got %s." % (type(p),))
            assert len(p) == len(a), (
                "Expected lengths of a and p to be identical, "
                "got %d and %d." % (len(a), len(p)))
        self.p = p

    def _draw_samples(self, size, random_state):
        if any([isinstance(a_i, StochasticParameter) for a_i in self.a]):
            rngs = random_state.duplicate(1+len(self.a))
            samples = rngs[0].choice(
                self.a, np.prod(size), replace=self.replace, p=self.p)

            # collect the sampled parameters and how many samples must be taken
            # from each of them
            params_counter = defaultdict(lambda: 0)
            for sample in samples:
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    params_counter[key] += 1

            # collect per parameter once the required number of samples
            # iterate here over self.a to always use the same seed for
            # the same parameter
            # TODO this might fail if the same parameter is added multiple
            #      times to self.a?
            # TODO this will fail if a parameter cant handle size=(N,)
            param_to_samples = dict()
            for i, param in enumerate(self.a):
                key = str(param)
                if key in params_counter:
                    param_to_samples[key] = param.draw_samples(
                        size=(params_counter[key],),
                        random_state=rngs[1+i]
                    )

            # assign the values sampled from the parameters to the `samples`
            # array by replacing the respective parameter
            param_to_readcount = defaultdict(lambda: 0)
            for i, sample in enumerate(samples):
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    readcount = param_to_readcount[key]
                    samples[i] = param_to_samples[key][readcount]
                    param_to_readcount[key] += 1

            samples = samples.reshape(size)
        else:
            samples = random_state.choice(self.a, size, replace=self.replace,
                                          p=self.p)

        dtype = samples.dtype
        if dtype.itemsize*8 > 32:
            # strings have kind "U"
            kind = dtype.kind
            if kind == "i":
                samples = samples.astype(np.int32)
            elif kind == "u":
                samples = samples.astype(np.uint32)
            elif kind == "f":
                samples = samples.astype(np.float32)

        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Choice(a=%s, replace=%s, p=%s)" % (
            str(self.a), str(self.replace), str(self.p),)


class Binomial(StochasticParameter):
    """Binomial distribution.

    Parameters
    ----------
    p : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Probability of the binomial distribution. Expected to be in the
        interval ``[0.0, 1.0]``.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`Binomial.draw_sample` or
        :func:`Binomial.draw_samples`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Binomial(Uniform(0.01, 0.2))

    Create a binomial distribution that uses a varying probability between
    ``0.01`` and ``0.2``, randomly and uniformly estimated once per sampling
    call.

    """

    def __init__(self, p):
        super(Binomial, self).__init__()
        self.p = handle_continuous_param(p, "p")

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        assert 0 <= p <= 1.0, (
            "Expected probability p to be in the interval [0.0, 1.0], "
            "got %.4f." % (p,))
        return random_state.binomial(1, p, size).astype(np.int32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Binomial(%s)" % (self.p,)


class DiscreteUniform(StochasticParameter):
    """Uniform distribution over the discrete interval ``[a..b]``.

    Parameters
    ----------
    a : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Lower bound of the interval.
        If ``a>b``, `a` and `b` will automatically be flipped.
        If ``a==b``, all generated values will be identical to `a`.

            * If a single ``int``, this ``int`` will be used as a
              constant value.
            * If a ``tuple`` of two ``int`` s ``(a, b)``, the value will be
              sampled from the discrete interval ``[a..b]`` once per call.
            * If a ``list`` of ``int``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`DiscreteUniform.draw_sample` or
        :func:`DiscreteUniform.draw_samples`.

    b : int or imgaug.parameters.StochasticParameter
        Upper bound of the interval. Analogous to `a`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.DiscreteUniform(10, Choice([20, 30, 40]))
    >>> sample = param.draw_sample()
    >>> assert 10 <= sample <= 40

    Create a discrete uniform distribution which's interval differs between
    calls and can be ``[10..20]``, ``[10..30]`` or ``[10..40]``.

    """

    def __init__(self, a, b):
        # pylint: disable=invalid-name
        super(DiscreteUniform, self).__init__()

        self.a = handle_discrete_param(a, "a")
        self.b = handle_discrete_param(b, "b")

    def _draw_samples(self, size, random_state):
        # pylint: disable=invalid-name
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.full(size, a, dtype=np.int32)
        return random_state.integers(a, b + 1, size, dtype=np.int32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DiscreteUniform(%s, %s)" % (self.a, self.b)


class Poisson(StochasticParameter):
    """Parameter that resembles a poisson distribution.

    A poisson distribution with ``lambda=0`` has its highest probability at
    point ``0`` and decreases quickly from there.
    Poisson distributions are discrete and never negative.

    Parameters
    ----------
    lam : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Lambda parameter of the poisson distribution.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`Poisson.draw_sample` or
        :func:`Poisson.draw_samples`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Poisson(1)
    >>> sample = param.draw_sample()
    >>> assert sample >= 0

    Create a poisson distribution with ``lambda=1`` and sample a value from
    it.

    """

    def __init__(self, lam):
        super(Poisson, self).__init__()

        self.lam = handle_continuous_param(lam, "lam")

    def _draw_samples(self, size, random_state):
        lam = self.lam.draw_sample(random_state=random_state)
        lam = max(lam, 0)

        return random_state.poisson(lam=lam, size=size).astype(np.int32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Poisson(%s)" % (self.lam,)


class Normal(StochasticParameter):
    """Parameter that resembles a normal/gaussian distribution.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The mean of the normal distribution.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`Laplace.draw_sample` or
        :func:`Laplace.draw_samples`.

    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The standard deviation of the normal distribution.
        If this parameter reaches ``0``, the output array will be filled with
        `loc`.
        Datatype behaviour is the analogous to `loc`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Normal(Choice([-1.0, 1.0]), 1.0)

    Create a gaussian distribution with a mean that differs by call.
    Samples values may sometimes follow ``N(-1.0, 1.0)`` and sometimes
    ``N(1.0, 1.0)``.

    """
    def __init__(self, loc, scale):
        super(Normal, self).__init__()

        self.loc = handle_continuous_param(loc, "loc")
        self.scale = handle_continuous_param(scale, "scale",
                                             value_range=(0, None))

    def _draw_samples(self, size, random_state):
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        assert scale >= 0, "Expected scale to be >=0, got %.4f." % (scale,)
        if scale == 0:
            return np.full(size, loc, dtype=np.float32)
        return random_state.normal(loc, scale, size=size).astype(np.float32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % (self.loc, self.scale)


# TODO docstring for parameters is outdated
class TruncatedNormal(StochasticParameter):
    """Parameter that resembles a truncated normal distribution.

    A truncated normal distribution is similar to a normal distribution,
    except the domain is smoothly bounded to a min and max value.

    This is a wrapper around :func:`scipy.stats.truncnorm`.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The mean of the normal distribution.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`TruncatedNormal.draw_sample` or
        :func:`TruncatedNormal.draw_samples`.

    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The standard deviation of the normal distribution.
        If this parameter reaches ``0``, the output array will be filled with
        `loc`.
        Datatype behaviour is the same as for `loc`.

    low : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The minimum value of the truncated normal distribution.
        Datatype behaviour is the same as for `loc`.

    high : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The maximum value of the truncated normal distribution.
        Datatype behaviour is the same as for `loc`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.TruncatedNormal(0, 5.0, low=-10, high=10)
    >>> samples = param.draw_samples(100, random_state=0)
    >>> assert np.all(samples >= -10)
    >>> assert np.all(samples <= 10)

    Create a truncated normal distribution with its minimum at ``-10.0``
    and its maximum at ``10.0``.

    """

    def __init__(self, loc, scale, low=-np.inf, high=np.inf):
        super(TruncatedNormal, self).__init__()

        self.loc = handle_continuous_param(loc, "loc")
        self.scale = handle_continuous_param(scale, "scale",
                                             value_range=(0, None))
        self.low = handle_continuous_param(low, "low")
        self.high = handle_continuous_param(high, "high")

    def _draw_samples(self, size, random_state):
        # pylint: disable=invalid-name
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        low = self.low.draw_sample(random_state=random_state)
        high = self.high.draw_sample(random_state=random_state)
        seed = random_state.generate_seed_()
        if low > high:
            low, high = high, low
        assert scale >= 0, "Expected scale to be >=0, got %.4f." % (scale,)
        if scale == 0:
            return np.full(size, fill_value=loc, dtype=np.float32)
        a = (low - loc) / scale
        b = (high - loc) / scale
        tnorm = scipy.stats.truncnorm(a=a, b=b, loc=loc, scale=scale)

        # Using a seed here works with both np.random interfaces.
        # Last time tried, scipy crashed when providing just
        # random_state.generator on the new np.random interface.
        return tnorm.rvs(size=size, random_state=seed).astype(np.float32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "TruncatedNormal(loc=%s, scale=%s, low=%s, high=%s)" % (
            self.loc, self.scale, self.low, self.high)


class Laplace(StochasticParameter):
    """Parameter that resembles a (continuous) laplace distribution.

    This is a wrapper around numpy's :func:`numpy.random.laplace`.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The position of the distribution peak, similar to the mean in normal
        distributions.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`Laplace.draw_sample` or
        :func:`Laplace.draw_samples`.

    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The exponential decay factor, similar to the standard deviation in
        gaussian distributions.
        If this parameter reaches ``0``, the output array will be filled with
        `loc`.
        Datatype behaviour is the analogous to `loc`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Laplace(0, 1.0)

    Create a laplace distribution, which's peak is at ``0`` and decay is
    ``1.0``.

    """
    def __init__(self, loc, scale):
        super(Laplace, self).__init__()

        self.loc = handle_continuous_param(loc, "loc")
        self.scale = handle_continuous_param(scale, "scale",
                                             value_range=(0, None))

    def _draw_samples(self, size, random_state):
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        assert scale >= 0, "Expected scale to be >=0, got %s." % (scale,)
        if scale == 0:
            return np.full(size, loc, dtype=np.float32)
        return random_state.laplace(loc, scale, size=size).astype(np.float32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Laplace(loc=%s, scale=%s)" % (self.loc, self.scale)


class ChiSquare(StochasticParameter):
    """Parameter that resembles a (continuous) chi-square distribution.

    This is a wrapper around numpy's :func:`numpy.random.chisquare`.

    Parameters
    ----------
    df : int or tuple of two int or list of int or imgaug.parameters.StochasticParameter
        Degrees of freedom. Expected value range is ``[1, inf)``.

            * If a single ``int``, this ``int`` will be used as a
              constant value.
            * If a ``tuple`` of two ``int`` s ``(a, b)``, the value will be
              sampled from the discrete interval ``[a..b]`` once per call.
            * If a ``list`` of ``int``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`ChiSquare.draw_sample` or
        :func:`ChiSquare.draw_samples`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.ChiSquare(df=2)

    Create a chi-square distribution with two degrees of freedom.

    """
    def __init__(self, df):
        # pylint: disable=invalid-name
        super(ChiSquare, self).__init__()

        self.df = handle_discrete_param(df, "df", value_range=(1, None))

    def _draw_samples(self, size, random_state):
        # pylint: disable=invalid-name
        df = self.df.draw_sample(random_state=random_state)
        assert df >= 1, "Expected df to be >=1, got %d." % (df,)
        return random_state.chisquare(df, size=size).astype(np.float32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "ChiSquare(df=%s)" % (self.df,)


class Weibull(StochasticParameter):
    """
    Parameter that resembles a (continuous) weibull distribution.

    This is a wrapper around numpy's :func:`numpy.random.weibull`.

    Parameters
    ----------
    a : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Shape parameter of the distribution.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`Weibull.draw_sample` or
        :func:`Weibull.draw_samples`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Weibull(a=0.5)

    Create a weibull distribution with shape 0.5.

    """
    def __init__(self, a):
        # pylint: disable=invalid-name
        super(Weibull, self).__init__()

        self.a = handle_continuous_param(a, "a", value_range=(0.0001, None))

    def _draw_samples(self, size, random_state):
        # pylint: disable=invalid-name
        a = self.a.draw_sample(random_state=random_state)
        assert a > 0, "Expected a to be >0, got %.4f." % (a,)
        return random_state.weibull(a, size=size).astype(np.float32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Weibull(a=%s)" % (self.a,)


# TODO rename (a, b) to (low, high) as in numpy?
class Uniform(StochasticParameter):
    """Parameter that resembles a uniform distribution over ``[a, b)``.

    Parameters
    ----------
    a : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Lower bound of the interval.
        If ``a>b``, `a` and `b` will automatically be flipped.
        If ``a==b``, all generated values will be identical to `a`.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`Uniform.draw_sample` or
        :func:`Uniform.draw_samples`.

    b : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Upper bound of the interval. Analogous to `a`.


    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Uniform(0, 10.0)
    >>> sample = param.draw_sample()
    >>> assert 0 <= sample < 10.0

    Create and sample from a uniform distribution over ``[0, 10.0)``.

    """
    def __init__(self, a, b):
        # pylint: disable=invalid-name
        super(Uniform, self).__init__()

        self.a = handle_continuous_param(a, "a")
        self.b = handle_continuous_param(b, "b")

    def _draw_samples(self, size, random_state):
        # pylint: disable=invalid-name
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.full(size, a, dtype=np.float32)
        return random_state.uniform(a, b, size).astype(np.float32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Uniform(%s, %s)" % (self.a, self.b)


class Beta(StochasticParameter):
    """Parameter that resembles a (continuous) beta distribution.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        alpha parameter of the beta distribution.
        Expected value range is ``(0, inf)``. Values below ``0`` are
        automatically clipped to ``0+epsilon``.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`Beta.draw_sample` or
        :func:`Beta.draw_samples`.

    beta : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Beta parameter of the beta distribution. Analogous to `alpha`.

    epsilon : number
        Clipping parameter. If `alpha` or `beta` end up ``<=0``, they are clipped to ``0+epsilon``.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Beta(0.4, 0.6)

    Create a beta distribution with ``alpha=0.4`` and ``beta=0.6``.

    """
    def __init__(self, alpha, beta, epsilon=0.0001):
        super(Beta, self).__init__()

        self.alpha = handle_continuous_param(alpha, "alpha")
        self.beta = handle_continuous_param(beta, "beta")

        assert ia.is_single_number(epsilon), (
            "Expected epsilon to a number, got type %s." % (type(epsilon),))
        self.epsilon = epsilon

    def _draw_samples(self, size, random_state):
        alpha = self.alpha.draw_sample(random_state=random_state)
        beta = self.beta.draw_sample(random_state=random_state)
        alpha = max(alpha, self.epsilon)
        beta = max(beta, self.epsilon)
        return random_state.beta(alpha, beta, size=size).astype(np.float32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Beta(%s, %s)" % (self.alpha, self.beta)


class FromLowerResolution(StochasticParameter):
    """Parameter to sample from other parameters at lower image resolutions.

    This parameter is intended to be used with parameters that would usually
    sample one value per pixel (or one value per pixel and channel). Instead
    of sampling from the other parameter at full resolution, it samples at
    lower resolution, e.g. ``0.5*H x 0.5*W`` with ``H`` being the height and
    ``W`` being the width. After the low-resolution sampling this parameter
    then upscales the result to ``HxW``.

    This parameter is intended to produce coarse samples. E.g. combining
    this with :class:`Binomial` can lead to large rectangular areas of
    ``1`` s and ``0`` s.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        The other parameter which is to be sampled on a coarser image.

    size_percent : None or number or iterable of number or imgaug.parameters.StochasticParameter, optional
        Size of the 2d sampling plane in percent of the requested size.
        I.e. this is relative to the size provided in the call to
        ``draw_samples(size)``. Lower values will result in smaller sampling
        planes, which are then upsampled to `size`. This means that lower
        values will result in larger rectangles. The size may be provided as
        a constant value or a tuple ``(a, b)``, which will automatically be
        converted to the continuous uniform range ``[a, b)`` or a
        :class:`StochasticParameter`, which will be queried per call to
        :func:`FromLowerResolution.draw_sample` and
        :func:`FromLowerResolution.draw_samples`.

    size_px : None or number or iterable of numbers or imgaug.parameters.StochasticParameter, optional
        Size of the 2d sampling plane in pixels.
        Lower values will result in smaller sampling planes, which are then
        upsampled to the input `size` of ``draw_samples(size)``.
        This means that lower values will result in larger rectangles.
        The size may be provided as a constant value or a tuple ``(a, b)``,
        which will automatically be converted to the discrete uniform
        range ``[a..b]`` or a :class:`StochasticParameter`, which will be
        queried once per call to :func:`FromLowerResolution.draw_sample` and
        :func:`FromLowerResolution.draw_samples`.

    method : str or int or imgaug.parameters.StochasticParameter, optional
        Upsampling/interpolation method to use. This is used after the sampling
        is finished and the low resolution plane has to be upsampled to the
        requested `size` in ``draw_samples(size, ...)``. The method may be
        the same as in :func:`~imgaug.imgaug.imresize_many_images`. Usually
        ``nearest`` or ``linear`` are good choices. ``nearest`` will result
        in rectangles with sharp edges and ``linear`` in rectangles with
        blurry and round edges. The method may be provided as a
        :class:`StochasticParameter`, which will be queried once per call to
        :func:`FromLowerResolution.draw_sample` and
        :func:`FromLowerResolution.draw_samples`.

    min_size : int, optional
        Minimum size in pixels of the low resolution sampling plane.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.FromLowerResolution(
    >>>     Binomial(0.05),
    >>>     size_px=(2, 16),
    >>>     method=Choice(["nearest", "linear"]))

    Samples from a binomial distribution with ``p=0.05``. The sampling plane
    will always have a size HxWxC with H and W being independently sampled
    from ``[2..16]`` (i.e. it may range from ``2x2xC`` up to ``16x16xC`` max,
    but may also be e.g. ``4x8xC``). The upsampling method will be ``nearest``
    in ``50%`` of all cases and ``linear`` in the other 50 percent. The result
    will sometimes be rectangular patches of sharp ``1`` s surrounded by
    ``0`` s and sometimes blurry blobs of ``1``s, surrounded by values
    ``<1.0``.

    """
    def __init__(self, other_param, size_percent=None, size_px=None,
                 method="nearest", min_size=1):
        super(FromLowerResolution, self).__init__()

        assert size_percent is not None or size_px is not None, (
            "Expected either 'size_percent' or 'size_px' to be provided, "
            "got neither of them.")

        if size_percent is not None:
            self.size_method = "percent"
            self.size_px = None
            if ia.is_single_number(size_percent):
                self.size_percent = Deterministic(size_percent)
            elif ia.is_iterable(size_percent):
                assert len(size_percent) == 2, (
                    "Expected iterable 'size_percent' to contain exactly 2 "
                    "values, got %d." % (len(size_percent),))
                self.size_percent = Uniform(size_percent[0], size_percent[1])
            elif isinstance(size_percent, StochasticParameter):
                self.size_percent = size_percent
            else:
                raise Exception(
                    "Expected int, float, tuple of two ints/floats or "
                    "StochasticParameter for size_percent, "
                    "got %s." % (type(size_percent),))
        else:  # = elif size_px is not None:
            self.size_method = "px"
            self.size_percent = None
            if ia.is_single_integer(size_px):
                self.size_px = Deterministic(size_px)
            elif ia.is_iterable(size_px):
                assert len(size_px) == 2, (
                    "Expected iterable 'size_px' to contain exactly 2 "
                    "values, got %d." % (len(size_px),))
                self.size_px = DiscreteUniform(size_px[0], size_px[1])
            elif isinstance(size_px, StochasticParameter):
                self.size_px = size_px
            else:
                raise Exception(
                    "Expected int, float, tuple of two ints/floats or "
                    "StochasticParameter for size_px, "
                    "got %s." % (type(size_px),))

        self.other_param = other_param

        if ia.is_string(method) or ia.is_single_integer(method):
            self.method = Deterministic(method)
        elif isinstance(method, StochasticParameter):
            self.method = method
        else:
            raise Exception("Expected string or StochasticParameter, "
                            "got %s." % (type(method),))

        self.min_size = min_size

    def _draw_samples(self, size, random_state):
        if len(size) == 3:
            n = 1
            h, w, c = size
        elif len(size) == 4:
            n, h, w, c = size
        else:
            raise Exception("FromLowerResolution can only generate samples "
                            "of shape (H, W, C) or (N, H, W, C), "
                            "requested was %s." % (str(size),))

        if self.size_method == "percent":
            hw_percents = self.size_percent.draw_samples(
                (n, 2), random_state=random_state)
            hw_pxs = (hw_percents * np.array([h, w])).astype(np.int32)
        else:
            hw_pxs = self.size_px.draw_samples(
                (n, 2), random_state=random_state)

        methods = self.method.draw_samples((n,), random_state=random_state)
        result = None
        for i, (hw_px, method) in enumerate(zip(hw_pxs, methods)):
            h_small = max(hw_px[0], self.min_size)
            w_small = max(hw_px[1], self.min_size)
            samples = self.other_param.draw_samples(
                (1, h_small, w_small, c), random_state=random_state)

            # This (1) makes sure that samples are of dtypes supported by
            # imresize_many_images, and (2) forces samples to be float-kind
            # if the requested interpolation is something else than nearest
            # neighbour interpolation. (2) is a bit hacky and makes sure that
            # continuous values are produced for e.g. cubic interpolation.
            # This is particularly important for e.g. binomial distributios
            # used in FromLowerResolution and thereby in e.g. CoarseDropout,
            # where integer-kinds would lead to sharp edges despite using
            # cubic interpolation.
            if samples.dtype.kind == "f":
                samples = iadt.restore_dtypes_(samples, np.float32)
            elif samples.dtype.kind == "i":
                if method == "nearest":
                    samples = iadt.restore_dtypes_(samples, np.int32)
                else:
                    samples = iadt.restore_dtypes_(samples, np.float32)
            else:
                assert samples.dtype.kind == "u", (
                    "FromLowerResolution can only process outputs of kind "
                    "f (float), i (int) or u (uint), got %s." % (
                        samples.dtype.kind))
                if method == "nearest":
                    samples = iadt.restore_dtypes_(samples, np.uint16)
                else:
                    samples = iadt.restore_dtypes_(samples, np.float32)

            samples_upscaled = ia.imresize_many_images(
                samples, (h, w), interpolation=method)

            if result is None:
                result = np.zeros((n, h, w, c), dtype=samples_upscaled.dtype)
            result[i] = samples_upscaled

        if len(size) == 3:
            return result[0]
        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.size_method == "percent":
            pattern = (
                "FromLowerResolution("
                "size_percent=%s, method=%s, other_param=%s"
                ")")
            return pattern % (self.size_percent, self.method, self.other_param)

        pattern = (
            "FromLowerResolution("
            "size_px=%s, method=%s, other_param=%s"
            ")")
        return pattern % (self.size_px, self.method, self.other_param)


class Clip(StochasticParameter):
    """Clip another parameter to a defined value range.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        The other parameter, which's values are to be clipped.

    minval : None or number, optional
        The minimum value to use.
        If ``None``, no minimum will be used.

    maxval : None or number, optional
        The maximum value to use.
        If ``None``, no maximum will be used.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Clip(Normal(0, 1.0), minval=-2.0, maxval=2.0)

    Create a standard gaussian distribution, which's values never go below
    ``-2.0`` or above ``2.0``. Note that this will lead to small "bumps" of
    higher probability at ``-2.0`` and ``2.0``, as values below/above these
    will be clipped to them. For smoother limitations on gaussian
    distributions, see :class:`TruncatedNormal`.

    """

    def __init__(self, other_param, minval=None, maxval=None):
        super(Clip, self).__init__()

        _assert_arg_is_stoch_param("other_param", other_param)
        assert minval is None or ia.is_single_number(minval), (
            "Expected 'minval' to be None or a number, got type %s." % (
                type(minval),))
        assert maxval is None or ia.is_single_number(maxval), (
            "Expected 'maxval' to be None or a number, got type %s." % (
                type(maxval),))

        self.other_param = other_param
        self.minval = minval
        self.maxval = maxval

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(size, random_state=random_state)
        if self.minval is not None or self.maxval is not None:
            # Note that this would produce a warning if 'samples' is int64
            # or uint64
            samples = np.clip(samples, self.minval, self.maxval, out=samples)
        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        if self.minval is not None and self.maxval is not None:
            return "Clip(%s, %.6f, %.6f)" % (
                opstr, float(self.minval), float(self.maxval))
        if self.minval is not None:
            return "Clip(%s, %.6f, None)" % (opstr, float(self.minval))
        if self.maxval is not None:
            return "Clip(%s, None, %.6f)" % (opstr, float(self.maxval))
        return "Clip(%s, None, None)" % (opstr,)


class Discretize(StochasticParameter):
    """Convert a continuous distribution to a discrete one.

    This will round the values and then cast them to integers.
    Values sampled from already discrete distributions are not changed.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        The other parameter, which's values are to be discretized.

    round : bool, optional
        Whether to round before converting to integer dtype.

        Added in 0.4.0.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Discretize(iap.Normal(0, 1.0))

    Create a discrete standard gaussian distribution.

    """
    def __init__(self, other_param, round=True):
        # pylint: disable=redefined-builtin
        super(Discretize, self).__init__()
        _assert_arg_is_stoch_param("other_param", other_param)
        self.other_param = other_param
        self.round = round

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(size, random_state=random_state)
        assert samples.dtype.kind in ["u", "i", "b", "f"], (
            "Expected to get uint, int, bool or float dtype as samples in "
            "Discretize(), but got dtype '%s' (kind '%s') instead." % (
                samples.dtype.name, samples.dtype.kind))

        if samples.dtype.kind in ["u", "i", "b"]:
            return samples

        # floats seem to reliably cover ints that have half the number of
        # bits -- probably not the case for float128 though as that is
        # really float96
        bitsize = 8 * samples.dtype.itemsize // 2
        # in case some weird system knows something like float8 we set a
        # lower bound here -- shouldn't happen though
        bitsize = max(bitsize, 8)
        dtype = np.dtype("int%d" % (bitsize,))
        if self.round:
            samples = np.round(samples)
        return samples.astype(dtype)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Discretize(%s, round=%s)" % (opstr, str(self.round))


class Multiply(StochasticParameter):
    """Multiply the samples of another stochastic parameter.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Other parameter which's sampled values are to be multiplied with `val`.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a single ``number``, this ``number`` will be used as a
              constant value to fill an array of shape ``S``.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, an array of
              shape ``S`` will be filled with uniformly sampled values from
              the continuous interval ``[a, b)``.
            * If a ``list`` of ``number``, an array of shape ``S`` will be
              filled with randomly picked values from the ``list``.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of :func:`Multiply.draw_sample` or
        :func:`Multiply.draw_samples`.

    val : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Multiplier to use.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant multiplier.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and multiplied elementwise with the samples of `other_param`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Multiply(iap.Uniform(0.0, 1.0), -1)

    Convert a uniform distribution from ``[0.0, 1.0)`` to ``(-1.0, 0.0]``.

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Multiply, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = (
            self.elementwise
            and not isinstance(self.val, Deterministic))

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=rngs[1])
        else:
            val_samples = self.val.draw_sample(random_state=rngs[1])

        if elementwise:
            return np.multiply(samples, val_samples)
        return samples * val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Multiply(%s, %s, %s)" % (
            str(self.other_param), str(self.val), self.elementwise)


class Divide(StochasticParameter):
    """Divide the samples of another stochastic parameter.

    This parameter will automatically prevent division by zero (uses 1.0)
    as the denominator in these cases.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Other parameter which's sampled values are to be divided by `val`.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a single ``number``, this ``number`` will be used as a
              constant value to fill an array of shape ``S``.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, an array of
              shape ``S`` will be filled with uniformly sampled values from
              the continuous interval ``[a, b)``.
            * If a ``list`` of ``number``, an array of shape ``S`` will be
              filled with randomly picked values from the ``list``.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of :func:`Divide.draw_sample` or
        :func:`Divide.draw_samples`.

    val : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Denominator to use.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant denominator.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and used to divide the samples of `other_param` elementwise.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Divide(iap.Uniform(0.0, 1.0), 2)

    Convert a uniform distribution ``[0.0, 1.0)`` to ``[0, 0.5)``.

    """

    def __init__(self, other_param, val, elementwise=False):
        super(Divide, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        # pylint: disable=no-else-return
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = (
            self.elementwise
            and not isinstance(self.val, Deterministic))

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=rngs[1])

            # prevent division by zero
            val_samples[val_samples == 0] = 1

            return np.divide(
                force_np_float_dtype(samples),
                force_np_float_dtype(val_samples)
            )
        else:
            val_sample = self.val.draw_sample(random_state=rngs[1])

            # prevent division by zero
            if val_sample == 0:
                val_sample = 1

            return force_np_float_dtype(samples) / float(val_sample)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Divide(%s, %s, %s)" % (
            str(self.other_param), str(self.val), self.elementwise)


# TODO sampling (N,) from something like 10+Uniform(0, 1) will return
#      N times the same value as (N,) values will be sampled from 10, but only
#      one from Uniform() unless elementwise=True is explicitly set. That
#      seems unintuitive. How can this be prevented?
class Add(StochasticParameter):
    """Add to the samples of another stochastic parameter.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Samples of `val` will be added to samples of this parameter.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a single ``number``, this ``number`` will be used as a
              constant value to fill an array of shape ``S``.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, an array of
              shape ``S`` will be filled with uniformly sampled values from
              the continuous interval ``[a, b)``.
            * If a ``list`` of ``number``, an array of shape ``S`` will be
              filled with randomly picked values from the ``list``.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of :func:`Add.draw_sample` or
        :func:`Add.draw_samples`.

    val : number or tuple of two number or list of number or imgaug.parameters.StochasticParameter
        Value to add to the samples of `other_param`.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant multiplier.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and added elementwise with the samples of `other_param`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Add(Uniform(0.0, 1.0), 1.0)

    Convert a uniform distribution from ``[0.0, 1.0)`` to ``[1.0, 2.0)``.

    """

    def __init__(self, other_param, val, elementwise=False):
        super(Add, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = (
            self.elementwise and not isinstance(self.val, Deterministic))

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=rngs[1])
        else:
            val_samples = self.val.draw_sample(random_state=rngs[1])

        if elementwise:
            return np.add(samples, val_samples)
        return samples + val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Add(%s, %s, %s)" % (
            str(self.other_param), str(self.val), self.elementwise)


class Subtract(StochasticParameter):
    """Subtract from the samples of another stochastic parameter.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Samples of `val` will be subtracted from samples of this parameter.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a single ``number``, this ``number`` will be used as a
              constant value to fill an array of shape ``S``.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, an array of
              shape ``S`` will be filled with uniformly sampled values from
              the continuous interval ``[a, b)``.
            * If a ``list`` of ``number``, an array of shape ``S`` will be
              filled with randomly picked values from the ``list``.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of :func:`Subtract.draw_sample` or
        :func:`Subtract.draw_samples`.

    val : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Value to subtract from the other parameter.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant multiplier.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and subtracted elementwise from the samples of `other_param`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Subtract(iap.Uniform(0.0, 1.0), 1.0)

    Convert a uniform distribution from ``[0.0, 1.0)`` to ``[-1.0, 0.0)``.

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Subtract, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = (self.elementwise
                       and not isinstance(self.val, Deterministic))

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=rngs[1])
        else:
            val_samples = self.val.draw_sample(random_state=rngs[1])

        if elementwise:
            return np.subtract(samples, val_samples)
        return samples - val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Subtract(%s, %s, %s)" % (
            str(self.other_param), str(self.val), self.elementwise)


class Power(StochasticParameter):
    """Exponentiate the samples of another stochastic parameter.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Other parameter which's sampled values are to be exponentiated by `val`.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a single ``number``, this ``number`` will be used as a
              constant value to fill an array of shape ``S``.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, an array of
              shape ``S`` will be filled with uniformly sampled values from
              the continuous interval ``[a, b)``.
            * If a ``list`` of ``number``, an array of shape ``S`` will be
              filled with randomly picked values from the ``list``.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of :func:`Power.draw_sample` or
        :func:`Power.draw_samples`.

    val : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Value to use exponentiate the samples of `other_param`.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant multiplier.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and used to exponentiate elementwise the samples of `other_param`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Power(iap.Uniform(0.0, 1.0), 2)

    Converts a uniform range ``[0.0, 1.0)`` to a distribution that is peaked
    towards 1.0.

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Power, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = (
            self.elementwise
            and not isinstance(self.val, Deterministic))

        if elementwise:
            exponents = self.val.draw_samples(size, random_state=rngs[1])
        else:
            exponents = self.val.draw_sample(random_state=rngs[1])

        # without this we get int results in the case of
        # Power(<int>, <stochastic float param>)
        samples, exponents = both_np_float_if_one_is_float(samples, exponents)
        samples_dtype = samples.dtype

        # TODO switch to this as numpy>=1.15 is now a requirement
        #      float_power requires numpy>=1.12
        # result = np.float_power(samples, exponents)
        # TODO why was float32 type here replaced with complex number
        #      formulation?
        result = np.power(samples.astype(np.complex), exponents).real
        if result.dtype != samples_dtype:
            result = result.astype(samples_dtype)

        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Power(%s, %s, %s)" % (
            str(self.other_param), str(self.val), self.elementwise)


class Absolute(StochasticParameter):
    """Convert the samples of another parameter to their absolute values.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        Other parameter which's sampled values are to be modified.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Absolute(iap.Uniform(-1.0, 1.0))

    Convert a uniform distribution from ``[-1.0, 1.0)`` to ``[0.0, 1.0]``.

    """
    def __init__(self, other_param):
        super(Absolute, self).__init__()

        _assert_arg_is_stoch_param("other_param", other_param)

        self.other_param = other_param

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(size, random_state=random_state)
        return np.absolute(samples)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Absolute(%s)" % (opstr,)


class RandomSign(StochasticParameter):
    """Convert a parameter's samples randomly to positive or negative values.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        Other parameter which's sampled values are to be modified.

    p_positive : number
        Fraction of values that are supposed to be turned to positive values.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.RandomSign(iap.Poisson(1))

    Create a poisson distribution with ``alpha=1`` that is mirrored/copied (not
    flipped) at the y-axis.

    """

    def __init__(self, other_param, p_positive=0.5):
        super(RandomSign, self).__init__()

        _assert_arg_is_stoch_param("other_param", other_param)
        assert ia.is_single_number(p_positive), (
            "Expected 'p_positive' to be a number, got %s." % (
                type(p_positive)))
        assert 0.0 <= p_positive <= 1.0, (
            "Expected 'p_positive' to be in the interval [0.0, 1.0], "
            "got %.4f." % (p_positive,))

        self.other_param = other_param
        self.p_positive = p_positive

    def _draw_samples(self, size, random_state):
        rss = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rss[0])
        # TODO add method to change from uint to int here instead of assert
        assert samples.dtype.kind in ["f", "i"], (
            "Expected to get samples of kind float or int, but got dtype %s "
            "of kind %s." % (samples.dtype.name, samples.dtype.kind))
        # TODO convert to same kind as samples
        coinflips = rss[1].binomial(
            1, self.p_positive, size=size).astype(np.int8)
        signs = coinflips * 2 - 1
        # Add absolute here to guarantee that we get p_positive percent of
        # positive values. Otherwise we would merely flip p_positive percent
        # of all signs.
        # TODO test if
        #          result[coinflips_mask] *= (-1)
        #      is faster  (with protection against mask being empty?)
        result = np.absolute(samples) * signs
        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "RandomSign(%s, %.2f)" % (opstr, self.p_positive)


class ForceSign(StochasticParameter):
    """Convert a parameter's samples to either positive or negative values.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        Other parameter which's sampled values are to be modified.

    positive : bool
        Whether to force all signs to be positive (``True``) or
        negative (``False``).

    mode : {'invert', 'reroll'}, optional
        Method to change the signs. Valid values are ``invert`` and ``reroll``.
        ``invert`` means that wrong signs are simply flipped.
        ``reroll`` means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional
        If `mode` is set to ``reroll``, this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        ``mode="invert"``). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.ForceSign(iap.Poisson(1), positive=False)

    Create a poisson distribution with ``alpha=1`` that is flipped towards
    negative values.

    """

    def __init__(self, other_param, positive, mode="invert",
                 reroll_count_max=2):
        super(ForceSign, self).__init__()

        _assert_arg_is_stoch_param("other_param", other_param)
        self.other_param = other_param

        assert positive in [True, False], (
            "Expected 'positive' to be True or False, got type %s." % (
                type(positive),))
        self.positive = positive

        assert mode in ["invert", "reroll"], (
            "Expected 'mode' to be \"invert\" or \"reroll\", got %s." % (mode,))
        self.mode = mode

        assert ia.is_single_integer(reroll_count_max), (
            "Expected 'reroll_count_max' to be an integer, got type %s." % (
                type(reroll_count_max)))
        self.reroll_count_max = reroll_count_max

    def _draw_samples(self, size, random_state):
        rngs = random_state.duplicate(1+self.reroll_count_max)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        if self.mode == "invert":
            if self.positive:
                samples[samples < 0] *= (-1)
            else:
                samples[samples > 0] *= (-1)
        else:
            if self.positive:
                bad_samples = np.where(samples < 0)[0]
            else:
                bad_samples = np.where(samples > 0)[0]

            reroll_count = 0
            while len(bad_samples) > 0 and reroll_count < self.reroll_count_max:
                # This rerolls the full input size, even when only a tiny
                # fraction of the values were wrong. That is done, because not
                # all parameters necessarily support any number of dimensions
                # for `size`, so we cant just resample size=N for N values
                # with wrong signs.
                # There is still quite some room for improvement here.
                samples_reroll = self.other_param.draw_samples(
                    size,
                    random_state=rngs[1+reroll_count]
                )
                samples[bad_samples] = samples_reroll[bad_samples]

                reroll_count += 1
                if self.positive:
                    bad_samples = np.where(samples < 0)[0]
                else:
                    bad_samples = np.where(samples > 0)[0]

            if len(bad_samples) > 0:
                samples[bad_samples] *= (-1)

        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "ForceSign(%s, %s, %s, %d)" % (
            opstr, str(self.positive), self.mode, self.reroll_count_max)


def Positive(other_param, mode="invert", reroll_count_max=2):
    """Convert another parameter's results to positive values.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    mode : {'invert', 'reroll'}, optional
        How to change the signs. Valid values are ``invert`` and ``reroll``.
        ``invert`` means that wrong signs are simply flipped.
        ``reroll`` means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional
        If `mode` is set to ``reroll``, this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        ``mode="invert"``). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Positive(iap.Normal(0, 1), mode="reroll")

    Create a gaussian distribution that has only positive values.
    If any negative value is sampled in the process, that sample is resampled
    up to two times to get a positive one. If it isn't positive after the
    second resampling step, the sign is simply flipped.

    """
    # pylint: disable=invalid-name
    return ForceSign(
        other_param=other_param,
        positive=True,
        mode=mode,
        reroll_count_max=reroll_count_max
    )


def Negative(other_param, mode="invert", reroll_count_max=2):
    """Convert another parameter's results to negative values.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    mode : {'invert', 'reroll'}, optional
        How to change the signs. Valid values are ``invert`` and ``reroll``.
        ``invert`` means that wrong signs are simply flipped.
        ``reroll`` means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional
        If `mode` is set to ``reroll``, this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        ``mode="invert"``). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Negative(iap.Normal(0, 1), mode="reroll")

    Create a gaussian distribution that has only negative values.
    If any positive value is sampled in the process, that sample is resampled
    up to two times to get a negative one. If it isn't negative after the
    second resampling step, the sign is simply flipped.

    """
    # pylint: disable=invalid-name
    return ForceSign(
        other_param=other_param,
        positive=False,
        mode=mode,
        reroll_count_max=reroll_count_max
    )


# TODO this always aggregates the result in high resolution space, instead of
#      aggregating them in low resolution and then only upscaling the final
#      image (for N iterations that would save up to N-1 upscales)
class IterativeNoiseAggregator(StochasticParameter):
    """Aggregate multiple iterations of samples from another parameter.

    This is supposed to be used in conjunction with :class:`SimplexNoise` or
    :class:`FrequencyNoise`. If a shape ``S`` is requested, it will request
    ``I`` times ``S`` samples from the underlying parameter, where ``I`` is
    the number of iterations. The ``I`` arrays will be combined to a single
    array of shape ``S`` using an aggregation method, e.g. simple averaging.

    Parameters
    ----------
    other_param : StochasticParameter
        The other parameter from which to sample one or more times.

    iterations : int or iterable of int or list of int or imgaug.parameters.StochasticParameter, optional
        The number of iterations.

            * If a single ``int``, this ``int`` will be used as a
              constant value.
            * If a ``tuple`` of two ``int`` s ``(a, b)``, the value will be
              sampled from the discrete interval ``[a..b]`` once per call.
            * If a ``list`` of ``int``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of
        :func:`IterativeNoiseAggregator.draw_sample` or
        :func:`IterativeNoiseAggregator.draw_samples`.

    aggregation_method : imgaug.ALL or {'min', 'avg', 'max'} or list of str or imgaug.parameters.StochasticParameter, optional
        The method to use to aggregate the samples of multiple iterations
        to a single output array. All methods combine several arrays of
        shape ``S`` each to a single array of shape ``S`` and hence work
        elementwise. Known methods are ``min`` (take the minimum over all
        iterations), ``max`` (take the maximum) and ``avg`` (take the average).

            * If an ``str``, it must be one of the described methods and
              will be used for all calls..
            * If a ``list`` of ``str``, it must contain one or more of the
              described methods and a random one will be samples once per call.
            * If ``imgaug.ALL``, then equivalent to the ``list``
              ``["min", "max", "avg"]``.
            * If :class:`StochasticParameter`, a value will be sampled from
              that parameter once per call and must be one of the described
              methods..

        "per call" denotes a call of
        :func:`IterativeNoiseAggregator.draw_sample` or
        :func:`IterativeNoiseAggregator.draw_samples`.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> noise = iap.IterativeNoiseAggregator(
    >>>     iap.SimplexNoise(),
    >>>     iterations=(2, 5),
    >>>     aggregation_method="max")

    Create a parameter that -- upon each call -- generates ``2`` to ``5``
    arrays of simplex noise with the same shape. Then it combines these
    noise maps to a single map using elementwise maximum.

    """

    def __init__(self, other_param, iterations=(1, 3),
                 aggregation_method=["max", "avg"]):
        # pylint: disable=dangerous-default-value
        super(IterativeNoiseAggregator, self).__init__()
        _assert_arg_is_stoch_param("other_param", other_param)
        self.other_param = other_param

        def _assert_within_bounds(_iterations):
            assert all([1 <= val <= 10000 for val in _iterations]), (
                "Expected 'iterations' to only contain values within "
                "the interval [1, 1000], got values %s." % (
                    ", ".join([str(val) for val in _iterations]),))

        if ia.is_single_integer(iterations):
            _assert_within_bounds([iterations])
            self.iterations = Deterministic(iterations)
        elif isinstance(iterations, list):
            assert len(iterations) > 0, (
                "Expected 'iterations' of type list to contain at least one "
                "entry, got %d." % (len(iterations),))
            _assert_within_bounds(iterations)
            self.iterations = Choice(iterations)
        elif ia.is_iterable(iterations):
            assert len(iterations) == 2, (
                "Expected iterable non-list 'iteratons' to contain exactly "
                "two entries, got %d." % (len(iterations),))
            assert all([ia.is_single_integer(val) for val in iterations]), (
                "Expected iterable non-list 'iterations' to only contain "
                "integers, got types %s." % (
                    ", ".join([str(type(val)) for val in iterations]),))
            _assert_within_bounds(iterations)
            self.iterations = DiscreteUniform(iterations[0], iterations[1])
        elif isinstance(iterations, StochasticParameter):
            self.iterations = iterations
        else:
            raise Exception(
                "Expected iterations to be int or tuple of two ints or "
                "StochasticParameter, got %s." % (type(iterations),))

        if aggregation_method == ia.ALL:
            self.aggregation_method = Choice(["min", "max", "avg"])
        elif ia.is_string(aggregation_method):
            self.aggregation_method = Deterministic(aggregation_method)
        elif isinstance(aggregation_method, list):
            assert len(aggregation_method) >= 1, (
                "Expected at least one aggregation method got %d." % (
                    len(aggregation_method),))
            assert all([ia.is_string(val) for val in aggregation_method]), (
                "Expected aggregation methods provided as strings, "
                "got types %s." % (
                    ", ".join([str(type(v)) for v in aggregation_method])))
            self.aggregation_method = Choice(aggregation_method)
        elif isinstance(aggregation_method, StochasticParameter):
            self.aggregation_method = aggregation_method
        else:
            raise Exception(
                "Expected aggregation_method to be string or list of strings "
                "or StochasticParameter, got %s." % (
                    type(aggregation_method),))

    def _draw_samples(self, size, random_state):
        rngs = random_state.duplicate(2)
        aggregation_method = self.aggregation_method.draw_sample(
            random_state=rngs[0])
        iterations = self.iterations.draw_sample(random_state=rngs[1])
        assert iterations > 0, (
            "Expected to sample at least one iteration of aggregation. "
            "Got %d." % (iterations,))

        rngs_iterations = rngs[1].duplicate(iterations)

        result = np.zeros(size, dtype=np.float32)
        for i in sm.xrange(iterations):
            noise_iter = self.other_param.draw_samples(
                size, random_state=rngs_iterations[i])

            if aggregation_method == "avg":
                result += noise_iter
            elif aggregation_method == "min":
                if i == 0:
                    result = noise_iter
                else:
                    result = np.minimum(result, noise_iter)
            else:  # self.aggregation_method == "max"
                if i == 0:
                    result = noise_iter
                else:
                    result = np.maximum(result, noise_iter)

        if aggregation_method == "avg":
            result = result / iterations

        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "IterativeNoiseAggregator(%s, %s, %s)" % (
            opstr, str(self.iterations), str(self.aggregation_method))


class Sigmoid(StochasticParameter):
    """Apply a sigmoid function to the outputs of another parameter.

    This is intended to be used in combination with :class:`SimplexNoise` or
    :class:`FrequencyNoise`. It pushes the noise values away from ``~0.5`` and
    towards ``0.0`` or ``1.0``, making the noise maps more binary.

    Parameters
    ----------
    other_param : imgaug.parameters.StochasticParameter
        The other parameter to which the sigmoid will be applied.

    threshold : number or tuple of number or iterable of number or imgaug.parameters.StochasticParameter, optional
        Sets the value of the sigmoid's saddle point, i.e. where values
        start to quickly shift from ``0.0`` to ``1.0``.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`Sigmoid.draw_sample` or
        :func:`Sigmoid.draw_samples`.

    activated : bool or number, optional
        Defines whether the sigmoid is activated. If this is ``False``, the
        results of `other_param` will not be altered. This may be set to a
        ``float`` ``p`` in value range``[0.0, 1.0]``, which will result in
        `activated` being ``True`` in ``p`` percent of all calls.

    mul : number, optional
        The results of `other_param` will be multiplied with this value before
        applying the sigmoid. For noise values (range ``[0.0, 1.0]``) this
        should be set to about ``20``.

    add : number, optional
        This value will be added to the results of `other_param` before
        applying the sigmoid. For noise values (range ``[0.0, 1.0]``) this
        should be set to about ``-10.0``, provided `mul` was set to ``20``.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.Sigmoid(
    >>>     iap.SimplexNoise(),
    >>>     activated=0.5,
    >>>     mul=20,
    >>>     add=-10)

    Applies a sigmoid to simplex noise in ``50%`` of all calls. The noise
    results are modified to match the sigmoid's expected value range. The
    sigmoid's outputs are in the range ``[0.0, 1.0]``.

    """

    def __init__(self, other_param, threshold=(-10, 10), activated=True,
                 mul=1, add=0):
        super(Sigmoid, self).__init__()
        _assert_arg_is_stoch_param("other_param", other_param)
        self.other_param = other_param

        self.threshold = handle_continuous_param(threshold, "threshold")
        self.activated = handle_probability_param(activated, "activated")

        assert ia.is_single_number(mul), (
            "Expected 'mul' to be a number, got type %s." % (type(mul),))
        assert mul > 0, (
            "Expected 'mul' to be greater than zero, got %.4f." % (mul,))
        self.mul = mul

        assert ia.is_single_number(add), (
            "Expected 'add' to be a number, got type %s." % (type(add),))
        self.add = add

    @staticmethod
    def create_for_noise(other_param, threshold=(-10, 10), activated=True):
        """Create a Sigmoid adjusted for noise parameters.

        "noise" here denotes :class:`SimplexNoise` and :class:`FrequencyNoise`.

        Parameters
        ----------
        other_param : imgaug.parameters.StochasticParameter
            See :func:`~imgaug.parameters.Sigmoid.__init__`.

        threshold : number or tuple of number or iterable of number or imgaug.parameters.StochasticParameter, optional
            See :func:`~imgaug.parameters.Sigmoid.__init__`.

        activated : bool or number, optional
            See :func:`~imgaug.parameters.Sigmoid.__init__`.

        Returns
        -------
        Sigmoid
            A sigmoid adjusted to be used with noise.

        """
        return Sigmoid(other_param, threshold, activated, mul=20, add=-10)

    def _draw_samples(self, size, random_state):
        rngs = random_state.duplicate(3)
        result = self.other_param.draw_samples(size, random_state=rngs[0])
        if result.dtype.kind != "f":
            result = result.astype(np.float32)
        activated = self.activated.draw_sample(random_state=rngs[1])
        threshold = self.threshold.draw_sample(random_state=rngs[2])
        if activated > 0.5:
            # threshold must be subtracted here, not added
            # higher threshold = move threshold of sigmoid towards the right
            #                  = make it harder to pass the threshold
            #                  = more 0.0s / less 1.0s
            # by subtracting a high value, it moves each x towards the left,
            # leading to more values being left of the threshold, leading
            # to more 0.0s
            return 1 / (1 + np.exp(-(result * self.mul + self.add - threshold)))
        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Sigmoid(%s, %s, %s, %s, %s)" % (
            opstr, str(self.threshold), str(self.activated), str(self.mul),
            str(self.add))


class SimplexNoise(StochasticParameter):
    """Parameter that generates simplex noise of varying resolutions.

    This parameter expects to sample noise for 2d planes, i.e. for
    sizes ``(H, W, [C])`` and will return a value in the range ``[0.0, 1.0]``
    per spatial location in that plane.

    The noise is sampled from low resolution planes and
    upscaled to the requested height and width. The size of the low
    resolution plane may be defined (large values can be slow) and the
    interpolation method for upscaling can be set.

    Parameters
    ----------
    size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Maximum height and width in pixels of the low resolution plane.
        Upon any sampling call, the requested shape will be downscaled until
        the height or width (whichever is larger) does not exceed this maximum
        value anymore. Then the noise will be sampled at that shape and later
        upscaled back to the requested shape.

            * If a single ``int``, this ``int`` will be used as a
              constant value.
            * If a ``tuple`` of two ``int`` s ``(a, b)``, the value will be
              sampled from the discrete interval ``[a..b]`` once per call.
            * If a ``list`` of ``int``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`SimplexNoise.draw_sample` or
        :func:`SimplexNoise.draw_samples`.

    upscale_method : str or int or list of str or list of int or imgaug.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the originally requested shape (i.e. usually
        the image size). This parameter controls the interpolation method to
        use. See also :func:`~imgaug.imgaug.imresize_many_images` for a
        description of possible values.

            * If ``imgaug.ALL``, then either ``nearest`` or ``linear`` or
              ``area`` or ``cubic`` is picked per iteration (all same
              probability).
            * If ``str``, then that value will always be used as the method
              (must be ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If ``list`` of ``str``, then a random value will be picked from
              that list per call.
            * If :class:`StochasticParameter`, then a random value will be
              sampled from that parameter per call.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.SimplexNoise(upscale_method="linear")

    Create a parameter that produces smooth simplex noise of varying sizes.

    >>> param = iap.SimplexNoise(
    >>>     size_px_max=(8, 16),
    >>>     upscale_method="nearest")

    Create a parameter that produces rectangular simplex noise of rather
    high detail.

    """

    def __init__(self, size_px_max=(2, 16),
                 upscale_method=["linear", "nearest"]):
        # pylint: disable=dangerous-default-value
        super(SimplexNoise, self).__init__()
        self.size_px_max = handle_discrete_param(
            size_px_max, "size_px_max", value_range=(1, 10000))

        if upscale_method == ia.ALL:
            self.upscale_method = Choice(["nearest", "linear", "area",
                                          "cubic"])
        elif ia.is_string(upscale_method):
            self.upscale_method = Deterministic(upscale_method)
        elif isinstance(upscale_method, list):
            assert len(upscale_method) >= 1, (
                "Expected at least one upscale method, "
                "got %d." % (len(upscale_method),))
            assert all([ia.is_string(val) for val in upscale_method]), (
                "Expected all upscale methods to be strings, got types %s." % (
                    ", ".join([str(type(v)) for v in upscale_method])))
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception(
                "Expected upscale_method to be string or list of strings or "
                "StochasticParameter, got %s." % (type(upscale_method),))

    def _draw_samples(self, size, random_state):
        assert len(size) in [2, 3], (
            "Expected requested noise to have shape (H, W) or (H, W, C), "
            "got shape %s." % (size,))
        height, width = size[0:2]
        nb_channels = 1 if len(size) == 2 else size[2]

        channels = [self._draw_samples_hw(height, width, random_state)
                    for _ in np.arange(nb_channels)]

        if len(size) == 2:
            return channels[0]
        return np.stack(channels, axis=-1)

    def _draw_samples_hw(self, height, width, random_state):
        iterations = 1
        rngs = random_state.duplicate(1+iterations)
        aggregation_method = "max"
        upscale_methods = self.upscale_method.draw_samples(
            (iterations,), random_state=rngs[0])
        result = np.zeros((height, width), dtype=np.float32)
        for i in sm.xrange(iterations):
            noise_iter = self._draw_samples_iteration(
                height, width, rngs[1+i], upscale_methods[i])
            if aggregation_method == "avg":
                result += noise_iter
            elif aggregation_method == "min":
                if i == 0:
                    result = noise_iter
                else:
                    result = np.minimum(result, noise_iter)
            else:  # self.aggregation_method == "max"
                if i == 0:
                    result = noise_iter
                else:
                    result = np.maximum(result, noise_iter)

        if aggregation_method == "avg":
            result = result / iterations

        return result

    def _draw_samples_iteration(self, height, width, rng, upscale_method):
        opensimplex_seed = rng.generate_seed_()

        # we have to use int(.) here, otherwise we can get warnings about
        # value overflows in OpenSimplex L103
        generator = OpenSimplex(seed=int(opensimplex_seed))

        maxlen = max(height, width)
        size_px_max = self.size_px_max.draw_sample(random_state=rng)
        if maxlen > size_px_max:
            downscale_factor = size_px_max / maxlen
            h_small = int(height * downscale_factor)
            w_small = int(width * downscale_factor)
        else:
            h_small = height
            w_small = width

        # don't go below Hx1 or 1xW
        h_small = max(h_small, 1)
        w_small = max(w_small, 1)

        noise = np.zeros((h_small, w_small), dtype=np.float32)
        for y in sm.xrange(h_small):
            for x in sm.xrange(w_small):
                noise[y, x] = generator.noise2d(y=y, x=x)

        # TODO this was previously (noise+0.5)/2, which was wrong as the noise
        #      here is in range [-1.0, 1.0], but this new normalization might
        #      lead to bad masks due to too many values being significantly
        #      above 0.0 instead of being clipped to 0?
        noise_0to1 = (noise + 1.0) / 2
        noise_0to1 = np.clip(noise_0to1, 0.0, 1.0)

        if noise_0to1.shape != (height, width):
            noise_0to1_uint8 = (noise_0to1 * 255).astype(np.uint8)
            noise_0to1_3d = np.tile(
                noise_0to1_uint8[..., np.newaxis], (1, 1, 3))
            noise_0to1 = ia.imresize_single_image(
                noise_0to1_3d, (height, width), interpolation=upscale_method)
            noise_0to1 = (noise_0to1[..., 0] / 255.0).astype(np.float32)

        return noise_0to1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "SimplexNoise(%s, %s)" % (
            str(self.size_px_max),
            str(self.upscale_method)
        )


class FrequencyNoise(StochasticParameter):
    """Parameter to generate noise of varying frequencies.

    This parameter expects to sample noise for 2d planes, i.e. for
    sizes ``(H, W, [C])`` and will return a value in the range ``[0.0, 1.0]``
    per spatial location in that plane.

    The exponent controls the frequencies and therefore noise patterns.
    Small values (around ``-4.0``) will result in large blobs. Large values
    (around ``4.0``) will result in small, repetitive patterns.

    The noise is sampled from low resolution planes and
    upscaled to the requested height and width. The size of the low
    resolution plane may be defined (high values can be slow) and the
    interpolation method for upscaling can be set.

    Parameters
    ----------
    exponent : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Exponent to use when scaling in the frequency domain.
        Sane values are in the range ``-4`` (large blobs) to ``4`` (small
        patterns). To generate cloud-like structures, use roughly ``-2``.

            * If a single ``number``, this ``number`` will be used as a
              constant value.
            * If a ``tuple`` of two ``number`` s ``(a, b)``, the value will be
              sampled from the continuous interval ``[a, b)`` once per call.
            * If a ``list`` of ``number``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

    size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Maximum height and width in pixels of the low resolution plane.
        Upon any sampling call, the requested shape will be downscaled until
        the height or width (whichever is larger) does not exceed this maximum
        value anymore. Then the noise will be sampled at that shape and later
        upscaled back to the requested shape.

            * If a single ``int``, this ``int`` will be used as a
              constant value.
            * If a ``tuple`` of two ``int`` s ``(a, b)``, the value will be
              sampled from the discrete interval ``[a..b]`` once per call.
            * If a ``list`` of ``int``, a random value will be picked from
              the ``list`` once per call.
            * If a :class:`StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of :func:`FrequencyNoise.draw_sample` or
        :func:`FrequencyNoise.draw_samples`.

    upscale_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the originally requested shape (i.e. usually
        the image size). This parameter controls the interpolation method to
        use. See also :func:`~imgaug.imgaug.imresize_many_images` for a
        description of possible values.

            * If ``imgaug.ALL``, then either ``nearest`` or ``linear`` or
              ``area`` or ``cubic`` is picked per iteration (all same
              probability).
            * If ``str``, then that value will always be used as the method
              (must be ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If ``list`` of ``str``, then a random value will be picked from
              that list per call.
            * If :class:`StochasticParameter`, then a random value will be
              sampled from that parameter per call.

    Examples
    --------
    >>> import imgaug.parameters as iap
    >>> param = iap.FrequencyNoise(
    >>>     exponent=-2,
    >>>     size_px_max=(16, 32),
    >>>     upscale_method="linear")

    Create a parameter that produces noise with cloud-like patterns.

    """

    def __init__(self, exponent=(-4, 4), size_px_max=(4, 32),
                 upscale_method=["linear", "nearest"]):
        # pylint: disable=dangerous-default-value
        super(FrequencyNoise, self).__init__()
        self.exponent = handle_continuous_param(exponent, "exponent")
        self.size_px_max = handle_discrete_param(
            size_px_max, "size_px_max", value_range=(1, 10000))

        if upscale_method == ia.ALL:
            self.upscale_method = Choice(["nearest", "linear", "area",
                                          "cubic"])
        elif ia.is_string(upscale_method):
            self.upscale_method = Deterministic(upscale_method)
        elif isinstance(upscale_method, list):
            assert len(upscale_method) >= 1, (
                "Expected at least one upscale method, "
                "got %d." % (len(upscale_method),))
            assert all([ia.is_string(val) for val in upscale_method]), (
                "Expected all upscale methods to be strings, got types %s." % (
                    ", ".join([str(type(v)) for v in upscale_method])))
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception(
                "Expected upscale_method to be string or list of strings or "
                "StochasticParameter, got %s." % (type(upscale_method),))

    # TODO this is the same as in SimplexNoise, make DRY
    def _draw_samples(self, size, random_state):
        # code here is similar to:
        #   http://www.redblobgames.com/articles/noise/2d/
        #   http://www.redblobgames.com/articles/noise/2d/2d-noise.js

        assert len(size) in [2, 3], (
            "Expected requested noise to have shape (H, W) or (H, W, C), "
            "got shape %s." % (size,))
        height, width = size[0:2]
        nb_channels = 1 if len(size) == 2 else size[2]

        channels = [self._draw_samples_hw(height, width, random_state)
                    for _ in np.arange(nb_channels)]

        if len(size) == 2:
            return channels[0]
        return np.stack(channels, axis=-1)

    def _draw_samples_hw(self, height, width, random_state):
        rngs = random_state.duplicate(5)
        maxlen = max(height, width)
        size_px_max = self.size_px_max.draw_sample(random_state=rngs[0])
        if maxlen > size_px_max:
            downscale_factor = size_px_max / maxlen
            h_small = int(height * downscale_factor)
            w_small = int(width * downscale_factor)
        else:
            h_small = height
            w_small = width

        # don't go below Hx4 or 4xW
        h_small = max(h_small, 4)
        w_small = max(w_small, 4)

        # generate random base matrix
        # TODO use a single RNG with a single call here
        wn_r = rngs[1].random(size=(h_small, w_small))
        wn_a = rngs[2].random(size=(h_small, w_small))

        wn_r = wn_r * (max(h_small, w_small) ** 2)
        wn_a = wn_a * 2 * np.pi

        wn_r = wn_r * np.cos(wn_a)
        wn_a = wn_r * np.sin(wn_a)

        # pronounce some frequencies
        exponent = self.exponent.draw_sample(random_state=rngs[3])
        # this has some similarity with a distance map from the center, but
        # looks a bit more like a cross
        f = self._create_distance_matrix((h_small, w_small))
        f[0, 0] = 1 # necessary to prevent -inf from appearing
        scale = f ** exponent
        scale[0, 0] = 0
        treal = wn_r * scale
        timag = wn_a * scale

        wn_freqs_mul = np.zeros(treal.shape, dtype=np.complex)
        wn_freqs_mul.real = treal
        wn_freqs_mul.imag = timag

        wn_inv = np.fft.ifft2(wn_freqs_mul).real

        # normalize to 0 to 1
        wn_inv_min = np.min(wn_inv)
        wn_inv_max = np.max(wn_inv)
        noise_0to1 = (wn_inv - wn_inv_min) / (wn_inv_max - wn_inv_min)

        # upscale from low resolution to image size
        upscale_method = self.upscale_method.draw_sample(random_state=rngs[4])
        if noise_0to1.shape != (height, width):
            noise_0to1_uint8 = (noise_0to1 * 255).astype(np.uint8)
            noise_0to1_3d = np.tile(
                noise_0to1_uint8[..., np.newaxis], (1, 1, 3))
            noise_0to1 = ia.imresize_single_image(
                noise_0to1_3d,
                (height, width),
                interpolation=upscale_method)
            noise_0to1 = (noise_0to1[..., 0] / 255.0).astype(np.float32)

        return noise_0to1

    @classmethod
    def _create_distance_matrix(cls, size):
        h, w = size

        def _freq(yy, xx):
            hdist = np.minimum(yy, h-yy)
            wdist = np.minimum(xx, w-xx)
            return np.sqrt(hdist**2 + wdist**2)

        return np.fromfunction(_freq, (h, w))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "FrequencyNoise(%s, %s, %s)" % (
            str(self.exponent),
            str(self.size_px_max),
            str(self.upscale_method))


def _assert_arg_is_stoch_param(arg_name, arg_value):
    assert isinstance(arg_value, StochasticParameter), (
        "Expected '%s' to be a StochasticParameter, "
        "got type %s." % (arg_name, arg_value,))
