# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# Copyright (c) 2012-2016 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See COPYING for license details.

"""1D and 2D Wavelet packet transform module."""

from __future__ import division, print_function, absolute_import

__all__ = ["BaseNode", "Node", "WaveletPacket", "Node2D", "WaveletPacket2D",
           "NodeND", "WaveletPacketND"]

from itertools import product
from collections import OrderedDict
import numpy as np

from ._extensions._pywt import Wavelet, _check_dtype
from ._dwt import dwt, idwt, dwt_max_level
from ._multidim import dwt2, idwt2, dwtn, idwtn


def get_graycode_order(level, x='a', y='d'):
    graycode_order = [x, y]
    for i in range(level - 1):
        graycode_order = [x + path for path in graycode_order] + \
                         [y + path for path in graycode_order[::-1]]
    return graycode_order


class BaseNode(object):
    """
    BaseNode for wavelet packet 1D and 2D tree nodes.

    The BaseNode is a base class for `Node` and `Node2D`.
    It should not be used directly unless creating a new transformation
    type. It is included here to document the common interface of 1D
    and 2D node and wavelet packet transform classes.

    Parameters
    ----------
    parent :
        Parent node. If parent is None then the node is considered detached
        (ie root).
    data : 1D or 2D array
        Data associated with the node. 1D or 2D numeric array, depending on the
        transform type.
    node_name :
        A name identifying the coefficients type.
        See `Node.node_name` and `Node2D.node_name`
        for information on the accepted subnodes names.
    """

    # PART_LEN and PARTS attributes that define path tokens for node[] lookup
    # must be defined in subclasses.
    PART_LEN = None
    PARTS = None

    def __init__(self, parent, data, node_name):
        self.parent = parent
        if parent is not None:
            self.wavelet = parent.wavelet
            self.mode = parent.mode
            self.level = parent.level + 1
            self._maxlevel = parent.maxlevel
            self.path = parent.path + node_name
            self.axes = parent.axes
        else:
            self.wavelet = None
            self.mode = None
            self.axes = None
            self.path = ""
            self.level = 0

        # data - signal on level 0, coeffs on higher levels
        self.data = data
        # Need to retain original data size/shape so we can trim any excess
        # boundary coefficients from the inverse transform.
        if self.data is None:
            self._data_shape = None
        else:
            self._data_shape = np.asarray(data).shape

        self._init_subnodes()

    def _init_subnodes(self):
        for part in self.PARTS:
            self._set_node(part, None)

    def _create_subnode(self, part, data=None, overwrite=True):
        raise NotImplementedError()

    def _create_subnode_base(self, node_cls, part, data=None, overwrite=True,
                             **kwargs):
        self._validate_node_name(part)
        if not overwrite and self._get_node(part) is not None:
            return self._get_node(part)
        node = node_cls(self, data, part, **kwargs)
        self._set_node(part, node)
        return node

    def _get_node(self, part):
        return getattr(self, part)

    def _set_node(self, part, node):
        setattr(self, part, node)

    def _delete_node(self, part):
        self._set_node(part, None)

    def _validate_node_name(self, part):
        if part not in self.PARTS:
            raise ValueError("Subnode name must be in [%s], not '%s'." %
                             (', '.join("'%s'" % p for p in self.PARTS), part))

    @property
    def path_tuple(self):
        """The path to the current node in tuple form.

        The length of the tuple is equal to the number of decomposition levels.
        """
        path = self.path
        nlev = len(path)//self.PART_LEN
        return tuple([path[(n-1)*self.PART_LEN:n*self.PART_LEN]
                      for n in range(1, nlev+1)])

    def _evaluate_maxlevel(self, evaluate_from='parent'):
        """
        Try to find the value of maximum decomposition level if it is not
        specified explicitly.

        Parameters
        ----------
        evaluate_from : {'parent', 'subnodes'}
        """
        assert evaluate_from in ('parent', 'subnodes')

        if self._maxlevel is not None:
            return self._maxlevel
        elif self.data is not None:
            return self.level + dwt_max_level(
                min(self.data.shape), self.wavelet)

        if evaluate_from == 'parent':
            if self.parent is not None:
                return self.parent._evaluate_maxlevel(evaluate_from)
        elif evaluate_from == 'subnodes':
            for node_name in self.PARTS:
                node = getattr(self, node_name, None)
                if node is not None:
                    level = node._evaluate_maxlevel(evaluate_from)
                    if level is not None:
                        return level
        return None

    @property
    def maxlevel(self):
        if self._maxlevel is not None:
            return self._maxlevel

        # Try getting the maxlevel from parents first
        self._maxlevel = self._evaluate_maxlevel(evaluate_from='parent')

        # If not found, check whether it can be evaluated from subnodes
        if self._maxlevel is None:
            self._maxlevel = self._evaluate_maxlevel(evaluate_from='subnodes')
        return self._maxlevel

    @property
    def node_name(self):
        return self.path[-self.PART_LEN:]

    def decompose(self):
        """
        Decompose node data creating DWT coefficients subnodes.

        Performs Discrete Wavelet Transform on the `~BaseNode.data` and
        returns transform coefficients.

        Note
        ----
        Descends to subnodes and recursively
        calls `~BaseNode.reconstruct` on them.

        """
        if self.level < self.maxlevel:
            return self._decompose()
        else:
            raise ValueError("Maximum decomposition level reached.")

    def _decompose(self):
        raise NotImplementedError()

    def reconstruct(self, update=False):
        """
        Reconstruct node from subnodes.

        Parameters
        ----------
        update : bool, optional
            If True, then reconstructed data replaces the current
            node data (default: False).

        Returns:
            - original node data if subnodes do not exist
            - IDWT of subnodes otherwise.
        """
        if not self.has_any_subnode:
            return self.data
        return self._reconstruct(update)

    def _reconstruct(self):
        raise NotImplementedError()  # override this in subclasses

    def get_subnode(self, part, decompose=True):
        """
        Returns subnode or None (see `decomposition` flag description).

        Parameters
        ----------
        part :
            Subnode name
        decompose : bool, optional
            If the param is True and corresponding subnode does not
            exist, the subnode will be created using coefficients
            from the DWT decomposition of the current node.
            (default: True)
        """
        self._validate_node_name(part)
        subnode = self._get_node(part)
        if subnode is None and decompose and not self.is_empty:
            self.decompose()
            subnode = self._get_node(part)
        return subnode

    def __getitem__(self, path):
        """
        Find node represented by the given path.

        Similar to `~BaseNode.get_subnode` method with `decompose=True`, but
        can access nodes on any level in the decomposition tree.

        Parameters
        ----------
        path : str
            String composed of node names. See `Node.node_name` and
            `Node2D.node_name` for node naming convention.

        Notes
        -----
        If node does not exist yet, it will be created by decomposition of its
        parent node.
        """
        errmsg = ("Invalid path parameter type - expected string or "
                  "tuple of strings but got %s." % type(path))
        if isinstance(path, tuple):
            # concatenate tuple of strings into a single string
            try:
                path = ''.join(path)
            except TypeError:
                raise TypeError(errmsg)
        if isinstance(path, str):
            if (self.maxlevel is not None and
                    len(path) > self.maxlevel * self.PART_LEN):
                raise IndexError("Path length is out of range.")
            if path:
                return self.get_subnode(path[0:self.PART_LEN], True)[
                    path[self.PART_LEN:]]
            else:
                return self
        else:
            raise TypeError(errmsg)

    def __setitem__(self, path, data):
        """
        Set node or node's data in the decomposition tree. Nodes are
        identified by string `path`.

        Parameters
        ----------
        path : str
            String composed of node names.
        data : array or BaseNode subclass.
        """

        if isinstance(path, str):
            if (
                self.maxlevel is not None and
                len(self.path) + len(path) > self.maxlevel * self.PART_LEN
            ):
                raise IndexError("Path length out of range.")
            if path:
                subnode = self.get_subnode(path[0:self.PART_LEN], False)
                if subnode is None:
                    self._create_subnode(path[0:self.PART_LEN], None)
                    subnode = self.get_subnode(path[0:self.PART_LEN], False)
                subnode[path[self.PART_LEN:]] = data
            else:
                if isinstance(data, BaseNode):
                    self.data = np.asarray(data.data)
                else:
                    self.data = np.asarray(data)
                # convert data to nearest supported dtype
                dtype = _check_dtype(data)
                if self.data.dtype != dtype:
                    self.data = self.data.astype(dtype)
        else:
            raise TypeError("Invalid path parameter type - expected string but"
                            " got %s." % type(path))

    def __delitem__(self, path):
        """
        Remove node from the tree.

        Parameters
        ----------
        path : str
            String composed of node names.
        """
        node = self[path]
        # don't clear node value and subnodes (node may still exist outside
        # the tree)
        # # node._init_subnodes()
        # # node.data = None
        parent = node.parent
        node.parent = None  # TODO
        if parent and node.node_name:
            parent._delete_node(node.node_name)

    @property
    def is_empty(self):
        return self.data is None

    @property
    def has_any_subnode(self):
        for part in self.PARTS:
            if self._get_node(part) is not None:  # and not .is_empty
                return True
        return False

    def get_leaf_nodes(self, decompose=False):
        """
        Returns leaf nodes.

        Parameters
        ----------
        decompose : bool, optional
            (default: True)
        """
        result = []

        def collect(node):
            if node.level == node.maxlevel and not node.is_empty:
                result.append(node)
                return False
            if not decompose and not node.has_any_subnode:
                result.append(node)
                return False
            return True
        self.walk(collect, decompose=decompose)
        return result

    def walk(self, func, args=(), kwargs=None, decompose=True):
        """
        Traverses the decomposition tree and calls
        ``func(node, *args, **kwargs)`` on every node. If `func` returns True,
        descending to subnodes will continue.

        Parameters
        ----------
        func : callable
            Callable accepting `BaseNode` as the first param and
            optional positional and keyword arguments
        args :
            func params
        kwargs :
            func keyword params
        decompose : bool, optional
            If True (default), the method will also try to decompose the tree
            up to the `maximum level <BaseNode.maxlevel>`.
        """
        if kwargs is None:
            kwargs = {}
        if func(self, *args, **kwargs) and self.level < self.maxlevel:
            for part in self.PARTS:
                subnode = self.get_subnode(part, decompose)
                if subnode is not None:
                    subnode.walk(func, args, kwargs, decompose)

    def walk_depth(self, func, args=(), kwargs=None, decompose=True):
        """
        Walk tree and call func on every node starting from the bottom-most
        nodes.

        Parameters
        ----------
        func : callable
            Callable accepting :class:`BaseNode` as the first param and
            optional positional and keyword arguments
        args :
            func params
        kwargs :
            func keyword params
        decompose : bool, optional
            (default: False)
        """
        if kwargs is None:
            kwargs = {}
        if self.level < self.maxlevel:
            for part in self.PARTS:
                subnode = self.get_subnode(part, decompose)
                if subnode is not None:
                    subnode.walk_depth(func, args, kwargs, decompose)
        func(self, *args, **kwargs)

    def __str__(self):
        return self.path + ": " + str(self.data)


class Node(BaseNode):
    """
    WaveletPacket tree node.

    Subnodes are called `a` and `d`, just like approximation
    and detail coefficients in the Discrete Wavelet Transform.
    """

    A = 'a'
    D = 'd'
    PARTS = A, D
    PART_LEN = 1

    def _create_subnode(self, part, data=None, overwrite=True):
        return self._create_subnode_base(node_cls=Node, part=part, data=data,
                                         overwrite=overwrite)

    def _decompose(self):
        """

        See also
        --------
        dwt : for 1D Discrete Wavelet Transform output coefficients.
        """
        if self.is_empty:
            data_a, data_d = None, None
            if self._get_node(self.A) is None:
                self._create_subnode(self.A, data_a)
            if self._get_node(self.D) is None:
                self._create_subnode(self.D, data_d)
        else:
            data_a, data_d = dwt(self.data, self.wavelet, self.mode,
                                 axis=self.axes)
            self._create_subnode(self.A, data_a)
            self._create_subnode(self.D, data_d)
        return self._get_node(self.A), self._get_node(self.D)

    def _reconstruct(self, update):
        data_a, data_d = None, None
        node_a, node_d = self._get_node(self.A), self._get_node(self.D)

        if node_a is not None:
            data_a = node_a.reconstruct()  # TODO: (update) ???
        if node_d is not None:
            data_d = node_d.reconstruct()  # TODO: (update) ???

        if data_a is None and data_d is None:
            raise ValueError("Node is a leaf node and cannot be reconstructed"
                             " from subnodes.")
        else:
            rec = idwt(data_a, data_d, self.wavelet, self.mode, axis=self.axes)
            if self._data_shape is not None and (
                    rec.shape != self._data_shape):
                rec = rec[tuple([slice(sz) for sz in self._data_shape])]
            if update:
                self.data = rec
            return rec


class Node2D(BaseNode):
    """
    WaveletPacket tree node.

    Subnodes are called 'a' (LL), 'h' (HL), 'v' (LH) and  'd' (HH), like
    approximation and detail coefficients in the 2D Discrete Wavelet Transform
    """

    LL = 'a'
    HL = 'h'
    LH = 'v'
    HH = 'd'

    PARTS = LL, HL, LH, HH
    PART_LEN = 1

    def _create_subnode(self, part, data=None, overwrite=True):
        return self._create_subnode_base(node_cls=Node2D, part=part, data=data,
                                         overwrite=overwrite)

    def _decompose(self):
        """
        See also
        --------
        dwt2 : for 2D Discrete Wavelet Transform output coefficients.
        """
        if self.is_empty:
            data_ll, data_lh, data_hl, data_hh = None, None, None, None
        else:
            data_ll, (data_hl, data_lh, data_hh) =\
                dwt2(self.data, self.wavelet, self.mode, axes=self.axes)
        self._create_subnode(self.LL, data_ll)
        self._create_subnode(self.LH, data_lh)
        self._create_subnode(self.HL, data_hl)
        self._create_subnode(self.HH, data_hh)
        return (self._get_node(self.LL), self._get_node(self.HL),
                self._get_node(self.LH), self._get_node(self.HH))

    def _reconstruct(self, update):
        data_ll, data_lh, data_hl, data_hh = None, None, None, None

        node_ll, node_lh, node_hl, node_hh =\
            self._get_node(self.LL), self._get_node(self.LH),\
            self._get_node(self.HL), self._get_node(self.HH)

        if node_ll is not None:
            data_ll = node_ll.reconstruct()
        if node_lh is not None:
            data_lh = node_lh.reconstruct()
        if node_hl is not None:
            data_hl = node_hl.reconstruct()
        if node_hh is not None:
            data_hh = node_hh.reconstruct()

        if (data_ll is None and data_lh is None and
                data_hl is None and data_hh is None):
            raise ValueError(
                "Tree is missing data - all subnodes of `%s` node "
                "are None. Cannot reconstruct node." % self.path
            )
        else:
            coeffs = data_ll, (data_hl, data_lh, data_hh)
            rec = idwt2(coeffs, self.wavelet, self.mode, axes=self.axes)
            if self._data_shape is not None and (
                    rec.shape != self._data_shape):
                rec = rec[tuple([slice(sz) for sz in self._data_shape])]
            if update:
                self.data = rec
            return rec

    def expand_2d_path(self, path):
        expanded_paths = {
            self.HH: 'hh',
            self.HL: 'hl',
            self.LH: 'lh',
            self.LL: 'll'
        }
        return (''.join([expanded_paths[p][0] for p in path]),
                ''.join([expanded_paths[p][1] for p in path]))


class NodeND(BaseNode):
    """
    WaveletPacket tree node.

    Unlike Node and Node2D self.PARTS is a dictionary.
    For 1D:  self.PARTS has keys 'a' and 'd'
    For 2D:  self.PARTS has keys 'aa', 'ad', 'da', 'dd'
    For 3D:  self.PARTS has keys 'aaa', 'aad', 'ada', 'daa', ..., 'ddd'

    Parameters
    ----------
    parent :
        Parent node. If parent is None then the node is considered detached
        (ie root).
    data : 1D or 2D array
        Data associated with the node. 1D or 2D numeric array, depending on the
        transform type.
    node_name : string
        A name identifying the coefficients type.
        See `Node.node_name` and `Node2D.node_name`
        for information on the accepted subnodes names.
    ndim : int
        The number of data dimensions.
    ndim_transform : int
        The number of dimensions that are to be transformed.

    """
    def __init__(self, parent, data, node_name, ndim, ndim_transform):
        super(NodeND, self).__init__(parent=parent, data=data,
                                     node_name=node_name)
        self.PART_LEN = ndim_transform
        self.PARTS = OrderedDict()
        for key in product(*(('ad', )*self.PART_LEN)):
            self.PARTS[''.join(key)] = None
        self.ndim = ndim
        self.ndim_transform = ndim_transform

    def _init_subnodes(self):
        # need this empty so BaseNode's _init_subnodes isn't called during
        # __init__.  We use a dictionary for PARTS instead for the nd case.
        pass

    def _get_node(self, part):
        return self.PARTS[part]

    def _set_node(self, part, node):
        if part not in self.PARTS:
            raise ValueError("invalid part")
        self.PARTS[part] = node

    def _delete_node(self, part):
        self._set_node(part, None)

    def _validate_node_name(self, part):
        if part not in self.PARTS:
            raise ValueError(
                "Subnode name must be in [%s], not '%s'." %
                (', '.join("'%s'" % p for p in list(self.PARTS.keys())), part))

    def _create_subnode(self, part, data=None, overwrite=True):
        return self._create_subnode_base(node_cls=NodeND, part=part, data=data,
                                         overwrite=overwrite, ndim=self.ndim,
                                         ndim_transform=self.ndim_transform)

    def _evaluate_maxlevel(self, evaluate_from='parent'):
        """
        Try to find the value of maximum decomposition level if it is not
        specified explicitly.

        Parameters
        ----------
        evaluate_from : {'parent', 'subnodes'}
        """
        assert evaluate_from in ('parent', 'subnodes')

        if self._maxlevel is not None:
            return self._maxlevel
        elif self.data is not None:
            return self.level + dwt_max_level(
                min(self.data.shape), self.wavelet)

        if evaluate_from == 'parent':
            if self.parent is not None:
                return self.parent._evaluate_maxlevel(evaluate_from)
        elif evaluate_from == 'subnodes':
            for node_name, node in self.PARTS.items():
                if node is not None:
                    level = node._evaluate_maxlevel(evaluate_from)
                    if level is not None:
                        return level
        return None

    def _decompose(self):
        """
        See also
        --------
        dwt2 : for 2D Discrete Wavelet Transform output coefficients.
        """
        if self.is_empty:
            coefs = {key: None for key in self.PARTS.keys()}
        else:
            coefs = dwtn(self.data, self.wavelet, self.mode, axes=self.axes)

        for key, data in coefs.items():
            self._create_subnode(key, data)
        return (self._get_node(key) for key in self.PARTS.keys())

    def _reconstruct(self, update):
        coeffs = {key: None for key in self.PARTS.keys()}

        nnodes = 0
        for key in self.PARTS.keys():
            node = self._get_node(key)
            if node is not None:
                nnodes += 1
                coeffs[key] = node.reconstruct()

        if nnodes == 0:
            raise ValueError(
                "Tree is missing data - all subnodes of `%s` node "
                "are None. Cannot reconstruct node." % self.path
            )
        else:
            rec = idwtn(coeffs, self.wavelet, self.mode, axes=self.axes)
            if update:
                self.data = rec
            return rec


class WaveletPacket(Node):
    """
    Data structure representing Wavelet Packet decomposition of signal.

    Parameters
    ----------
    data : 1D ndarray
        Original data (signal)
    wavelet : Wavelet object or name string
        Wavelet used in DWT decomposition and reconstruction
    mode : str, optional
        Signal extension mode for the `dwt` and `idwt` decomposition and
        reconstruction functions.
    maxlevel : int, optional
        Maximum level of decomposition.
        If None, it will be calculated based on the `wavelet` and `data`
        length using `pywt.dwt_max_level`.
    axis : int, optional
        The axis to transform.
    """
    def __init__(self, data, wavelet, mode='symmetric', maxlevel=None,
                 axis=-1):
        super(WaveletPacket, self).__init__(None, data, "")

        if not isinstance(wavelet, Wavelet):
            wavelet = Wavelet(wavelet)
        self.wavelet = wavelet
        self.mode = mode
        self.axes = axis  # self.axes is just an integer for 1D transforms

        if data is not None:
            data = np.asarray(data)
            if self.axes < 0:
                self.axes = self.axes + data.ndim
            if not 0 <= self.axes < data.ndim:
                raise ValueError("Axis greater than data dimensions")
            self.data_size = data.shape
            if maxlevel is None:
                maxlevel = dwt_max_level(data.shape[self.axes], self.wavelet)
        else:
            self.data_size = None

        self._maxlevel = maxlevel

    def __reduce__(self):
        return (WaveletPacket,
                (self.data, self.wavelet, self.mode, self.maxlevel))

    def reconstruct(self, update=True):
        """
        Reconstruct data value using coefficients from subnodes.

        Parameters
        ----------
        update : bool, optional
            If True (default), then data values will be replaced by
            reconstruction values, also in subnodes.
        """
        if self.has_any_subnode:
            data = super(WaveletPacket, self).reconstruct(update)
            if self.data_size is not None and (data.shape != self.data_size):
                data = data[[slice(sz) for sz in self.data_size]]
            if update:
                self.data = data
            return data
        return self.data  # return original data

    def get_level(self, level, order="natural", decompose=True):
        """
        Returns all nodes on the specified level.

        Parameters
        ----------
        level : int
            Specifies decomposition `level` from which the nodes will be
            collected.
        order : {'natural', 'freq'}, optional
            - "natural" - left to right in tree (default)
            - "freq" - band ordered
        decompose : bool, optional
            If set then the method will try to decompose the data up
            to the specified `level` (default: True).

        Notes
        -----
        If nodes at the given level are missing (i.e. the tree is partially
        decomposed) and `decompose` is set to False, only existing nodes
        will be returned.

        Frequency order (``order="freq"``) is also known as as sequency order
        and "natural" order is sometimes referred to as Paley order. A detailed
        discussion of these orderings is also given in [1]_, [2]_.

        References
        ----------
        ..[1] M.V. Wickerhauser. Adapted Wavelet Analysis from Theory to
              Software. Wellesley. Massachusetts: A K Peters. 1994.
        ..[2] D.B. Percival and A.T. Walden.  Wavelet Methods for Time Series
              Analysis. Cambridge University Press. 2000.
              DOI:10.1017/CBO9780511841040
        """
        if order not in ["natural", "freq"]:
            raise ValueError("Invalid order: {}".format(order))
        if level > self.maxlevel:
            raise ValueError("The level cannot be greater than the maximum"
                             " decomposition level value (%d)" % self.maxlevel)

        result = []

        def collect(node):
            if node.level == level:
                result.append(node)
                return False
            return True

        self.walk(collect, decompose=decompose)
        if order == "natural":
            return result
        elif order == "freq":
            result = dict((node.path, node) for node in result)
            graycode_order = get_graycode_order(level)
            return [result[path] for path in graycode_order if path in result]
        else:
            raise ValueError("Invalid order name - %s." % order)


class WaveletPacket2D(Node2D):
    """
    Data structure representing 2D Wavelet Packet decomposition of signal.

    Parameters
    ----------
    data : 2D ndarray
        Data associated with the node.
    wavelet : Wavelet object or name string
        Wavelet used in DWT decomposition and reconstruction
    mode : str, optional
        Signal extension mode for the `dwt` and `idwt` decomposition and
        reconstruction functions.
    maxlevel : int
        Maximum level of decomposition.
        If None, it will be calculated based on the `wavelet` and `data`
        length using `pywt.dwt_max_level`.
    axes : 2-tuple of ints, optional
        The axes that will be transformed.
    """
    def __init__(self, data, wavelet, mode='smooth', maxlevel=None,
                 axes=(-2, -1)):
        super(WaveletPacket2D, self).__init__(None, data, "")

        if not isinstance(wavelet, Wavelet):
            wavelet = Wavelet(wavelet)
        self.wavelet = wavelet
        self.mode = mode
        self.axes = tuple(axes)
        if len(np.unique(self.axes)) != 2:
            raise ValueError("Expected two unique axes.")
        if data is not None:
            data = np.asarray(data)
            if data.ndim < 2:
                raise ValueError(
                    "WaveletPacket2D requires data with 2 or more dimensions.")
            self.data_size = data.shape
            transform_size = [data.shape[ax] for ax in self.axes]
            if maxlevel is None:
                maxlevel = dwt_max_level(min(transform_size), self.wavelet)
        else:
            self.data_size = None
        self._maxlevel = maxlevel

    def __reduce__(self):
        return (WaveletPacket2D,
                (self.data, self.wavelet, self.mode, self.maxlevel))

    def reconstruct(self, update=True):
        """
        Reconstruct data using coefficients from subnodes.

        Parameters
        ----------
        update : bool, optional
            If True (default) then the coefficients of the current node
            and its subnodes will be replaced with values from reconstruction.
        """
        if self.has_any_subnode:
            data = super(WaveletPacket2D, self).reconstruct(update)
            if self.data_size is not None and (data.shape != self.data_size):
                data = data[[slice(sz) for sz in self.data_size]]
            if update:
                self.data = data
            return data
        return self.data  # return original data

    def get_level(self, level, order="natural", decompose=True):
        """
        Returns all nodes from specified level.

        Parameters
        ----------
        level : int
            Decomposition `level` from which the nodes will be
            collected.
        order : {'natural', 'freq'}, optional
            If `natural` (default) a flat list is returned.
            If `freq`, a 2d structure with rows and cols
            sorted by corresponding dimension frequency of 2d
            coefficient array (adapted from 1d case).
        decompose : bool, optional
            If set then the method will try to decompose the data up
            to the specified `level` (default: True).

        Notes
        -----
        Frequency order (``order="freq"``) is also known as as sequency order
        and "natural" order is sometimes referred to as Paley order. A detailed
        discussion of these orderings is also given in [1]_, [2]_.

        References
        ----------
        ..[1] M.V. Wickerhauser. Adapted Wavelet Analysis from Theory to
              Software. Wellesley. Massachusetts: A K Peters. 1994.
        ..[2] D.B. Percival and A.T. Walden.  Wavelet Methods for Time Series
              Analysis. Cambridge University Press. 2000.
              DOI:10.1017/CBO9780511841040
        """
        if order not in ["natural", "freq"]:
            raise ValueError("Invalid order: {}".format(order))
        if level > self.maxlevel:
            raise ValueError("The level cannot be greater than the maximum"
                             " decomposition level value (%d)" % self.maxlevel)

        result = []

        def collect(node):
            if node.level == level:
                result.append(node)
                return False
            return True

        self.walk(collect, decompose=decompose)

        if order == "freq":
            nodes = {}
            for (row_path, col_path), node in [
                (self.expand_2d_path(node.path), node) for node in result
            ]:
                nodes.setdefault(row_path, {})[col_path] = node
            graycode_order = get_graycode_order(level, x='l', y='h')
            nodes = [nodes[path] for path in graycode_order if path in nodes]
            result = []
            for row in nodes:
                result.append(
                    [row[path] for path in graycode_order if path in row]
                )
        return result


class WaveletPacketND(NodeND):
    """
    Data structure representing ND Wavelet Packet decomposition of signal.

    Parameters
    ----------
    data : ND ndarray
        Data associated with the node.
    wavelet : Wavelet object or name string
        Wavelet used in DWT decomposition and reconstruction
    mode : str, optional
        Signal extension mode for the `dwt` and `idwt` decomposition and
        reconstruction functions.
    maxlevel : int, optional
        Maximum level of decomposition.
        If None, it will be calculated based on the `wavelet` and `data`
        length using `pywt.dwt_max_level`.
    axes : tuple of int, optional
        The axes to transform.  The default value of `None` corresponds to all
        axes.
    """
    def __init__(self, data, wavelet, mode='smooth', maxlevel=None,
                 axes=None):
        if (data is None) and (axes is None):
            # ndim is required to create a NodeND object
            raise ValueError("If data is None, axes must be specified")

        # axes determines the number of transform dimensions
        if axes is None:
            axes = range(data.ndim)
        elif np.isscalar(axes):
            axes = (axes, )
        axes = tuple(axes)
        if len(np.unique(axes)) != len(axes):
            raise ValueError("Expected a set of unique axes.")
        ndim_transform = len(axes)

        if data is not None:
            data = np.asarray(data)
            if data.ndim == 0:
                raise ValueError("data must be at least 1D")
            ndim = data.ndim
        else:
            ndim = len(axes)

        super(WaveletPacketND, self).__init__(None, data, "", ndim,
                                              ndim_transform)
        if not isinstance(wavelet, Wavelet):
            wavelet = Wavelet(wavelet)
        self.wavelet = wavelet
        self.mode = mode
        self.axes = axes
        self.ndim_transform = ndim_transform
        if data is not None:
            if data.ndim < len(axes):
                raise ValueError("The number of axes exceeds the number of "
                                 "data dimensions.")
            self.data_size = data.shape
            transform_size = [data.shape[ax] for ax in self.axes]
            if maxlevel is None:
                maxlevel = dwt_max_level(min(transform_size), self.wavelet)
        else:
            self.data_size = None
        self._maxlevel = maxlevel

    def reconstruct(self, update=True):
        """
        Reconstruct data using coefficients from subnodes.

        Parameters
        ----------
        update : bool, optional
            If True (default) then the coefficients of the current node
            and its subnodes will be replaced with values from reconstruction.
        """
        if self.has_any_subnode:
            data = super(WaveletPacketND, self).reconstruct(update)
            if self.data_size is not None and (data.shape != self.data_size):
                data = data[[slice(sz) for sz in self.data_size]]
            if update:
                self.data = data
            return data
        return self.data  # return original data

    def get_level(self, level, decompose=True):
        """
        Returns all nodes from specified level.

        Parameters
        ----------
        level : int
            Decomposition `level` from which the nodes will be
            collected.
        decompose : bool, optional
            If set then the method will try to decompose the data up
            to the specified `level` (default: True).
        """
        if level > self.maxlevel:
            raise ValueError("The level cannot be greater than the maximum"
                             " decomposition level value (%d)" % self.maxlevel)

        result = []

        def collect(node):
            if node.level == level:
                result.append(node)
                return False
            return True

        self.walk(collect, decompose=decompose)

        return result
