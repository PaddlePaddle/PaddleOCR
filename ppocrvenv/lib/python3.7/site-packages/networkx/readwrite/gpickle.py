"""
**************
Pickled Graphs
**************
Read and write NetworkX graphs as Python pickles.

"The pickle module implements a fundamental, but powerful algorithm
for serializing and de-serializing a Python object
structure. "Pickling" is the process whereby a Python object hierarchy
is converted into a byte stream, and "unpickling" is the inverse
operation, whereby a byte stream is converted back into an object
hierarchy."

Note that NetworkX graphs can contain any hashable Python object as
node (not just integers and strings).  For arbitrary data types it may
be difficult to represent the data as text.  In that case using Python
pickles to store the graph data can be used.

Format
------
See https://docs.python.org/3/library/pickle.html
"""

__all__ = ["read_gpickle", "write_gpickle"]

from networkx.utils import open_file

import pickle
import warnings


@open_file(1, mode="wb")
def write_gpickle(G, path, protocol=pickle.HIGHEST_PROTOCOL):
    """Write graph in Python pickle format.

    Pickles are a serialized byte stream of a Python object [1]_.
    This format will preserve Python objects used as nodes or edges.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be compressed.

    protocol : integer
        Pickling protocol to use. Default value: ``pickle.HIGHEST_PROTOCOL``.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_gpickle(G, "test.gpickle")

    References
    ----------
    .. [1] https://docs.python.org/3/library/pickle.html

    .. deprecated:: 2.6
    """
    msg = (
        "write_gpickle is deprecated and will be removed in 3.0."
        "Use ``pickle.dump(G, path, protocol)``"
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    pickle.dump(G, path, protocol)


@open_file(0, mode="rb")
def read_gpickle(path):
    """Read graph object in Python pickle format.

    Pickles are a serialized byte stream of a Python object [1]_.
    This format will preserve Python objects used as nodes or edges.

    Parameters
    ----------
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be uncompressed.

    Returns
    -------
    G : graph
       A NetworkX graph

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_gpickle(G, "test.gpickle")
    >>> G = nx.read_gpickle("test.gpickle")

    References
    ----------
    .. [1] https://docs.python.org/3/library/pickle.html

    .. deprecated:: 2.6
    """
    msg = (
        "read_gpickle is deprecated and will be removed in 3.0."
        "Use ``pickle.load(path)``"
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return pickle.load(path)
