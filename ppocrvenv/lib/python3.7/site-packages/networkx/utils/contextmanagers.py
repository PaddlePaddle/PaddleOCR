from contextlib import contextmanager
import warnings

__all__ = ["reversed"]


@contextmanager
def reversed(G):
    """A context manager for temporarily reversing a directed graph in place.

    This is a no-op for undirected graphs.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    Warning
    -------
    The reversed context manager is deprecated in favor
    of G.reverse(copy=False). The view allows multiple threads to use the
    same graph without confusion while the context manager does not.
    This context manager is scheduled to be removed in version 3.0.
    """
    msg = (
        "context manager reversed is deprecated and to be removed in 3.0."
        "Use G.reverse(copy=False) if G.is_directed() else G instead."
    )
    warnings.warn(msg, DeprecationWarning)

    directed = G.is_directed()
    if directed:
        G._pred, G._succ = G._succ, G._pred
        G._adj = G._succ

    try:
        yield
    finally:
        if directed:
            # Reverse the reverse.
            G._pred, G._succ = G._succ, G._pred
            G._adj = G._succ
