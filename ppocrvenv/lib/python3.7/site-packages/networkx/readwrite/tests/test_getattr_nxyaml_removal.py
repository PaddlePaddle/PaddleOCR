"""Test that informative exception messages are raised when attempting to
access nx_yaml."""

import pytest

_msg_stub = "\n.* has been removed from NetworkX"


def test_access_from_module():
    with pytest.raises(ImportError, match=_msg_stub):
        from networkx.readwrite.nx_yaml import read_yaml
    with pytest.raises(ImportError, match=_msg_stub):
        from networkx.readwrite.nx_yaml import write_yaml


def test_access_from_nx_namespace():
    import networkx as nx

    with pytest.raises(ImportError, match=_msg_stub):
        nx.read_yaml
    with pytest.raises(ImportError, match=_msg_stub):
        nx.write_yaml


def test_access_from_readwrite_pkg():
    from networkx import readwrite

    with pytest.raises(ImportError, match=_msg_stub):
        readwrite.read_yaml
    with pytest.raises(ImportError, match=_msg_stub):
        readwrite.write_yaml


def test_accessing_nx_yaml():
    import networkx as nx

    with pytest.raises(ImportError, match=_msg_stub):
        nx.nx_yaml
