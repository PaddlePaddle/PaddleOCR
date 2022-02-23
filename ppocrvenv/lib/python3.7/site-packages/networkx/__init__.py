"""
NetworkX
========

NetworkX is a Python package for the creation, manipulation, and study of the
structure, dynamics, and functions of complex networks.

See https://networkx.org for complete documentation.
"""

__version__ = "2.6.3"


def __getattr__(name):
    """Remove functions and provide informative error messages."""
    if name == "nx_yaml":
        raise ImportError(
            "\nThe nx_yaml module has been removed from NetworkX.\n"
            "Please use the `yaml` package directly for working with yaml data.\n"
            "For example, a networkx.Graph `G` can be written to and loaded\n"
            "from a yaml file with:\n\n"
            "    import yaml\n\n"
            "    with open('path_to_yaml_file', 'w') as fh:\n"
            "        yaml.dump(G, fh)\n"
            "    with open('path_to_yaml_file', 'r') as fh:\n"
            "        G = yaml.load(fh, Loader=yaml.Loader)\n\n"
            "Note that yaml.Loader is considered insecure - see the pyyaml\n"
            "documentation for further details.\n\n"
            "This message will be removed in NetworkX 3.0."
        )
    if name == "read_yaml":
        raise ImportError(
            "\nread_yaml has been removed from NetworkX, please use `yaml`\n"
            "directly:\n\n"
            "    import yaml\n\n"
            "    with open('path', 'r') as fh:\n"
            "        yaml.load(fh, Loader=yaml.Loader)\n\n"
            "Note that yaml.Loader is considered insecure - see the pyyaml\n"
            "documentation for further details.\n\n"
            "This message will be removed in NetworkX 3.0."
        )
    if name == "write_yaml":
        raise ImportError(
            "\nwrite_yaml has been removed from NetworkX, please use `yaml`\n"
            "directly:\n\n"
            "    import yaml\n\n"
            "    with open('path_for_yaml_output', 'w') as fh:\n"
            "        yaml.dump(G_to_be_yaml, path_for_yaml_output, **kwds)\n\n"
            "This message will be removed in NetworkX 3.0."
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")


# These are import orderwise
from networkx.exception import *

from networkx import utils

from networkx import classes
from networkx.classes import filters
from networkx.classes import *

from networkx import convert
from networkx.convert import *

from networkx import convert_matrix
from networkx.convert_matrix import *

from networkx import relabel
from networkx.relabel import *

from networkx import generators
from networkx.generators import *

from networkx import readwrite
from networkx.readwrite import *

# Need to test with SciPy, when available
from networkx import algorithms
from networkx.algorithms import *

from networkx import linalg
from networkx.linalg import *

from networkx.testing.test import run as test

from networkx import drawing
from networkx.drawing import *
