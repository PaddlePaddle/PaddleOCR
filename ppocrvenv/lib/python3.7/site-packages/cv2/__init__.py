import importlib
import os
import sys

from .cv2 import *
from .data import *

# wildcard import above does not import "private" variables like __version__
# this makes them available
globals().update(importlib.import_module('cv2.cv2').__dict__)

if sys.platform == 'darwin':
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'qt', 'plugins'
    )
