# flake8: noqa
###############################################################################
# Compat file to load the correct wait function
#
# author: Thomas Moreau and Olivier grisel
#
import sys

# Compat wait
if sys.version_info < (3, 3):
    from ._posix_wait import wait
else:
    from multiprocessing.connection import wait
