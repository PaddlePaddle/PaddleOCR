"""fontTools.ttLib -- a package for dealing with TrueType fonts."""

from fontTools.misc.loggingTools import deprecateFunction
import logging


log = logging.getLogger(__name__)

class TTLibError(Exception): pass

@deprecateFunction("use logging instead", category=DeprecationWarning)
def debugmsg(msg):
	import time
	print(msg + time.strftime("  (%H:%M:%S)", time.localtime(time.time())))

from fontTools.ttLib.ttFont import *
from fontTools.ttLib.ttCollection import TTCollection
