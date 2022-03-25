
import sys

if sys.version_info[0] == 2:
	from cpuinfo import *
else:
	from cpuinfo.cpuinfo import *


