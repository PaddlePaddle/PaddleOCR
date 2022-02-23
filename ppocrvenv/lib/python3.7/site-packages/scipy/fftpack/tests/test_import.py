"""Test possibility of patching fftpack with pyfftw.

No module source outside of scipy.fftpack should contain an import of
the form `from scipy.fftpack import ...`, so that a simple replacement
of scipy.fftpack by the corresponding fftw interface completely swaps
the two FFT implementations.

Because this simply inspects source files, we only need to run the test
on one version of Python.
"""


from pathlib import Path
import re
import tokenize
from numpy.testing import assert_
import scipy

class TestFFTPackImport:
    def test_fftpack_import(self):
        base = Path(scipy.__file__).parent
        regexp = r"\s*from.+\.fftpack import .*\n"
        for path in base.rglob("*.py"):
            if base / "fftpack" in path.parents:
                continue
            # use tokenize to auto-detect encoding on systems where no
            # default encoding is defined (e.g., LANG='C')
            with tokenize.open(str(path)) as file:
                assert_(all(not re.fullmatch(regexp, line)
                            for line in file),
                        "{0} contains an import from fftpack".format(path))
