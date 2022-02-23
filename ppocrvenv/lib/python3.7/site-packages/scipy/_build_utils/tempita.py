import sys
import os

from Cython import Tempita as tempita
# XXX: If this import ever fails (does it really?), vendor either
# cython.tempita or numpy/npy_tempita.


def process_tempita(fromfile):
    """Process tempita templated file and write out the result.

    The template file is expected to end in `.c.in` or `.pyx.in`:
    E.g. processing `template.c.in` generates `template.c`.

    """
    if not fromfile.endswith('.in'):
        raise ValueError("Unexpected extension: %s" % fromfile)

    from_filename = tempita.Template.from_filename
    template = from_filename(fromfile,
                             encoding=sys.getdefaultencoding())    

    content = template.substitute()

    outfile = os.path.splitext(fromfile)[0]
    with open(outfile, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    process_tempita(sys.argv[1])
    
