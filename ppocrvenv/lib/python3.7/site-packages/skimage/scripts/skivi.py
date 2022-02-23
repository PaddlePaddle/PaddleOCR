"""skimage viewer"""


def main():
    import skimage.io as io
    import sys
    from warnings import warn

    warn('`skivi.py` script is deprecated and will be removed in 0.20. '
         'For alternatives, refer to '
         'https://scikit-image.org/docs/stable/user_guide/visualization.html',
         FutureWarning, stacklevel=2)

    if len(sys.argv) != 2:
        print("Usage: skivi <image-file>")
        sys.exit(-1)

    io.use_plugin('qt')
    io.imshow(io.imread(sys.argv[1]), fancy=True)
    io.show()
