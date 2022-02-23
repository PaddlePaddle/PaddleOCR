from os.path import dirname, join

def get_data_file_path(basename):
    """
    Returns the path to the test data file given by `basename`
    """
    return join(dirname(__file__), basename)
