def return_slice_of_data(arr, start_idx, end_idx):
    return arr[start_idx:end_idx]


def print_filename_and_raise(arr):
    from joblib._memmapping_reducer import _get_backing_memmap
    print(_get_backing_memmap(arr).filename)
    raise ValueError
