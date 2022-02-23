__all__ = ['imread', 'imsave']

try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError("SimpleITK could not be found. "
                      "Please try "
                      "  easy_install SimpleITK "
                      "or refer to "
                      "  http://simpleitk.org/ "
                      "for further instructions.")


def imread(fname):
    sitk_img = sitk.ReadImage(fname)
    return sitk.GetArrayFromImage(sitk_img)


def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr, isVector=True)
    sitk.WriteImage(sitk_img, fname)
