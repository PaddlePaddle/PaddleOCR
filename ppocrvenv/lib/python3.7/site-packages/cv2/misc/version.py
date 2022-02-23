import cv2


def get_ocv_version():
    return getattr(cv2, "__version__", "unavailable")
