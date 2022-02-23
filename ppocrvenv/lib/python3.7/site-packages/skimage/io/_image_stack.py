import numpy as np


__all__ = ['image_stack', 'push', 'pop']


# Shared image queue
image_stack = []


def push(img):
    """Push an image onto the shared image stack.

    Parameters
    ----------
    img : ndarray
        Image to push.

    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Can only push ndarrays to the image stack.")

    image_stack.append(img)


def pop():
    """Pop an image from the shared image stack.

    Returns
    -------
    img : ndarray
        Image popped from the stack.

    """
    return image_stack.pop()
