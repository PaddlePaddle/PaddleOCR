import numpy as np
from . import _colormixer
from . import _histograms
import threading
from ...util import img_as_ubyte

# utilities to make life easier for plugin writers.

import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()


class GuiLockError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class WindowManager(object):
    ''' A class to keep track of spawned windows,
    and make any needed callback once all the windows,
    are closed.'''
    def __init__(self):
        self._windows = []
        self._callback = None
        self._callback_args = ()
        self._callback_kwargs = {}
        self._gui_lock = False
        self._guikit = ''

    def _check_locked(self):
        if not self._gui_lock:
            raise GuiLockError(\
            'Must first acquire the gui lock before using this image manager')

    def _exec_callback(self):
        if self._callback:
            self._callback(*self._callback_args, **self._callback_kwargs)

    def acquire(self, kit):
        if self._gui_lock:
            raise GuiLockError(\
            'The gui lock can only be acquired by one toolkit per session. \
            The lock is already acquired by %s' % self._guikit)
        else:
            self._gui_lock = True
            self._guikit = str(kit)

    def _release(self, kit):
        # releaseing the lock will lose all references to currently
        # tracked images and the callback.
        # this function is private for reason!
        self._check_locked()
        if str(kit) == self._guikit:
            self._windows = []
            self._callback = None
            self._callback_args = ()
            self._callback_kwargs = {}
            self._gui_lock = False
            self._guikit = ''
        else:
            raise RuntimeError('Only the toolkit that owns the lock may '
                               'release it')

    def add_window(self, win):
        self._check_locked()
        self._windows.append(win)

    def remove_window(self, win):
        self._check_locked()
        try:
            self._windows.remove(win)
        except ValueError:
            print('Unable to find referenced window in tracked windows.')
            print('Ignoring...')
        else:
            if len(self._windows) == 0:
                self._exec_callback()

    def register_callback(self, cb, *cbargs, **cbkwargs):
        self._check_locked()
        self._callback = cb
        self._callback_args = cbargs
        self._callback_kwargs = cbkwargs

    def has_windows(self):
        if len(self._windows) > 0:
            return True
        else:
            return False

window_manager = WindowManager()


def prepare_for_display(npy_img):
    '''Convert a 2D or 3D numpy array of any dtype into a
    3D numpy array with dtype uint8. This array will
    be suitable for use in passing to gui toolkits for
    image display purposes.

    Parameters
    ----------
    npy_img : ndarray, 2D or 3D
        The image to convert for display

    Returns
    -------
    out : ndarray, 3D dtype=np.uint8
        The converted image. This is guaranteed to be a contiguous array.

    Notes
    -----
    If the input image is floating point, it is assumed that the data
    is in the range of 0.0 - 1.0. No check is made to assert this
    condition. The image is then scaled to be in the range 0 - 255
    and then cast to np.uint8

    For all other dtypes, the array is simply cast to np.uint8

    If a 2D array is passed, the single channel is replicated
    to the 2nd and 3rd channels.

    If the array contains an alpha channel, this channel is
    ignored.

    '''
    if npy_img.ndim < 2:
        raise ValueError('Image must be 2D or 3D array')

    height = npy_img.shape[0]
    width = npy_img.shape[1]

    out = np.empty((height, width, 3), dtype=np.uint8)
    npy_img = img_as_ubyte(npy_img)

    if npy_img.ndim == 2 or \
       (npy_img.ndim == 3 and npy_img.shape[2] == 1):
        npy_plane = npy_img.reshape((height, width))
        out[:, :, 0] = npy_plane
        out[:, :, 1] = npy_plane
        out[:, :, 2] = npy_plane

    elif npy_img.ndim == 3:
        if npy_img.shape[2] == 3 or npy_img.shape[2] == 4:
            out[:, :, :3] = npy_img[:, :, :3]
        else:
            raise ValueError('Image must have 1, 3, or 4 channels')

    else:
        raise ValueError('Image must have 2 or 3 dimensions')

    return out


def histograms(image, nbins):
    '''Calculate the channel histograms of the current image.

    Parameters
    ----------
    image : ndarray, ndim=3, dtype=np.uint8
        Input image.
    nbins : int
        The number of bins.

    Returns
    -------
    out : (rcounts, gcounts, bcounts, vcounts)
        The binned histograms of the RGB channels and intensity values.
    This is a NAIVE histogram routine, meant primarily for fast display.

    '''

    return _histograms.histograms(image, nbins)


class ImgThread(threading.Thread):
    def __init__(self, func, *args):
        super(ImgThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


class ThreadDispatch(object):
    def __init__(self, img, stateimg, func, *args):

        height = img.shape[0]
        self.cores = CPU_COUNT
        self.threads = []
        self.chunks = []

        if self.cores == 1:
            self.chunks.append((img, stateimg))

        elif self.cores >= 4:
            self.chunks.append((img[:(height // 4), :, :],
                                stateimg[:(height // 4), :, :]))
            self.chunks.append((img[(height // 4):(height // 2), :, :],
                                stateimg[(height // 4):(height // 2), :, :]))
            self.chunks.append((img[(height // 2):(3 * height // 4), :, :],
                                stateimg[(height // 2):(3 * height // 4), :, :]
                                ))
            self.chunks.append((img[(3 * height // 4):, :, :],
                                stateimg[(3 * height // 4):, :, :]))

        # if they don't have 1, or 4 or more, 2 is good.
        else:
            self.chunks.append((img[:(height // 2), :, :],
                                stateimg[:(height // 2), :, :]))
            self.chunks.append((img[(height // 2):, :, :],
                               stateimg[(height // 2):, :, :]))

        for i in range(len(self.chunks)):
            self.threads.append(ImgThread(func, self.chunks[i][0],
                                          self.chunks[i][1], *args))

    def run(self):
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()


class ColorMixer(object):
    ''' a class to manage mixing colors in an image.
    The input array must be an RGB uint8 image.

    The mixer maintains an original copy of the image,
    and uses this copy to query the pixel data for operations.
    It also makes a copy for sharing state across operations.
    That is, if you add to a channel, and multiply to same channel,
    the two operations are carried separately and the results
    averaged together.

    it modifies your array in place. This ensures that if you
    bust over a threshold, you can always come back down.

    The passed values to a function are always considered
    absolute. Thus to threshold a channel completely you
    can do mixer.add(RED, 255). Or to double the intensity
    of the blue channel: mixer.multiply(BLUE, 2.)

    To reverse these operations, respectively:
    mixer.add(RED, 0), mixer.multiply(BLUE, 1.)

    The majority of the backend is implemented in Cython,
    so it should be quite quick.
    '''

    RED = 0
    GREEN = 1
    BLUE = 2

    valid_channels = [RED, GREEN, BLUE]

    def __init__(self, img):
        if type(img) != np.ndarray:
            raise ValueError('Image must be a numpy array')
        if img.dtype != np.uint8:
            raise ValueError('Image must have dtype uint8')
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Image must be 3 channel MxNx3')

        self.img = img
        self.origimg = img.copy()
        self.stateimg = img.copy()

    def get_stateimage(self):
        return self.stateimg

    def commit_changes(self):
        self.stateimg[:] = self.img[:]

    def revert(self):
        self.stateimg[:] = self.origimg[:]
        self.img[:] = self.stateimg[:]

    def set_to_stateimg(self):
        self.img[:] = self.stateimg[:]

    def add(self, channel, ammount):
        '''Add the specified ammount to the specified channel.

        Parameters
        ----------
        channel : flag
            the color channel to operate on
            RED, GREED, or BLUE
        ammount : integer
            the ammount of color to add to the channel,
            can be positive or negative.

        '''
        if channel not in self.valid_channels:
            raise ValueError('assert_channel is not a valid channel.')
        pool = ThreadDispatch(self.img, self.stateimg,
                              _colormixer.add, channel, ammount)
        pool.run()

    def multiply(self, channel, ammount):
        '''Mutliply the indicated channel by the specified value.

        Parameters
        ----------
        channel : flag
            the color channel to operate on
            RED, GREED, or BLUE
        ammount : integer
            the ammount of color to add to the channel,
            can be positive or negative.

        '''
        if channel not in self.valid_channels:
            raise ValueError('assert_channel is not a valid channel.')
        pool = ThreadDispatch(self.img, self.stateimg,
                              _colormixer.multiply, channel, ammount)
        pool.run()

    def brightness(self, factor, offset):
        '''Adjust the brightness off an image with an offset and factor.

        Parameters
        ----------
        offset : integer
            The ammount to add to each channel.
        factor : float
            The factor to multiply each channel by.
        result = clip((pixel + offset)*factor)

        '''
        pool = ThreadDispatch(self.img, self.stateimg,
                              _colormixer.brightness, factor, offset)
        pool.run()

    def sigmoid_gamma(self, alpha, beta):
        pool = ThreadDispatch(self.img, self.stateimg,
                              _colormixer.sigmoid_gamma, alpha, beta)
        pool.run()

    def gamma(self, gamma):
        pool = ThreadDispatch(self.img, self.stateimg,
                              _colormixer.gamma, gamma)
        pool.run()

    def hsv_add(self, h_amt, s_amt, v_amt):
        '''Adjust the H, S, V channels of an image by a constant ammount.
        This is similar to the add() mixer function, but operates over the
        entire image at once. Thus all three additive values, H, S, V, must
        be supplied simultaneously.

        Parameters
        ----------
        h_amt : float
            The ammount to add to the hue (-180..180)
        s_amt : float
            The ammount to add to the saturation (-1..1)
        v_amt : float
            The ammount to add to the value (-1..1)

        '''
        pool = ThreadDispatch(self.img, self.stateimg,
                              _colormixer.hsv_add, h_amt, s_amt, v_amt)
        pool.run()

    def hsv_multiply(self, h_amt, s_amt, v_amt):
        '''Adjust the H, S, V channels of an image by a constant ammount.
        This is similar to the add() mixer function, but operates over the
        entire image at once. Thus all three additive values, H, S, V, must
        be supplied simultaneously.

        Note that since hue is in degrees, it makes no sense to multiply
        that channel, thus an add operation is performed on the hue. And the
        values given for h_amt, should be the same as for hsv_add

        Parameters
        ----------
        h_amt : float
            The ammount to to add to the hue (-180..180)
        s_amt : float
            The ammount to multiply to the saturation (0..1)
        v_amt : float
            The ammount to multiply to the value (0..1)

        '''
        pool = ThreadDispatch(self.img, self.stateimg,
                              _colormixer.hsv_multiply, h_amt, s_amt, v_amt)
        pool.run()

    def rgb_2_hsv_pixel(self, R, G, B):
        '''Convert an RGB value to HSV

        Parameters
        ----------
        R : int
            Red value
        G : int
            Green value
        B : int
            Blue value

        Returns
        -------
        out : (H, S, V) Floats
            The HSV values

        '''
        H, S, V = _colormixer.py_rgb_2_hsv(R, G, B)
        return (H, S, V)

    def hsv_2_rgb_pixel(self, H, S, V):
        '''Convert an HSV value to RGB

        Parameters
        ----------
        H : float
            Hue value
        S : float
            Saturation value
        V : float
            Intensity value

        Returns
        -------
        out : (R, G, B) ints
            The RGB values

        '''
        R, G, B = _colormixer.py_hsv_2_rgb(H, S, V)
        return (R, G, B)
