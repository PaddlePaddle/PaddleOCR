""":mod:`wand.sequence` --- Sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.3.0

"""
import contextlib
import ctypes
import numbers

from .api import libmagick, library
from .compat import abc, binary, xrange
from .image import BaseImage, ImageProperty
from .version import MAGICK_VERSION_INFO

__all__ = 'Sequence', 'SingleImage'


class Sequence(ImageProperty, abc.MutableSequence):
    """The list-like object that contains every :class:`SingleImage`
    in the :class:`~wand.image.Image` container.  It implements
    :class:`collections.abc.Sequence` protocol.

    .. versionadded:: 0.3.0

    """

    def __init__(self, image):
        super(Sequence, self).__init__(image)
        self.instances = []

    @property
    def current_index(self):
        """(:class:`numbers.Integral`) The current index of
        its internal iterator.

        .. note::

           It's only for internal use.

        """
        return library.MagickGetIteratorIndex(self.image.wand)

    @current_index.setter
    def current_index(self, index):
        library.MagickSetIteratorIndex(self.image.wand, index)

    @contextlib.contextmanager
    def index_context(self, index):
        """Scoped setter of :attr:`current_index`.  Should be
        used for :keyword:`with` statement e.g.::

            with image.sequence.index_context(3):
                print(image.size)

        .. note::

           It's only for internal use.

        """
        index = self.validate_position(index)
        tmp_idx = self.current_index
        self.current_index = index
        yield index
        self.current_index = tmp_idx

    def __len__(self):
        return library.MagickGetNumberImages(self.image.wand)

    def validate_position(self, index):
        if not isinstance(index, numbers.Integral):
            raise TypeError('index must be integer, not ' + repr(index))
        length = len(self)
        if index >= length or index < -length:
            raise IndexError(
                'out of index: {0} (total: {1})'.format(index, length)
            )
        if index < 0:
            index += length
        return index

    def validate_slice(self, slice_, as_range=False):
        if not (slice_.step is None or slice_.step == 1):
            raise ValueError('slicing with step is unsupported')
        length = len(self)
        if slice_.start is None:
            start = 0
        elif slice_.start < 0:
            start = length + slice_.start
        else:
            start = slice_.start
        start = min(length, start)
        if slice_.stop is None:
            stop = 0
        elif slice_.stop < 0:
            stop = length + slice_.stop
        else:
            stop = slice_.stop
        stop = min(length, stop or length)
        return xrange(start, stop) if as_range else slice(start, stop, None)

    def __getitem__(self, index):
        if isinstance(index, slice):
            slice_ = self.validate_slice(index)
            return [self[i] for i in xrange(slice_.start, slice_.stop)]
        index = self.validate_position(index)
        instances = self.instances
        instances_length = len(instances)
        if index < instances_length:
            instance = instances[index]
            if (instance is not None and
                    getattr(instance, 'c_resource', None) is not None):
                return instance
        else:
            number_to_extend = index - instances_length + 1
            instances.extend(None for _ in xrange(number_to_extend))
        wand = self.image.wand
        tmp_idx = library.MagickGetIteratorIndex(wand)
        library.MagickSetIteratorIndex(wand, index)
        image = library.GetImageFromMagickWand(wand)
        exc = libmagick.AcquireExceptionInfo()
        single_image = libmagick.CloneImages(image, binary(str(index)), exc)
        libmagick.DestroyExceptionInfo(exc)
        single_wand = library.NewMagickWandFromImage(single_image)
        single_image = libmagick.DestroyImage(single_image)
        library.MagickSetIteratorIndex(wand, tmp_idx)
        instance = SingleImage(single_wand, self.image, image)
        self.instances[index] = instance
        return instance

    def __setitem__(self, index, image):
        if isinstance(index, slice):
            tmp_idx = self.current_index
            slice_ = self.validate_slice(index)
            del self[slice_]
            self.extend(image, offset=slice_.start)
            self.current_index = tmp_idx
        else:
            if not isinstance(image, BaseImage):
                raise TypeError('image must be an instance of wand.image.'
                                'BaseImage, not ' + repr(image))
            with self.index_context(index) as index:
                if library.MagickHasNextImage(self.image.wand):
                    library.MagickAddImage(self.image.wand, image.wand)
                    library.MagickRemoveImage(self.image.wand)
                else:
                    library.MagickRemoveImage(self.image.wand)
                    library.MagickAddImage(self.image.wand, image.wand)

    def __delitem__(self, index):
        if isinstance(index, slice):
            range_ = self.validate_slice(index, as_range=True)
            for i in reversed(range_):
                del self[i]
        else:
            with self.index_context(index) as index:
                library.MagickRemoveImage(self.image.wand)
                if index < len(self.instances):
                    del self.instances[index]

    def insert(self, index, image):
        try:
            index = self.validate_position(index)
        except IndexError:
            index = len(self)
        if not isinstance(image, BaseImage):
            raise TypeError('image must be an instance of wand.image.'
                            'BaseImage, not ' + repr(image))
        if not self:
            library.MagickAddImage(self.image.wand, image.wand)
        elif index == 0:
            tmp_idx = self.current_index
            self_wand = self.image.wand
            wand = image.sequence[0].wand
            try:
                # Prepending image into the list using MagickSetFirstIterator()
                # and MagickAddImage() had not worked properly, but was fixed
                # since 6.7.6-0 (rev7106).
                if MAGICK_VERSION_INFO >= (6, 7, 6, 0):
                    library.MagickSetFirstIterator(self_wand)
                    library.MagickAddImage(self_wand, wand)
                else:  # pragma: no cover
                    self.current_index = 0
                    library.MagickAddImage(self_wand,
                                           self.image.sequence[0].wand)
                    self.current_index = 0
                    library.MagickAddImage(self_wand, wand)
                    self.current_index = 0
                    library.MagickRemoveImage(self_wand)
            finally:
                self.current_index = tmp_idx
        else:
            with self.index_context(index - 1):
                library.MagickAddImage(self.image.wand, image.sequence[0].wand)
        self.instances.insert(index, None)

    def append(self, image):
        if not isinstance(image, BaseImage):
            raise TypeError('image must be an instance of wand.image.'
                            'BaseImage, not ' + repr(image))
        wand = self.image.wand
        tmp_idx = self.current_index
        try:
            library.MagickSetLastIterator(wand)
            library.MagickAddImage(wand, image.sequence[0].wand)
        finally:
            self.current_index = tmp_idx
        self.instances.append(None)

    def extend(self, images, offset=None):
        tmp_idx = self.current_index
        wand = self.image.wand
        length = 0
        try:
            if offset is None:
                library.MagickSetLastIterator(self.image.wand)
            else:
                if offset == 0:
                    images = iter(images)
                    self.insert(0, next(images))
                    offset += 1
                self.current_index = offset - 1
            if isinstance(images, type(self)):
                library.MagickAddImage(wand, images.image.wand)
                length = len(images)
            else:
                delta = 1 if MAGICK_VERSION_INFO >= (6, 7, 6, 0) else 2
                for image in images:
                    if not isinstance(image, BaseImage):
                        raise TypeError(
                            'images must consist of only instances of '
                            'wand.image.BaseImage, not ' + repr(image)
                        )
                    else:
                        library.MagickAddImage(wand, image.sequence[0].wand)
                        self.instances = []
                        if offset is None:
                            library.MagickSetLastIterator(self.image.wand)
                        else:
                            self.current_index += delta
                        length += 1
        finally:
            self.current_index = tmp_idx
        null_list = [None] * length
        if offset is None:
            self.instances[offset:] = null_list
        else:
            self.instances[offset:offset] = null_list

    def _repr_png_(self):  # pragma: no cover
        library.MagickResetIterator(self.image.wand)
        repr_wand = library.MagickAppendImages(self.image.wand, 1)
        length = ctypes.c_size_t()
        blob_p = library.MagickGetImagesBlob(repr_wand,
                                             ctypes.byref(length))
        if blob_p and length.value:
            blob = ctypes.string_at(blob_p, length.value)
            blob_p = library.MagickRelinquishMemory(blob_p)
            return blob
        else:
            return None


class SingleImage(BaseImage):
    """Each single image in :class:`~wand.image.Image` container.
    For example, it can be a frame of GIF animation.

    Note that all changes on single images are invisible to their
    containers unless they are altered a ``with ...`` context manager.

        with Image(filename='animation.gif') as container:
            with container.sequence[0] as frame:
                frame.negate()

    .. versionadded:: 0.3.0

    .. versionchanged:: 0.5.1
       Only sync changes of a :class:`SingleImage` when exiting a ``with ...``
       context. Not when parent :class:`~wand.image.Image` closes.
    """

    #: (:class:`wand.image.Image`) The container image.
    container = None

    def __init__(self, wand, container, c_original_resource):
        super(SingleImage, self).__init__(wand)
        self.container = container
        self.c_original_resource = c_original_resource
        self._delay = None

    @property
    def sequence(self):
        return self,

    @property
    def index(self):
        """(:class:`numbers.Integral`) The index of the single image in
        the :attr:`container` image.

        """
        wand = self.container.wand
        library.MagickResetIterator(wand)
        image = library.GetImageFromMagickWand(wand)
        i = 0
        while self.c_original_resource != image and image:
            image = libmagick.GetNextImageInList(image)
            i += 1
        assert image
        assert self.c_original_resource == image
        return i

    @property
    def delay(self):
        """(:class:`numbers.Integral`) The delay to pause before display
        the next image (in the :attr:`~wand.image.BaseImage.sequence` of
        its :attr:`container`).  It's hundredths of a second.

        """
        if self._delay is None:
            container = self.container
            with container.sequence.index_context(self.index):
                self._delay = library.MagickGetImageDelay(container.wand)
        return self._delay

    @delay.setter
    def delay(self, delay):
        if not isinstance(delay, numbers.Integral):
            raise TypeError('delay must be an integer, not ' + repr(delay))
        elif delay < 0:
            raise ValueError('delay cannot be less than zero')
        container = self.container
        with container.sequence.index_context(self.index):
            library.MagickSetImageDelay(container.wand, delay)
        self._delay = delay

    def _sync_container_sequence(self):
        """If instances was flagged as :attr:`dirty` by any manipulation
        methods, then this instance will overwrite :attr:`container` internal
        version at :attr:`index`.

        .. versionadded:: 0.5.1
        """
        if self.dirty:
            self.container.sequence[self.index] = self
            self.dirty = False  # Reset dirty flag

    def __exit__(self, type_, value, traceback):
        self._sync_container_sequence()
        super(SingleImage, self).__exit__(type_, value, traceback)

    def __repr__(self):
        cls = type(self)
        if getattr(self, 'c_resource', None) is None:
            return '<{0}.{1}: (closed)>'.format(cls.__module__, cls.__name__)
        return '<{0}.{1}: {2} ({3}x{4})>'.format(
            cls.__module__, cls.__name__,
            self.signature[:7], self.width, self.height
        )
