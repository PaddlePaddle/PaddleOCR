# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Read/Write video using FFMPEG

Backend Library: https://github.com/imageio/imageio-ffmpeg

.. note::
    To use this plugin you have to install its backend::

        pip install imageio[ffmpeg]


The ffmpeg format provides reading and writing for a wide range of movie formats
such as .avi, .mpeg, .mp4, etc. as well as the ability to read streams from
webcams and USB cameras. It is based on ffmpeg and is inspired by/based `moviepy
<https://github.com/Zulko/moviepy/>`_ by Zulko.

Parameters for reading
----------------------
fps : scalar
    The number of frames per second to read the data at. Default None (i.e.
    read at the file's own fps). One can use this for files with a
    variable fps, or in cases where imageio is unable to correctly detect
    the fps. In case of trouble opening camera streams, it may help to set an
    explicit fps value matching a framerate supported by the camera.
loop : bool
    If True, the video will rewind as soon as a frame is requested
    beyond the last frame. Otherwise, IndexError is raised. Default False.
    Setting this to True will internally call ``count_frames()``,
    and set the reader's length to that value instead of inf.
size : str | tuple
    The frame size (i.e. resolution) to read the images, e.g.
    (100, 100) or "640x480". For camera streams, this allows setting
    the capture resolution. For normal video data, ffmpeg will
    rescale the data.
dtype : str | type
    The dtype for the output arrays. Determines the bit-depth that
    is requested from ffmpeg. Supported dtypes: uint8, uint16.
    Default: uint8.
pixelformat : str
    The pixel format for the camera to use (e.g. "yuyv422" or
    "gray"). The camera needs to support the format in order for
    this to take effect. Note that the images produced by this
    reader are always RGB.
input_params : list
    List additional arguments to ffmpeg for input file options.
    (Can also be provided as ``ffmpeg_params`` for backwards compatibility)
    Example ffmpeg arguments to use aggressive error handling:
    ['-err_detect', 'aggressive']
output_params : list
    List additional arguments to ffmpeg for output file options (i.e. the
    stream being read by imageio).
print_info : bool
    Print information about the video file as reported by ffmpeg.

Parameters for writing
----------------------
fps : scalar
    The number of frames per second. Default 10.
codec : str
    the video codec to use. Default 'libx264', which represents the
    widely available mpeg4. Except when saving .wmv files, then the
    defaults is 'msmpeg4' which is more commonly supported for windows
quality : float | None
    Video output quality. Default is 5. Uses variable bit rate. Highest
    quality is 10, lowest is 0. Set to None to prevent variable bitrate
    flags to FFMPEG so you can manually specify them using output_params
    instead. Specifying a fixed bitrate using 'bitrate' disables this
    parameter.
bitrate : int | None
    Set a constant bitrate for the video encoding. Default is None causing
    'quality' parameter to be used instead.  Better quality videos with
    smaller file sizes will result from using the 'quality'  variable
    bitrate parameter rather than specifiying a fixed bitrate with this
    parameter.
pixelformat: str
    The output video pixel format. Default is 'yuv420p' which most widely
    supported by video players.
input_params : list
    List additional arguments to ffmpeg for input file options (i.e. the
    stream that imageio provides).
output_params : list
    List additional arguments to ffmpeg for output file options.
    (Can also be provided as ``ffmpeg_params`` for backwards compatibility)
    Example ffmpeg arguments to use only intra frames and set aspect ratio:
    ['-intra', '-aspect', '16:9']
ffmpeg_log_level: str
    Sets ffmpeg output log level.  Default is "warning".
    Values can be "quiet", "panic", "fatal", "error", "warning", "info"
    "verbose", or "debug". Also prints the FFMPEG command being used by
    imageio if "info", "verbose", or "debug".
macro_block_size: int
    Size constraint for video. Width and height, must be divisible by this
    number. If not divisible by this number imageio will tell ffmpeg to
    scale the image up to the next closest size
    divisible by this number. Most codecs are compatible with a macroblock
    size of 16 (default), some can go smaller (4, 8). To disable this
    automatic feature set it to None or 1, however be warned many players
    can't decode videos that are odd in size and some codecs will produce
    poor results or fail. See https://en.wikipedia.org/wiki/Macroblock.


Notes
-----
If you are using anaconda and ``anaconda/ffmpeg`` you will not be able to
encode/decode H.264 (likely due to licensing concerns). If you need this
format on anaconda install ``conda-forge/ffmpeg`` instead.

You can use the ``IMAGEIO_FFMPEG_EXE`` environment variable to force using a
specific ffmpeg executable.

To get the number of frames before having read them all, you can use the
``reader.count_frames()`` method (the reader will then use
``imageio_ffmpeg.count_frames_and_secs()`` to get the exact number of frames,
note that this operation can take a few seconds on large files). Alternatively,
the number of frames can be estimated from the fps and duration in the meta data
(though these values themselves are not always present/reliable).

"""

import re
import sys
import time
import logging
import platform
import threading
import subprocess as sp

import numpy as np

from ..core import Format, image_as_uint

logger = logging.getLogger(__name__)

# Get camera format
if sys.platform.startswith("win"):
    CAM_FORMAT = "dshow"  # dshow or vfwcap
elif sys.platform.startswith("linux"):
    CAM_FORMAT = "video4linux2"
elif sys.platform.startswith("darwin"):
    CAM_FORMAT = "avfoundation"
else:  # pragma: no cover
    CAM_FORMAT = "unknown-cam-format"


def download(directory=None, force_download=False):  # pragma: no cover
    raise RuntimeError(
        "imageio.ffmpeg.download() has been deprecated. "
        "Use 'pip install imageio-ffmpeg' instead.'"
    )


# For backwards compatibility - we dont use this ourselves
def get_exe():  # pragma: no cover
    """Wrapper for imageio_ffmpeg.get_ffmpeg_exe()"""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


_ffmpeg_api = None


def _get_ffmpeg_api():
    global _ffmpeg_api
    if _ffmpeg_api is None:
        try:
            import imageio_ffmpeg
        except ImportError:
            raise ImportError(
                "To use the imageio ffmpeg plugin you need to "
                "'pip install imageio-ffmpeg'"
            )
        _ffmpeg_api = imageio_ffmpeg
    return _ffmpeg_api


class FfmpegFormat(Format):
    """Read/Write ImageResources using FFMPEG.

    See :mod:`imageio.plugins.ffmpeg`
    """

    def _can_read(self, request):
        if request.mode[1] not in "I?":
            return False

        # Read from video stream?
        # Note that we could write the _video flag here, but a user might
        # select this format explicitly (and this code is not run)
        if re.match(r"<video(\d+)>", request.filename):
            return True

        # Read from file that we know?
        if request.extension in self.extensions:
            return True

    def _can_write(self, request):
        if request.mode[1] in (self.modes + "?"):
            if request.extension in self.extensions:
                return True

    # --

    class Reader(Format.Reader):

        _frame_catcher = None
        _read_gen = None

        def _get_cam_inputname(self, index):
            if sys.platform.startswith("linux"):
                return "/dev/" + self.request._video[1:-1]

            elif sys.platform.startswith("win"):
                # Ask ffmpeg for list of dshow device names
                ffmpeg_api = _get_ffmpeg_api()
                cmd = [
                    ffmpeg_api.get_ffmpeg_exe(),
                    "-list_devices",
                    "true",
                    "-f",
                    CAM_FORMAT,
                    "-i",
                    "dummy",
                ]
                # Set `shell=True` in sp.run to prevent popup of a command
                # line window in frozen applications. Note: this would be a
                # security vulnerability if user-input goes into the cmd.
                # Note that the ffmpeg process returns with exit code 1 when
                # using `-list_devices` (or `-list_options`), even if the
                # command is successful, so we set `check=False` explicitly.
                completed_process = sp.run(
                    cmd,
                    stdout=sp.PIPE,
                    stderr=sp.PIPE,
                    encoding="utf-8",
                    shell=True,
                    check=False,
                )

                # Return device name at index
                try:
                    name = parse_device_names(completed_process.stderr)[index]
                except IndexError:
                    raise IndexError("No ffdshow camera at index %i." % index)
                return "video=%s" % name

            elif sys.platform.startswith("darwin"):
                # Appears that newer ffmpeg builds don't support -list-devices
                # on OS X. But you can directly open the camera by index.
                name = str(index)
                return name

            else:  # pragma: no cover
                return "??"

        def _open(
            self,
            loop=False,
            size=None,
            dtype=None,
            pixelformat=None,
            print_info=False,
            ffmpeg_params=None,
            input_params=None,
            output_params=None,
            fps=None,
        ):
            # Get generator functions
            self._ffmpeg_api = _get_ffmpeg_api()
            # Process input args
            self._arg_loop = bool(loop)
            if size is None:
                self._arg_size = None
            elif isinstance(size, tuple):
                self._arg_size = "%ix%i" % size
            elif isinstance(size, str) and "x" in size:
                self._arg_size = size
            else:
                raise ValueError('FFMPEG size must be tuple of "NxM"')
            if pixelformat is None:
                pass
            elif not isinstance(pixelformat, str):
                raise ValueError("FFMPEG pixelformat must be str")
            if dtype is None:
                self._dtype = np.dtype("uint8")
            else:
                self._dtype = np.dtype(dtype)
                allowed_dtypes = ["uint8", "uint16"]
                if self._dtype.name not in allowed_dtypes:
                    raise ValueError(
                        "dtype must be one of: {}".format(", ".join(allowed_dtypes))
                    )
            self._arg_pixelformat = pixelformat
            self._arg_input_params = input_params or []
            self._arg_output_params = output_params or []
            self._arg_input_params += ffmpeg_params or []  # backward compat
            # Write "_video"_arg - indicating webcam support
            self.request._video = None
            regex_match = re.match(r"<video(\d+)>", self.request.filename)
            if regex_match:
                self.request._video = self.request.filename
            # Specify input framerate? (only on macOS)
            if self.request._video and platform.system().lower() == "darwin":
                if "-framerate" not in str(self._arg_input_params):
                    self._arg_input_params.extend(["-framerate", str(float(fps or 15))])
            # Get local filename
            if self.request._video:
                index = int(regex_match.group(1))
                self._filename = self._get_cam_inputname(index)
            else:
                self._filename = self.request.get_local_filename()
                # When passed to ffmpeg on command line, carets need to be escaped.
                self._filename = self._filename.replace("^", "^^")
            # Determine pixel format and depth
            self._depth = 3
            if self._dtype.name == "uint8":
                self._pix_fmt = "rgb24"
                self._bytes_per_channel = 1
            else:
                self._pix_fmt = "rgb48le"
                self._bytes_per_channel = 2
            # Initialize parameters
            self._pos = -1
            self._meta = {"plugin": "ffmpeg"}
            self._lastread = None

            # Calculating this from fps and duration is not accurate,
            # and calculating it exactly with ffmpeg_api.count_frames_and_secs
            # takes too long to do for each video. But we need it for looping.
            self._nframes = float("inf")
            if self._arg_loop and not self.request._video:
                self._nframes = self.count_frames()
            self._meta["nframes"] = self._nframes

            # Start ffmpeg subprocess and get meta information
            self._initialize()

            # For cameras, create thread that keeps reading the images
            if self.request._video:
                self._frame_catcher = FrameCatcher(self._read_gen)

            # For reference - but disabled, because it is inaccurate
            # if self._meta["nframes"] == float("inf"):
            #     if self._meta.get("fps", 0) > 0:
            #         if self._meta.get("duration", 0) > 0:
            #             n = round(self._meta["duration"] * self._meta["fps"])
            #             self._meta["nframes"] = int(n)

        def _close(self):
            # First close the frame catcher, because we cannot close the gen
            # if the frame catcher thread is using it
            if self._frame_catcher is not None:
                self._frame_catcher.stop_me()
                self._frame_catcher = None
            if self._read_gen is not None:
                self._read_gen.close()
                self._read_gen = None

        def count_frames(self):
            """Count the number of frames. Note that this can take a few
            seconds for large files. Also note that it counts the number
            of frames in the original video and does not take a given fps
            into account.
            """
            # This would have been nice, but this does not work :(
            # oargs = []
            # if self.request.kwargs.get("fps", None):
            #     fps = float(self.request.kwargs["fps"])
            #     oargs += ["-r", "%.02f" % fps]
            cf = self._ffmpeg_api.count_frames_and_secs
            return cf(self._filename)[0]

        def _get_length(self):
            return self._nframes  # only not inf if loop is True

        def _get_data(self, index):
            """Reads a frame at index. Note for coders: getting an
            arbitrary frame in the video with ffmpeg can be painfully
            slow if some decoding has to be done. This function tries
            to avoid fectching arbitrary frames whenever possible, by
            moving between adjacent frames."""
            # Modulo index (for looping)
            if self._arg_loop and self._nframes < float("inf"):
                index %= self._nframes

            if index == self._pos:
                return self._lastread, dict(new=False)
            elif index < 0:
                raise IndexError("Frame index must be >= 0")
            elif index >= self._nframes:
                raise IndexError("Reached end of video")
            else:
                if (index < self._pos) or (index > self._pos + 100):
                    self._initialize(index)
                else:
                    self._skip_frames(index - self._pos - 1)
                result, is_new = self._read_frame()
                self._pos = index
                return result, dict(new=is_new)

        def _get_meta_data(self, index):
            return self._meta

        def _initialize(self, index=0):

            # Close the current generator, and thereby terminate its subprocess
            if self._read_gen is not None:
                self._read_gen.close()

            iargs = []
            oargs = []

            # Create input args
            iargs += self._arg_input_params
            if self.request._video:
                iargs += ["-f", CAM_FORMAT]
                if self._arg_pixelformat:
                    iargs += ["-pix_fmt", self._arg_pixelformat]
                if self._arg_size:
                    iargs += ["-s", self._arg_size]
            elif index > 0:  # re-initialize  / seek
                # Note: only works if we initialized earlier, and now have meta
                # Some info here: https://trac.ffmpeg.org/wiki/Seeking
                # There are two ways to seek, one before -i (input_params) and
                # after (output_params). The former is fast, because it uses
                # keyframes, the latter is slow but accurate. According to
                # the article above, the fast method should also be accurate
                # from ffmpeg version 2.1, however in version 4.1 our tests
                # start failing again. Not sure why, but we can solve this
                # by combining slow and fast. Seek the long stretch using
                # the fast method, and seek the last 10s the slow way.
                starttime = index / self._meta["fps"]
                seek_slow = min(10, starttime)
                seek_fast = starttime - seek_slow
                # We used to have this epsilon earlier, when we did not use
                # the slow seek. I don't think we need it anymore.
                # epsilon = -1 / self._meta["fps"] * 0.1
                iargs += ["-ss", "%.06f" % (seek_fast)]
                oargs += ["-ss", "%.06f" % (seek_slow)]

            # Output args, for writing to pipe
            if self._arg_size:
                oargs += ["-s", self._arg_size]
            if self.request.kwargs.get("fps", None):
                fps = float(self.request.kwargs["fps"])
                oargs += ["-r", "%.02f" % fps]
            oargs += self._arg_output_params

            # Get pixelformat and bytes per pixel
            pix_fmt = self._pix_fmt
            bpp = self._depth * self._bytes_per_channel

            # Create generator
            rf = self._ffmpeg_api.read_frames
            self._read_gen = rf(
                self._filename, pix_fmt, bpp, input_params=iargs, output_params=oargs
            )

            # Read meta data. This start the generator (and ffmpeg subprocess)
            if self.request._video:
                # With cameras, catch error and turn into IndexError
                try:
                    meta = self._read_gen.__next__()
                except IOError as err:
                    err_text = str(err)
                    if "darwin" in sys.platform:
                        if "Unknown input format: 'avfoundation'" in err_text:
                            err_text += (
                                "Try installing FFMPEG using "
                                "home brew to get a version with "
                                "support for cameras."
                            )
                    raise IndexError(
                        "No (working) camera at {}.\n\n{}".format(
                            self.request._video, err_text
                        )
                    )
                else:
                    self._meta.update(meta)
            elif index == 0:
                self._meta.update(self._read_gen.__next__())
            else:
                self._read_gen.__next__()  # we already have meta data

        def _skip_frames(self, n=1):
            """Reads and throws away n frames"""
            for i in range(n):
                self._read_gen.__next__()
            self._pos += n

        def _read_frame(self):
            # Read and convert to numpy array
            w, h = self._meta["size"]
            framesize = w * h * self._depth * self._bytes_per_channel
            # t0 = time.time()

            # Read frame
            if self._frame_catcher:  # pragma: no cover - camera thing
                s, is_new = self._frame_catcher.get_frame()
            else:
                s = self._read_gen.__next__()
                is_new = True

            # Check
            if len(s) != framesize:
                raise RuntimeError(
                    "Frame is %i bytes, but expected %i." % (len(s), framesize)
                )

            result = np.frombuffer(s, dtype=self._dtype).copy()
            result = result.reshape((h, w, self._depth))
            # t1 = time.time()
            # print('etime', t1-t0)

            # Store and return
            self._lastread = result
            return result, is_new

    # --

    class Writer(Format.Writer):

        _write_gen = None

        def _open(
            self,
            fps=10,
            codec="libx264",
            bitrate=None,
            pixelformat="yuv420p",
            ffmpeg_params=None,
            input_params=None,
            output_params=None,
            ffmpeg_log_level="quiet",
            quality=5,
            macro_block_size=16,
        ):
            self._ffmpeg_api = _get_ffmpeg_api()
            self._filename = self.request.get_local_filename()
            self._pix_fmt = None
            self._depth = None
            self._size = None

        def _close(self):
            if self._write_gen is not None:
                self._write_gen.close()
                self._write_gen = None

        def _append_data(self, im, meta):

            # Get props of image
            h, w = im.shape[:2]
            size = w, h
            depth = 1 if im.ndim == 2 else im.shape[2]

            # Ensure that image is in uint8
            im = image_as_uint(im, bitdepth=8)
            # To be written efficiently, ie. without creating an immutable
            # buffer, by calling im.tobytes() the array must be contiguous.
            if not im.flags.c_contiguous:
                # checkign the flag is a micro optimization.
                # the image will be a numpy subclass. See discussion
                # https://github.com/numpy/numpy/issues/11804
                im = np.ascontiguousarray(im)

            # Set size and initialize if not initialized yet
            if self._size is None:
                map = {1: "gray", 2: "gray8a", 3: "rgb24", 4: "rgba"}
                self._pix_fmt = map.get(depth, None)
                if self._pix_fmt is None:
                    raise ValueError("Image must have 1, 2, 3 or 4 channels")
                self._size = size
                self._depth = depth
                self._initialize()

            # Check size of image
            if size != self._size:
                raise ValueError("All images in a movie should have same size")
            if depth != self._depth:
                raise ValueError(
                    "All images in a movie should have same " "number of channels"
                )

            assert self._write_gen is not None  # Check status

            # Write. Yes, we can send the data in as a numpy array
            self._write_gen.send(im)

        def set_meta_data(self, meta):
            raise RuntimeError(
                "The ffmpeg format does not support setting " "meta data."
            )

        def _initialize(self):

            # Close existing generator
            if self._write_gen is not None:
                self._write_gen.close()

            # Get parameters
            # Use None to let imageio-ffmpeg (or ffmpeg) select good results
            fps = self.request.kwargs.get("fps", 10)
            codec = self.request.kwargs.get("codec", None)
            bitrate = self.request.kwargs.get("bitrate", None)
            quality = self.request.kwargs.get("quality", None)
            input_params = self.request.kwargs.get("input_params") or []
            output_params = self.request.kwargs.get("output_params") or []
            output_params += self.request.kwargs.get("ffmpeg_params") or []
            pixelformat = self.request.kwargs.get("pixelformat", None)
            macro_block_size = self.request.kwargs.get("macro_block_size", 16)
            ffmpeg_log_level = self.request.kwargs.get("ffmpeg_log_level", None)

            macro_block_size = macro_block_size or 1  # None -> 1

            # Create generator
            self._write_gen = self._ffmpeg_api.write_frames(
                self._filename,
                self._size,
                pix_fmt_in=self._pix_fmt,
                pix_fmt_out=pixelformat,
                fps=fps,
                quality=quality,
                bitrate=bitrate,
                codec=codec,
                macro_block_size=macro_block_size,
                ffmpeg_log_level=ffmpeg_log_level,
                input_params=input_params,
                output_params=output_params,
            )

            # Seed the generator (this is where the ffmpeg subprocess starts)
            self._write_gen.send(None)


class FrameCatcher(threading.Thread):
    """Thread to keep reading the frame data from stdout. This is
    useful when streaming from a webcam. Otherwise, if the user code
    does not grab frames fast enough, the buffer will fill up, leading
    to lag, and ffmpeg can also stall (experienced on Linux). The
    get_frame() method always returns the last available image.
    """

    def __init__(self, gen):
        self._gen = gen
        self._frame = None
        self._frame_is_new = False
        self._lock = threading.RLock()
        threading.Thread.__init__(self)
        self.setDaemon(True)  # do not let this thread hold up Python shutdown
        self._should_stop = False
        self.start()

    def stop_me(self):
        self._should_stop = True
        while self.is_alive():
            time.sleep(0.001)

    def get_frame(self):
        while self._frame is None:  # pragma: no cover - an init thing
            time.sleep(0.001)
        with self._lock:
            is_new = self._frame_is_new
            self._frame_is_new = False  # reset
            return self._frame, is_new

    def run(self):
        # This runs in the worker thread
        try:
            while not self._should_stop:
                time.sleep(0)  # give control to other threads
                frame = self._gen.__next__()
                with self._lock:
                    self._frame = frame
                    self._frame_is_new = True
        except (StopIteration, EOFError):
            pass


def parse_device_names(ffmpeg_output):
    """Parse the output of the ffmpeg -list-devices command"""
    # Collect device names - get [friendly_name, alt_name] of each
    device_names = []
    in_video_devices = False
    for line in ffmpeg_output.splitlines():
        if line.startswith("[dshow"):
            logger.debug(line)
            line = line.split("]", 1)[1].strip()
            if in_video_devices and line.startswith('"'):
                friendly_name = line[1:-1]
                device_names.append([friendly_name, ""])
            elif in_video_devices and line.lower().startswith("alternative name"):
                alt_name = line.split(" name ", 1)[1].strip()[1:-1]
                if sys.platform.startswith("win"):
                    alt_name = alt_name.replace("&", "^&")  # Tested to work
                else:
                    alt_name = alt_name.replace("&", "\\&")  # Does this work?
                device_names[-1][-1] = alt_name
            elif "video devices" in line:
                in_video_devices = True
            elif "devices" in line:
                # set False for subsequent "devices" sections
                in_video_devices = False
    # Post-process, see #441
    # prefer friendly names, use alt name if two cams have same friendly name
    device_names2 = []
    for friendly_name, alt_name in device_names:
        if friendly_name not in device_names2:
            device_names2.append(friendly_name)
        elif alt_name:
            device_names2.append(alt_name)
        else:
            device_names2.append(friendly_name)  # duplicate, but not much we can do
    return device_names2
