"""
    pdf2image is a light wrapper for the poppler-utils tools that can convert your
    PDFs into Pillow images.
"""

import os
import platform
import tempfile
import types
import shutil
import pathlib
import subprocess
from subprocess import Popen, PIPE, TimeoutExpired
from PIL import Image

from .generators import uuid_generator, counter_generator, ThreadSafeGenerator

from .parsers import (
    parse_buffer_to_pgm,
    parse_buffer_to_ppm,
    parse_buffer_to_jpeg,
    parse_buffer_to_png,
)

from .exceptions import (
    PopplerNotInstalledError,
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
    PDFPopplerTimeoutError,
)

TRANSPARENT_FILE_TYPES = ["png", "tiff"]
PDFINFO_CONVERT_TO_INT = ["Pages"]


def convert_from_path(
    pdf_path,
    dpi=200,
    output_folder=None,
    first_page=None,
    last_page=None,
    fmt="ppm",
    jpegopt=None,
    thread_count=1,
    userpw=None,
    use_cropbox=False,
    strict=False,
    transparent=False,
    single_file=False,
    output_file=uuid_generator(),
    poppler_path=None,
    grayscale=False,
    size=None,
    paths_only=False,
    use_pdftocairo=False,
    timeout=None,
    hide_annotations=False,
):
    """
        Description: Convert PDF to Image will throw whenever one of the condition is reached
        Parameters:
            pdf_path -> Path to the PDF that you want to convert
            dpi -> Image quality in DPI (default 200)
            output_folder -> Write the resulting images to a folder (instead of directly in memory)
            first_page -> First page to process
            last_page -> Last page to process before stopping
            fmt -> Output image format
            jpegopt -> jpeg options `quality`, `progressive`, and `optimize` (only for jpeg format)
            thread_count -> How many threads we are allowed to spawn for processing
            userpw -> PDF's password
            use_cropbox -> Use cropbox instead of mediabox
            strict -> When a Syntax Error is thrown, it will be raised as an Exception
            transparent -> Output with a transparent background instead of a white one.
            single_file -> Uses the -singlefile option from pdftoppm/pdftocairo
            output_file -> What is the output filename or generator
            poppler_path -> Path to look for poppler binaries
            grayscale -> Output grayscale image(s)
            size -> Size of the resulting image(s), uses the Pillow (width, height) standard
            paths_only -> Don't load image(s), return paths instead (requires output_folder)
            use_pdftocairo -> Use pdftocairo instead of pdftoppm, may help performance
            timeout -> Raise PDFPopplerTimeoutError after the given time
    """

    if use_pdftocairo and fmt == "ppm":
        fmt = "png"

    # We make sure that if passed arguments are Path objects, they're converted to strings
    if isinstance(pdf_path, pathlib.PurePath):
        pdf_path = pdf_path.as_posix()

    if isinstance(output_folder, pathlib.PurePath):
        output_folder = output_folder.as_posix()

    if isinstance(poppler_path, pathlib.PurePath):
        poppler_path = poppler_path.as_posix()

    page_count = pdfinfo_from_path(pdf_path, userpw, poppler_path=poppler_path)["Pages"]

    # We start by getting the output format, the buffer processing function and if we need pdftocairo
    parsed_fmt, final_extension, parse_buffer_func, use_pdfcairo_format = _parse_format(
        fmt, grayscale
    )

    # We use pdftocairo is the format requires it OR we need a transparent output
    use_pdfcairo = (
        use_pdftocairo
        or use_pdfcairo_format
        or (transparent and parsed_fmt in TRANSPARENT_FILE_TYPES)
    )

    poppler_version_major, poppler_version_minor = _get_poppler_version(
        "pdftocairo" if use_pdfcairo else "pdftoppm", poppler_path=poppler_path
    )

    if poppler_version_major == 0 and poppler_version_minor <= 57:
        jpegopt = None

    if poppler_version_major == 0 and poppler_version_minor <= 83:
        hide_annotations = False

    # If output_file isn't a generator, it will be turned into one
    if not isinstance(output_file, types.GeneratorType) and not isinstance(
        output_file, ThreadSafeGenerator
    ):
        if single_file:
            output_file = iter([output_file])
        else:
            output_file = counter_generator(output_file)

    if thread_count < 1:
        thread_count = 1

    if first_page is None or first_page < 1:
        first_page = 1

    if last_page is None or last_page > page_count:
        last_page = page_count

    if first_page > last_page:
        return []

    auto_temp_dir = False
    if output_folder is None and use_pdfcairo:
        auto_temp_dir = True
        output_folder = tempfile.mkdtemp()

    # Recalculate page count based on first and last page
    page_count = last_page - first_page + 1

    if thread_count > page_count:
        thread_count = page_count

    reminder = page_count % thread_count
    current_page = first_page
    processes = []
    for _ in range(thread_count):
        thread_output_file = next(output_file)

        # Get the number of pages the thread will be processing
        thread_page_count = page_count // thread_count + int(reminder > 0)
        # Build the command accordingly
        args = _build_command(
            ["-r", str(dpi), pdf_path],
            output_folder,
            current_page,
            current_page + thread_page_count - 1,
            parsed_fmt,
            jpegopt,
            thread_output_file,
            userpw,
            use_cropbox,
            transparent,
            single_file,
            grayscale,
            size,
            hide_annotations,
        )

        if use_pdfcairo:
            if hide_annotations:
                raise NotImplementedError("Hide annotations flag not implemented in pdftocairo.")
            args = [_get_command_path("pdftocairo", poppler_path)] + args
        else:
            args = [_get_command_path("pdftoppm", poppler_path)] + args

        # Update page values
        current_page = current_page + thread_page_count
        reminder -= int(reminder > 0)
        # Add poppler path to LD_LIBRARY_PATH
        env = os.environ.copy()
        if poppler_path is not None:
            env["LD_LIBRARY_PATH"] = poppler_path + ":" + env.get("LD_LIBRARY_PATH", "")
        # Spawn the process and save its uuid
        startupinfo=None
        if platform.system() == 'Windows':
            # this startupinfo structure prevents a console window from popping up on Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        processes.append(
            (thread_output_file, Popen(args, env=env, stdout=PIPE, stderr=PIPE, startupinfo=startupinfo))
        )

    images = []

    for uid, proc in processes:
        try:
            data, err = proc.communicate(timeout=timeout)
        except TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            raise PDFPopplerTimeoutError("Run poppler poppler timeout.")

        if b"Syntax Error" in err and strict:
            raise PDFSyntaxError(err.decode("utf8", "ignore"))

        if output_folder is not None:
            images += _load_from_output_folder(
                output_folder, uid, final_extension, paths_only, in_memory=auto_temp_dir
            )
        else:
            images += parse_buffer_func(data)

    if auto_temp_dir:
        shutil.rmtree(output_folder)

    return images


def convert_from_bytes(
    pdf_file,
    dpi=200,
    output_folder=None,
    first_page=None,
    last_page=None,
    fmt="ppm",
    jpegopt=None,
    thread_count=1,
    userpw=None,
    use_cropbox=False,
    strict=False,
    transparent=False,
    single_file=False,
    output_file=uuid_generator(),
    poppler_path=None,
    grayscale=False,
    size=None,
    paths_only=False,
    use_pdftocairo=False,
    timeout=None,
    hide_annotations=False,
):
    """
        Description: Convert PDF to Image will throw whenever one of the condition is reached
        Parameters:
            pdf_file -> Bytes representing the PDF file
            dpi -> Image quality in DPI
            output_folder -> Write the resulting images to a folder (instead of directly in memory)
            first_page -> First page to process
            last_page -> Last page to process before stopping
            fmt -> Output image format
            jpegopt -> jpeg options `quality`, `progressive`, and `optimize` (only for jpeg format)
            thread_count -> How many threads we are allowed to spawn for processing
            userpw -> PDF's password
            use_cropbox -> Use cropbox instead of mediabox
            strict -> When a Syntax Error is thrown, it will be raised as an Exception
            transparent -> Output with a transparent background instead of a white one.
            single_file -> Uses the -singlefile option from pdftoppm/pdftocairo
            output_file -> What is the output filename or generator
            poppler_path -> Path to look for poppler binaries
            grayscale -> Output grayscale image(s)
            size -> Size of the resulting image(s), uses the Pillow (width, height) standard
            paths_only -> Don't load image(s), return paths instead (requires output_folder)
            use_pdftocairo -> Use pdftocairo instead of pdftoppm, may help performance
            timeout -> Raise PDFPopplerTimeoutError after the given time
    """

    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(pdf_file)
            f.flush()
            return convert_from_path(
                f.name,
                dpi=dpi,
                output_folder=output_folder,
                first_page=first_page,
                last_page=last_page,
                fmt=fmt,
                jpegopt=jpegopt,
                thread_count=thread_count,
                userpw=userpw,
                use_cropbox=use_cropbox,
                strict=strict,
                transparent=transparent,
                single_file=single_file,
                output_file=output_file,
                poppler_path=poppler_path,
                grayscale=grayscale,
                size=size,
                paths_only=paths_only,
                use_pdftocairo=use_pdftocairo,
                timeout=timeout,
                hide_annotations=hide_annotations,
            )
    finally:
        os.close(fh)
        os.remove(temp_filename)


def _build_command(
    args,
    output_folder,
    first_page,
    last_page,
    fmt,
    jpegopt,
    output_file,
    userpw,
    use_cropbox,
    transparent,
    single_file,
    grayscale,
    size,
    hide_annotations,
):
    if use_cropbox:
        args.append("-cropbox")

    if hide_annotations:
        args.append("-hide-annotations")

    if transparent and fmt in TRANSPARENT_FILE_TYPES:
        args.append("-transp")

    if first_page is not None:
        args.extend(["-f", str(first_page)])

    if last_page is not None:
        args.extend(["-l", str(last_page)])

    if fmt not in ["pgm", "ppm"]:
        args.append("-" + fmt)

    if fmt in ["jpeg", "jpg"] and jpegopt:
        args.extend(["-jpegopt", _parse_jpegopt(jpegopt)])

    if single_file:
        args.append("-singlefile")

    if output_folder is not None:
        args.append(os.path.join(output_folder, output_file))

    if userpw is not None:
        args.extend(["-upw", userpw])

    if grayscale:
        args.append("-gray")

    if size is None:
        pass
    elif isinstance(size, tuple) and len(size) == 2:
        if size[0] is not None:
            args.extend(["-scale-to-x", str(int(size[0]))])
        else:
            args.extend(["-scale-to-x", str(-1)])
        if size[1] is not None:
            args.extend(["-scale-to-y", str(int(size[1]))])
        else:
            args.extend(["-scale-to-y", str(-1)])
    elif isinstance(size, tuple) and len(size) == 1:
        args.extend(["-scale-to", str(int(size[0]))])
    elif isinstance(size, int) or isinstance(size, float):
        args.extend(["-scale-to", str(int(size))])
    else:
        raise ValueError("Size {} is not a tuple or an integer")

    return args


def _parse_format(fmt, grayscale=False):
    fmt = fmt.lower()
    if fmt[0] == ".":
        fmt = fmt[1:]
    if fmt in ("jpeg", "jpg"):
        return "jpeg", "jpg", parse_buffer_to_jpeg, False
    if fmt == "png":
        return "png", "png", parse_buffer_to_png, False
    if fmt in ("tif", "tiff"):
        return "tiff", "tif", None, True
    if fmt == "ppm" and grayscale:
        return "pgm", "pgm", parse_buffer_to_pgm, False
    # Unable to parse the format so we'll use the default
    return "ppm", "ppm", parse_buffer_to_ppm, False


def _parse_jpegopt(jpegopt):
    parts = []
    for k, v in jpegopt.items():
        if v is True:
            v = "y"
        if v is False:
            v = "n"
        parts.append("{}={}".format(k, v))
    return ",".join(parts)


def _get_command_path(command, poppler_path=None):
    if platform.system() == "Windows":
        command = command + ".exe"

    if poppler_path is not None:
        command = os.path.join(poppler_path, command)

    return command


def _get_poppler_version(command, poppler_path=None, timeout=None):
    command = [_get_command_path(command, poppler_path), "-v"]

    env = os.environ.copy()
    if poppler_path is not None:
        env["LD_LIBRARY_PATH"] = poppler_path + ":" + env.get("LD_LIBRARY_PATH", "")
    proc = Popen(command, env=env, stdout=PIPE, stderr=PIPE)

    try:
        data, err = proc.communicate(timeout=timeout)
    except TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        raise PDFPopplerTimeoutError("Run poppler poppler timeout.")

    try:
        # TODO: Make this more robust
        version = err.decode("utf8", "ignore").split("\n")[0].split(" ")[-1].split(".")
        return int(version[0]), int(version[1])
    except:
        # Lowest version that includes pdftocairo (2011)
        return 0, 17


def pdfinfo_from_path(
    pdf_path, userpw=None, poppler_path=None, rawdates=False, timeout=None
):
    try:
        command = [_get_command_path("pdfinfo", poppler_path), pdf_path]

        if userpw is not None:
            command.extend(["-upw", userpw])

        if rawdates:
            command.extend(["-rawdates"])

        # Add poppler path to LD_LIBRARY_PATH
        env = os.environ.copy()
        if poppler_path is not None:
            env["LD_LIBRARY_PATH"] = poppler_path + ":" + env.get("LD_LIBRARY_PATH", "")
        proc = Popen(command, env=env, stdout=PIPE, stderr=PIPE)

        try:
            out, err = proc.communicate(timeout=timeout)
        except TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            raise PDFPopplerTimeoutError("Run poppler poppler timeout.")

        d = {}
        for field in out.decode("utf8", "ignore").split("\n"):
            sf = field.split(":")
            key, value = sf[0], ":".join(sf[1:])
            if key != "":
                d[key] = (
                    int(value.strip())
                    if key in PDFINFO_CONVERT_TO_INT
                    else value.strip()
                )

        if "Pages" not in d:
            raise ValueError

        return d

    except OSError:
        raise PDFInfoNotInstalledError(
            "Unable to get page count. Is poppler installed and in PATH?"
        )
    except ValueError:
        raise PDFPageCountError(
            "Unable to get page count.\n%s" % err.decode("utf8", "ignore")
        )


def pdfinfo_from_bytes(
    pdf_file, userpw=None, poppler_path=None, rawdates=False, timeout=None
):
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(pdf_file)
            f.flush()
        return pdfinfo_from_path(temp_filename, userpw=userpw, rawdates=rawdates,
                                 poppler_path=poppler_path)
    finally:
        os.close(fh)
        os.remove(temp_filename)


def _load_from_output_folder(
    output_folder, output_file, ext, paths_only, in_memory=False
):
    images = []
    for f in sorted(os.listdir(output_folder)):
        if f.startswith(output_file) and f.split(".")[-1] == ext:
            if paths_only:
                images.append(os.path.join(output_folder, f))
            else:
                images.append(Image.open(os.path.join(output_folder, f)))
                if in_memory:
                    images[-1].load()
    return images
