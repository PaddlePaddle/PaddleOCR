from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Iterator


def response_chunks(response, chunk_size=CONTENT_CHUNK_SIZE):
    # type: (Response, int) -> Iterator[bytes]
    """Given a requests Response, provide the data chunks.
    """
    try:
        # Special case for urllib3.
        for chunk in response.raw.stream(
            chunk_size,
            # We use decode_content=False here because we don't
            # want urllib3 to mess with the raw bytes we get
            # from the server. If we decompress inside of
            # urllib3 then we cannot verify the checksum
            # because the checksum will be of the compressed
            # file. This breakage will only occur if the
            # server adds a Content-Encoding header, which
            # depends on how the server was configured:
            # - Some servers will notice that the file isn't a
            #   compressible file and will leave the file alone
            #   and with an empty Content-Encoding
            # - Some servers will notice that the file is
            #   already compressed and will leave the file
            #   alone and will add a Content-Encoding: gzip
            #   header
            # - Some servers won't notice anything at all and
            #   will take a file that's already been compressed
            #   and compress it again and set the
            #   Content-Encoding: gzip header
            #
            # By setting this not to decode automatically we
            # hope to eliminate problems with the second case.
            decode_content=False,
        ):
            yield chunk
    except AttributeError:
        # Standard file-like object.
        while True:
            chunk = response.raw.read(chunk_size)
            if not chunk:
                break
            yield chunk
