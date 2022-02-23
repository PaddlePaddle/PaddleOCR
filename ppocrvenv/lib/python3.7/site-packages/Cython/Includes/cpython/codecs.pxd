cdef extern from "Python.h":

    ###########################################################################
    # Codec registry and support functions
    ###########################################################################

    int PyCodec_Register(object search_function)
    # Register a new codec search function.

    # As side effect, this tries to load the encodings package, if not yet
    # done, to make sure that it is always first in the list of search
    # functions.

    int PyCodec_KnownEncoding(const char *encoding)
    # Return 1 or 0 depending on whether there is a registered codec for the
    # given encoding. This function always succeeds.

    object PyCodec_Encode(object o, const char *encoding, const char *errors)
    # Return value: New reference.
    # Generic codec based encoding API.

    # o is passed through the encoder function found for the given encoding
    # using the error handling method defined by errors. errors may be NULL
    # to use the default method defined for the codec. Raises a LookupError
    # if no encoder can be found.

    object PyCodec_Decode(object o, const char *encoding, const char *errors)
    # Return value: New reference.
    # Generic codec based decoding API.

    # o is passed through the decoder function found for the given encoding
    # using the error handling method defined by errors. errors may be NULL
    # to use the default method defined for the codec. Raises a LookupError
    # if no encoder can be found.


    # Codec lookup API

    # In the following functions, the encoding string is looked up converted
    # to all lower-case characters, which makes encodings looked up through
    # this mechanism effectively case-insensitive. If no codec is found, a
    # KeyError is set and NULL returned.

    object PyCodec_Encoder(const char *encoding)
    # Return value: New reference.
    # Get an encoder function for the given encoding.

    object PyCodec_Decoder(const char *encoding)
    # Return value: New reference.
    # Get a decoder function for the given encoding.

    object PyCodec_IncrementalEncoder(const char *encoding, const char *errors)
    # Return value: New reference.
    # Get an IncrementalEncoder object for the given encoding.

    object PyCodec_IncrementalDecoder(const char *encoding, const char *errors)
    # Return value: New reference.
    # Get an IncrementalDecoder object for the given encoding.

    object PyCodec_StreamReader(const char *encoding, object stream, const char *errors)
    # Return value: New reference.
    # Get a StreamReader factory function for the given encoding.

    object PyCodec_StreamWriter(const char *encoding, object stream, const char *errors)
    # Return value: New reference.
    # Get a StreamWriter factory function for the given encoding.


    # Registry API for Unicode encoding error handlers

    int PyCodec_RegisterError(const char *name, object error) except? -1
    # Register the error handling callback function error under the given
    # name. This callback function will be called by a codec when it
    # encounters unencodable characters/undecodable bytes and name is
    # specified as the error parameter in the call to the encode/decode
    # function.

    # The callback gets a single argument, an instance of
    # UnicodeEncodeError, UnicodeDecodeError or UnicodeTranslateError that
    # holds information about the problematic sequence of characters or bytes
    # and their offset in the original string (see Unicode Exception Objects
    # for functions to extract this information). The callback must either
    # raise the given exception, or return a two-item tuple containing the
    # replacement for the problematic sequence, and an integer giving the
    # offset in the original string at which encoding/decoding should be
    # resumed.

    # Return 0 on success, -1 on error.

    object PyCodec_LookupError(const char *name)
    # Return value: New reference.
    # Lookup the error handling callback function registered under name. As a
    # special case NULL can be passed, in which case the error handling
    # callback for "strict" will be returned.

    object PyCodec_StrictErrors(object exc)
    # Return value: Always NULL.
    # Raise exc as an exception.

    object PyCodec_IgnoreErrors(object exc)
    # Return value: New reference.
    # Ignore the unicode error, skipping the faulty input.

    object PyCodec_ReplaceErrors(object exc)
    # Return value: New reference.
    # Replace the unicode encode error with "?" or "U+FFFD".

    object PyCodec_XMLCharRefReplaceErrors(object exc)
    # Return value: New reference.
    # Replace the unicode encode error with XML character references.

    object PyCodec_BackslashReplaceErrors(object exc)
    # Return value: New reference.
    # Replace the unicode encode error with backslash escapes ("\x", "\u"
    # and "\U").

    object PyCodec_NameReplaceErrors(object exc)
    # Return value: New reference.
    # Replace the unicode encode error with "\N{...}" escapes.

    # New in version 3.5.
