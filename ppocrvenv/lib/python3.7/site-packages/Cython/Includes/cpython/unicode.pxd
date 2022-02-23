cdef extern from *:
    # Return true if the object o is a Unicode object or an instance
    # of a Unicode subtype. Changed in version 2.2: Allowed subtypes
    # to be accepted.
    bint PyUnicode_Check(object o)

    # Return true if the object o is a Unicode object, but not an
    # instance of a subtype. New in version 2.2.
    bint PyUnicode_CheckExact(object o)

    # Return the size of the object. o has to be a PyUnicodeObject
    # (not checked).
    #
    # Deprecated since version 3.3, will be removed in version 3.10:
    # Part of the old-style Unicode API, please migrate to using
    # PyUnicode_GET_LENGTH().
    Py_ssize_t PyUnicode_GET_SIZE(object o)

    # Return the length of the Unicode string, in code points. o has
    # to be a Unicode object in the “canonical” representation (not
    # checked).
    #
    # New in version 3.3.
    Py_ssize_t PyUnicode_GET_LENGTH(object o)

    # Return the size of the object's internal buffer in bytes. o has
    # to be a PyUnicodeObject (not checked).
    Py_ssize_t PyUnicode_GET_DATA_SIZE(object o)

    # Return a pointer to the internal Py_UNICODE buffer of the
    # object. o has to be a PyUnicodeObject (not checked).
    Py_UNICODE* PyUnicode_AS_UNICODE(object o)

    # Return a pointer to the internal buffer of the object. o has to
    # be a PyUnicodeObject (not checked).
    char* PyUnicode_AS_DATA(object o)

    # Return 1 or 0 depending on whether ch is a whitespace character.
    bint Py_UNICODE_ISSPACE(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is a lowercase character.
    bint Py_UNICODE_ISLOWER(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is an uppercase character.
    bint Py_UNICODE_ISUPPER(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is a titlecase character.
    bint Py_UNICODE_ISTITLE(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is a linebreak character.
    bint Py_UNICODE_ISLINEBREAK(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is a decimal character.
    bint Py_UNICODE_ISDECIMAL(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is a digit character.
    bint Py_UNICODE_ISDIGIT(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is a numeric character.
    bint Py_UNICODE_ISNUMERIC(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is an alphabetic character.
    bint Py_UNICODE_ISALPHA(Py_UCS4 ch)

    # Return 1 or 0 depending on whether ch is an alphanumeric character.
    bint Py_UNICODE_ISALNUM(Py_UCS4 ch)

    # Return the character ch converted to lower case.
    # Used to return a Py_UNICODE value before Py3.3.
    Py_UCS4 Py_UNICODE_TOLOWER(Py_UCS4 ch)

    # Return the character ch converted to upper case.
    # Used to return a Py_UNICODE value before Py3.3.
    Py_UCS4 Py_UNICODE_TOUPPER(Py_UCS4 ch)

    # Return the character ch converted to title case.
    # Used to return a Py_UNICODE value before Py3.3.
    Py_UCS4 Py_UNICODE_TOTITLE(Py_UCS4 ch)

    # Return the character ch converted to a decimal positive
    # integer. Return -1 if this is not possible. This macro does not
    # raise exceptions.
    int Py_UNICODE_TODECIMAL(Py_UCS4 ch)

    # Return the character ch converted to a single digit
    # integer. Return -1 if this is not possible. This macro does not
    # raise exceptions.
    int Py_UNICODE_TODIGIT(Py_UCS4 ch)

    # Return the character ch converted to a double. Return -1.0 if
    # this is not possible. This macro does not raise exceptions.
    double Py_UNICODE_TONUMERIC(Py_UCS4 ch)

    # To create Unicode objects and access their basic sequence
    # properties, use these APIs:

    # Create a Unicode Object from the Py_UNICODE buffer u of the
    # given size. u may be NULL which causes the contents to be
    # undefined. It is the user's responsibility to fill in the needed
    # data. The buffer is copied into the new object. If the buffer is
    # not NULL, the return value might be a shared object. Therefore,
    # modification of the resulting Unicode object is only allowed
    # when u is NULL.
    unicode PyUnicode_FromUnicode(Py_UNICODE *u, Py_ssize_t size)

    # Create a Unicode Object from the given Unicode code point ordinal.
    #
    # The ordinal must be in range(0x10000) on narrow Python builds
    # (UCS2), and range(0x110000) on wide builds (UCS4). A ValueError
    # is raised in case it is not.
    unicode PyUnicode_FromOrdinal(int ordinal)

    # Return a read-only pointer to the Unicode object's internal
    # Py_UNICODE buffer, NULL if unicode is not a Unicode object.
    Py_UNICODE* PyUnicode_AsUnicode(object o) except NULL

    # Return the length of the Unicode object.
    Py_ssize_t PyUnicode_GetSize(object o) except -1

    # Coerce an encoded object obj to an Unicode object and return a
    # reference with incremented refcount.
    # String and other char buffer compatible objects are decoded
    # according to the given encoding and using the error handling
    # defined by errors. Both can be NULL to have the interface use
    # the default values (see the next section for details).
    # All other objects, including Unicode objects, cause a TypeError
    # to be set.
    object PyUnicode_FromEncodedObject(object o, char *encoding, char *errors)

    # Shortcut for PyUnicode_FromEncodedObject(obj, NULL, "strict")
    # which is used throughout the interpreter whenever coercion to
    # Unicode is needed.
    object PyUnicode_FromObject(object obj)

    # If the platform supports wchar_t and provides a header file
    # wchar.h, Python can interface directly to this type using the
    # following functions. Support is optimized if Python's own
    # Py_UNICODE type is identical to the system's wchar_t.

    #ctypedef int wchar_t

    # Create a Unicode object from the wchar_t buffer w of the given
    # size. Return NULL on failure.
    #PyObject* PyUnicode_FromWideChar(wchar_t *w, Py_ssize_t size)

    #Py_ssize_t PyUnicode_AsWideChar(object o, wchar_t *w, Py_ssize_t size)


# Unicode Methods

    # Concat two strings giving a new Unicode string.
    # Return value: New reference.
    unicode PyUnicode_Concat(object left, object right)

    # Split a string giving a list of Unicode strings. If sep is NULL,
    # splitting will be done at all whitespace substrings. Otherwise,
    # splits occur at the given separator. At most maxsplit splits will
    # be done. If negative, no limit is set. Separators are not included
    # in the resulting list.
    # Return value: New reference.
    list PyUnicode_Split(object s, object sep, Py_ssize_t maxsplit)

    # Split a Unicode string at line breaks, returning a list of Unicode
    # strings. CRLF is considered to be one line break. If keepend is 0,
    # the Line break characters are not included in the resulting strings.
    # Return value: New reference.
    list PyUnicode_Splitlines(object s, bint keepend)

    # Translate a string by applying a character mapping table to it and
    # return the resulting Unicode object.
    #
    # The mapping table must map Unicode ordinal integers to Unicode ordinal
    # integers or None (causing deletion of the character).
    #
    # Mapping tables need only provide the __getitem__() interface;
    # dictionaries and sequences work well. Unmapped character ordinals (ones
    # which cause a LookupError) are left untouched and are copied as-is.
    #
    # errors has the usual meaning for codecs. It may be NULL which indicates
    # to use the default error handling.
    # Return value: New reference.
    unicode PyUnicode_Translate(object str, object table, const char *errors)

    # Join a sequence of strings using the given separator and return the
    # resulting Unicode string.
    # Return value: New reference.
    unicode PyUnicode_Join(object separator, object seq)

    # Return 1 if substr matches str[start:end] at the given tail end
    # (direction == -1 means to do a prefix match, direction == 1 a
    # suffix match), 0 otherwise.
    # Return -1 if an error occurred.
    Py_ssize_t PyUnicode_Tailmatch(object str, object substr,
                                   Py_ssize_t start, Py_ssize_t end, int direction) except -1

    # Return the first position of substr in str[start:end] using the given
    # direction (direction == 1 means to do a forward search, direction == -1
    # a backward search). The return value is the index of the first match;
    # a value of -1 indicates that no match was found, and -2 indicates that an
    # error occurred and an exception has been set.
    Py_ssize_t PyUnicode_Find(object str, object substr, Py_ssize_t start, Py_ssize_t end, int direction) except -2

    # Return the first position of the character ch in str[start:end] using
    # the given direction (direction == 1 means to do a forward search,
    # direction == -1 a backward search). The return value is the index of
    # the first match; a value of -1 indicates that no match was found, and
    # -2 indicates that an error occurred and an exception has been set.
    # New in version 3.3.
    Py_ssize_t PyUnicode_FindChar(object str, Py_UCS4 ch, Py_ssize_t start, Py_ssize_t end, int direction) except -2

    # Return the number of non-overlapping occurrences of substr in
    # str[start:end]. Return -1 if an error occurred.
    Py_ssize_t PyUnicode_Count(object str, object substr, Py_ssize_t start, Py_ssize_t end) except -1

    # Replace at most maxcount occurrences of substr in str with replstr and
    # return the resulting Unicode object. maxcount == -1 means replace all
    # occurrences.
    # Return value: New reference.
    unicode PyUnicode_Replace(object str, object substr, object replstr, Py_ssize_t maxcount)

    # Compare two strings and return -1, 0, 1 for less than,
    # equal, and greater than, respectively.
    int PyUnicode_Compare(object left, object right) except? -1

    # Compare a unicode object, uni, with string and return -1, 0, 1 for less than,
    # equal, and greater than, respectively. It is best to pass only ASCII-encoded
    # strings, but the function interprets the input string as ISO-8859-1 if it
    # contains non-ASCII characters.
    int PyUnicode_CompareWithASCIIString(object uni, const char *string)

    # Rich compare two unicode strings and return one of the following:
    #
    #    NULL in case an exception was raised
    #    Py_True or Py_False for successful comparisons
    #    Py_NotImplemented in case the type combination is unknown
    #
    # Note that Py_EQ and Py_NE comparisons can cause a UnicodeWarning in case
    # the conversion of the arguments to Unicode fails with a UnicodeDecodeError.
    #
    # Possible values for op are Py_GT, Py_GE, Py_EQ, Py_NE, Py_LT, and Py_LE.
    object PyUnicode_RichCompare(object left, object right, int op)

    # Return a new string object from format and args; this is analogous to
    # format % args.
    # Return value: New reference.
    unicode PyUnicode_Format(object format, object args)

    # Check whether element is contained in container and return true or false
    # accordingly.
    #
    # element has to coerce to a one element Unicode string. -1 is returned
    # if there was an error.
    int PyUnicode_Contains(object container, object element) except -1

    # Intern the argument *string in place. The argument must be the address
    # of a pointer variable pointing to a Python unicode string object. If
    # there is an existing interned string that is the same as *string, it sets
    # *string to it (decrementing the reference count of the old string object
    # and incrementing the reference count of the interned string object),
    # otherwise it leaves *string alone and interns it (incrementing its reference
    # count). (Clarification: even though there is a lot of talk about reference
    # counts, think of this function as reference-count-neutral; you own the object
    # after the call if and only if you owned it before the call.)
    #void PyUnicode_InternInPlace(PyObject **string)

    # A combination of PyUnicode_FromString() and PyUnicode_InternInPlace(),
    # returning either a new unicode string object that has been interned, or
    # a new ("owned") reference to an earlier interned string object with the
    # same value.
    unicode PyUnicode_InternFromString(const char *v)


# Codecs

    # Create a Unicode object by decoding size bytes of the encoded
    # string s. encoding and errors have the same meaning as the
    # parameters of the same name in the unicode() builtin
    # function. The codec to be used is looked up using the Python
    # codec registry. Return NULL if an exception was raised by the
    # codec.
    object PyUnicode_Decode(char *s, Py_ssize_t size, char *encoding, char *errors)

    # Encode the Py_UNICODE buffer of the given size and return a
    # Python string object. encoding and errors have the same meaning
    # as the parameters of the same name in the Unicode encode()
    # method. The codec to be used is looked up using the Python codec
    # registry. Return NULL if an exception was raised by the codec.
    object PyUnicode_Encode(Py_UNICODE *s, Py_ssize_t size,
                            char *encoding, char *errors)

    # Encode a Unicode object and return the result as Python string
    # object. encoding and errors have the same meaning as the
    # parameters of the same name in the Unicode encode() method. The
    # codec to be used is looked up using the Python codec
    # registry. Return NULL if an exception was raised by the codec.
    object PyUnicode_AsEncodedString(object unicode, char *encoding, char *errors)

# These are the UTF-8 codec APIs:

    # Create a Unicode object by decoding size bytes of the UTF-8
    # encoded string s. Return NULL if an exception was raised by the
    # codec.
    unicode PyUnicode_DecodeUTF8(char *s, Py_ssize_t size, char *errors)

    # If consumed is NULL, behave like PyUnicode_DecodeUTF8(). If
    # consumed is not NULL, trailing incomplete UTF-8 byte sequences
    # will not be treated as an error. Those bytes will not be decoded
    # and the number of bytes that have been decoded will be stored in
    # consumed. New in version 2.4.
    unicode PyUnicode_DecodeUTF8Stateful(char *s, Py_ssize_t size, char *errors, Py_ssize_t *consumed)

    # Encode the Py_UNICODE buffer of the given size using UTF-8 and
    # return a Python string object. Return NULL if an exception was
    # raised by the codec.
    bytes PyUnicode_EncodeUTF8(Py_UNICODE *s, Py_ssize_t size, char *errors)

    # Encode a Unicode objects using UTF-8 and return the result as Python string object. Error handling is ``strict''. Return NULL if an exception was raised by the codec.
    bytes PyUnicode_AsUTF8String(object unicode)

# These are the UTF-16 codec APIs:

    # Decode length bytes from a UTF-16 encoded buffer string and
    # return the corresponding Unicode object. errors (if non-NULL)
    # defines the error handling. It defaults to ``strict''.
    #
    # If byteorder is non-NULL, the decoder starts decoding using the
    # given byte order:
    #
    #   *byteorder == -1: little endian
    #   *byteorder == 0:  native order
    #   *byteorder == 1:  big endian
    #
    # and then switches if the first two bytes of the input data are a
    # byte order mark (BOM) and the specified byte order is native
    # order. This BOM is not copied into the resulting Unicode
    # string. After completion, *byteorder is set to the current byte
    # order at the.
    #
    # If byteorder is NULL, the codec starts in native order mode.
    unicode PyUnicode_DecodeUTF16(char *s, Py_ssize_t size, char *errors, int *byteorder)

    # If consumed is NULL, behave like PyUnicode_DecodeUTF16(). If
    # consumed is not NULL, PyUnicode_DecodeUTF16Stateful() will not
    # treat trailing incomplete UTF-16 byte sequences (such as an odd
    # number of bytes or a split surrogate pair) as an error. Those
    # bytes will not be decoded and the number of bytes that have been
    # decoded will be stored in consumed. New in version 2.4.
    unicode PyUnicode_DecodeUTF16Stateful(char *s, Py_ssize_t size, char *errors, int *byteorder, Py_ssize_t *consumed)

    # Return a Python string object holding the UTF-16 encoded value
    # of the Unicode data in s. If byteorder is not 0, output is
    # written according to the following byte order:
    #
    #   byteorder == -1: little endian
    #   byteorder == 0:  native byte order (writes a BOM mark)
    #   byteorder == 1:  big endian
    #
    # If byteorder is 0, the output string will always start with the
    # Unicode BOM mark (U+FEFF). In the other two modes, no BOM mark
    # is prepended.
    #
    # If Py_UNICODE_WIDE is defined, a single Py_UNICODE value may get
    # represented as a surrogate pair. If it is not defined, each
    # Py_UNICODE values is interpreted as an UCS-2 character.
    bytes PyUnicode_EncodeUTF16(Py_UNICODE *s, Py_ssize_t size, char *errors, int byteorder)

    # Return a Python string using the UTF-16 encoding in native byte
    # order. The string always starts with a BOM mark. Error handling
    # is ``strict''. Return NULL if an exception was raised by the
    # codec.
    bytes PyUnicode_AsUTF16String(object unicode)

# These are the ``Unicode Escape'' codec APIs:

    # Create a Unicode object by decoding size bytes of the
    # Unicode-Escape encoded string s. Return NULL if an exception was
    # raised by the codec.
    object PyUnicode_DecodeUnicodeEscape(char *s, Py_ssize_t size, char *errors)

    # Encode the Py_UNICODE buffer of the given size using
    # Unicode-Escape and return a Python string object. Return NULL if
    # an exception was raised by the codec.
    object PyUnicode_EncodeUnicodeEscape(Py_UNICODE *s, Py_ssize_t size)

    # Encode a Unicode objects using Unicode-Escape and return the
    # result as Python string object. Error handling is
    # ``strict''. Return NULL if an exception was raised by the codec.
    object PyUnicode_AsUnicodeEscapeString(object unicode)

# These are the ``Raw Unicode Escape'' codec APIs:

    # Create a Unicode object by decoding size bytes of the
    # Raw-Unicode-Escape encoded string s. Return NULL if an exception
    # was raised by the codec.
    object PyUnicode_DecodeRawUnicodeEscape(char *s, Py_ssize_t size, char *errors)

    # Encode the Py_UNICODE buffer of the given size using
    # Raw-Unicode-Escape and return a Python string object. Return
    # NULL if an exception was raised by the codec.
    object PyUnicode_EncodeRawUnicodeEscape(Py_UNICODE *s, Py_ssize_t size, char *errors)

    # Encode a Unicode objects using Raw-Unicode-Escape and return the
    # result as Python string object. Error handling is
    # ``strict''. Return NULL if an exception was raised by the codec.
    object PyUnicode_AsRawUnicodeEscapeString(object unicode)

# These are the Latin-1 codec APIs: Latin-1 corresponds to the first 256 Unicode ordinals and only these are accepted by the codecs during encoding.

    # Create a Unicode object by decoding size bytes of the Latin-1
    # encoded string s. Return NULL if an exception was raised by the
    # codec.
    unicode PyUnicode_DecodeLatin1(char *s, Py_ssize_t size, char *errors)

    # Encode the Py_UNICODE buffer of the given size using Latin-1 and
    # return a Python bytes object. Return NULL if an exception was
    # raised by the codec.
    bytes PyUnicode_EncodeLatin1(Py_UNICODE *s, Py_ssize_t size, char *errors)

    # Encode a Unicode objects using Latin-1 and return the result as
    # Python bytes object. Error handling is ``strict''. Return NULL
    # if an exception was raised by the codec.
    bytes PyUnicode_AsLatin1String(object unicode)

# These are the ASCII codec APIs. Only 7-bit ASCII data is
# accepted. All other codes generate errors.

    # Create a Unicode object by decoding size bytes of the ASCII
    # encoded string s. Return NULL if an exception was raised by the
    # codec.
    unicode PyUnicode_DecodeASCII(char *s, Py_ssize_t size, char *errors)

    # Encode the Py_UNICODE buffer of the given size using ASCII and
    # return a Python bytes object. Return NULL if an exception was
    # raised by the codec.
    bytes PyUnicode_EncodeASCII(Py_UNICODE *s, Py_ssize_t size, char *errors)

    # Encode a Unicode objects using ASCII and return the result as
    # Python bytes object. Error handling is ``strict''. Return NULL
    # if an exception was raised by the codec.
    bytes PyUnicode_AsASCIIString(object o)

# These are the mapping codec APIs:
#
# This codec is special in that it can be used to implement many
# different codecs (and this is in fact what was done to obtain most
# of the standard codecs included in the encodings package). The codec
# uses mapping to encode and decode characters.
#
# Decoding mappings must map single string characters to single
# Unicode characters, integers (which are then interpreted as Unicode
# ordinals) or None (meaning "undefined mapping" and causing an
# error).
#
# Encoding mappings must map single Unicode characters to single
# string characters, integers (which are then interpreted as Latin-1
# ordinals) or None (meaning "undefined mapping" and causing an
# error).
#
# The mapping objects provided must only support the __getitem__
# mapping interface.
#
# If a character lookup fails with a LookupError, the character is
# copied as-is meaning that its ordinal value will be interpreted as
# Unicode or Latin-1 ordinal resp. Because of this, mappings only need
# to contain those mappings which map characters to different code
# points.

    # Create a Unicode object by decoding size bytes of the encoded
    # string s using the given mapping object. Return NULL if an
    # exception was raised by the codec. If mapping is NULL latin-1
    # decoding will be done. Else it can be a dictionary mapping byte
    # or a unicode string, which is treated as a lookup table. Byte
    # values greater that the length of the string and U+FFFE
    # "characters" are treated as "undefined mapping". Changed in
    # version 2.4: Allowed unicode string as mapping argument.
    object PyUnicode_DecodeCharmap(char *s, Py_ssize_t size, object mapping, char *errors)

    # Encode the Py_UNICODE buffer of the given size using the given
    # mapping object and return a Python string object. Return NULL if
    # an exception was raised by the codec.
    #
    # Deprecated since version 3.3, will be removed in version 4.0.
    object PyUnicode_EncodeCharmap(Py_UNICODE *s, Py_ssize_t size, object mapping, char *errors)

    # Encode a Unicode objects using the given mapping object and
    # return the result as Python string object. Error handling is
    # ``strict''. Return NULL if an exception was raised by the codec.
    object PyUnicode_AsCharmapString(object o, object mapping)

# The following codec API is special in that maps Unicode to Unicode.

    # Translate a Py_UNICODE buffer of the given length by applying a
    # character mapping table to it and return the resulting Unicode
    # object. Return NULL when an exception was raised by the codec.
    #
    # The mapping table must map Unicode ordinal integers to Unicode
    # ordinal integers or None (causing deletion of the character).
    #
    # Mapping tables need only provide the __getitem__() interface;
    # dictionaries and sequences work well. Unmapped character
    # ordinals (ones which cause a LookupError) are left untouched and
    # are copied as-is.
    #
    # Deprecated since version 3.3, will be removed in version 4.0.
    object PyUnicode_TranslateCharmap(Py_UNICODE *s, Py_ssize_t size,
                                      object table, char *errors)

# These are the MBCS codec APIs. They are currently only available on
# Windows and use the Win32 MBCS converters to implement the
# conversions. Note that MBCS (or DBCS) is a class of encodings, not
# just one. The target encoding is defined by the user settings on the
# machine running the codec.

    # Create a Unicode object by decoding size bytes of the MBCS
    # encoded string s. Return NULL if an exception was raised by the
    # codec.
    unicode PyUnicode_DecodeMBCS(char *s, Py_ssize_t size, char *errors)

    # If consumed is NULL, behave like PyUnicode_DecodeMBCS(). If
    # consumed is not NULL, PyUnicode_DecodeMBCSStateful() will not
    # decode trailing lead byte and the number of bytes that have been
    # decoded will be stored in consumed. New in version 2.5.
    # NOTE: Python 2.x uses 'int' values for 'size' and 'consumed' (changed in 3.0)
    unicode PyUnicode_DecodeMBCSStateful(char *s, Py_ssize_t size, char *errors, Py_ssize_t *consumed)

    # Encode the Py_UNICODE buffer of the given size using MBCS and
    # return a Python string object. Return NULL if an exception was
    # raised by the codec.
    bytes PyUnicode_EncodeMBCS(Py_UNICODE *s, Py_ssize_t size, char *errors)

    # Encode a Unicode objects using MBCS and return the result as
    # Python string object. Error handling is ``strict''. Return NULL
    # if an exception was raised by the codec.
    bytes PyUnicode_AsMBCSString(object o)

    # Encode the Unicode object using the specified code page and return
    # a Python bytes object. Return NULL if an exception was raised by the
    # codec. Use CP_ACP code page to get the MBCS encoder.
    #
    # New in version 3.3.
    bytes PyUnicode_EncodeCodePage(int code_page, object unicode, const char *errors)


# Py_UCS4 helpers (new in CPython 3.3)

    # These utility functions work on strings of Py_UCS4 characters and
    # otherwise behave like the C standard library functions with the same name.

    size_t Py_UCS4_strlen(const Py_UCS4 *u)
    Py_UCS4* Py_UCS4_strcpy(Py_UCS4 *s1, const Py_UCS4 *s2)
    Py_UCS4* Py_UCS4_strncpy(Py_UCS4 *s1, const Py_UCS4 *s2, size_t n)
    Py_UCS4* Py_UCS4_strcat(Py_UCS4 *s1, const Py_UCS4 *s2)
    int Py_UCS4_strcmp(const Py_UCS4 *s1, const Py_UCS4 *s2)
    int Py_UCS4_strncmp(const Py_UCS4 *s1, const Py_UCS4 *s2, size_t n)
    Py_UCS4* Py_UCS4_strchr(const Py_UCS4 *s, Py_UCS4 c)
    Py_UCS4* Py_UCS4_strrchr(const Py_UCS4 *s, Py_UCS4 c)
