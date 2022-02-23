
//////////////////// IncludeStringH.proto ////////////////////

#include <string.h>

//////////////////// IncludeCppStringH.proto ////////////////////

#include <string>

//////////////////// InitStrings.proto ////////////////////

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t); /*proto*/

//////////////////// InitStrings ////////////////////

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t) {
    while (t->p) {
        #if PY_MAJOR_VERSION < 3
        if (t->is_unicode) {
            *t->p = PyUnicode_DecodeUTF8(t->s, t->n - 1, NULL);
        } else if (t->intern) {
            *t->p = PyString_InternFromString(t->s);
        } else {
            *t->p = PyString_FromStringAndSize(t->s, t->n - 1);
        }
        #else  /* Python 3+ has unicode identifiers */
        if (t->is_unicode | t->is_str) {
            if (t->intern) {
                *t->p = PyUnicode_InternFromString(t->s);
            } else if (t->encoding) {
                *t->p = PyUnicode_Decode(t->s, t->n - 1, t->encoding, NULL);
            } else {
                *t->p = PyUnicode_FromStringAndSize(t->s, t->n - 1);
            }
        } else {
            *t->p = PyBytes_FromStringAndSize(t->s, t->n - 1);
        }
        #endif
        if (!*t->p)
            return -1;
        // initialise cached hash value
        if (PyObject_Hash(*t->p) == -1)
            return -1;
        ++t;
    }
    return 0;
}

//////////////////// BytesContains.proto ////////////////////

static CYTHON_INLINE int __Pyx_BytesContains(PyObject* bytes, char character); /*proto*/

//////////////////// BytesContains ////////////////////
//@requires: IncludeStringH

static CYTHON_INLINE int __Pyx_BytesContains(PyObject* bytes, char character) {
    const Py_ssize_t length = PyBytes_GET_SIZE(bytes);
    char* char_start = PyBytes_AS_STRING(bytes);
    return memchr(char_start, (unsigned char)character, (size_t)length) != NULL;
}


//////////////////// PyUCS4InUnicode.proto ////////////////////

static CYTHON_INLINE int __Pyx_UnicodeContainsUCS4(PyObject* unicode, Py_UCS4 character); /*proto*/

//////////////////// PyUCS4InUnicode ////////////////////

#if PY_VERSION_HEX < 0x03090000 || (defined(PyUnicode_WCHAR_KIND) && defined(PyUnicode_AS_UNICODE))

#if PY_VERSION_HEX < 0x03090000
#define __Pyx_PyUnicode_AS_UNICODE(op) PyUnicode_AS_UNICODE(op)
#define __Pyx_PyUnicode_GET_SIZE(op) PyUnicode_GET_SIZE(op)
#else
// Avoid calling deprecated C-API functions in Py3.9+ that PEP-623 schedules for removal in Py3.12.
// https://www.python.org/dev/peps/pep-0623/
#define __Pyx_PyUnicode_AS_UNICODE(op) (((PyASCIIObject *)(op))->wstr)
#define __Pyx_PyUnicode_GET_SIZE(op) ((PyCompactUnicodeObject *)(op))->wstr_length
#endif

#if !defined(Py_UNICODE_SIZE) || Py_UNICODE_SIZE == 2
static int __Pyx_PyUnicodeBufferContainsUCS4_SP(Py_UNICODE* buffer, Py_ssize_t length, Py_UCS4 character) {
    /* handle surrogate pairs for Py_UNICODE buffers in 16bit Unicode builds */
    Py_UNICODE high_val, low_val;
    Py_UNICODE* pos;
    high_val = (Py_UNICODE) (0xD800 | (((character - 0x10000) >> 10) & ((1<<10)-1)));
    low_val  = (Py_UNICODE) (0xDC00 | ( (character - 0x10000)        & ((1<<10)-1)));
    for (pos=buffer; pos < buffer+length-1; pos++) {
        if (unlikely((high_val == pos[0]) & (low_val == pos[1]))) return 1;
    }
    return 0;
}
#endif

static int __Pyx_PyUnicodeBufferContainsUCS4_BMP(Py_UNICODE* buffer, Py_ssize_t length, Py_UCS4 character) {
    Py_UNICODE uchar;
    Py_UNICODE* pos;
    uchar = (Py_UNICODE) character;
    for (pos=buffer; pos < buffer+length; pos++) {
        if (unlikely(uchar == pos[0])) return 1;
    }
    return 0;
}
#endif

static CYTHON_INLINE int __Pyx_UnicodeContainsUCS4(PyObject* unicode, Py_UCS4 character) {
#if CYTHON_PEP393_ENABLED
    const int kind = PyUnicode_KIND(unicode);
    #ifdef PyUnicode_WCHAR_KIND
    if (likely(kind != PyUnicode_WCHAR_KIND))
    #endif
    {
        Py_ssize_t i;
        const void* udata = PyUnicode_DATA(unicode);
        const Py_ssize_t length = PyUnicode_GET_LENGTH(unicode);
        for (i=0; i < length; i++) {
            if (unlikely(character == PyUnicode_READ(kind, udata, i))) return 1;
        }
        return 0;
    }
#elif PY_VERSION_HEX >= 0x03090000
    #error Cannot use "UChar in Unicode" in Python 3.9 without PEP-393 unicode strings.
#elif !defined(PyUnicode_AS_UNICODE)
    #error Cannot use "UChar in Unicode" in Python < 3.9 without Py_UNICODE support.
#endif

#if PY_VERSION_HEX < 0x03090000 || (defined(PyUnicode_WCHAR_KIND) && defined(PyUnicode_AS_UNICODE))
#if !defined(Py_UNICODE_SIZE) || Py_UNICODE_SIZE == 2
    if ((sizeof(Py_UNICODE) == 2) && unlikely(character > 65535)) {
        return __Pyx_PyUnicodeBufferContainsUCS4_SP(
            __Pyx_PyUnicode_AS_UNICODE(unicode),
            __Pyx_PyUnicode_GET_SIZE(unicode),
            character);
    } else
#endif
    {
        return __Pyx_PyUnicodeBufferContainsUCS4_BMP(
            __Pyx_PyUnicode_AS_UNICODE(unicode),
            __Pyx_PyUnicode_GET_SIZE(unicode),
            character);

    }
#endif
}


//////////////////// PyUnicodeContains.proto ////////////////////

static CYTHON_INLINE int __Pyx_PyUnicode_ContainsTF(PyObject* substring, PyObject* text, int eq) {
    int result = PyUnicode_Contains(text, substring);
    return unlikely(result < 0) ? result : (result == (eq == Py_EQ));
}


//////////////////// CStringEquals.proto ////////////////////

static CYTHON_INLINE int __Pyx_StrEq(const char *, const char *); /*proto*/

//////////////////// CStringEquals ////////////////////

static CYTHON_INLINE int __Pyx_StrEq(const char *s1, const char *s2) {
    while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
    return *s1 == *s2;
}


//////////////////// StrEquals.proto ////////////////////
//@requires: BytesEquals
//@requires: UnicodeEquals

#if PY_MAJOR_VERSION >= 3
#define __Pyx_PyString_Equals __Pyx_PyUnicode_Equals
#else
#define __Pyx_PyString_Equals __Pyx_PyBytes_Equals
#endif


//////////////////// UnicodeEquals.proto ////////////////////

static CYTHON_INLINE int __Pyx_PyUnicode_Equals(PyObject* s1, PyObject* s2, int equals); /*proto*/

//////////////////// UnicodeEquals ////////////////////
//@requires: BytesEquals

static CYTHON_INLINE int __Pyx_PyUnicode_Equals(PyObject* s1, PyObject* s2, int equals) {
#if CYTHON_COMPILING_IN_PYPY
    return PyObject_RichCompareBool(s1, s2, equals);
#else
#if PY_MAJOR_VERSION < 3
    PyObject* owned_ref = NULL;
#endif
    int s1_is_unicode, s2_is_unicode;
    if (s1 == s2) {
        /* as done by PyObject_RichCompareBool(); also catches the (interned) empty string */
        goto return_eq;
    }
    s1_is_unicode = PyUnicode_CheckExact(s1);
    s2_is_unicode = PyUnicode_CheckExact(s2);
#if PY_MAJOR_VERSION < 3
    if ((s1_is_unicode & (!s2_is_unicode)) && PyString_CheckExact(s2)) {
        owned_ref = PyUnicode_FromObject(s2);
        if (unlikely(!owned_ref))
            return -1;
        s2 = owned_ref;
        s2_is_unicode = 1;
    } else if ((s2_is_unicode & (!s1_is_unicode)) && PyString_CheckExact(s1)) {
        owned_ref = PyUnicode_FromObject(s1);
        if (unlikely(!owned_ref))
            return -1;
        s1 = owned_ref;
        s1_is_unicode = 1;
    } else if (((!s2_is_unicode) & (!s1_is_unicode))) {
        return __Pyx_PyBytes_Equals(s1, s2, equals);
    }
#endif
    if (s1_is_unicode & s2_is_unicode) {
        Py_ssize_t length;
        int kind;
        void *data1, *data2;
        if (unlikely(__Pyx_PyUnicode_READY(s1) < 0) || unlikely(__Pyx_PyUnicode_READY(s2) < 0))
            return -1;
        length = __Pyx_PyUnicode_GET_LENGTH(s1);
        if (length != __Pyx_PyUnicode_GET_LENGTH(s2)) {
            goto return_ne;
        }
#if CYTHON_USE_UNICODE_INTERNALS
        {
            Py_hash_t hash1, hash2;
        #if CYTHON_PEP393_ENABLED
            hash1 = ((PyASCIIObject*)s1)->hash;
            hash2 = ((PyASCIIObject*)s2)->hash;
        #else
            hash1 = ((PyUnicodeObject*)s1)->hash;
            hash2 = ((PyUnicodeObject*)s2)->hash;
        #endif
            if (hash1 != hash2 && hash1 != -1 && hash2 != -1) {
                goto return_ne;
            }
        }
#endif
        // len(s1) == len(s2) >= 1  (empty string is interned, and "s1 is not s2")
        kind = __Pyx_PyUnicode_KIND(s1);
        if (kind != __Pyx_PyUnicode_KIND(s2)) {
            goto return_ne;
        }
        data1 = __Pyx_PyUnicode_DATA(s1);
        data2 = __Pyx_PyUnicode_DATA(s2);
        if (__Pyx_PyUnicode_READ(kind, data1, 0) != __Pyx_PyUnicode_READ(kind, data2, 0)) {
            goto return_ne;
        } else if (length == 1) {
            goto return_eq;
        } else {
            int result = memcmp(data1, data2, (size_t)(length * kind));
            #if PY_MAJOR_VERSION < 3
            Py_XDECREF(owned_ref);
            #endif
            return (equals == Py_EQ) ? (result == 0) : (result != 0);
        }
    } else if ((s1 == Py_None) & s2_is_unicode) {
        goto return_ne;
    } else if ((s2 == Py_None) & s1_is_unicode) {
        goto return_ne;
    } else {
        int result;
        PyObject* py_result = PyObject_RichCompare(s1, s2, equals);
        #if PY_MAJOR_VERSION < 3
        Py_XDECREF(owned_ref);
        #endif
        if (!py_result)
            return -1;
        result = __Pyx_PyObject_IsTrue(py_result);
        Py_DECREF(py_result);
        return result;
    }
return_eq:
    #if PY_MAJOR_VERSION < 3
    Py_XDECREF(owned_ref);
    #endif
    return (equals == Py_EQ);
return_ne:
    #if PY_MAJOR_VERSION < 3
    Py_XDECREF(owned_ref);
    #endif
    return (equals == Py_NE);
#endif
}


//////////////////// BytesEquals.proto ////////////////////

static CYTHON_INLINE int __Pyx_PyBytes_Equals(PyObject* s1, PyObject* s2, int equals); /*proto*/

//////////////////// BytesEquals ////////////////////
//@requires: IncludeStringH

static CYTHON_INLINE int __Pyx_PyBytes_Equals(PyObject* s1, PyObject* s2, int equals) {
#if CYTHON_COMPILING_IN_PYPY
    return PyObject_RichCompareBool(s1, s2, equals);
#else
    if (s1 == s2) {
        /* as done by PyObject_RichCompareBool(); also catches the (interned) empty string */
        return (equals == Py_EQ);
    } else if (PyBytes_CheckExact(s1) & PyBytes_CheckExact(s2)) {
        const char *ps1, *ps2;
        Py_ssize_t length = PyBytes_GET_SIZE(s1);
        if (length != PyBytes_GET_SIZE(s2))
            return (equals == Py_NE);
        // len(s1) == len(s2) >= 1  (empty string is interned, and "s1 is not s2")
        ps1 = PyBytes_AS_STRING(s1);
        ps2 = PyBytes_AS_STRING(s2);
        if (ps1[0] != ps2[0]) {
            return (equals == Py_NE);
        } else if (length == 1) {
            return (equals == Py_EQ);
        } else {
            int result;
#if CYTHON_USE_UNICODE_INTERNALS
            Py_hash_t hash1, hash2;
            hash1 = ((PyBytesObject*)s1)->ob_shash;
            hash2 = ((PyBytesObject*)s2)->ob_shash;
            if (hash1 != hash2 && hash1 != -1 && hash2 != -1) {
                return (equals == Py_NE);
            }
#endif
            result = memcmp(ps1, ps2, (size_t)length);
            return (equals == Py_EQ) ? (result == 0) : (result != 0);
        }
    } else if ((s1 == Py_None) & PyBytes_CheckExact(s2)) {
        return (equals == Py_NE);
    } else if ((s2 == Py_None) & PyBytes_CheckExact(s1)) {
        return (equals == Py_NE);
    } else {
        int result;
        PyObject* py_result = PyObject_RichCompare(s1, s2, equals);
        if (!py_result)
            return -1;
        result = __Pyx_PyObject_IsTrue(py_result);
        Py_DECREF(py_result);
        return result;
    }
#endif
}

//////////////////// GetItemIntByteArray.proto ////////////////////

#define __Pyx_GetItemInt_ByteArray(o, i, type, is_signed, to_py_func, is_list, wraparound, boundscheck) \
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ? \
    __Pyx_GetItemInt_ByteArray_Fast(o, (Py_ssize_t)i, wraparound, boundscheck) : \
    (PyErr_SetString(PyExc_IndexError, "bytearray index out of range"), -1))

static CYTHON_INLINE int __Pyx_GetItemInt_ByteArray_Fast(PyObject* string, Py_ssize_t i,
                                                         int wraparound, int boundscheck);

//////////////////// GetItemIntByteArray ////////////////////

static CYTHON_INLINE int __Pyx_GetItemInt_ByteArray_Fast(PyObject* string, Py_ssize_t i,
                                                         int wraparound, int boundscheck) {
    Py_ssize_t length;
    if (wraparound | boundscheck) {
        length = PyByteArray_GET_SIZE(string);
        if (wraparound & unlikely(i < 0)) i += length;
        if ((!boundscheck) || likely(__Pyx_is_valid_index(i, length))) {
            return (unsigned char) (PyByteArray_AS_STRING(string)[i]);
        } else {
            PyErr_SetString(PyExc_IndexError, "bytearray index out of range");
            return -1;
        }
    } else {
        return (unsigned char) (PyByteArray_AS_STRING(string)[i]);
    }
}


//////////////////// SetItemIntByteArray.proto ////////////////////

#define __Pyx_SetItemInt_ByteArray(o, i, v, type, is_signed, to_py_func, is_list, wraparound, boundscheck) \
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ? \
    __Pyx_SetItemInt_ByteArray_Fast(o, (Py_ssize_t)i, v, wraparound, boundscheck) : \
    (PyErr_SetString(PyExc_IndexError, "bytearray index out of range"), -1))

static CYTHON_INLINE int __Pyx_SetItemInt_ByteArray_Fast(PyObject* string, Py_ssize_t i, unsigned char v,
                                                         int wraparound, int boundscheck);

//////////////////// SetItemIntByteArray ////////////////////

static CYTHON_INLINE int __Pyx_SetItemInt_ByteArray_Fast(PyObject* string, Py_ssize_t i, unsigned char v,
                                                         int wraparound, int boundscheck) {
    Py_ssize_t length;
    if (wraparound | boundscheck) {
        length = PyByteArray_GET_SIZE(string);
        if (wraparound & unlikely(i < 0)) i += length;
        if ((!boundscheck) || likely(__Pyx_is_valid_index(i, length))) {
            PyByteArray_AS_STRING(string)[i] = (char) v;
            return 0;
        } else {
            PyErr_SetString(PyExc_IndexError, "bytearray index out of range");
            return -1;
        }
    } else {
        PyByteArray_AS_STRING(string)[i] = (char) v;
        return 0;
    }
}


//////////////////// GetItemIntUnicode.proto ////////////////////

#define __Pyx_GetItemInt_Unicode(o, i, type, is_signed, to_py_func, is_list, wraparound, boundscheck) \
    (__Pyx_fits_Py_ssize_t(i, type, is_signed) ? \
    __Pyx_GetItemInt_Unicode_Fast(o, (Py_ssize_t)i, wraparound, boundscheck) : \
    (PyErr_SetString(PyExc_IndexError, "string index out of range"), (Py_UCS4)-1))

static CYTHON_INLINE Py_UCS4 __Pyx_GetItemInt_Unicode_Fast(PyObject* ustring, Py_ssize_t i,
                                                           int wraparound, int boundscheck);

//////////////////// GetItemIntUnicode ////////////////////

static CYTHON_INLINE Py_UCS4 __Pyx_GetItemInt_Unicode_Fast(PyObject* ustring, Py_ssize_t i,
                                                           int wraparound, int boundscheck) {
    Py_ssize_t length;
    if (unlikely(__Pyx_PyUnicode_READY(ustring) < 0)) return (Py_UCS4)-1;
    if (wraparound | boundscheck) {
        length = __Pyx_PyUnicode_GET_LENGTH(ustring);
        if (wraparound & unlikely(i < 0)) i += length;
        if ((!boundscheck) || likely(__Pyx_is_valid_index(i, length))) {
            return __Pyx_PyUnicode_READ_CHAR(ustring, i);
        } else {
            PyErr_SetString(PyExc_IndexError, "string index out of range");
            return (Py_UCS4)-1;
        }
    } else {
        return __Pyx_PyUnicode_READ_CHAR(ustring, i);
    }
}


/////////////// decode_c_string_utf16.proto ///////////////

static CYTHON_INLINE PyObject *__Pyx_PyUnicode_DecodeUTF16(const char *s, Py_ssize_t size, const char *errors) {
    int byteorder = 0;
    return PyUnicode_DecodeUTF16(s, size, errors, &byteorder);
}
static CYTHON_INLINE PyObject *__Pyx_PyUnicode_DecodeUTF16LE(const char *s, Py_ssize_t size, const char *errors) {
    int byteorder = -1;
    return PyUnicode_DecodeUTF16(s, size, errors, &byteorder);
}
static CYTHON_INLINE PyObject *__Pyx_PyUnicode_DecodeUTF16BE(const char *s, Py_ssize_t size, const char *errors) {
    int byteorder = 1;
    return PyUnicode_DecodeUTF16(s, size, errors, &byteorder);
}

/////////////// decode_cpp_string.proto ///////////////
//@requires: IncludeCppStringH
//@requires: decode_c_bytes

static CYTHON_INLINE PyObject* __Pyx_decode_cpp_string(
         std::string cppstring, Py_ssize_t start, Py_ssize_t stop,
         const char* encoding, const char* errors,
         PyObject* (*decode_func)(const char *s, Py_ssize_t size, const char *errors)) {
    return __Pyx_decode_c_bytes(
        cppstring.data(), cppstring.size(), start, stop, encoding, errors, decode_func);
}

/////////////// decode_c_string.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_decode_c_string(
         const char* cstring, Py_ssize_t start, Py_ssize_t stop,
         const char* encoding, const char* errors,
         PyObject* (*decode_func)(const char *s, Py_ssize_t size, const char *errors));

/////////////// decode_c_string ///////////////
//@requires: IncludeStringH
//@requires: decode_c_string_utf16
//@substitute: naming

/* duplicate code to avoid calling strlen() if start >= 0 and stop >= 0 */
static CYTHON_INLINE PyObject* __Pyx_decode_c_string(
         const char* cstring, Py_ssize_t start, Py_ssize_t stop,
         const char* encoding, const char* errors,
         PyObject* (*decode_func)(const char *s, Py_ssize_t size, const char *errors)) {
    Py_ssize_t length;
    if (unlikely((start < 0) | (stop < 0))) {
        size_t slen = strlen(cstring);
        if (unlikely(slen > (size_t) PY_SSIZE_T_MAX)) {
            PyErr_SetString(PyExc_OverflowError,
                            "c-string too long to convert to Python");
            return NULL;
        }
        length = (Py_ssize_t) slen;
        if (start < 0) {
            start += length;
            if (start < 0)
                start = 0;
        }
        if (stop < 0)
            stop += length;
    }
    if (unlikely(stop <= start))
        return __Pyx_NewRef($empty_unicode);
    length = stop - start;
    cstring += start;
    if (decode_func) {
        return decode_func(cstring, length, errors);
    } else {
        return PyUnicode_Decode(cstring, length, encoding, errors);
    }
}

/////////////// decode_c_bytes.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_decode_c_bytes(
         const char* cstring, Py_ssize_t length, Py_ssize_t start, Py_ssize_t stop,
         const char* encoding, const char* errors,
         PyObject* (*decode_func)(const char *s, Py_ssize_t size, const char *errors));

/////////////// decode_c_bytes ///////////////
//@requires: decode_c_string_utf16
//@substitute: naming

static CYTHON_INLINE PyObject* __Pyx_decode_c_bytes(
         const char* cstring, Py_ssize_t length, Py_ssize_t start, Py_ssize_t stop,
         const char* encoding, const char* errors,
         PyObject* (*decode_func)(const char *s, Py_ssize_t size, const char *errors)) {
    if (unlikely((start < 0) | (stop < 0))) {
        if (start < 0) {
            start += length;
            if (start < 0)
                start = 0;
        }
        if (stop < 0)
            stop += length;
    }
    if (stop > length)
        stop = length;
    if (unlikely(stop <= start))
        return __Pyx_NewRef($empty_unicode);
    length = stop - start;
    cstring += start;
    if (decode_func) {
        return decode_func(cstring, length, errors);
    } else {
        return PyUnicode_Decode(cstring, length, encoding, errors);
    }
}

/////////////// decode_bytes.proto ///////////////
//@requires: decode_c_bytes

static CYTHON_INLINE PyObject* __Pyx_decode_bytes(
         PyObject* string, Py_ssize_t start, Py_ssize_t stop,
         const char* encoding, const char* errors,
         PyObject* (*decode_func)(const char *s, Py_ssize_t size, const char *errors)) {
    return __Pyx_decode_c_bytes(
        PyBytes_AS_STRING(string), PyBytes_GET_SIZE(string),
        start, stop, encoding, errors, decode_func);
}

/////////////// decode_bytearray.proto ///////////////
//@requires: decode_c_bytes

static CYTHON_INLINE PyObject* __Pyx_decode_bytearray(
         PyObject* string, Py_ssize_t start, Py_ssize_t stop,
         const char* encoding, const char* errors,
         PyObject* (*decode_func)(const char *s, Py_ssize_t size, const char *errors)) {
    return __Pyx_decode_c_bytes(
        PyByteArray_AS_STRING(string), PyByteArray_GET_SIZE(string),
        start, stop, encoding, errors, decode_func);
}

/////////////// PyUnicode_Substring.proto ///////////////

static CYTHON_INLINE PyObject* __Pyx_PyUnicode_Substring(
            PyObject* text, Py_ssize_t start, Py_ssize_t stop);

/////////////// PyUnicode_Substring ///////////////
//@substitute: naming

static CYTHON_INLINE PyObject* __Pyx_PyUnicode_Substring(
            PyObject* text, Py_ssize_t start, Py_ssize_t stop) {
    Py_ssize_t length;
    if (unlikely(__Pyx_PyUnicode_READY(text) == -1)) return NULL;
    length = __Pyx_PyUnicode_GET_LENGTH(text);
    if (start < 0) {
        start += length;
        if (start < 0)
            start = 0;
    }
    if (stop < 0)
        stop += length;
    else if (stop > length)
        stop = length;
    if (stop <= start)
        return __Pyx_NewRef($empty_unicode);
#if CYTHON_PEP393_ENABLED
    return PyUnicode_FromKindAndData(PyUnicode_KIND(text),
        PyUnicode_1BYTE_DATA(text) + start*PyUnicode_KIND(text), stop-start);
#else
    return PyUnicode_FromUnicode(PyUnicode_AS_UNICODE(text)+start, stop-start);
#endif
}


/////////////// py_unicode_istitle.proto ///////////////

// Py_UNICODE_ISTITLE() doesn't match unicode.istitle() as the latter
// additionally allows character that comply with Py_UNICODE_ISUPPER()

#if PY_VERSION_HEX < 0x030200A2
static CYTHON_INLINE int __Pyx_Py_UNICODE_ISTITLE(Py_UNICODE uchar)
#else
static CYTHON_INLINE int __Pyx_Py_UNICODE_ISTITLE(Py_UCS4 uchar)
#endif
{
    return Py_UNICODE_ISTITLE(uchar) || Py_UNICODE_ISUPPER(uchar);
}


/////////////// unicode_tailmatch.proto ///////////////

static int __Pyx_PyUnicode_Tailmatch(
    PyObject* s, PyObject* substr, Py_ssize_t start, Py_ssize_t end, int direction); /*proto*/

/////////////// unicode_tailmatch ///////////////

// Python's unicode.startswith() and unicode.endswith() support a
// tuple of prefixes/suffixes, whereas it's much more common to
// test for a single unicode string.

static int __Pyx_PyUnicode_TailmatchTuple(PyObject* s, PyObject* substrings,
                                          Py_ssize_t start, Py_ssize_t end, int direction) {
    Py_ssize_t i, count = PyTuple_GET_SIZE(substrings);
    for (i = 0; i < count; i++) {
        Py_ssize_t result;
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        result = PyUnicode_Tailmatch(s, PyTuple_GET_ITEM(substrings, i),
                                     start, end, direction);
#else
        PyObject* sub = PySequence_ITEM(substrings, i);
        if (unlikely(!sub)) return -1;
        result = PyUnicode_Tailmatch(s, sub, start, end, direction);
        Py_DECREF(sub);
#endif
        if (result) {
            return (int) result;
        }
    }
    return 0;
}

static int __Pyx_PyUnicode_Tailmatch(PyObject* s, PyObject* substr,
                                     Py_ssize_t start, Py_ssize_t end, int direction) {
    if (unlikely(PyTuple_Check(substr))) {
        return __Pyx_PyUnicode_TailmatchTuple(s, substr, start, end, direction);
    }
    return (int) PyUnicode_Tailmatch(s, substr, start, end, direction);
}


/////////////// bytes_tailmatch.proto ///////////////

static int __Pyx_PyBytes_SingleTailmatch(PyObject* self, PyObject* arg,
                                         Py_ssize_t start, Py_ssize_t end, int direction); /*proto*/
static int __Pyx_PyBytes_Tailmatch(PyObject* self, PyObject* substr,
                                   Py_ssize_t start, Py_ssize_t end, int direction); /*proto*/

/////////////// bytes_tailmatch ///////////////

static int __Pyx_PyBytes_SingleTailmatch(PyObject* self, PyObject* arg,
                                         Py_ssize_t start, Py_ssize_t end, int direction) {
    const char* self_ptr = PyBytes_AS_STRING(self);
    Py_ssize_t self_len = PyBytes_GET_SIZE(self);
    const char* sub_ptr;
    Py_ssize_t sub_len;
    int retval;

    Py_buffer view;
    view.obj = NULL;

    if ( PyBytes_Check(arg) ) {
        sub_ptr = PyBytes_AS_STRING(arg);
        sub_len = PyBytes_GET_SIZE(arg);
    }
#if PY_MAJOR_VERSION < 3
    // Python 2.x allows mixing unicode and str
    else if ( PyUnicode_Check(arg) ) {
        return (int) PyUnicode_Tailmatch(self, arg, start, end, direction);
    }
#endif
    else {
        if (unlikely(PyObject_GetBuffer(self, &view, PyBUF_SIMPLE) == -1))
            return -1;
        sub_ptr = (const char*) view.buf;
        sub_len = view.len;
    }

    if (end > self_len)
        end = self_len;
    else if (end < 0)
        end += self_len;
    if (end < 0)
        end = 0;
    if (start < 0)
        start += self_len;
    if (start < 0)
        start = 0;

    if (direction > 0) {
        /* endswith */
        if (end-sub_len > start)
            start = end - sub_len;
    }

    if (start + sub_len <= end)
        retval = !memcmp(self_ptr+start, sub_ptr, (size_t)sub_len);
    else
        retval = 0;

    if (view.obj)
        PyBuffer_Release(&view);

    return retval;
}

static int __Pyx_PyBytes_TailmatchTuple(PyObject* self, PyObject* substrings,
                                        Py_ssize_t start, Py_ssize_t end, int direction) {
    Py_ssize_t i, count = PyTuple_GET_SIZE(substrings);
    for (i = 0; i < count; i++) {
        int result;
#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
        result = __Pyx_PyBytes_SingleTailmatch(self, PyTuple_GET_ITEM(substrings, i),
                                               start, end, direction);
#else
        PyObject* sub = PySequence_ITEM(substrings, i);
        if (unlikely(!sub)) return -1;
        result = __Pyx_PyBytes_SingleTailmatch(self, sub, start, end, direction);
        Py_DECREF(sub);
#endif
        if (result) {
            return result;
        }
    }
    return 0;
}

static int __Pyx_PyBytes_Tailmatch(PyObject* self, PyObject* substr,
                                   Py_ssize_t start, Py_ssize_t end, int direction) {
    if (unlikely(PyTuple_Check(substr))) {
        return __Pyx_PyBytes_TailmatchTuple(self, substr, start, end, direction);
    }

    return __Pyx_PyBytes_SingleTailmatch(self, substr, start, end, direction);
}


/////////////// str_tailmatch.proto ///////////////

static CYTHON_INLINE int __Pyx_PyStr_Tailmatch(PyObject* self, PyObject* arg, Py_ssize_t start,
                                               Py_ssize_t end, int direction); /*proto*/

/////////////// str_tailmatch ///////////////
//@requires: bytes_tailmatch
//@requires: unicode_tailmatch

static CYTHON_INLINE int __Pyx_PyStr_Tailmatch(PyObject* self, PyObject* arg, Py_ssize_t start,
                                               Py_ssize_t end, int direction)
{
    // We do not use a C compiler macro here to avoid "unused function"
    // warnings for the *_Tailmatch() function that is not being used in
    // the specific CPython version.  The C compiler will generate the same
    // code anyway, and will usually just remove the unused function.
    if (PY_MAJOR_VERSION < 3)
        return __Pyx_PyBytes_Tailmatch(self, arg, start, end, direction);
    else
        return __Pyx_PyUnicode_Tailmatch(self, arg, start, end, direction);
}


/////////////// bytes_index.proto ///////////////

static CYTHON_INLINE char __Pyx_PyBytes_GetItemInt(PyObject* bytes, Py_ssize_t index, int check_bounds); /*proto*/

/////////////// bytes_index ///////////////

static CYTHON_INLINE char __Pyx_PyBytes_GetItemInt(PyObject* bytes, Py_ssize_t index, int check_bounds) {
    if (index < 0)
        index += PyBytes_GET_SIZE(bytes);
    if (check_bounds) {
        Py_ssize_t size = PyBytes_GET_SIZE(bytes);
        if (unlikely(!__Pyx_is_valid_index(index, size))) {
            PyErr_SetString(PyExc_IndexError, "string index out of range");
            return (char) -1;
        }
    }
    return PyBytes_AS_STRING(bytes)[index];
}


//////////////////// StringJoin.proto ////////////////////

#if PY_MAJOR_VERSION < 3
#define __Pyx_PyString_Join __Pyx_PyBytes_Join
#define __Pyx_PyBaseString_Join(s, v) (PyUnicode_CheckExact(s) ? PyUnicode_Join(s, v) : __Pyx_PyBytes_Join(s, v))
#else
#define __Pyx_PyString_Join PyUnicode_Join
#define __Pyx_PyBaseString_Join PyUnicode_Join
#endif

#if CYTHON_COMPILING_IN_CPYTHON
    #if PY_MAJOR_VERSION < 3
    #define __Pyx_PyBytes_Join _PyString_Join
    #else
    #define __Pyx_PyBytes_Join _PyBytes_Join
    #endif
#else
static CYTHON_INLINE PyObject* __Pyx_PyBytes_Join(PyObject* sep, PyObject* values); /*proto*/
#endif


//////////////////// StringJoin ////////////////////

#if !CYTHON_COMPILING_IN_CPYTHON
static CYTHON_INLINE PyObject* __Pyx_PyBytes_Join(PyObject* sep, PyObject* values) {
    return PyObject_CallMethodObjArgs(sep, PYIDENT("join"), values, NULL);
}
#endif


/////////////// JoinPyUnicode.proto ///////////////

static PyObject* __Pyx_PyUnicode_Join(PyObject* value_tuple, Py_ssize_t value_count, Py_ssize_t result_ulength,
                                      Py_UCS4 max_char);

/////////////// JoinPyUnicode ///////////////
//@requires: IncludeStringH
//@substitute: naming

static PyObject* __Pyx_PyUnicode_Join(PyObject* value_tuple, Py_ssize_t value_count, Py_ssize_t result_ulength,
                                      CYTHON_UNUSED Py_UCS4 max_char) {
#if CYTHON_USE_UNICODE_INTERNALS && CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS
    PyObject *result_uval;
    int result_ukind;
    Py_ssize_t i, char_pos;
    void *result_udata;
#if CYTHON_PEP393_ENABLED
    // Py 3.3+  (post PEP-393)
    result_uval = PyUnicode_New(result_ulength, max_char);
    if (unlikely(!result_uval)) return NULL;
    result_ukind = (max_char <= 255) ? PyUnicode_1BYTE_KIND : (max_char <= 65535) ? PyUnicode_2BYTE_KIND : PyUnicode_4BYTE_KIND;
    result_udata = PyUnicode_DATA(result_uval);
#else
    // Py 2.x/3.2  (pre PEP-393)
    result_uval = PyUnicode_FromUnicode(NULL, result_ulength);
    if (unlikely(!result_uval)) return NULL;
    result_ukind = sizeof(Py_UNICODE);
    result_udata = PyUnicode_AS_UNICODE(result_uval);
#endif

    char_pos = 0;
    for (i=0; i < value_count; i++) {
        int ukind;
        Py_ssize_t ulength;
        void *udata;
        PyObject *uval = PyTuple_GET_ITEM(value_tuple, i);
        if (unlikely(__Pyx_PyUnicode_READY(uval)))
            goto bad;
        ulength = __Pyx_PyUnicode_GET_LENGTH(uval);
        if (unlikely(!ulength))
            continue;
        if (unlikely(char_pos + ulength < 0))
            goto overflow;
        ukind = __Pyx_PyUnicode_KIND(uval);
        udata = __Pyx_PyUnicode_DATA(uval);
        if (!CYTHON_PEP393_ENABLED || ukind == result_ukind) {
            memcpy((char *)result_udata + char_pos * result_ukind, udata, (size_t) (ulength * result_ukind));
        } else {
            #if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX >= 0x030300F0 || defined(_PyUnicode_FastCopyCharacters)
            _PyUnicode_FastCopyCharacters(result_uval, char_pos, uval, 0, ulength);
            #else
            Py_ssize_t j;
            for (j=0; j < ulength; j++) {
                Py_UCS4 uchar = __Pyx_PyUnicode_READ(ukind, udata, j);
                __Pyx_PyUnicode_WRITE(result_ukind, result_udata, char_pos+j, uchar);
            }
            #endif
        }
        char_pos += ulength;
    }
    return result_uval;
overflow:
    PyErr_SetString(PyExc_OverflowError, "join() result is too long for a Python string");
bad:
    Py_DECREF(result_uval);
    return NULL;
#else
    // non-CPython fallback
    result_ulength++;
    value_count++;
    return PyUnicode_Join($empty_unicode, value_tuple);
#endif
}


/////////////// BuildPyUnicode.proto ///////////////

static PyObject* __Pyx_PyUnicode_BuildFromAscii(Py_ssize_t ulength, char* chars, int clength,
                                                int prepend_sign, char padding_char);

/////////////// BuildPyUnicode ///////////////

// Create a PyUnicode object from an ASCII char*, e.g. a formatted number.

static PyObject* __Pyx_PyUnicode_BuildFromAscii(Py_ssize_t ulength, char* chars, int clength,
                                                int prepend_sign, char padding_char) {
    PyObject *uval;
    Py_ssize_t uoffset = ulength - clength;
#if CYTHON_USE_UNICODE_INTERNALS
    Py_ssize_t i;
#if CYTHON_PEP393_ENABLED
    // Py 3.3+  (post PEP-393)
    void *udata;
    uval = PyUnicode_New(ulength, 127);
    if (unlikely(!uval)) return NULL;
    udata = PyUnicode_DATA(uval);
#else
    // Py 2.x/3.2  (pre PEP-393)
    Py_UNICODE *udata;
    uval = PyUnicode_FromUnicode(NULL, ulength);
    if (unlikely(!uval)) return NULL;
    udata = PyUnicode_AS_UNICODE(uval);
#endif
    if (uoffset > 0) {
        i = 0;
        if (prepend_sign) {
            __Pyx_PyUnicode_WRITE(PyUnicode_1BYTE_KIND, udata, 0, '-');
            i++;
        }
        for (; i < uoffset; i++) {
            __Pyx_PyUnicode_WRITE(PyUnicode_1BYTE_KIND, udata, i, padding_char);
        }
    }
    for (i=0; i < clength; i++) {
        __Pyx_PyUnicode_WRITE(PyUnicode_1BYTE_KIND, udata, uoffset+i, chars[i]);
    }

#else
    // non-CPython
    {
        PyObject *sign = NULL, *padding = NULL;
        uval = NULL;
        if (uoffset > 0) {
            prepend_sign = !!prepend_sign;
            if (uoffset > prepend_sign) {
                padding = PyUnicode_FromOrdinal(padding_char);
                if (likely(padding) && uoffset > prepend_sign + 1) {
                    PyObject *tmp;
                    PyObject *repeat = PyInt_FromSize_t(uoffset - prepend_sign);
                    if (unlikely(!repeat)) goto done_or_error;
                    tmp = PyNumber_Multiply(padding, repeat);
                    Py_DECREF(repeat);
                    Py_DECREF(padding);
                    padding = tmp;
                }
                if (unlikely(!padding)) goto done_or_error;
            }
            if (prepend_sign) {
                sign = PyUnicode_FromOrdinal('-');
                if (unlikely(!sign)) goto done_or_error;
            }
        }

        uval = PyUnicode_DecodeASCII(chars, clength, NULL);
        if (likely(uval) && padding) {
            PyObject *tmp = PyNumber_Add(padding, uval);
            Py_DECREF(uval);
            uval = tmp;
        }
        if (likely(uval) && sign) {
            PyObject *tmp = PyNumber_Add(sign, uval);
            Py_DECREF(uval);
            uval = tmp;
        }
done_or_error:
        Py_XDECREF(padding);
        Py_XDECREF(sign);
    }
#endif

    return uval;
}


//////////////////// ByteArrayAppendObject.proto ////////////////////

static CYTHON_INLINE int __Pyx_PyByteArray_AppendObject(PyObject* bytearray, PyObject* value);

//////////////////// ByteArrayAppendObject ////////////////////
//@requires: ByteArrayAppend

static CYTHON_INLINE int __Pyx_PyByteArray_AppendObject(PyObject* bytearray, PyObject* value) {
    Py_ssize_t ival;
#if PY_MAJOR_VERSION < 3
    if (unlikely(PyString_Check(value))) {
        if (unlikely(PyString_GET_SIZE(value) != 1)) {
            PyErr_SetString(PyExc_ValueError, "string must be of size 1");
            return -1;
        }
        ival = (unsigned char) (PyString_AS_STRING(value)[0]);
    } else
#endif
#if CYTHON_USE_PYLONG_INTERNALS
    if (likely(PyLong_CheckExact(value)) && likely(Py_SIZE(value) == 1 || Py_SIZE(value) == 0)) {
        if (Py_SIZE(value) == 0) {
            ival = 0;
        } else {
            ival = ((PyLongObject*)value)->ob_digit[0];
            if (unlikely(ival > 255)) goto bad_range;
        }
    } else
#endif
    {
        // CPython calls PyNumber_Index() internally
        ival = __Pyx_PyIndex_AsSsize_t(value);
        if (unlikely(!__Pyx_is_valid_index(ival, 256))) {
            if (ival == -1 && PyErr_Occurred())
                return -1;
            goto bad_range;
        }
    }
    return __Pyx_PyByteArray_Append(bytearray, ival);
bad_range:
    PyErr_SetString(PyExc_ValueError, "byte must be in range(0, 256)");
    return -1;
}

//////////////////// ByteArrayAppend.proto ////////////////////

static CYTHON_INLINE int __Pyx_PyByteArray_Append(PyObject* bytearray, int value);

//////////////////// ByteArrayAppend ////////////////////
//@requires: ObjectHandling.c::PyObjectCallMethod1

static CYTHON_INLINE int __Pyx_PyByteArray_Append(PyObject* bytearray, int value) {
    PyObject *pyval, *retval;
#if CYTHON_COMPILING_IN_CPYTHON
    if (likely(__Pyx_is_valid_index(value, 256))) {
        Py_ssize_t n = Py_SIZE(bytearray);
        if (likely(n != PY_SSIZE_T_MAX)) {
            if (unlikely(PyByteArray_Resize(bytearray, n + 1) < 0))
                return -1;
            PyByteArray_AS_STRING(bytearray)[n] = value;
            return 0;
        }
    } else {
        PyErr_SetString(PyExc_ValueError, "byte must be in range(0, 256)");
        return -1;
    }
#endif
    pyval = PyInt_FromLong(value);
    if (unlikely(!pyval))
        return -1;
    retval = __Pyx_PyObject_CallMethod1(bytearray, PYIDENT("append"), pyval);
    Py_DECREF(pyval);
    if (unlikely(!retval))
        return -1;
    Py_DECREF(retval);
    return 0;
}


//////////////////// PyObjectFormat.proto ////////////////////

#if CYTHON_USE_UNICODE_WRITER
static PyObject* __Pyx_PyObject_Format(PyObject* s, PyObject* f);
#else
#define __Pyx_PyObject_Format(s, f) PyObject_Format(s, f)
#endif

//////////////////// PyObjectFormat ////////////////////

#if CYTHON_USE_UNICODE_WRITER
static PyObject* __Pyx_PyObject_Format(PyObject* obj, PyObject* format_spec) {
    int ret;
    _PyUnicodeWriter writer;

    if (likely(PyFloat_CheckExact(obj))) {
        // copied from CPython 3.5 "float__format__()" in floatobject.c
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX < 0x03040000
        _PyUnicodeWriter_Init(&writer, 0);
#else
        _PyUnicodeWriter_Init(&writer);
#endif
        ret = _PyFloat_FormatAdvancedWriter(
            &writer,
            obj,
            format_spec, 0, PyUnicode_GET_LENGTH(format_spec));
    } else if (likely(PyLong_CheckExact(obj))) {
        // copied from CPython 3.5 "long__format__()" in longobject.c
#if CYTHON_COMPILING_IN_CPYTHON && PY_VERSION_HEX < 0x03040000
        _PyUnicodeWriter_Init(&writer, 0);
#else
        _PyUnicodeWriter_Init(&writer);
#endif
        ret = _PyLong_FormatAdvancedWriter(
            &writer,
            obj,
            format_spec, 0, PyUnicode_GET_LENGTH(format_spec));
    } else {
        return PyObject_Format(obj, format_spec);
    }

    if (unlikely(ret == -1)) {
        _PyUnicodeWriter_Dealloc(&writer);
        return NULL;
    }
    return _PyUnicodeWriter_Finish(&writer);
}
#endif


//////////////////// PyObjectFormatSimple.proto ////////////////////

#if CYTHON_COMPILING_IN_PYPY
    #define __Pyx_PyObject_FormatSimple(s, f) ( \
        likely(PyUnicode_CheckExact(s)) ? (Py_INCREF(s), s) : \
        PyObject_Format(s, f))
#elif PY_MAJOR_VERSION < 3
    // str is common in Py2, but formatting must return a Unicode string
    #define __Pyx_PyObject_FormatSimple(s, f) ( \
        likely(PyUnicode_CheckExact(s)) ? (Py_INCREF(s), s) : \
        likely(PyString_CheckExact(s)) ? PyUnicode_FromEncodedObject(s, NULL, "strict") : \
        PyObject_Format(s, f))
#elif CYTHON_USE_TYPE_SLOTS
    // Py3 nicely returns unicode strings from str() which makes this quite efficient for builtin types
    #define __Pyx_PyObject_FormatSimple(s, f) ( \
        likely(PyUnicode_CheckExact(s)) ? (Py_INCREF(s), s) : \
        likely(PyLong_CheckExact(s)) ? PyLong_Type.tp_str(s) : \
        likely(PyFloat_CheckExact(s)) ? PyFloat_Type.tp_str(s) : \
        PyObject_Format(s, f))
#else
    #define __Pyx_PyObject_FormatSimple(s, f) ( \
        likely(PyUnicode_CheckExact(s)) ? (Py_INCREF(s), s) : \
        PyObject_Format(s, f))
#endif


//////////////////// PyObjectFormatAndDecref.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyObject_FormatSimpleAndDecref(PyObject* s, PyObject* f);
static CYTHON_INLINE PyObject* __Pyx_PyObject_FormatAndDecref(PyObject* s, PyObject* f);

//////////////////// PyObjectFormatAndDecref ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyObject_FormatSimpleAndDecref(PyObject* s, PyObject* f) {
    if (unlikely(!s)) return NULL;
    if (likely(PyUnicode_CheckExact(s))) return s;
    #if PY_MAJOR_VERSION < 3
    // str is common in Py2, but formatting must return a Unicode string
    if (likely(PyString_CheckExact(s))) {
        PyObject *result = PyUnicode_FromEncodedObject(s, NULL, "strict");
        Py_DECREF(s);
        return result;
    }
    #endif
    return __Pyx_PyObject_FormatAndDecref(s, f);
}

static CYTHON_INLINE PyObject* __Pyx_PyObject_FormatAndDecref(PyObject* s, PyObject* f) {
    PyObject *result = PyObject_Format(s, f);
    Py_DECREF(s);
    return result;
}


//////////////////// PyUnicode_Unicode.proto ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyUnicode_Unicode(PyObject *obj);/*proto*/

//////////////////// PyUnicode_Unicode ////////////////////

static CYTHON_INLINE PyObject* __Pyx_PyUnicode_Unicode(PyObject *obj) {
    if (unlikely(obj == Py_None))
        obj = PYUNICODE("None");
    return __Pyx_NewRef(obj);
}


//////////////////// PyObject_Unicode.proto ////////////////////

#if PY_MAJOR_VERSION >= 3
#define __Pyx_PyObject_Unicode(obj) \
    (likely(PyUnicode_CheckExact(obj)) ? __Pyx_NewRef(obj) : PyObject_Str(obj))
#else
#define __Pyx_PyObject_Unicode(obj) \
    (likely(PyUnicode_CheckExact(obj)) ? __Pyx_NewRef(obj) : PyObject_Unicode(obj))
#endif
