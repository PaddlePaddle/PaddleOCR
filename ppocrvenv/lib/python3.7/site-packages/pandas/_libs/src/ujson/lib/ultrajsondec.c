/*
Copyright (c) 2011-2013, ESN Social Software AB and Jonas Tarnstrom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the ESN Social Software AB nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ESN SOCIAL SOFTWARE AB OR JONAS TARNSTROM BE
LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Portions of code from MODP_ASCII - Ascii transformations (upper/lower, etc)
https://github.com/client9/stringencoders
Copyright (c) 2007  Nick Galbreath -- nickg [at] modp [dot] com. All rights
reserved.

Numeric decoder derived from from TCL library
https://www.opensource.apple.com/source/tcl/tcl-14/tcl/license.terms
* Copyright (c) 1988-1993 The Regents of the University of California.
* Copyright (c) 1994 Sun Microsystems, Inc.
*/

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "ultrajson.h"

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#ifndef NULL
#define NULL 0
#endif

struct DecoderState {
    char *start;
    char *end;
    wchar_t *escStart;
    wchar_t *escEnd;
    int escHeap;
    int lastType;
    JSUINT32 objDepth;
    void *prv;
    JSONObjectDecoder *dec;
};

JSOBJ FASTCALL_MSVC decode_any(struct DecoderState *ds);
typedef JSOBJ (*PFN_DECODER)(struct DecoderState *ds);

static JSOBJ SetError(struct DecoderState *ds, int offset,
                      const char *message) {
    ds->dec->errorOffset = ds->start + offset;
    ds->dec->errorStr = (char *)message;
    return NULL;
}

double createDouble(double intNeg, double intValue, double frcValue,
                    int frcDecimalCount) {
    static const double g_pow10[] = {1.0,
                                     0.1,
                                     0.01,
                                     0.001,
                                     0.0001,
                                     0.00001,
                                     0.000001,
                                     0.0000001,
                                     0.00000001,
                                     0.000000001,
                                     0.0000000001,
                                     0.00000000001,
                                     0.000000000001,
                                     0.0000000000001,
                                     0.00000000000001,
                                     0.000000000000001};
    return (intValue + (frcValue * g_pow10[frcDecimalCount])) * intNeg;
}

JSOBJ FASTCALL_MSVC decodePreciseFloat(struct DecoderState *ds) {
    char *end;
    double value;
    errno = 0;

    value = strtod(ds->start, &end);

    if (errno == ERANGE) {
        return SetError(ds, -1, "Range error when decoding numeric as double");
    }

    ds->start = end;
    return ds->dec->newDouble(ds->prv, value);
}

JSOBJ FASTCALL_MSVC decode_numeric(struct DecoderState *ds) {
    int intNeg = 1;
    int mantSize = 0;
    JSUINT64 intValue;
    int chr;
    int decimalCount = 0;
    double frcValue = 0.0;
    double expNeg;
    double expValue;
    char *offset = ds->start;

    JSUINT64 overflowLimit = LLONG_MAX;

    if (*(offset) == 'I') {
      goto DECODE_INF;
    } else if (*(offset) == 'N') {
      goto DECODE_NAN;
    } else if (*(offset) == '-') {
        offset++;
        intNeg = -1;
        if (*(offset) == 'I') {
          goto DECODE_INF;
        }
        overflowLimit = LLONG_MIN;
    }

    // Scan integer part
    intValue = 0;

    while (1) {
        chr = (int)(unsigned char)*(offset);

        switch (chr) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9': {
                // FIXME: Check for arithmetic overflow here
                // PERF: Don't do 64-bit arithmetic here unless we know we have
                // to
                intValue = intValue * 10ULL + (JSLONG)(chr - 48);

                if (intValue > overflowLimit) {
                    return SetError(ds, -1, overflowLimit == LLONG_MAX
                                                ? "Value is too big"
                                                : "Value is too small");
                }

                offset++;
                mantSize++;
                break;
            }
            case '.': {
                offset++;
                goto DECODE_FRACTION;
                break;
            }
            case 'e':
            case 'E': {
                offset++;
                goto DECODE_EXPONENT;
                break;
            }

            default: {
                goto BREAK_INT_LOOP;
                break;
            }
        }
    }

BREAK_INT_LOOP:

    ds->lastType = JT_INT;
    ds->start = offset;

    if ((intValue >> 31)) {
        return ds->dec->newLong(ds->prv, (JSINT64)(intValue * (JSINT64)intNeg));
    } else {
        return ds->dec->newInt(ds->prv, (JSINT32)(intValue * intNeg));
    }

DECODE_FRACTION:

    if (ds->dec->preciseFloat) {
        return decodePreciseFloat(ds);
    }

    // Scan fraction part
    frcValue = 0.0;
    for (;;) {
        chr = (int)(unsigned char)*(offset);

        switch (chr) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9': {
                if (decimalCount < JSON_DOUBLE_MAX_DECIMALS) {
                    frcValue = frcValue * 10.0 + (double)(chr - 48);
                    decimalCount++;
                }
                offset++;
                break;
            }
            case 'e':
            case 'E': {
                offset++;
                goto DECODE_EXPONENT;
                break;
            }
            default: { goto BREAK_FRC_LOOP; }
        }
    }

BREAK_FRC_LOOP:
    // FIXME: Check for arithmetic overflow here
    ds->lastType = JT_DOUBLE;
    ds->start = offset;
    return ds->dec->newDouble(
        ds->prv,
        createDouble((double)intNeg, (double)intValue, frcValue, decimalCount));

DECODE_EXPONENT:
    if (ds->dec->preciseFloat) {
        return decodePreciseFloat(ds);
    }

    expNeg = 1.0;

    if (*(offset) == '-') {
        expNeg = -1.0;
        offset++;
    } else if (*(offset) == '+') {
        expNeg = +1.0;
        offset++;
    }

    expValue = 0.0;

    for (;;) {
        chr = (int)(unsigned char)*(offset);

        switch (chr) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9': {
                expValue = expValue * 10.0 + (double)(chr - 48);
                offset++;
                break;
            }
            default: { goto BREAK_EXP_LOOP; }
        }
    }

DECODE_NAN:
    offset++;
    if (*(offset++) != 'a') goto SET_NAN_ERROR;
    if (*(offset++) != 'N') goto SET_NAN_ERROR;

    ds->lastType = JT_NULL;
    ds->start = offset;
    return ds->dec->newNull(ds->prv);

SET_NAN_ERROR:
    return SetError(ds, -1, "Unexpected character found when decoding 'NaN'");

DECODE_INF:
    offset++;
    if (*(offset++) != 'n') goto SET_INF_ERROR;
    if (*(offset++) != 'f') goto SET_INF_ERROR;
    if (*(offset++) != 'i') goto SET_INF_ERROR;
    if (*(offset++) != 'n') goto SET_INF_ERROR;
    if (*(offset++) != 'i') goto SET_INF_ERROR;
    if (*(offset++) != 't') goto SET_INF_ERROR;
    if (*(offset++) != 'y') goto SET_INF_ERROR;

    ds->start = offset;

    if (intNeg == 1) {
      ds->lastType = JT_POS_INF;
      return ds->dec->newPosInf(ds->prv);
    } else {
      ds->lastType = JT_NEG_INF;
      return ds->dec->newNegInf(ds->prv);
    }

SET_INF_ERROR:
    if (intNeg == 1) {
      const char *msg = "Unexpected character found when decoding 'Infinity'";
      return SetError(ds, -1, msg);
    } else {
      const char *msg = "Unexpected character found when decoding '-Infinity'";
      return SetError(ds, -1, msg);
    }


BREAK_EXP_LOOP:
    // FIXME: Check for arithmetic overflow here
    ds->lastType = JT_DOUBLE;
    ds->start = offset;
    return ds->dec->newDouble(
        ds->prv,
        createDouble((double)intNeg, (double)intValue, frcValue, decimalCount) *
            pow(10.0, expValue * expNeg));
}

JSOBJ FASTCALL_MSVC decode_true(struct DecoderState *ds) {
    char *offset = ds->start;
    offset++;

    if (*(offset++) != 'r') goto SETERROR;
    if (*(offset++) != 'u') goto SETERROR;
    if (*(offset++) != 'e') goto SETERROR;

    ds->lastType = JT_TRUE;
    ds->start = offset;
    return ds->dec->newTrue(ds->prv);

SETERROR:
    return SetError(ds, -1, "Unexpected character found when decoding 'true'");
}

JSOBJ FASTCALL_MSVC decode_false(struct DecoderState *ds) {
    char *offset = ds->start;
    offset++;

    if (*(offset++) != 'a') goto SETERROR;
    if (*(offset++) != 'l') goto SETERROR;
    if (*(offset++) != 's') goto SETERROR;
    if (*(offset++) != 'e') goto SETERROR;

    ds->lastType = JT_FALSE;
    ds->start = offset;
    return ds->dec->newFalse(ds->prv);

SETERROR:
    return SetError(ds, -1, "Unexpected character found when decoding 'false'");
}

JSOBJ FASTCALL_MSVC decode_null(struct DecoderState *ds) {
    char *offset = ds->start;
    offset++;

    if (*(offset++) != 'u') goto SETERROR;
    if (*(offset++) != 'l') goto SETERROR;
    if (*(offset++) != 'l') goto SETERROR;

    ds->lastType = JT_NULL;
    ds->start = offset;
    return ds->dec->newNull(ds->prv);

SETERROR:
    return SetError(ds, -1, "Unexpected character found when decoding 'null'");
}

void FASTCALL_MSVC SkipWhitespace(struct DecoderState *ds) {
    char *offset;

    for (offset = ds->start; (ds->end - offset) > 0; offset++) {
        switch (*offset) {
            case ' ':
            case '\t':
            case '\r':
            case '\n':
                break;

            default:
                ds->start = offset;
                return;
        }
    }

    if (offset == ds->end) {
        ds->start = ds->end;
    }
}

enum DECODESTRINGSTATE {
    DS_ISNULL = 0x32,
    DS_ISQUOTE,
    DS_ISESCAPE,
    DS_UTFLENERROR,
};

static const JSUINT8 g_decoderLookup[256] = {
    /* 0x00 */ DS_ISNULL,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x10 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x20 */ 1,
    1,
    DS_ISQUOTE,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x30 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x40 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x50 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    DS_ISESCAPE,
    1,
    1,
    1,
    /* 0x60 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x70 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x80 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0x90 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0xa0 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0xb0 */ 1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    /* 0xc0 */ 2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    /* 0xd0 */ 2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    /* 0xe0 */ 3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    /* 0xf0 */ 4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    DS_UTFLENERROR,
    DS_UTFLENERROR,
    DS_UTFLENERROR,
    DS_UTFLENERROR,
    DS_UTFLENERROR,
    DS_UTFLENERROR,
    DS_UTFLENERROR,
    DS_UTFLENERROR,
};

JSOBJ FASTCALL_MSVC decode_string(struct DecoderState *ds) {
    JSUTF16 sur[2] = {0};
    int iSur = 0;
    int index;
    wchar_t *escOffset;
    wchar_t *escStart;
    size_t escLen = (ds->escEnd - ds->escStart);
    JSUINT8 *inputOffset;
    JSUINT8 oct;
    JSUTF32 ucs;
    ds->lastType = JT_INVALID;
    ds->start++;

    if ((size_t)(ds->end - ds->start) > escLen) {
        size_t newSize = (ds->end - ds->start);

        if (ds->escHeap) {
            if (newSize > (SIZE_MAX / sizeof(wchar_t))) {
                return SetError(ds, -1, "Could not reserve memory block");
            }
            escStart = (wchar_t *)ds->dec->realloc(ds->escStart,
                                                   newSize * sizeof(wchar_t));
            if (!escStart) {
                ds->dec->free(ds->escStart);
                return SetError(ds, -1, "Could not reserve memory block");
            }
            ds->escStart = escStart;
        } else {
            wchar_t *oldStart = ds->escStart;
            if (newSize > (SIZE_MAX / sizeof(wchar_t))) {
                return SetError(ds, -1, "Could not reserve memory block");
            }
            ds->escStart =
                (wchar_t *)ds->dec->malloc(newSize * sizeof(wchar_t));
            if (!ds->escStart) {
                return SetError(ds, -1, "Could not reserve memory block");
            }
            ds->escHeap = 1;
            memcpy(ds->escStart, oldStart, escLen * sizeof(wchar_t));
        }

        ds->escEnd = ds->escStart + newSize;
    }

    escOffset = ds->escStart;
    inputOffset = (JSUINT8 *)ds->start;

    for (;;) {
        switch (g_decoderLookup[(JSUINT8)(*inputOffset)]) {
            case DS_ISNULL: {
                return SetError(ds, -1,
                                "Unmatched ''\"' when when decoding 'string'");
            }
            case DS_ISQUOTE: {
                ds->lastType = JT_UTF8;
                inputOffset++;
                ds->start += ((char *)inputOffset - (ds->start));
                return ds->dec->newString(ds->prv, ds->escStart, escOffset);
            }
            case DS_UTFLENERROR: {
                return SetError(
                    ds, -1,
                    "Invalid UTF-8 sequence length when decoding 'string'");
            }
            case DS_ISESCAPE:
                inputOffset++;
                switch (*inputOffset) {
                    case '\\':
                        *(escOffset++) = L'\\';
                        inputOffset++;
                        continue;
                    case '\"':
                        *(escOffset++) = L'\"';
                        inputOffset++;
                        continue;
                    case '/':
                        *(escOffset++) = L'/';
                        inputOffset++;
                        continue;
                    case 'b':
                        *(escOffset++) = L'\b';
                        inputOffset++;
                        continue;
                    case 'f':
                        *(escOffset++) = L'\f';
                        inputOffset++;
                        continue;
                    case 'n':
                        *(escOffset++) = L'\n';
                        inputOffset++;
                        continue;
                    case 'r':
                        *(escOffset++) = L'\r';
                        inputOffset++;
                        continue;
                    case 't':
                        *(escOffset++) = L'\t';
                        inputOffset++;
                        continue;

                    case 'u': {
                        int index;
                        inputOffset++;

                        for (index = 0; index < 4; index++) {
                            switch (*inputOffset) {
                                case '\0':
                                    return SetError(ds, -1,
                                                    "Unterminated unicode "
                                                    "escape sequence when "
                                                    "decoding 'string'");
                                default:
                                    return SetError(ds, -1,
                                                    "Unexpected character in "
                                                    "unicode escape sequence "
                                                    "when decoding 'string'");

                                case '0':
                                case '1':
                                case '2':
                                case '3':
                                case '4':
                                case '5':
                                case '6':
                                case '7':
                                case '8':
                                case '9':
                                    sur[iSur] = (sur[iSur] << 4) +
                                                (JSUTF16)(*inputOffset - '0');
                                    break;

                                case 'a':
                                case 'b':
                                case 'c':
                                case 'd':
                                case 'e':
                                case 'f':
                                    sur[iSur] = (sur[iSur] << 4) + 10 +
                                                (JSUTF16)(*inputOffset - 'a');
                                    break;

                                case 'A':
                                case 'B':
                                case 'C':
                                case 'D':
                                case 'E':
                                case 'F':
                                    sur[iSur] = (sur[iSur] << 4) + 10 +
                                                (JSUTF16)(*inputOffset - 'A');
                                    break;
                            }

                            inputOffset++;
                        }

                        if (iSur == 0) {
                            if ((sur[iSur] & 0xfc00) == 0xd800) {
                                // First of a surrogate pair, continue parsing
                                iSur++;
                                break;
                            }
                            (*escOffset++) = (wchar_t)sur[iSur];
                            iSur = 0;
                        } else {
                            // Decode pair
                            if ((sur[1] & 0xfc00) != 0xdc00) {
                                return SetError(ds, -1,
                                                "Unpaired high surrogate when "
                                                "decoding 'string'");
                            }
#if WCHAR_MAX == 0xffff
                            (*escOffset++) = (wchar_t)sur[0];
                            (*escOffset++) = (wchar_t)sur[1];
#else
                            (*escOffset++) =
                                (wchar_t)0x10000 +
                                (((sur[0] - 0xd800) << 10) | (sur[1] - 0xdc00));
#endif
                            iSur = 0;
                        }
                        break;
                    }

                    case '\0':
                        return SetError(ds, -1,
                                        "Unterminated escape sequence when "
                                        "decoding 'string'");
                    default:
                        return SetError(ds, -1,
                                        "Unrecognized escape sequence when "
                                        "decoding 'string'");
                }
                break;

            case 1: {
                *(escOffset++) = (wchar_t)(*inputOffset++);
                break;
            }

            case 2: {
                ucs = (*inputOffset++) & 0x1f;
                ucs <<= 6;
                if (((*inputOffset) & 0x80) != 0x80) {
                    return SetError(ds, -1,
                                    "Invalid octet in UTF-8 sequence when "
                                    "decoding 'string'");
                }
                ucs |= (*inputOffset++) & 0x3f;
                if (ucs < 0x80)
                    return SetError(ds, -1,
                                    "Overlong 2 byte UTF-8 sequence detected "
                                    "when decoding 'string'");
                *(escOffset++) = (wchar_t)ucs;
                break;
            }

            case 3: {
                JSUTF32 ucs = 0;
                ucs |= (*inputOffset++) & 0x0f;

                for (index = 0; index < 2; index++) {
                    ucs <<= 6;
                    oct = (*inputOffset++);

                    if ((oct & 0x80) != 0x80) {
                        return SetError(ds, -1,
                                        "Invalid octet in UTF-8 sequence when "
                                        "decoding 'string'");
                    }

                    ucs |= oct & 0x3f;
                }

                if (ucs < 0x800)
                    return SetError(ds, -1,
                                    "Overlong 3 byte UTF-8 sequence detected "
                                    "when encoding string");
                *(escOffset++) = (wchar_t)ucs;
                break;
            }

            case 4: {
                JSUTF32 ucs = 0;
                ucs |= (*inputOffset++) & 0x07;

                for (index = 0; index < 3; index++) {
                    ucs <<= 6;
                    oct = (*inputOffset++);

                    if ((oct & 0x80) != 0x80) {
                        return SetError(ds, -1,
                                        "Invalid octet in UTF-8 sequence when "
                                        "decoding 'string'");
                    }

                    ucs |= oct & 0x3f;
                }

                if (ucs < 0x10000)
                    return SetError(ds, -1,
                                    "Overlong 4 byte UTF-8 sequence detected "
                                    "when decoding 'string'");

#if WCHAR_MAX == 0xffff
                if (ucs >= 0x10000) {
                    ucs -= 0x10000;
                    *(escOffset++) = (wchar_t)(ucs >> 10) + 0xd800;
                    *(escOffset++) = (wchar_t)(ucs & 0x3ff) + 0xdc00;
                } else {
                    *(escOffset++) = (wchar_t)ucs;
                }
#else
                *(escOffset++) = (wchar_t)ucs;
#endif
                break;
            }
        }
    }
}

JSOBJ FASTCALL_MSVC decode_array(struct DecoderState *ds) {
    JSOBJ itemValue;
    JSOBJ newObj;
    int len;
    ds->objDepth++;
    if (ds->objDepth > JSON_MAX_OBJECT_DEPTH) {
        return SetError(ds, -1, "Reached object decoding depth limit");
    }

    newObj = ds->dec->newArray(ds->prv, ds->dec);
    len = 0;

    ds->lastType = JT_INVALID;
    ds->start++;

    for (;;) {
        SkipWhitespace(ds);

        if ((*ds->start) == ']') {
            ds->objDepth--;
            if (len == 0) {
                ds->start++;
                return ds->dec->endArray(ds->prv, newObj);
            }

            ds->dec->releaseObject(ds->prv, newObj, ds->dec);
            return SetError(
                ds, -1,
                "Unexpected character found when decoding array value (1)");
        }

        itemValue = decode_any(ds);

        if (itemValue == NULL) {
            ds->dec->releaseObject(ds->prv, newObj, ds->dec);
            return NULL;
        }

        if (!ds->dec->arrayAddItem(ds->prv, newObj, itemValue)) {
            ds->dec->releaseObject(ds->prv, newObj, ds->dec);
            return NULL;
        }

        SkipWhitespace(ds);

        switch (*(ds->start++)) {
            case ']': {
                ds->objDepth--;
                return ds->dec->endArray(ds->prv, newObj);
            }
            case ',':
                break;

            default:
                ds->dec->releaseObject(ds->prv, newObj, ds->dec);
                return SetError(
                    ds, -1,
                    "Unexpected character found when decoding array value (2)");
        }

        len++;
    }
}

JSOBJ FASTCALL_MSVC decode_object(struct DecoderState *ds) {
    JSOBJ itemName;
    JSOBJ itemValue;
    JSOBJ newObj;

    ds->objDepth++;
    if (ds->objDepth > JSON_MAX_OBJECT_DEPTH) {
        return SetError(ds, -1, "Reached object decoding depth limit");
    }

    newObj = ds->dec->newObject(ds->prv, ds->dec);

    ds->start++;

    for (;;) {
        SkipWhitespace(ds);

        if ((*ds->start) == '}') {
            ds->objDepth--;
            ds->start++;
            return ds->dec->endObject(ds->prv, newObj);
        }

        ds->lastType = JT_INVALID;
        itemName = decode_any(ds);

        if (itemName == NULL) {
            ds->dec->releaseObject(ds->prv, newObj, ds->dec);
            return NULL;
        }

        if (ds->lastType != JT_UTF8) {
            ds->dec->releaseObject(ds->prv, newObj, ds->dec);
            ds->dec->releaseObject(ds->prv, itemName, ds->dec);
            return SetError(
                ds, -1,
                "Key name of object must be 'string' when decoding 'object'");
        }

        SkipWhitespace(ds);

        if (*(ds->start++) != ':') {
            ds->dec->releaseObject(ds->prv, newObj, ds->dec);
            ds->dec->releaseObject(ds->prv, itemName, ds->dec);
            return SetError(ds, -1, "No ':' found when decoding object value");
        }

        SkipWhitespace(ds);

        itemValue = decode_any(ds);

        if (itemValue == NULL) {
            ds->dec->releaseObject(ds->prv, newObj, ds->dec);
            ds->dec->releaseObject(ds->prv, itemName, ds->dec);
            return NULL;
        }

        if (!ds->dec->objectAddKey(ds->prv, newObj, itemName, itemValue)) {
            ds->dec->releaseObject(ds->prv, newObj, ds->dec);
            ds->dec->releaseObject(ds->prv, itemName, ds->dec);
            ds->dec->releaseObject(ds->prv, itemValue, ds->dec);
            return NULL;
        }

        SkipWhitespace(ds);

        switch (*(ds->start++)) {
            case '}': {
                ds->objDepth--;
                return ds->dec->endObject(ds->prv, newObj);
            }
            case ',':
                break;

            default:
                ds->dec->releaseObject(ds->prv, newObj, ds->dec);
                return SetError(
                    ds, -1,
                    "Unexpected character found when decoding object value");
        }
    }
}

JSOBJ FASTCALL_MSVC decode_any(struct DecoderState *ds) {
    for (;;) {
        switch (*ds->start) {
            case '\"':
                return decode_string(ds);
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            case 'I':
            case 'N':
            case '-':
                return decode_numeric(ds);

            case '[':
                return decode_array(ds);
            case '{':
                return decode_object(ds);
            case 't':
                return decode_true(ds);
            case 'f':
                return decode_false(ds);
            case 'n':
                return decode_null(ds);

            case ' ':
            case '\t':
            case '\r':
            case '\n':
                // White space
                ds->start++;
                break;

            default:
                return SetError(ds, -1, "Expected object or value");
        }
    }
}

JSOBJ JSON_DecodeObject(JSONObjectDecoder *dec, const char *buffer,
                        size_t cbBuffer) {
    /*
    FIXME: Base the size of escBuffer of that of cbBuffer so that the unicode
    escaping doesn't run into the wall each time */
    char *locale;
    struct DecoderState ds;
    wchar_t escBuffer[(JSON_MAX_STACK_BUFFER_SIZE / sizeof(wchar_t))];
    JSOBJ ret;

    ds.start = (char *)buffer;
    ds.end = ds.start + cbBuffer;

    ds.escStart = escBuffer;
    ds.escEnd = ds.escStart + (JSON_MAX_STACK_BUFFER_SIZE / sizeof(wchar_t));
    ds.escHeap = 0;
    ds.prv = dec->prv;
    ds.dec = dec;
    ds.dec->errorStr = NULL;
    ds.dec->errorOffset = NULL;
    ds.objDepth = 0;

    ds.dec = dec;

    locale = setlocale(LC_NUMERIC, NULL);
    if (strcmp(locale, "C")) {
        locale = strdup(locale);
        if (!locale) {
            return SetError(&ds, -1, "Could not reserve memory block");
        }
        setlocale(LC_NUMERIC, "C");
        ret = decode_any(&ds);
        setlocale(LC_NUMERIC, locale);
        free(locale);
    } else {
        ret = decode_any(&ds);
    }

    if (ds.escHeap) {
        dec->free(ds.escStart);
    }

    SkipWhitespace(&ds);

    if (ds.start != ds.end && ret) {
        dec->releaseObject(ds.prv, ret, ds.dec);
        return SetError(&ds, -1, "Trailing data");
    }

    return ret;
}
