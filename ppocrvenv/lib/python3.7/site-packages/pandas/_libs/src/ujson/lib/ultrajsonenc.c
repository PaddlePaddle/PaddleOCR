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
#include <float.h>
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ultrajson.h"

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/*
Worst cases being:

Control characters (ASCII < 32)
0x00 (1 byte) input => \u0000 output (6 bytes)
1 * 6 => 6 (6 bytes required)

or UTF-16 surrogate pairs
4 bytes input in UTF-8 => \uXXXX\uYYYY (12 bytes).

4 * 6 => 24 bytes (12 bytes required)

The extra 2 bytes are for the quotes around the string

*/
#define RESERVE_STRING(_len) (2 + ((_len)*6))

static const double g_pow10[] = {1,
                                 10,
                                 100,
                                 1000,
                                 10000,
                                 100000,
                                 1000000,
                                 10000000,
                                 100000000,
                                 1000000000,
                                 10000000000,
                                 100000000000,
                                 1000000000000,
                                 10000000000000,
                                 100000000000000,
                                 1000000000000000};
static const char g_hexChars[] = "0123456789abcdef";
static const char g_escapeChars[] = "0123456789\\b\\t\\n\\f\\r\\\"\\\\\\/";

/*
FIXME: While this is fine dandy and working it's a magic value mess which
probably only the author understands.
Needs a cleanup and more documentation */

/*
Table for pure ascii output escaping all characters above 127 to \uXXXX */
static const JSUINT8 g_asciiOutputTable[256] = {
    /* 0x00 */ 0,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    10,
    12,
    14,
    30,
    16,
    18,
    30,
    30,
    /* 0x10 */ 30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    /* 0x20 */ 1,
    1,
    20,
    1,
    1,
    1,
    29,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    24,
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
    29,
    1,
    29,
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
    22,
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
    5,
    5,
    5,
    5,
    6,
    6,
    1,
    1};

static void SetError(JSOBJ obj, JSONObjectEncoder *enc, const char *message) {
    enc->errorMsg = message;
    enc->errorObj = obj;
}

/*
FIXME: Keep track of how big these get across several encoder calls and try to
make an estimate
That way we won't run our head into the wall each call */
void Buffer_Realloc(JSONObjectEncoder *enc, size_t cbNeeded) {
    size_t curSize = enc->end - enc->start;
    size_t newSize = curSize * 2;
    size_t offset = enc->offset - enc->start;

    while (newSize < curSize + cbNeeded) {
        newSize *= 2;
    }

    if (enc->heap) {
        enc->start = (char *)enc->realloc(enc->start, newSize);
        if (!enc->start) {
            SetError(NULL, enc, "Could not reserve memory block");
            return;
        }
    } else {
        char *oldStart = enc->start;
        enc->heap = 1;
        enc->start = (char *)enc->malloc(newSize);
        if (!enc->start) {
            SetError(NULL, enc, "Could not reserve memory block");
            return;
        }
        memcpy(enc->start, oldStart, offset);
    }
    enc->offset = enc->start + offset;
    enc->end = enc->start + newSize;
}

INLINE_PREFIX void FASTCALL_MSVC
Buffer_AppendShortHexUnchecked(char *outputOffset, unsigned short value) {
    *(outputOffset++) = g_hexChars[(value & 0xf000) >> 12];
    *(outputOffset++) = g_hexChars[(value & 0x0f00) >> 8];
    *(outputOffset++) = g_hexChars[(value & 0x00f0) >> 4];
    *(outputOffset++) = g_hexChars[(value & 0x000f) >> 0];
}

int Buffer_EscapeStringUnvalidated(JSONObjectEncoder *enc, const char *io,
                                   const char *end) {
    char *of = (char *)enc->offset;

    for (;;) {
        switch (*io) {
            case 0x00: {
                if (io < end) {
                    *(of++) = '\\';
                    *(of++) = 'u';
                    *(of++) = '0';
                    *(of++) = '0';
                    *(of++) = '0';
                    *(of++) = '0';
                    break;
                } else {
                    enc->offset += (of - enc->offset);
                    return TRUE;
                }
            }
            case '\"':
                (*of++) = '\\';
                (*of++) = '\"';
                break;
            case '\\':
                (*of++) = '\\';
                (*of++) = '\\';
                break;
            case '/':
                (*of++) = '\\';
                (*of++) = '/';
                break;
            case '\b':
                (*of++) = '\\';
                (*of++) = 'b';
                break;
            case '\f':
                (*of++) = '\\';
                (*of++) = 'f';
                break;
            case '\n':
                (*of++) = '\\';
                (*of++) = 'n';
                break;
            case '\r':
                (*of++) = '\\';
                (*of++) = 'r';
                break;
            case '\t':
                (*of++) = '\\';
                (*of++) = 't';
                break;

            case 0x26:  // '/'
            case 0x3c:  // '<'
            case 0x3e:  // '>'
            {
                if (enc->encodeHTMLChars) {
                    // Fall through to \u00XX case below.
                } else {
                    // Same as default case below.
                    (*of++) = (*io);
                    break;
                }
            }
            case 0x01:
            case 0x02:
            case 0x03:
            case 0x04:
            case 0x05:
            case 0x06:
            case 0x07:
            case 0x0b:
            case 0x0e:
            case 0x0f:
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17:
            case 0x18:
            case 0x19:
            case 0x1a:
            case 0x1b:
            case 0x1c:
            case 0x1d:
            case 0x1e:
            case 0x1f: {
                *(of++) = '\\';
                *(of++) = 'u';
                *(of++) = '0';
                *(of++) = '0';
                *(of++) = g_hexChars[(unsigned char)(((*io) & 0xf0) >> 4)];
                *(of++) = g_hexChars[(unsigned char)((*io) & 0x0f)];
                break;
            }
            default:
                (*of++) = (*io);
                break;
        }
        io++;
    }
}

int Buffer_EscapeStringValidated(JSOBJ obj, JSONObjectEncoder *enc,
                                 const char *io, const char *end) {
    JSUTF32 ucs;
    char *of = (char *)enc->offset;

    for (;;) {
        JSUINT8 utflen = g_asciiOutputTable[(unsigned char)*io];

        switch (utflen) {
            case 0: {
                if (io < end) {
                    *(of++) = '\\';
                    *(of++) = 'u';
                    *(of++) = '0';
                    *(of++) = '0';
                    *(of++) = '0';
                    *(of++) = '0';
                    io++;
                    continue;
                } else {
                    enc->offset += (of - enc->offset);
                    return TRUE;
                }
            }

            case 1: {
                *(of++) = (*io++);
                continue;
            }

            case 2: {
                JSUTF32 in;
                JSUTF16 in16;

                if (end - io < 1) {
                    enc->offset += (of - enc->offset);
                    SetError(
                        obj, enc,
                        "Unterminated UTF-8 sequence when encoding string");
                    return FALSE;
                }

                memcpy(&in16, io, sizeof(JSUTF16));
                in = (JSUTF32)in16;

#ifdef __LITTLE_ENDIAN__
                ucs = ((in & 0x1f) << 6) | ((in >> 8) & 0x3f);
#else
                ucs = ((in & 0x1f00) >> 2) | (in & 0x3f);
#endif

                if (ucs < 0x80) {
                    enc->offset += (of - enc->offset);
                    SetError(obj, enc,
                             "Overlong 2 byte UTF-8 sequence detected when "
                             "encoding string");
                    return FALSE;
                }

                io += 2;
                break;
            }

            case 3: {
                JSUTF32 in;
                JSUTF16 in16;
                JSUINT8 in8;

                if (end - io < 2) {
                    enc->offset += (of - enc->offset);
                    SetError(
                        obj, enc,
                        "Unterminated UTF-8 sequence when encoding string");
                    return FALSE;
                }

                memcpy(&in16, io, sizeof(JSUTF16));
                memcpy(&in8, io + 2, sizeof(JSUINT8));
#ifdef __LITTLE_ENDIAN__
                in = (JSUTF32)in16;
                in |= in8 << 16;
                ucs = ((in & 0x0f) << 12) | ((in & 0x3f00) >> 2) |
                      ((in & 0x3f0000) >> 16);
#else
                in = in16 << 8;
                in |= in8;
                ucs =
                    ((in & 0x0f0000) >> 4) | ((in & 0x3f00) >> 2) | (in & 0x3f);
#endif

                if (ucs < 0x800) {
                    enc->offset += (of - enc->offset);
                    SetError(obj, enc,
                             "Overlong 3 byte UTF-8 sequence detected when "
                             "encoding string");
                    return FALSE;
                }

                io += 3;
                break;
            }
            case 4: {
                JSUTF32 in;

                if (end - io < 3) {
                    enc->offset += (of - enc->offset);
                    SetError(
                        obj, enc,
                        "Unterminated UTF-8 sequence when encoding string");
                    return FALSE;
                }

                memcpy(&in, io, sizeof(JSUTF32));
#ifdef __LITTLE_ENDIAN__
                ucs = ((in & 0x07) << 18) | ((in & 0x3f00) << 4) |
                      ((in & 0x3f0000) >> 10) | ((in & 0x3f000000) >> 24);
#else
                ucs = ((in & 0x07000000) >> 6) | ((in & 0x3f0000) >> 4) |
                      ((in & 0x3f00) >> 2) | (in & 0x3f);
#endif
                if (ucs < 0x10000) {
                    enc->offset += (of - enc->offset);
                    SetError(obj, enc,
                             "Overlong 4 byte UTF-8 sequence detected when "
                             "encoding string");
                    return FALSE;
                }

                io += 4;
                break;
            }

            case 5:
            case 6: {
                enc->offset += (of - enc->offset);
                SetError(
                    obj, enc,
                    "Unsupported UTF-8 sequence length when encoding string");
                return FALSE;
            }

            case 29: {
                if (enc->encodeHTMLChars) {
                    // Fall through to \u00XX case 30 below.
                } else {
                    // Same as case 1 above.
                    *(of++) = (*io++);
                    continue;
                }
            }

            case 30: {
                // \uXXXX encode
                *(of++) = '\\';
                *(of++) = 'u';
                *(of++) = '0';
                *(of++) = '0';
                *(of++) = g_hexChars[(unsigned char)(((*io) & 0xf0) >> 4)];
                *(of++) = g_hexChars[(unsigned char)((*io) & 0x0f)];
                io++;
                continue;
            }
            case 10:
            case 12:
            case 14:
            case 16:
            case 18:
            case 20:
            case 22:
            case 24: {
                *(of++) = *((char *)(g_escapeChars + utflen + 0));
                *(of++) = *((char *)(g_escapeChars + utflen + 1));
                io++;
                continue;
            }
            // This can never happen, it's here to make L4 VC++ happy
            default: {
                ucs = 0;
                break;
            }
        }

        /*
        If the character is a UTF8 sequence of length > 1 we end up here */
        if (ucs >= 0x10000) {
            ucs -= 0x10000;
            *(of++) = '\\';
            *(of++) = 'u';
            Buffer_AppendShortHexUnchecked(
                of, (unsigned short)(ucs >> 10) + 0xd800);
            of += 4;

            *(of++) = '\\';
            *(of++) = 'u';
            Buffer_AppendShortHexUnchecked(
                of, (unsigned short)(ucs & 0x3ff) + 0xdc00);
            of += 4;
        } else {
            *(of++) = '\\';
            *(of++) = 'u';
            Buffer_AppendShortHexUnchecked(of, (unsigned short)ucs);
            of += 4;
        }
    }
}

#define Buffer_Reserve(__enc, __len) \
    if ( (size_t) ((__enc)->end - (__enc)->offset) < (size_t) (__len))  \
    {   \
      Buffer_Realloc((__enc), (__len));\
    }   \

#define Buffer_AppendCharUnchecked(__enc, __chr) *((__enc)->offset++) = __chr;

INLINE_PREFIX void FASTCALL_MSVC strreverse(char *begin,
                                                          char *end) {
    char aux;
    while (end > begin) aux = *end, *end-- = *begin, *begin++ = aux;
}

void Buffer_AppendIndentNewlineUnchecked(JSONObjectEncoder *enc) {
  if (enc->indent > 0) Buffer_AppendCharUnchecked(enc, '\n');
}

// This function could be refactored to only accept enc as an argument,
// but this is a straight vendor from ujson source
void Buffer_AppendIndentUnchecked(JSONObjectEncoder *enc, JSINT32 value) {
  int i;
  if (enc->indent > 0) {
    while (value-- > 0)
      for (i = 0; i < enc->indent; i++)
        Buffer_AppendCharUnchecked(enc, ' ');
  }
}

void Buffer_AppendIntUnchecked(JSONObjectEncoder *enc, JSINT32 value) {
    char *wstr;
    JSUINT32 uvalue = (value < 0) ? -value : value;
    wstr = enc->offset;

    // Conversion. Number is reversed.
    do {
        *wstr++ = (char)(48 + (uvalue % 10));
    } while (uvalue /= 10);
    if (value < 0) *wstr++ = '-';

    // Reverse string
    strreverse(enc->offset, wstr - 1);
    enc->offset += (wstr - (enc->offset));
}

void Buffer_AppendLongUnchecked(JSONObjectEncoder *enc, JSINT64 value) {
    char *wstr;
    JSUINT64 uvalue = (value < 0) ? -value : value;

    wstr = enc->offset;
    // Conversion. Number is reversed.

    do {
        *wstr++ = (char)(48 + (uvalue % 10ULL));
    } while (uvalue /= 10ULL);
    if (value < 0) *wstr++ = '-';

    // Reverse string
    strreverse(enc->offset, wstr - 1);
    enc->offset += (wstr - (enc->offset));
}

int Buffer_AppendDoubleUnchecked(JSOBJ obj, JSONObjectEncoder *enc,
                                 double value) {
    /* if input is beyond the thresholds, revert to exponential */
    const double thres_max = (double)1e16 - 1;
    const double thres_min = (double)1e-15;
    char precision_str[20];
    int count;
    double diff = 0.0;
    char *str = enc->offset;
    char *wstr = str;
    unsigned long long whole;
    double tmp;
    unsigned long long frac;
    int neg;
    double pow10;

    if (value == HUGE_VAL || value == -HUGE_VAL) {
        SetError(obj, enc, "Invalid Inf value when encoding double");
        return FALSE;
    }

    if (!(value == value)) {
        SetError(obj, enc, "Invalid Nan value when encoding double");
        return FALSE;
    }

    /* we'll work in positive values and deal with the
    negative sign issue later */
    neg = 0;
    if (value < 0) {
        neg = 1;
        value = -value;
    }

    /*
    for very large or small numbers switch back to native sprintf for
    exponentials.  anyone want to write code to replace this? */
    if (value > thres_max || (value != 0.0 && fabs(value) < thres_min)) {
        precision_str[0] = '%';
        precision_str[1] = '.';
#if defined(_WIN32) && defined(_MSC_VER)
        sprintf_s(precision_str + 2, sizeof(precision_str) - 2, "%ug",
                  enc->doublePrecision);
        enc->offset += sprintf_s(str, enc->end - enc->offset, precision_str,
                                 neg ? -value : value);
#else
        snprintf(precision_str + 2, sizeof(precision_str) - 2, "%ug",
                 enc->doublePrecision);
        enc->offset += snprintf(str, enc->end - enc->offset, precision_str,
                                neg ? -value : value);
#endif
        return TRUE;
    }

    pow10 = g_pow10[enc->doublePrecision];

    whole = (unsigned long long)value;
    tmp = (value - whole) * pow10;
    frac = (unsigned long long)(tmp);
    diff = tmp - frac;

    if (diff > 0.5) {
        ++frac;
    } else if (diff == 0.5 && ((frac == 0) || (frac & 1))) {
        /* if halfway, round up if odd, OR
        if last digit is 0.  That last part is strange */
        ++frac;
    }

    // handle rollover, e.g.
    // case 0.99 with prec 1 is 1.0 and case 0.95 with prec is 1.0 as well
    if (frac >= pow10) {
        frac = 0;
        ++whole;
    }

    if (enc->doublePrecision == 0) {
        diff = value - whole;

        if (diff > 0.5) {
            /* greater than 0.5, round up, e.g. 1.6 -> 2 */
            ++whole;
        } else if (diff == 0.5 && (whole & 1)) {
            /* exactly 0.5 and ODD, then round up */
            /* 1.5 -> 2, but 2.5 -> 2 */
            ++whole;
        }

        // vvvvvvvvvvvvvvvvvvv  Diff from modp_dto2
    } else if (frac) {
        count = enc->doublePrecision;
        // now do fractional part, as an unsigned number
        // we know it is not 0 but we can have leading zeros, these
        // should be removed
        while (!(frac % 10)) {
            --count;
            frac /= 10;
        }
        //^^^^^^^^^^^^^^^^^^^  Diff from modp_dto2

        // now do fractional part, as an unsigned number
        do {
            --count;
            *wstr++ = (char)(48 + (frac % 10));
        } while (frac /= 10);
        // add extra 0s
        while (count-- > 0) {
            *wstr++ = '0';
        }
        // add decimal
        *wstr++ = '.';
    } else {
        *wstr++ = '0';
        *wstr++ = '.';
    }

    // Do whole part. Take care of sign
    // conversion. Number is reversed.
    do {
        *wstr++ = (char)(48 + (whole % 10));
    } while (whole /= 10);

    if (neg) {
        *wstr++ = '-';
    }
    strreverse(str, wstr - 1);
    enc->offset += (wstr - (enc->offset));

    return TRUE;
}

/*
FIXME:
Handle integration functions returning NULL here */

/*
FIXME:
Perhaps implement recursion detection */

void encode(JSOBJ obj, JSONObjectEncoder *enc, const char *name,
            size_t cbName) {
    const char *value;
    char *objName;
    int count;
    JSOBJ iterObj;
    size_t szlen;
    JSONTypeContext tc;
    tc.encoder = enc;

    if (enc->level > enc->recursionMax) {
        SetError(obj, enc, "Maximum recursion level reached");
        return;
    }

    /*
    This reservation must hold

    length of _name as encoded worst case +
    maxLength of double to string OR maxLength of JSLONG to string
    */

    Buffer_Reserve(enc, 256 + RESERVE_STRING(cbName));
    if (enc->errorMsg) {
        return;
    }

    if (name) {
        Buffer_AppendCharUnchecked(enc, '\"');

        if (enc->forceASCII) {
            if (!Buffer_EscapeStringValidated(obj, enc, name, name + cbName)) {
                return;
            }
        } else {
            if (!Buffer_EscapeStringUnvalidated(enc, name, name + cbName)) {
                return;
            }
        }

        Buffer_AppendCharUnchecked(enc, '\"');

        Buffer_AppendCharUnchecked(enc, ':');
#ifndef JSON_NO_EXTRA_WHITESPACE
        Buffer_AppendCharUnchecked(enc, ' ');
#endif
    }

    enc->beginTypeContext(obj, &tc);

    switch (tc.type) {
        case JT_INVALID: {
            return;
        }

        case JT_ARRAY: {
            count = 0;
            enc->iterBegin(obj, &tc);

            Buffer_AppendCharUnchecked(enc, '[');
            Buffer_AppendIndentNewlineUnchecked(enc);

            while (enc->iterNext(obj, &tc)) {
                if (count > 0) {
                    Buffer_AppendCharUnchecked(enc, ',');
#ifndef JSON_NO_EXTRA_WHITESPACE
                    Buffer_AppendCharUnchecked(buffer, ' ');
#endif
                    Buffer_AppendIndentNewlineUnchecked(enc);
                }

                iterObj = enc->iterGetValue(obj, &tc);

                enc->level++;
                Buffer_AppendIndentUnchecked(enc, enc->level);
                encode(iterObj, enc, NULL, 0);
                count++;
            }

            enc->iterEnd(obj, &tc);
            Buffer_AppendIndentNewlineUnchecked(enc);
            Buffer_AppendIndentUnchecked(enc, enc->level);
            Buffer_AppendCharUnchecked(enc, ']');
            break;
        }

        case JT_OBJECT: {
            count = 0;
            enc->iterBegin(obj, &tc);

            Buffer_AppendCharUnchecked(enc, '{');
            Buffer_AppendIndentNewlineUnchecked(enc);

            while (enc->iterNext(obj, &tc)) {
                if (count > 0) {
                    Buffer_AppendCharUnchecked(enc, ',');
#ifndef JSON_NO_EXTRA_WHITESPACE
                    Buffer_AppendCharUnchecked(enc, ' ');
#endif
                    Buffer_AppendIndentNewlineUnchecked(enc);
                }

                iterObj = enc->iterGetValue(obj, &tc);
                objName = enc->iterGetName(obj, &tc, &szlen);

                enc->level++;
                Buffer_AppendIndentUnchecked(enc, enc->level);
                encode(iterObj, enc, objName, szlen);
                count++;
            }

            enc->iterEnd(obj, &tc);
            Buffer_AppendIndentNewlineUnchecked(enc);
            Buffer_AppendIndentUnchecked(enc, enc->level);
            Buffer_AppendCharUnchecked(enc, '}');
            break;
        }

        case JT_LONG: {
            Buffer_AppendLongUnchecked(enc, enc->getLongValue(obj, &tc));
            break;
        }

        case JT_INT: {
            Buffer_AppendIntUnchecked(enc, enc->getIntValue(obj, &tc));
            break;
        }

        case JT_TRUE: {
            Buffer_AppendCharUnchecked(enc, 't');
            Buffer_AppendCharUnchecked(enc, 'r');
            Buffer_AppendCharUnchecked(enc, 'u');
            Buffer_AppendCharUnchecked(enc, 'e');
            break;
        }

        case JT_FALSE: {
            Buffer_AppendCharUnchecked(enc, 'f');
            Buffer_AppendCharUnchecked(enc, 'a');
            Buffer_AppendCharUnchecked(enc, 'l');
            Buffer_AppendCharUnchecked(enc, 's');
            Buffer_AppendCharUnchecked(enc, 'e');
            break;
        }

        case JT_NULL: {
            Buffer_AppendCharUnchecked(enc, 'n');
            Buffer_AppendCharUnchecked(enc, 'u');
            Buffer_AppendCharUnchecked(enc, 'l');
            Buffer_AppendCharUnchecked(enc, 'l');
            break;
        }

        case JT_DOUBLE: {
            if (!Buffer_AppendDoubleUnchecked(obj, enc,
                                              enc->getDoubleValue(obj, &tc))) {
                enc->endTypeContext(obj, &tc);
                enc->level--;
                return;
            }
            break;
        }

        case JT_UTF8: {
            value = enc->getStringValue(obj, &tc, &szlen);
            Buffer_Reserve(enc, RESERVE_STRING(szlen));
            if (enc->errorMsg) {
                enc->endTypeContext(obj, &tc);
                return;
            }
            Buffer_AppendCharUnchecked(enc, '\"');

            if (enc->forceASCII) {
                if (!Buffer_EscapeStringValidated(obj, enc, value,
                                                  value + szlen)) {
                    enc->endTypeContext(obj, &tc);
                    enc->level--;
                    return;
                }
            } else {
                if (!Buffer_EscapeStringUnvalidated(enc, value,
                                                    value + szlen)) {
                    enc->endTypeContext(obj, &tc);
                    enc->level--;
                    return;
                }
            }

            Buffer_AppendCharUnchecked(enc, '\"');
            break;
        }

        case JT_BIGNUM: {
            value = enc->getBigNumStringValue(obj, &tc, &szlen);

            Buffer_Reserve(enc, RESERVE_STRING(szlen));
            if (enc->errorMsg) {
                enc->endTypeContext(obj, &tc);
                return;
            }

            if (enc->forceASCII) {
                if (!Buffer_EscapeStringValidated(obj, enc, value,
                                                  value + szlen)) {
                    enc->endTypeContext(obj, &tc);
                    enc->level--;
                    return;
                }
            } else {
                if (!Buffer_EscapeStringUnvalidated(enc, value,
                                                    value + szlen)) {
                    enc->endTypeContext(obj, &tc);
                    enc->level--;
                    return;
                }
            }

            break;
        }
    }

    enc->endTypeContext(obj, &tc);
    enc->level--;
}

char *JSON_EncodeObject(JSOBJ obj, JSONObjectEncoder *enc, char *_buffer,
                        size_t _cbBuffer) {
    char *locale;
    enc->malloc = enc->malloc ? enc->malloc : malloc;
    enc->free = enc->free ? enc->free : free;
    enc->realloc = enc->realloc ? enc->realloc : realloc;
    enc->errorMsg = NULL;
    enc->errorObj = NULL;
    enc->level = 0;

    if (enc->recursionMax < 1) {
        enc->recursionMax = JSON_MAX_RECURSION_DEPTH;
    }

    if (enc->doublePrecision < 0 ||
        enc->doublePrecision > JSON_DOUBLE_MAX_DECIMALS) {
        enc->doublePrecision = JSON_DOUBLE_MAX_DECIMALS;
    }

    if (_buffer == NULL) {
        _cbBuffer = 32768;
        enc->start = (char *)enc->malloc(_cbBuffer);
        if (!enc->start) {
            SetError(obj, enc, "Could not reserve memory block");
            return NULL;
        }
        enc->heap = 1;
    } else {
        enc->start = _buffer;
        enc->heap = 0;
    }

    enc->end = enc->start + _cbBuffer;
    enc->offset = enc->start;

    locale = setlocale(LC_NUMERIC, NULL);
    if (strcmp(locale, "C")) {
        locale = strdup(locale);
        if (!locale) {
            SetError(NULL, enc, "Could not reserve memory block");
            return NULL;
        }
        setlocale(LC_NUMERIC, "C");
        encode(obj, enc, NULL, 0);
        setlocale(LC_NUMERIC, locale);
        free(locale);
    } else {
        encode(obj, enc, NULL, 0);
    }

    Buffer_Reserve(enc, 1);
    if (enc->errorMsg) {
        return NULL;
    }
    Buffer_AppendCharUnchecked(enc, '\0');

    return enc->start;
}
