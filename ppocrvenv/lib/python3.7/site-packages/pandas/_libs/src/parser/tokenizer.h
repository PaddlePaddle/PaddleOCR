/*

Copyright (c) 2012, Lambda Foundry, Inc., except where noted

Incorporates components of WarrenWeckesser/textreader, licensed under 3-clause
BSD

See LICENSE for the license

*/

#ifndef PANDAS__LIBS_SRC_PARSER_TOKENIZER_H_
#define PANDAS__LIBS_SRC_PARSER_TOKENIZER_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define ERROR_NO_DIGITS 1
#define ERROR_OVERFLOW 2
#define ERROR_INVALID_CHARS 3

#include "../headers/stdint.h"
#include "../inline_helper.h"
#include "../headers/portable.h"

#include "khash.h"

#define STREAM_INIT_SIZE 32

#define REACHED_EOF 1
#define CALLING_READ_FAILED 2


/*

  C flat file parsing low level code for pandas / NumPy

 */

/*
 *  Common set of error types for the read_rows() and tokenize()
 *  functions.
 */

// #define VERBOSE
#if defined(VERBOSE)
#define TRACE(X) printf X;
#else
#define TRACE(X)
#endif  // VERBOSE

#define PARSER_OUT_OF_MEMORY -1

/*
 *  TODO: Might want to couple count_rows() with read_rows() to avoid
 *        duplication of some file I/O.
 */

typedef enum {
    START_RECORD,
    START_FIELD,
    ESCAPED_CHAR,
    IN_FIELD,
    IN_QUOTED_FIELD,
    ESCAPE_IN_QUOTED_FIELD,
    QUOTE_IN_QUOTED_FIELD,
    EAT_CRNL,
    EAT_CRNL_NOP,
    EAT_WHITESPACE,
    EAT_COMMENT,
    EAT_LINE_COMMENT,
    WHITESPACE_LINE,
    START_FIELD_IN_SKIP_LINE,
    IN_FIELD_IN_SKIP_LINE,
    IN_QUOTED_FIELD_IN_SKIP_LINE,
    QUOTE_IN_QUOTED_FIELD_IN_SKIP_LINE,
    FINISHED
} ParserState;

typedef enum {
    QUOTE_MINIMAL,
    QUOTE_ALL,
    QUOTE_NONNUMERIC,
    QUOTE_NONE
} QuoteStyle;

typedef enum {
    ERROR,
    WARN,
    SKIP
} BadLineHandleMethod;

typedef void *(*io_callback)(void *src, size_t nbytes, size_t *bytes_read,
                             int *status, const char *encoding_errors);
typedef int (*io_cleanup)(void *src);

typedef struct parser_t {
    void *source;
    io_callback cb_io;
    io_cleanup cb_cleanup;

    int64_t chunksize;      // Number of bytes to prepare for each chunk
    char *data;             // pointer to data to be processed
    int64_t datalen;        // amount of data available
    int64_t datapos;

    // where to write out tokenized data
    char *stream;
    uint64_t stream_len;
    uint64_t stream_cap;

    // Store words in (potentially ragged) matrix for now, hmm
    char **words;
    int64_t *word_starts;   // where we are in the stream
    uint64_t words_len;
    uint64_t words_cap;
    uint64_t max_words_cap;  // maximum word cap encountered

    char *pword_start;      // pointer to stream start of current field
    int64_t word_start;     // position start of current field

    int64_t *line_start;    // position in words for start of line
    int64_t *line_fields;   // Number of fields in each line
    uint64_t lines;         // Number of (good) lines observed
    uint64_t file_lines;    // Number of lines (including bad or skipped)
    uint64_t lines_cap;     // Vector capacity

    // Tokenizing stuff
    ParserState state;
    int doublequote;      /* is " represented by ""? */
    char delimiter;       /* field separator */
    int delim_whitespace; /* delimit by consuming space/tabs instead */
    char quotechar;       /* quote character */
    char escapechar;      /* escape character */
    char lineterminator;
    int skipinitialspace; /* ignore spaces following delimiter? */
    int quoting;          /* style of quoting to write */

    char commentchar;
    int allow_embedded_newline;

    int usecols;  // Boolean: 1: usecols provided, 0: none provided

    int expected_fields;
    BadLineHandleMethod on_bad_lines;

    // floating point options
    char decimal;
    char sci;

    // thousands separator (comma, period)
    char thousands;

    int header;            // Boolean: 1: has header, 0: no header
    int64_t header_start;  // header row start
    uint64_t header_end;   // header row end

    void *skipset;
    PyObject *skipfunc;
    int64_t skip_first_N_rows;
    int64_t skip_footer;
    double (*double_converter)(const char *, char **,
                               char, char, char, int, int *, int *);

    // error handling
    char *warn_msg;
    char *error_msg;

    int skip_empty_lines;
} parser_t;

typedef struct coliter_t {
    char **words;
    int64_t *line_start;
    int64_t col;
} coliter_t;

void coliter_setup(coliter_t *self, parser_t *parser, int i, int start);

#define COLITER_NEXT(iter, word)                           \
    do {                                                   \
        const int64_t i = *iter.line_start++ + iter.col;   \
        word = i >= *iter.line_start ? "" : iter.words[i]; \
    } while (0)

parser_t *parser_new(void);

int parser_init(parser_t *self);

int parser_consume_rows(parser_t *self, size_t nrows);

int parser_trim_buffers(parser_t *self);

int parser_add_skiprow(parser_t *self, int64_t row);

int parser_set_skipfirstnrows(parser_t *self, int64_t nrows);

void parser_free(parser_t *self);

void parser_del(parser_t *self);

void parser_set_default_options(parser_t *self);

int tokenize_nrows(parser_t *self, size_t nrows, const char *encoding_errors);

int tokenize_all_rows(parser_t *self, const char *encoding_errors);

// Have parsed / type-converted a chunk of data
// and want to free memory from the token stream

typedef struct uint_state {
    int seen_sint;
    int seen_uint;
    int seen_null;
} uint_state;

void uint_state_init(uint_state *self);

int uint64_conflict(uint_state *self);

uint64_t str_to_uint64(uint_state *state, const char *p_item, int64_t int_max,
                       uint64_t uint_max, int *error, char tsep);
int64_t str_to_int64(const char *p_item, int64_t int_min, int64_t int_max,
                     int *error, char tsep);
double xstrtod(const char *p, char **q, char decimal, char sci, char tsep,
               int skip_trailing, int *error, int *maybe_int);
double precise_xstrtod(const char *p, char **q, char decimal,
                       char sci, char tsep, int skip_trailing,
                       int *error, int *maybe_int);

// GH-15140 - round_trip requires and acquires the GIL on its own
double round_trip(const char *p, char **q, char decimal, char sci, char tsep,
                  int skip_trailing, int *error, int *maybe_int);
int to_boolean(const char *item, uint8_t *val);

#endif  // PANDAS__LIBS_SRC_PARSER_TOKENIZER_H_
