/* @(#) $Id: Levenshtein.h,v 1.22 2005/01/13 20:02:56 yeti Exp $ */
#ifndef LEVENSHTEIN_H
#define LEVENSHTEIN_H

#ifndef size_t
#  include <stdlib.h>
#endif

/* A bit dirty. */
#ifndef _LEV_STATIC_PY
#  define _LEV_STATIC_PY /* */
#endif

/* In C, this is just wchar_t and unsigned char, in Python, lev_wchar can
 * be anything.  If you really want to cheat, define wchar_t to any integer
 * type you like before including Levenshtein.h and recompile it. */
#ifndef lev_wchar
#  ifndef wchar_t
#    include <wchar.h>
#  endif
#  define lev_wchar wchar_t
#endif
typedef unsigned char lev_byte;

/* Edit opration type
 * DON'T CHANGE! used ad arrays indices and the bits are occasionally used
 * as flags */
typedef enum {
  LEV_EDIT_KEEP = 0,
  LEV_EDIT_REPLACE = 1,
  LEV_EDIT_INSERT = 2,
  LEV_EDIT_DELETE = 3,
  LEV_EDIT_LAST  /* sometimes returned when an error occurs */
} LevEditType;

/* Error codes returned by editop check functions */
typedef enum {
  LEV_EDIT_ERR_OK = 0,
  LEV_EDIT_ERR_TYPE,  /* nonexistent edit type */
  LEV_EDIT_ERR_OUT,  /* edit out of string bounds */
  LEV_EDIT_ERR_ORDER,  /* ops are not ordered */
  LEV_EDIT_ERR_BLOCK,  /* incosistent block boundaries (block ops) */
  LEV_EDIT_ERR_SPAN,  /* sequence is not a full transformation (block ops) */
  LEV_EDIT_ERR_LAST
} LevEditOpError;

/* string averaging method (UNUSED yet) */
typedef enum {
  LEV_AVG_HEAD = 0,  /* take operations from the head */
  LEV_AVG_TAIL,  /* take operations from the tail */
  LEV_AVG_SPREAD,  /* take a equidistantly distributed subset */
  LEV_AVG_BLOCK,  /* take a random continuous block */
  LEV_AVG_RANDOM,  /* take a random subset */
  LEV_AVG_LAST
} LevAveragingType;

/* Edit operation (atomic).
 * This is the `native' atomic edit operation.  It differs from the difflib
 * one's because it represents a change of one character, not a block.  And
 * we usually don't care about LEV_EDIT_KEEP, though the functions can handle
 * them.  The positions are interpreted as at the left edge of a character.
 */
typedef struct {
  LevEditType type;  /* editing operation type */
  size_t spos;  /* source block position */
  size_t dpos;  /* destination position */
} LevEditOp;

/* Edit operation (difflib-compatible).
 * This is not `native', but conversion functions exist.  These fields exactly
 * correspond to the codeops() tuples fields (and this method is also the
 * source of the silly OpCode name).  Sequences must span over complete
 * strings, subsequences are simply edit sequences with more (or larger)
 * LEV_EDIT_KEEP blocks.
 */
typedef struct {
  LevEditType type;  /* editing operation type */
  size_t sbeg, send;  /* source block begin, end */
  size_t dbeg, dend;  /* destination block begin, end */
} LevOpCode;

/* Matching block (difflib-compatible). */
typedef struct {
  size_t spos;
  size_t dpos;
  size_t len;
} LevMatchingBlock;

_LEV_STATIC_PY
size_t
lev_edit_distance(size_t len1,
                  const lev_byte *string1,
                  size_t len2,
                  const lev_byte *string2,
                  int xcost);

_LEV_STATIC_PY
size_t
lev_u_edit_distance(size_t len1,
                    const lev_wchar *string1,
                    size_t len2,
                    const lev_wchar *string2,
                    int xcost);

_LEV_STATIC_PY
size_t
lev_hamming_distance(size_t len,
                     const lev_byte *string1,
                     const lev_byte *string2);

_LEV_STATIC_PY
size_t
lev_u_hamming_distance(size_t len,
                       const lev_wchar *string1,
                       const lev_wchar *string2);

_LEV_STATIC_PY
double
lev_jaro_ratio(size_t len1,
               const lev_byte *string1,
               size_t len2,
               const lev_byte *string2);

_LEV_STATIC_PY
double
lev_u_jaro_ratio(size_t len1,
                 const lev_wchar *string1,
                 size_t len2,
                 const lev_wchar *string2);

_LEV_STATIC_PY
double
lev_jaro_winkler_ratio(size_t len1, const lev_byte *string1,
                       size_t len2, const lev_byte *string2,
                       double pfweight);

_LEV_STATIC_PY
double
lev_u_jaro_winkler_ratio(size_t len1, const lev_wchar *string1,
                         size_t len2, const lev_wchar *string2,
                         double pfweight);

_LEV_STATIC_PY
lev_byte*
lev_greedy_median(size_t n,
                  const size_t *lengths,
                  const lev_byte *strings[],
                  const double *weights,
                  size_t *medlength);

_LEV_STATIC_PY
lev_wchar*
lev_u_greedy_median(size_t n,
                    const size_t *lengths,
                    const lev_wchar *strings[],
                    const double *weights,
                    size_t *medlength);

_LEV_STATIC_PY
lev_byte*
lev_median_improve(size_t len, const lev_byte *s,
                   size_t n, const size_t *lengths,
                   const lev_byte *strings[],
                   const double *weights,
                   size_t *medlength);

_LEV_STATIC_PY
lev_wchar*
lev_u_median_improve(size_t len, const lev_wchar *s,
                     size_t n, const size_t *lengths,
                     const lev_wchar *strings[],
                     const double *weights,
                     size_t *medlength);

_LEV_STATIC_PY
lev_byte*
lev_quick_median(size_t n,
                 const size_t *lengths,
                 const lev_byte *strings[],
                 const double *weights,
                 size_t *medlength);

_LEV_STATIC_PY
lev_wchar*
lev_u_quick_median(size_t n,
                   const size_t *lengths,
                   const lev_wchar *strings[],
                   const double *weights,
                   size_t *medlength);

_LEV_STATIC_PY
lev_byte*
lev_set_median(size_t n,
               const size_t *lengths,
               const lev_byte *strings[],
               const double *weights,
               size_t *medlength);

_LEV_STATIC_PY
size_t
lev_set_median_index(size_t n, const size_t *lengths,
                     const lev_byte *strings[],
                     const double *weights);

_LEV_STATIC_PY
lev_wchar*
lev_u_set_median(size_t n,
                 const size_t *lengths,
                 const lev_wchar *strings[],
                 const double *weights,
                 size_t *medlength);

_LEV_STATIC_PY
size_t
lev_u_set_median_index(size_t n, const size_t *lengths,
                       const lev_wchar *strings[],
                       const double *weights);

_LEV_STATIC_PY
double
lev_edit_seq_distance(size_t n1,
                      const size_t *lengths1,
                      const lev_byte *strings1[],
                      size_t n2,
                      const size_t *lengths2,
                      const lev_byte *strings2[]);

_LEV_STATIC_PY
double
lev_u_edit_seq_distance(size_t n1,
                        const size_t *lengths1,
                        const lev_wchar *strings1[],
                        size_t n2,
                        const size_t *lengths2,
                        const lev_wchar *strings2[]);

_LEV_STATIC_PY
double
lev_set_distance(size_t n1,
                 const size_t *lengths1,
                 const lev_byte *strings1[],
                 size_t n2,
                 const size_t *lengths2,
                 const lev_byte *strings2[]);

_LEV_STATIC_PY
double
lev_u_set_distance(size_t n1,
                   const size_t *lengths1,
                   const lev_wchar *strings1[],
                   size_t n2,
                   const size_t *lengths2,
                   const lev_wchar *strings2[]);

_LEV_STATIC_PY
int
lev_editops_check_errors(size_t len1,
                         size_t len2,
                         size_t n,
                         const LevEditOp *ops);

_LEV_STATIC_PY
int
lev_opcodes_check_errors(size_t len1,
                         size_t len2,
                         size_t nb,
                         const LevOpCode *bops);

_LEV_STATIC_PY
void
lev_editops_invert(size_t n,
                   LevEditOp *ops);

_LEV_STATIC_PY
void
lev_opcodes_invert(size_t nb,
                   LevOpCode *bops);

_LEV_STATIC_PY
LevMatchingBlock*
lev_editops_matching_blocks(size_t len1,
                            size_t len2,
                            size_t n,
                            const LevEditOp *ops,
                            size_t *nmblocks);

_LEV_STATIC_PY
LevMatchingBlock*
lev_opcodes_matching_blocks(size_t len1,
                            size_t len2,
                            size_t nb,
                            const LevOpCode *bops,
                            size_t *nmblocks);

_LEV_STATIC_PY
lev_byte*
lev_editops_apply(size_t len1,
                  const lev_byte* string1,
                  size_t len2,
                  const lev_byte* string2,
                  size_t n,
                  const LevEditOp *ops,
                  size_t *len);

_LEV_STATIC_PY
lev_wchar*
lev_u_editops_apply(size_t len1,
                    const lev_wchar* string1,
                    size_t len2,
                    const lev_wchar* string2,
                    size_t n,
                    const LevEditOp *ops,
                    size_t *len);

_LEV_STATIC_PY
lev_byte*
lev_opcodes_apply(size_t len1,
                  const lev_byte* string1,
                  size_t len2,
                  const lev_byte* string2,
                  size_t nb,
                  const LevOpCode *bops,
                  size_t *len);

_LEV_STATIC_PY
lev_wchar*
lev_u_opcodes_apply(size_t len1,
                    const lev_wchar* string1,
                    size_t len2,
                    const lev_wchar* string2,
                    size_t nb,
                    const LevOpCode *bops,
                    size_t *len);

_LEV_STATIC_PY
LevEditOp*
lev_editops_find(size_t len1,
                 const lev_byte *string1,
                 size_t len2,
                 const lev_byte *string2,
                 size_t *n);

_LEV_STATIC_PY
LevEditOp*
lev_u_editops_find(size_t len1,
                   const lev_wchar *string1,
                   size_t len2,
                   const lev_wchar *string2,
                   size_t *n);

_LEV_STATIC_PY
LevEditOp*
lev_opcodes_to_editops(size_t nb,
                       const LevOpCode *bops,
                       size_t *n,
                       int keepkeep);

_LEV_STATIC_PY
LevOpCode*
lev_editops_to_opcodes(size_t n,
                       const LevEditOp *ops,
                       size_t *nb,
                       size_t len1,
                       size_t len2);

_LEV_STATIC_PY
size_t
lev_editops_total_cost(size_t n,
                       const LevEditOp *ops);

_LEV_STATIC_PY
size_t
lev_opcodes_total_cost(size_t nb,
                       const LevOpCode *bops);

_LEV_STATIC_PY
LevEditOp*
lev_editops_normalize(size_t n,
                      const LevEditOp *ops,
                      size_t *nnorm);

_LEV_STATIC_PY LevEditOp*
lev_editops_subtract(size_t n,
                     const LevEditOp *ops,
                     size_t ns,
                     const LevEditOp *sub,
                     size_t *nrem);

/* UNUSED yet */
_LEV_STATIC_PY
void
lev_init_rng(unsigned long int seed);

#endif /* not LEVENSHTEIN_H */
