/*
 * Levenshtein.c
 * @(#) $Id: Levenshtein.c,v 1.41 2005/01/13 20:05:36 yeti Exp $
 * Python extension computing Levenshtein distances, string similarities,
 * median strings and other goodies.
 *
 * Copyright (C) 2002-2003 David Necas (Yeti) <yeti@physics.muni.cz>.
 *
 * The Taus113 random generator:
 * Copyright (C) 2002 Atakan Gurkan
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 James Theiler, Brian Gough
 * (see below for more)
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 **/

/**
 * TODO:
 *
 * - Implement weighted string averaging, see:
 *   H. Bunke et. al.: On the Weighted Mean of a Pair of Strings,
 *         Pattern Analysis and Applications 2002, 5(1): 23-30.
 *   X. Jiang et. al.: Dynamic Computations of Generalized Median Strings,
 *         Pattern Analysis and Applications 2002, ???.
 *   The latter also contains an interesting median-search algorithm.
 *
 * - Deal with stray symbols in greedy median() and median_improve().
 *   There are two possibilities:
 *    (i) Remember which strings contain which symbols.  This allows certain
 *        small optimizations when processing them.
 *   (ii) Use some overall heuristics to find symbols which don't worth
 *        trying.  This is very appealing, but hard to do properly
 *        (requires some inequality strong enough to allow practical exclusion
 *        of certain symbols -- at certain positions)
 *
 * - Editops should be an object that only *looks* like a list (which means
 *   it is a list in duck typing) to avoid never-ending conversions from
 *   Python lists to LevEditOp arrays and back
 *
 * - Optimize munkers_blackman(), it's pretty dumb (no memory of visited
 *   columns/rows)
 *
 * - Make it really usable as a C library (needs some wrappers, headers, ...,
 *   and maybe even documentation ;-)
 *
 * - Add interface to various interesting auxiliary results, namely
 *   set and sequence distance (only ratio is exported), the map from
 *   munkers_blackman() itself, ...
 *
 * - Generalizations:
 *   - character weight matrix/function
 *   - arbitrary edit operation costs, decomposable edit operations
 *
 * - Create a test suite
 *
 * - Add more interesting algorithms ;-)
 *
 * Postponed TODO (investigated, and a big `but' was found):
 *
 * - A linear approximate set median algorithm:
 *   P. Indyk: Sublinear time algorithms for metric space problems,
 *         STOC 1999, http://citeseer.nj.nec.com/indyk00sublinear.html.
 *   BUT: The algorithm seems to be advantageous only in the case of very
 *   large sets -- if my estimates are correct (the article itself is quite
 *   `asymptotic'), say 10^5 at least.  On smaller sets either one would get
 *   only an extermely rough median estimate, or the number of distance
 *   computations would be in fact higher than in the dumb O(n^2) algorithm.
 *
 * - Improve setmedian() speed with triangular inequality, see:
 *   Juan, A., E. Vidal: An Algorithm for Fast Median Search,
 *         1997, http://citeseer.nj.nec.com/article/juan97algorithm.html
 *   BUT: It doesn't seem to help much in spaces of high dimension (see the
 *   discussion and graphs in the article itself), a few percents at most,
 *   and strings behave like a space with a very high dimension (locally), so
 *   who knows, it probably wouldn't help much.
 *
 **/
#ifdef NO_PYTHON
#define _GNU_SOURCE
#include <string.h>
#include <math.h>
/* for debugging */
#include <stdio.h>
#else /* NO_PYTHON */
#define _LEV_STATIC_PY static
#define lev_wchar Py_UNICODE
#include <Python.h>
#endif /* NO_PYTHON */

#if PY_MAJOR_VERSION >= 3
#define LEV_PYTHON3
#define PyString_Type PyBytes_Type
#define PyString_GET_SIZE PyBytes_GET_SIZE
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_Check PyBytes_Check
#define PyString_FromStringAndSize PyBytes_FromStringAndSize
#define PyString_InternFromString PyUnicode_InternFromString
#define PyInt_AS_LONG PyLong_AsLong
#define PyInt_FromLong PyLong_FromLong
#define PyInt_Check PyLong_Check
#define PY_INIT_MOD(module, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        module = PyModule_Create(&moduledef);
    #define PY_MOD_INIT_FUNC_DEF(name) PyObject* PyInit_##name(void)
#else
    #define PY_INIT_MOD(module, name, doc, methods) \
            Py_InitModule3(name, methods, doc);
    #define PY_MOD_INIT_FUNC_DEF(name) void init##name(void)
#endif /* PY_MAJOR_VERSION */

#include <assert.h>
#include "_levenshtein.h"

/* FIXME: inline avaliability should be solved in setup.py, somehow, or
 * even better in Python.h, like const is...
 * this should inline at least with gcc and msvc */
#ifndef __GNUC__
#  ifdef _MSC_VER
#    define inline __inline
#  else
#    define inline /* */
#  endif
#  define __attribute__(x) /* */
#endif

#define LEV_EPSILON 1e-14
#define LEV_INFINITY 1e100

/* Me thinks the second argument of PyArg_UnpackTuple() should be const.
 * Anyway I habitually pass a constant string.
 * A cast to placate the compiler. */
#define PYARGCFIX(x) (char*)(x)

/* local functions */
static size_t*
munkers_blackman(size_t n1,
                 size_t n2,
                 double *dists);

#define TAUS113_LCG(n) ((69069UL * n) & 0xffffffffUL)
#define TAUS113_MASK 0xffffffffUL

typedef struct {
  unsigned long int z1, z2, z3, z4;
} taus113_state_t;

static inline unsigned long int
taus113_get(taus113_state_t *state);

static void
taus113_set(taus113_state_t *state,
            unsigned long int s);

#ifndef NO_PYTHON
/* python interface and wrappers */
/* declarations and docstrings {{{ */
static PyObject* distance_py(PyObject *self, PyObject *args);
static PyObject* ratio_py(PyObject *self, PyObject *args);
static PyObject* hamming_py(PyObject *self, PyObject *args);
static PyObject* jaro_py(PyObject *self, PyObject *args);
static PyObject* jaro_winkler_py(PyObject *self, PyObject *args);
static PyObject* median_py(PyObject *self, PyObject *args);
static PyObject* median_improve_py(PyObject *self, PyObject *args);
static PyObject* quickmedian_py(PyObject *self, PyObject *args);
static PyObject* setmedian_py(PyObject *self, PyObject *args);
static PyObject* seqratio_py(PyObject *self, PyObject *args);
static PyObject* setratio_py(PyObject *self, PyObject *args);
static PyObject* editops_py(PyObject *self, PyObject *args);
static PyObject* opcodes_py(PyObject *self, PyObject *args);
static PyObject* inverse_py(PyObject *self, PyObject *args);
static PyObject* apply_edit_py(PyObject *self, PyObject *args);
static PyObject* matching_blocks_py(PyObject *self, PyObject *args);
static PyObject* subtract_edit_py(PyObject *self, PyObject *args);

#define Levenshtein_DESC \
  "A C extension module for fast computation of:\n" \
  "- Levenshtein (edit) distance and edit sequence manipulation\n" \
  "- string similarity\n" \
  "- approximate median strings, and generally string averaging\n" \
  "- string sequence and set similarity\n" \
  "\n" \
  "Levenshtein has a some overlap with difflib (SequenceMatcher).  It\n" \
  "supports only strings, not arbitrary sequence types, but on the\n" \
  "other hand it's much faster.\n" \
  "\n" \
  "It supports both normal and Unicode strings, but can't mix them, all\n" \
  "arguments to a function (method) have to be of the same type (or its\n" \
  "subclasses).\n"

#define distance_DESC \
  "Compute absolute Levenshtein distance of two strings.\n" \
  "\n" \
  "distance(string1, string2)\n" \
  "\n" \
  "Examples (it's hard to spell Levenshtein correctly):\n" \
  "\n" \
  ">>> distance('Levenshtein', 'Lenvinsten')\n" \
  "4\n" \
  ">>> distance('Levenshtein', 'Levensthein')\n" \
  "2\n" \
  ">>> distance('Levenshtein', 'Levenshten')\n" \
  "1\n" \
  ">>> distance('Levenshtein', 'Levenshtein')\n" \
  "0\n" \
  "\n" \
  "Yeah, we've managed it at last.\n"

#define ratio_DESC \
  "Compute similarity of two strings.\n" \
  "\n" \
  "ratio(string1, string2)\n" \
  "\n" \
  "The similarity is a number between 0 and 1, it's usually equal or\n" \
  "somewhat higher than difflib.SequenceMatcher.ratio(), because it's\n" \
  "based on real minimal edit distance.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> ratio('Hello world!', 'Holly grail!')  # doctest: +ELLIPSIS\n" \
  "0.583333...\n" \
  ">>> ratio('Brian', 'Jesus')\n" \
  "0.0\n" \
  "\n" \
  "Really?  I thought there was some similarity.\n"

#define hamming_DESC \
  "Compute Hamming distance of two strings.\n" \
  "\n" \
  "hamming(string1, string2)\n" \
  "\n" \
  "The Hamming distance is simply the number of differing characters.\n" \
  "That means the length of the strings must be the same.\n" \
  "\n" \
  "Examples:\n" \
  ">>> hamming('Hello world!', 'Holly grail!')\n" \
  "7\n" \
  ">>> hamming('Brian', 'Jesus')\n" \
  "5\n"

#define jaro_DESC \
  "Compute Jaro string similarity metric of two strings.\n" \
  "\n" \
  "jaro(string1, string2)\n" \
  "\n" \
  "The Jaro string similarity metric is intended for short strings like\n" \
  "personal last names.  It is 0 for completely different strings and\n" \
  "1 for identical strings.\n" \
  "\n" \
  "Examples:\n" \
  ">>> jaro('Brian', 'Jesus')\n" \
  "0.0\n" \
  ">>> jaro('Thorkel', 'Thorgier')  # doctest: +ELLIPSIS\n" \
  "0.779761...\n" \
  ">>> jaro('Dinsdale', 'D')  # doctest: +ELLIPSIS\n" \
  "0.708333...\n"

#define jaro_winkler_DESC \
  "Compute Jaro string similarity metric of two strings.\n" \
  "\n" \
  "jaro_winkler(string1, string2[, prefix_weight])\n" \
  "\n" \
  "The Jaro-Winkler string similarity metric is a modification of Jaro\n" \
  "metric giving more weight to common prefix, as spelling mistakes are\n" \
  "more likely to occur near ends of words.\n" \
  "\n" \
  "The prefix weight is inverse value of common prefix length sufficient\n" \
  "to consider the strings *identical*.  If no prefix weight is\n" \
  "specified, 1/10 is used.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> jaro_winkler('Brian', 'Jesus')\n" \
  "0.0\n" \
  ">>> jaro_winkler('Thorkel', 'Thorgier')  # doctest: +ELLIPSIS\n" \
  "0.867857...\n" \
  ">>> jaro_winkler('Dinsdale', 'D')  # doctest: +ELLIPSIS\n" \
  "0.7375...\n" \
  ">>> jaro_winkler('Thorkel', 'Thorgier', 0.25)\n" \
  "1.0\n"

#define median_DESC \
  "Find an approximate generalized median string using greedy algorithm.\n" \
  "\n" \
  "median(string_sequence[, weight_sequence])\n" \
  "\n" \
  "You can optionally pass a weight for each string as the second\n" \
  "argument.  The weights are interpreted as item multiplicities,\n" \
  "although any non-negative real numbers are accepted.  Use them to\n" \
  "improve computation speed when strings often appear multiple times\n" \
  "in the sequence.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> median(['SpSm', 'mpamm', 'Spam', 'Spa', 'Sua', 'hSam'])\n" \
  "'Spam'\n" \
  ">>> fixme = ['Levnhtein', 'Leveshein', 'Leenshten',\n" \
  "...          'Leveshtei', 'Lenshtein', 'Lvenstein',\n" \
  "...          'Levenhtin', 'evenshtei']\n" \
  ">>> median(fixme)\n" \
  "'Levenshtein'\n" \
  "\n" \
  "Hm.  Even a computer program can spell Levenshtein better than me.\n"

#define median_improve_DESC \
  "Improve an approximate generalized median string by perturbations.\n" \
  "\n" \
  "median_improve(string, string_sequence[, weight_sequence])\n" \
  "\n" \
  "The first argument is the estimated generalized median string you\n" \
  "want to improve, the others are the same as in median().  It returns\n" \
  "a string with total distance less or equal to that of the given string.\n" \
  "\n" \
  "Note this is much slower than median().  Also note it performs only\n" \
  "one improvement step, calling median_improve() again on the result\n" \
  "may improve it further, though this is unlikely to happen unless the\n" \
  "given string was not very similar to the actual generalized median.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> fixme = ['Levnhtein', 'Leveshein', 'Leenshten',\n" \
  "...          'Leveshtei', 'Lenshtein', 'Lvenstein',\n" \
  "...          'Levenhtin', 'evenshtei']\n" \
  ">>> median_improve('spam', fixme)\n" \
  "'enhtein'\n" \
  ">>> median_improve(median_improve('spam', fixme), fixme)\n" \
  "'Levenshtein'\n" \
  "\n" \
  "It takes some work to change spam to Levenshtein.\n"

#define quickmedian_DESC \
  "Find a very approximate generalized median string, but fast.\n" \
  "\n" \
  "quickmedian(string[, weight_sequence])\n" \
  "\n" \
  "See median() for argument description.\n" \
  "\n" \
  "This method is somewhere between setmedian() and picking a random\n" \
  "string from the set; both speedwise and quality-wise.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> fixme = ['Levnhtein', 'Leveshein', 'Leenshten',\n" \
  "...          'Leveshtei', 'Lenshtein', 'Lvenstein',\n" \
  "...          'Levenhtin', 'evenshtei']\n" \
  ">>> quickmedian(fixme)\n" \
  "'Levnshein'\n"

#define setmedian_DESC \
  "Find set median of a string set (passed as a sequence).\n" \
  "\n" \
  "setmedian(string[, weight_sequence])\n" \
  "\n" \
  "See median() for argument description.\n" \
  "\n" \
  "The returned string is always one of the strings in the sequence.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> setmedian(['ehee', 'cceaes', 'chees', 'chreesc',\n" \
  "...            'chees', 'cheesee', 'cseese', 'chetese'])\n" \
  "'chees'\n" \
  "\n" \
  "You haven't asked me about Limburger, sir.\n"

#define seqratio_DESC \
  "Compute similarity ratio of two sequences of strings.\n" \
  "\n" \
  "seqratio(string_sequence1, string_sequence2)\n" \
  "\n" \
  "This is like ratio(), but for string sequences.  A kind of ratio()\n" \
  "is used to to measure the cost of item change operation for the\n" \
  "strings.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> seqratio(['newspaper', 'litter bin', 'tinny', 'antelope'],\n" \
  "...          ['caribou', 'sausage', 'gorn', 'woody'])\n" \
  "0.21517857142857144\n"

#define setratio_DESC \
  "Compute similarity ratio of two strings sets (passed as sequences).\n" \
  "\n" \
  "setratio(string_sequence1, string_sequence2)\n" \
  "\n" \
  "The best match between any strings in the first set and the second\n" \
  "set (passed as sequences) is attempted.  I.e., the order doesn't\n" \
  "matter here.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> setratio(['newspaper', 'litter bin', 'tinny', 'antelope'],\n" \
  "...          ['caribou', 'sausage', 'gorn', 'woody'])  # doctest: +ELLIPSIS\n" \
  "0.281845...\n" \
  "\n" \
  "No, even reordering doesn't help the tinny words to match the\n" \
  "woody ones.\n"

#define editops_DESC \
  "Find sequence of edit operations transforming one string to another.\n" \
  "\n" \
  "editops(source_string, destination_string)\n" \
  "editops(edit_operations, source_length, destination_length)\n" \
  "\n" \
  "The result is a list of triples (operation, spos, dpos), where\n" \
  "operation is one of 'equal', 'replace', 'insert', or 'delete';  spos\n" \
  "and dpos are position of characters in the first (source) and the\n" \
  "second (destination) strings.  These are operations on signle\n" \
  "characters.  In fact the returned list doesn't contain the 'equal',\n" \
  "but all the related functions accept both lists with and without\n" \
  "'equal's.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> editops('spam', 'park')\n" \
  "[('delete', 0, 0), ('insert', 3, 2), ('replace', 3, 3)]\n" \
  "\n" \
  "The alternate form editops(opcodes, source_string, destination_string)\n" \
  "can be used for conversion from opcodes (5-tuples) to editops (you can\n" \
  "pass strings or their lengths, it doesn't matter).\n"

#define opcodes_DESC \
  "Find sequence of edit operations transforming one string to another.\n" \
  "\n" \
  "opcodes(source_string, destination_string)\n" \
  "opcodes(edit_operations, source_length, destination_length)\n" \
  "\n" \
  "The result is a list of 5-tuples with the same meaning as in\n" \
  "SequenceMatcher's get_opcodes() output.  But since the algorithms\n" \
  "differ, the actual sequences from Levenshtein and SequenceMatcher\n" \
  "may differ too.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> for x in opcodes('spam', 'park'):\n" \
  "...     print(x)\n" \
  "...\n" \
  "('delete', 0, 1, 0, 0)\n" \
  "('equal', 1, 3, 0, 2)\n" \
  "('insert', 3, 3, 2, 3)\n" \
  "('replace', 3, 4, 3, 4)\n" \
  "\n" \
  "The alternate form opcodes(editops, source_string, destination_string)\n" \
  "can be used for conversion from editops (triples) to opcodes (you can\n" \
  "pass strings or their lengths, it doesn't matter).\n"

#define inverse_DESC \
  "Invert the sense of an edit operation sequence.\n" \
  "\n" \
  "inverse(edit_operations)\n" \
  "\n" \
  "In other words, it returns a list of edit operations transforming the\n" \
  "second (destination) string to the first (source).  It can be used\n" \
  "with both editops and opcodes.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> inverse(editops('spam', 'park'))\n" \
  "[('insert', 0, 0), ('delete', 2, 3), ('replace', 3, 3)]\n" \
  ">>> editops('park', 'spam')\n" \
  "[('insert', 0, 0), ('delete', 2, 3), ('replace', 3, 3)]\n"

#define apply_edit_DESC \
  "Apply a sequence of edit operations to a string.\n" \
  "\n" \
  "apply_edit(edit_operations, source_string, destination_string)\n" \
  "\n" \
  "In the case of editops, the sequence can be arbitrary ordered subset\n" \
  "of the edit sequence transforming source_string to destination_string.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> e = editops('man', 'scotsman')\n" \
  ">>> apply_edit(e, 'man', 'scotsman')\n" \
  "'scotsman'\n" \
  ">>> apply_edit(e[:3], 'man', 'scotsman')\n" \
  "'scoman'\n" \
  "\n" \
  "The other form of edit operations, opcodes, is not very suitable for\n" \
  "such a tricks, because it has to always span over complete strings,\n" \
  "subsets can be created by carefully replacing blocks with 'equal'\n" \
  "blocks, or by enlarging 'equal' block at the expense of other blocks\n" \
  "and adjusting the other blocks accordingly.\n" \
  "\n" \
  "Examples:\n" \
  ">>> a, b = 'spam and eggs', 'foo and bar'\n" \
  ">>> e = opcodes(a, b)\n" \
  ">>> apply_edit(inverse(e), b, a)\n" \
  "'spam and eggs'\n" \
  ">>> e[4] = ('equal', 10, 13, 8, 11)\n" \
  ">>> e\n" \
  "[('delete', 0, 1, 0, 0), ('replace', 1, 4, 0, 3), ('equal', 4, 9, 3, 8), ('delete', 9, 10, 8, 8), ('equal', 10, 13, 8, 11)]\n" \
  ">>> apply_edit(e, a, b)\n" \
  "'foo and ggs'\n"

#define matching_blocks_DESC \
  "Find identical blocks in two strings.\n" \
  "\n" \
  "matching_blocks(edit_operations, source_length, destination_length)\n" \
  "\n" \
  "The result is a list of triples with the same meaning as in\n" \
  "SequenceMatcher's get_matching_blocks() output.  It can be used with\n" \
  "both editops and opcodes.  The second and third arguments don't\n" \
  "have to be actually strings, their lengths are enough.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> a, b = 'spam', 'park'\n" \
  ">>> matching_blocks(editops(a, b), a, b)\n" \
  "[(1, 0, 2), (4, 4, 0)]\n" \
  ">>> matching_blocks(editops(a, b), len(a), len(b))\n" \
  "[(1, 0, 2), (4, 4, 0)]\n" \
  "\n" \
  "The last zero-length block is not an error, but it's there for\n" \
  "compatibility with difflib which always emits it.\n" \
  "\n" \
  "One can join the matching blocks to get two identical strings:\n" \
  ">>> a, b = 'dog kennels', 'mattresses'\n" \
  ">>> mb = matching_blocks(editops(a,b), a, b)\n" \
  ">>> ''.join([a[x[0]:x[0]+x[2]] for x in mb])\n" \
  "'ees'\n" \
  ">>> ''.join([b[x[1]:x[1]+x[2]] for x in mb])\n" \
  "'ees'\n"

#define subtract_edit_DESC \
  "Subtract an edit subsequence from a sequence.\n" \
  "\n" \
  "subtract_edit(edit_operations, subsequence)\n" \
  "\n" \
  "The result is equivalent to\n" \
  "editops(apply_edit(subsequence, s1, s2), s2), except that is\n" \
  "constructed directly from the edit operations.  That is, if you apply\n" \
  "it to the result of subsequence application, you get the same final\n" \
  "string as from application complete edit_operations.  It may be not\n" \
  "identical, though (in amibuous cases, like insertion of a character\n" \
  "next to the same character).\n" \
  "\n" \
  "The subtracted subsequence must be an ordered subset of\n" \
  "edit_operations.\n" \
  "\n" \
  "Note this function does not accept difflib-style opcodes as no one in\n" \
  "his right mind wants to create subsequences from them.\n" \
  "\n" \
  "Examples:\n" \
  "\n" \
  ">>> e = editops('man', 'scotsman')\n" \
  ">>> e1 = e[:3]\n" \
  ">>> bastard = apply_edit(e1, 'man', 'scotsman')\n" \
  ">>> bastard\n" \
  "'scoman'\n" \
  ">>> apply_edit(subtract_edit(e, e1), bastard, 'scotsman')\n" \
  "'scotsman'\n" \

#define METHODS_ITEM(x) { #x, x##_py, METH_VARARGS, x##_DESC }
static PyMethodDef methods[] = {
  METHODS_ITEM(distance),
  METHODS_ITEM(ratio),
  METHODS_ITEM(hamming),
  METHODS_ITEM(jaro),
  METHODS_ITEM(jaro_winkler),
  METHODS_ITEM(median),
  METHODS_ITEM(median_improve),
  METHODS_ITEM(quickmedian),
  METHODS_ITEM(setmedian),
  METHODS_ITEM(seqratio),
  METHODS_ITEM(setratio),
  METHODS_ITEM(editops),
  METHODS_ITEM(opcodes),
  METHODS_ITEM(inverse),
  METHODS_ITEM(apply_edit),
  METHODS_ITEM(matching_blocks),
  METHODS_ITEM(subtract_edit),
  { NULL, NULL, 0, NULL },
};

/* opcode names, these are to be initialized in the init func,
 * indexed by LevEditType values */
struct {
  PyObject* pystring;
  const char *cstring;
  size_t len;
}
static opcode_names[] = {
  { NULL, "equal", 0 },
  { NULL, "replace", 0 },
  { NULL, "insert", 0 },
  { NULL, "delete", 0 },
};
#define N_OPCODE_NAMES ((sizeof(opcode_names)/sizeof(opcode_names[0])))

typedef lev_byte *(*MedianFuncString)(size_t n,
                                      const size_t *lengths,
                                      const lev_byte *strings[],
                                      const double *weights,
                                      size_t *medlength);
typedef Py_UNICODE *(*MedianFuncUnicode)(size_t n,
                                         const size_t *lengths,
                                         const Py_UNICODE *strings[],
                                         const double *weights,
                                         size_t *medlength);
typedef struct {
  MedianFuncString s;
  MedianFuncUnicode u;
} MedianFuncs;

typedef lev_byte *(*MedianImproveFuncString)(size_t len, const lev_byte *s,
                                             size_t n,
                                             const size_t *lengths,
                                             const lev_byte *strings[],
                                             const double *weights,
                                             size_t *medlength);
typedef Py_UNICODE *(*MedianImproveFuncUnicode)(size_t len, const Py_UNICODE *s,
                                                size_t n,
                                                const size_t *lengths,
                                                const Py_UNICODE *strings[],
                                                const double *weights,
                                                size_t *medlength);
typedef struct {
  MedianImproveFuncString s;
  MedianImproveFuncUnicode u;
} MedianImproveFuncs;

typedef double (*SetSeqFuncString)(size_t n1,
                                   const size_t *lengths1,
                                   const lev_byte *strings1[],
                                   size_t n2,
                                   const size_t *lengths2,
                                   const lev_byte *strings2[]);
typedef double (*SetSeqFuncUnicode)(size_t n1,
                                    const size_t *lengths1,
                                    const Py_UNICODE *strings1[],
                                    size_t n2,
                                    const size_t *lengths2,
                                    const Py_UNICODE *strings2[]);

typedef struct {
  SetSeqFuncString s;
  SetSeqFuncUnicode u;
} SetSeqFuncs;

static long int
levenshtein_common(PyObject *args,
                   const char *name,
                   size_t xcost,
                   size_t *lensum);

static int
extract_stringlist(PyObject *list,
                   const char *name,
                   size_t n,
                   size_t **sizelist,
                   void *strlist);

static double*
extract_weightlist(PyObject *wlist,
                   const char *name,
                   size_t n);

static PyObject*
median_common(PyObject *args,
              const char *name,
              MedianFuncs foo);

static PyObject*
median_improve_common(PyObject *args,
                      const char *name,
                      MedianImproveFuncs foo);

static double
setseq_common(PyObject *args,
              const char *name,
              SetSeqFuncs foo,
              size_t *lensum);

static void *
safe_malloc(size_t nmemb, size_t size) {
  /* extra-conservative overflow check */
  if (SIZE_MAX / size <= nmemb) {
    return NULL;
  }
  return malloc(nmemb * size);
}

static void *
safe_malloc_3(size_t nmemb1, size_t nmemb2, size_t size) {
  /* extra-conservative overflow check */
  if (SIZE_MAX / size <= nmemb2) {
    return NULL;
  }
  return safe_malloc(nmemb1, nmemb2 * size);
}

/* }}} */

/****************************************************************************
 *
 * Python interface and subroutines
 *
 ****************************************************************************/
/* {{{ */

static long int
levenshtein_common(PyObject *args, const char *name, size_t xcost,
                   size_t *lensum)
{
  PyObject *arg1, *arg2;
  size_t len1, len2;

  if (!PyArg_UnpackTuple(args, PYARGCFIX(name), 2, 2, &arg1, &arg2))
    return -1;

  if (PyObject_TypeCheck(arg1, &PyString_Type)
      && PyObject_TypeCheck(arg2, &PyString_Type)) {
    lev_byte *string1, *string2;

    len1 = PyString_GET_SIZE(arg1);
    len2 = PyString_GET_SIZE(arg2);
    *lensum = len1 + len2;
    string1 = PyString_AS_STRING(arg1);
    string2 = PyString_AS_STRING(arg2);
    {
      size_t d = lev_edit_distance(len1, string1, len2, string2, xcost);
      if (d == (size_t)(-1)) {
        PyErr_NoMemory();
        return -1;
      }
      return d;
    }
  }
  else if (PyObject_TypeCheck(arg1, &PyUnicode_Type)
      && PyObject_TypeCheck(arg2, &PyUnicode_Type)) {
    Py_UNICODE *string1, *string2;

    len1 = PyUnicode_GET_SIZE(arg1);
    len2 = PyUnicode_GET_SIZE(arg2);
    *lensum = len1 + len2;
    string1 = PyUnicode_AS_UNICODE(arg1);
    string2 = PyUnicode_AS_UNICODE(arg2);
    {
      size_t d = lev_u_edit_distance(len1, string1, len2, string2, xcost);
      if (d == (size_t)(-1)) {
        PyErr_NoMemory();
        return -1;
      }
      return d;
    }
  }
  else {
    PyErr_Format(PyExc_TypeError,
                 "%s expected two Strings or two Unicodes", name);
    return -1;
  }
}

static PyObject*
distance_py(PyObject *self, PyObject *args)
{
  size_t lensum;
  long int ldist;

  if ((ldist = levenshtein_common(args, "distance", 0, &lensum)) < 0)
    return NULL;

  return PyInt_FromLong((long)ldist);
}

static PyObject*
ratio_py(PyObject *self, PyObject *args)
{
  size_t lensum;
  long int ldist;

  if ((ldist = levenshtein_common(args, "ratio", 1, &lensum)) < 0)
    return NULL;

  if (lensum == 0)
    return PyFloat_FromDouble(1.0);

  return PyFloat_FromDouble((double)(lensum - ldist)/(lensum));
}

static PyObject*
hamming_py(PyObject *self, PyObject *args)
{
  PyObject *arg1, *arg2;
  const char *name = "hamming";
  size_t len1, len2;
  long int dist;

  if (!PyArg_UnpackTuple(args, PYARGCFIX(name), 2, 2, &arg1, &arg2))
    return NULL;

  if (PyObject_TypeCheck(arg1, &PyString_Type)
      && PyObject_TypeCheck(arg2, &PyString_Type)) {
    lev_byte *string1, *string2;

    len1 = PyString_GET_SIZE(arg1);
    len2 = PyString_GET_SIZE(arg2);
    if (len1 != len2) {
      PyErr_Format(PyExc_ValueError,
                   "%s expected two strings of the same length", name);
      return NULL;
    }
    string1 = PyString_AS_STRING(arg1);
    string2 = PyString_AS_STRING(arg2);
    dist = lev_hamming_distance(len1, string1, string2);
    return PyInt_FromLong(dist);
  }
  else if (PyObject_TypeCheck(arg1, &PyUnicode_Type)
      && PyObject_TypeCheck(arg2, &PyUnicode_Type)) {
    Py_UNICODE *string1, *string2;

    len1 = PyUnicode_GET_SIZE(arg1);
    len2 = PyUnicode_GET_SIZE(arg2);
    if (len1 != len2) {
      PyErr_Format(PyExc_ValueError,
                   "%s expected two unicodes of the same length", name);
      return NULL;
    }
    string1 = PyUnicode_AS_UNICODE(arg1);
    string2 = PyUnicode_AS_UNICODE(arg2);
    dist = lev_u_hamming_distance(len1, string1, string2);
    return PyInt_FromLong(dist);
  }
  else {
    PyErr_Format(PyExc_TypeError,
                 "%s expected two Strings or two Unicodes", name);
    return NULL;
  }
}

static PyObject*
jaro_py(PyObject *self, PyObject *args)
{
  PyObject *arg1, *arg2;
  const char *name = "jaro";
  size_t len1, len2;

  if (!PyArg_UnpackTuple(args, PYARGCFIX(name), 2, 2, &arg1, &arg2))
    return NULL;

  if (PyObject_TypeCheck(arg1, &PyString_Type)
      && PyObject_TypeCheck(arg2, &PyString_Type)) {
    lev_byte *string1, *string2;

    len1 = PyString_GET_SIZE(arg1);
    len2 = PyString_GET_SIZE(arg2);
    string1 = PyString_AS_STRING(arg1);
    string2 = PyString_AS_STRING(arg2);
    return PyFloat_FromDouble(lev_jaro_ratio(len1, string1, len2, string2));
  }
  else if (PyObject_TypeCheck(arg1, &PyUnicode_Type)
      && PyObject_TypeCheck(arg2, &PyUnicode_Type)) {
    Py_UNICODE *string1, *string2;

    len1 = PyUnicode_GET_SIZE(arg1);
    len2 = PyUnicode_GET_SIZE(arg2);
    string1 = PyUnicode_AS_UNICODE(arg1);
    string2 = PyUnicode_AS_UNICODE(arg2);
    return PyFloat_FromDouble(lev_u_jaro_ratio(len1, string1, len2, string2));
  }
  else {
    PyErr_Format(PyExc_TypeError,
                 "%s expected two Strings or two Unicodes", name);
    return NULL;
  }
}

static PyObject*
jaro_winkler_py(PyObject *self, PyObject *args)
{
  PyObject *arg1, *arg2, *arg3 = NULL;
  double pfweight = 0.1;
  const char *name = "jaro_winkler";
  size_t len1, len2;

  if (!PyArg_UnpackTuple(args, PYARGCFIX(name), 2, 3, &arg1, &arg2, &arg3))
    return NULL;

  if (arg3) {
    if (!PyObject_TypeCheck(arg3, &PyFloat_Type)) {
      PyErr_Format(PyExc_TypeError, "%s third argument must be a Float", name);
      return NULL;
    }
    pfweight = PyFloat_AS_DOUBLE(arg3);
    if (pfweight < 0.0) {
      PyErr_Format(PyExc_ValueError, "%s negative prefix weight", name);
      return NULL;
    }
  }

  if (PyObject_TypeCheck(arg1, &PyString_Type)
      && PyObject_TypeCheck(arg2, &PyString_Type)) {
    lev_byte *string1, *string2;

    len1 = PyString_GET_SIZE(arg1);
    len2 = PyString_GET_SIZE(arg2);
    string1 = PyString_AS_STRING(arg1);
    string2 = PyString_AS_STRING(arg2);
    return PyFloat_FromDouble(lev_jaro_winkler_ratio(len1, string1,
                                                     len2, string2,
                                                     pfweight));
  }
  else if (PyObject_TypeCheck(arg1, &PyUnicode_Type)
      && PyObject_TypeCheck(arg2, &PyUnicode_Type)) {
    Py_UNICODE *string1, *string2;

    len1 = PyUnicode_GET_SIZE(arg1);
    len2 = PyUnicode_GET_SIZE(arg2);
    string1 = PyUnicode_AS_UNICODE(arg1);
    string2 = PyUnicode_AS_UNICODE(arg2);
    return PyFloat_FromDouble(lev_u_jaro_winkler_ratio(len1, string1,
                                                       len2, string2,
                                                       pfweight));
  }
  else {
    PyErr_Format(PyExc_TypeError,
                 "%s expected two Strings or two Unicodes", name);
    return NULL;
  }
}

static PyObject*
median_py(PyObject *self, PyObject *args)
{
  MedianFuncs engines = { lev_greedy_median, lev_u_greedy_median };
  return median_common(args, "median", engines);
}

static PyObject*
median_improve_py(PyObject *self, PyObject *args)
{
  MedianImproveFuncs engines = { lev_median_improve, lev_u_median_improve };
  return median_improve_common(args, "median_improve", engines);
}

static PyObject*
quickmedian_py(PyObject *self, PyObject *args)
{
  MedianFuncs engines = { lev_quick_median, lev_u_quick_median };
  return median_common(args, "quickmedian", engines);
}

static PyObject*
setmedian_py(PyObject *self, PyObject *args)
{
  MedianFuncs engines = { lev_set_median, lev_u_set_median };
  return median_common(args, "setmedian", engines);
}

static PyObject*
median_common(PyObject *args, const char *name, MedianFuncs foo)
{
  size_t n, len;
  void *strings = NULL;
  size_t *sizes = NULL;
  PyObject *strlist = NULL;
  PyObject *wlist = NULL;
  PyObject *strseq = NULL;
  double *weights;
  int stringtype;
  PyObject *result = NULL;

  if (!PyArg_UnpackTuple(args, PYARGCFIX(name), 1, 2, &strlist, &wlist))
    return NULL;

  if (!PySequence_Check(strlist)) {
    PyErr_Format(PyExc_TypeError,
                 "%s first argument must be a Sequence", name);
    return NULL;
  }
  strseq = PySequence_Fast(strlist, name);

  n = PySequence_Fast_GET_SIZE(strseq);
  if (n == 0) {
    Py_INCREF(Py_None);
    Py_DECREF(strseq);
    return Py_None;
  }

  /* get (optional) weights, use 1 if none specified. */
  weights = extract_weightlist(wlist, name, n);
  if (!weights) {
    Py_DECREF(strseq);
    return NULL;
  }

  stringtype = extract_stringlist(strseq, name, n, &sizes, &strings);
  Py_DECREF(strseq);
  if (stringtype < 0) {
    free(weights);
    return NULL;
  }

  if (stringtype == 0) {
    lev_byte *medstr = foo.s(n, sizes, strings, weights, &len);
    if (!medstr && len)
      result = PyErr_NoMemory();
    else {
      result = PyString_FromStringAndSize(medstr, len);
      free(medstr);
    }
  }
  else if (stringtype == 1) {
    Py_UNICODE *medstr = foo.u(n, sizes, strings, weights, &len);
    if (!medstr && len)
      result = PyErr_NoMemory();
    else {
      result = PyUnicode_FromUnicode(medstr, len);
      free(medstr);
    }
  }
  else
    PyErr_Format(PyExc_SystemError, "%s internal error", name);

  free(strings);
  free(weights);
  free(sizes);
  return result;
}

static PyObject*
median_improve_common(PyObject *args, const char *name, MedianImproveFuncs foo)
{
  size_t n, len;
  void *strings = NULL;
  size_t *sizes = NULL;
  PyObject *arg1 = NULL;
  PyObject *strlist = NULL;
  PyObject *wlist = NULL;
  PyObject *strseq = NULL;
  double *weights;
  int stringtype;
  PyObject *result = NULL;

  if (!PyArg_UnpackTuple(args, PYARGCFIX(name), 2, 3, &arg1, &strlist, &wlist))
    return NULL;

  if (PyObject_TypeCheck(arg1, &PyString_Type))
    stringtype = 0;
  else if (PyObject_TypeCheck(arg1, &PyUnicode_Type))
    stringtype = 1;
  else {
    PyErr_Format(PyExc_TypeError,
                 "%s first argument must be a String or Unicode", name);
    return NULL;
  }

  if (!PySequence_Check(strlist)) {
    PyErr_Format(PyExc_TypeError,
                 "%s second argument must be a Sequence", name);
    return NULL;
  }
  strseq = PySequence_Fast(strlist, name);

  n = PySequence_Fast_GET_SIZE(strseq);
  if (n == 0) {
    Py_INCREF(Py_None);
    Py_DECREF(strseq);
    return Py_None;
  }

  /* get (optional) weights, use 1 if none specified. */
  weights = extract_weightlist(wlist, name, n);
  if (!weights) {
    Py_DECREF(strseq);
    return NULL;
  }

  if (extract_stringlist(strseq, name, n, &sizes, &strings) != stringtype) {
    PyErr_Format(PyExc_TypeError,
                 "%s argument types don't match", name);
    free(weights);
    return NULL;
  }

  Py_DECREF(strseq);
  if (stringtype == 0) {
    lev_byte *s = PyString_AS_STRING(arg1);
    size_t l = PyString_GET_SIZE(arg1);
    lev_byte *medstr = foo.s(l, s, n, sizes, strings, weights, &len);
    if (!medstr && len)
      result = PyErr_NoMemory();
    else {
      result = PyString_FromStringAndSize(medstr, len);
      free(medstr);
    }
  }
  else if (stringtype == 1) {
    Py_UNICODE *s = PyUnicode_AS_UNICODE(arg1);
    size_t l = PyUnicode_GET_SIZE(arg1);
    Py_UNICODE *medstr = foo.u(l, s, n, sizes, strings, weights, &len);
    if (!medstr && len)
      result = PyErr_NoMemory();
    else {
      result = PyUnicode_FromUnicode(medstr, len);
      free(medstr);
    }
  }
  else
    PyErr_Format(PyExc_SystemError, "%s internal error", name);

  free(strings);
  free(weights);
  free(sizes);
  return result;
}

static double*
extract_weightlist(PyObject *wlist, const char *name, size_t n)
{
  size_t i;
  double *weights = NULL;
  PyObject *seq;

  if (wlist) {
    if (!PySequence_Check(wlist)) {
      PyErr_Format(PyExc_TypeError,
                  "%s second argument must be a Sequence", name);
      return NULL;
    }
    seq = PySequence_Fast(wlist, name);
    if (PySequence_Fast_GET_SIZE(wlist) != n) {
      PyErr_Format(PyExc_ValueError,
                   "%s got %i strings but %i weights",
                   name, n, PyList_GET_SIZE(wlist));
      Py_DECREF(seq);
      return NULL;
    }
    weights = (double*)safe_malloc(n, sizeof(double));
    if (!weights)
      return (double*)PyErr_NoMemory();
    for (i = 0; i < n; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(wlist, i);
      PyObject *number = PyNumber_Float(item);

      if (!number) {
        free(weights);
        PyErr_Format(PyExc_TypeError,
                     "%s weight #%i is not a Number", name, i);
        Py_DECREF(seq);
        return NULL;
      }
      weights[i] = PyFloat_AS_DOUBLE(number);
      Py_DECREF(number);
      if (weights[i] < 0) {
        free(weights);
        PyErr_Format(PyExc_ValueError,
                     "%s weight #%i is negative", name, i);
        Py_DECREF(seq);
        return NULL;
      }
    }
    Py_DECREF(seq);
  }
  else {
    weights = (double*)safe_malloc(n, sizeof(double));
    if (!weights)
      return (double*)PyErr_NoMemory();
    for (i = 0; i < n; i++)
      weights[i] = 1.0;
  }

  return weights;
}

/* extract a list of strings or unicode strings, returns
 * 0 -- strings
 * 1 -- unicode strings
 * <0 -- failure
 */
static int
extract_stringlist(PyObject *list, const char *name,
                   size_t n, size_t **sizelist, void *strlist)
{
  size_t i;
  PyObject *first;

  /* check first object type.  when it's a string then all others must be
   * strings too; when it's a unicode string then all others must be unicode
   * strings too. */
  first = PySequence_Fast_GET_ITEM(list, 0);
  /* i don't exactly understand why the problem doesn't exhibit itself earlier
   * but a queer error message is better than a segfault :o) */
  if (first == (PyObject*)-1) {
    PyErr_Format(PyExc_TypeError,
                 "%s undecomposable Sequence???", name);
    return -1;
  }

  if (PyObject_TypeCheck(first, &PyString_Type)) {
    lev_byte **strings;
    size_t *sizes;

    strings = (lev_byte**)safe_malloc(n, sizeof(lev_byte*));
    if (!strings) {
      PyErr_Format(PyExc_MemoryError,
                   "%s cannot allocate memory", name);
      return -1;
    }
    sizes = (size_t*)safe_malloc(n, sizeof(size_t));
    if (!sizes) {
      free(strings);
      PyErr_Format(PyExc_MemoryError,
                   "%s cannot allocate memory", name);
      return -1;
    }

    strings[0] = PyString_AS_STRING(first);
    sizes[0] = PyString_GET_SIZE(first);
    for (i = 1; i < n; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(list, i);

      if (!PyObject_TypeCheck(item, &PyString_Type)) {
        free(strings);
        free(sizes);
        PyErr_Format(PyExc_TypeError,
                     "%s item #%i is not a String", name, i);
        return -1;
      }
      strings[i] = PyString_AS_STRING(item);
      sizes[i] = PyString_GET_SIZE(item);
    }

    *(lev_byte***)strlist = strings;
    *sizelist = sizes;
    return 0;
  }
  if (PyObject_TypeCheck(first, &PyUnicode_Type)) {
    Py_UNICODE **strings;
    size_t *sizes;

    strings = (Py_UNICODE**)safe_malloc(n, sizeof(Py_UNICODE*));
    if (!strings) {
      PyErr_NoMemory();
      return -1;
    }
    sizes = (size_t*)safe_malloc(n, sizeof(size_t));
    if (!sizes) {
      free(strings);
      PyErr_NoMemory();
      return -1;
    }

    strings[0] = PyUnicode_AS_UNICODE(first);
    sizes[0] = PyUnicode_GET_SIZE(first);
    for (i = 1; i < n; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(list, i);

      if (!PyObject_TypeCheck(item, &PyUnicode_Type)) {
        free(strings);
        free(sizes);
        PyErr_Format(PyExc_TypeError,
                     "%s item #%i is not a Unicode", name, i);
        return -1;
      }
      strings[i] = PyUnicode_AS_UNICODE(item);
      sizes[i] = PyUnicode_GET_SIZE(item);
    }

    *(Py_UNICODE***)strlist = strings;
    *sizelist = sizes;
    return 1;
  }

  PyErr_Format(PyExc_TypeError,
               "%s expected list of Strings or Unicodes", name);
  return -1;
}

static PyObject*
seqratio_py(PyObject *self, PyObject *args)
{
  SetSeqFuncs engines = { lev_edit_seq_distance, lev_u_edit_seq_distance };
  size_t lensum;
  double r = setseq_common(args, "seqratio", engines, &lensum);
  if (r < 0)
    return NULL;
  if (lensum == 0)
    return PyFloat_FromDouble(1.0);
  return PyFloat_FromDouble((lensum - r)/lensum);
}

static PyObject*
setratio_py(PyObject *self, PyObject *args)
{
  SetSeqFuncs engines = { lev_set_distance, lev_u_set_distance };
  size_t lensum;
  double r = setseq_common(args, "setratio", engines, &lensum);
  if (r < 0)
    return NULL;
  if (lensum == 0)
    return PyFloat_FromDouble(1.0);
  return PyFloat_FromDouble((lensum - r)/lensum);
}

static double
setseq_common(PyObject *args, const char *name, SetSeqFuncs foo,
              size_t *lensum)
{
  size_t n1, n2;
  void *strings1 = NULL;
  void *strings2 = NULL;
  size_t *sizes1 = NULL;
  size_t *sizes2 = NULL;
  PyObject *strlist1;
  PyObject *strlist2;
  PyObject *strseq1;
  PyObject *strseq2;
  int stringtype1, stringtype2;
  double r = -1.0;

  if (!PyArg_UnpackTuple(args, PYARGCFIX(name), 2, 2, &strlist1, &strlist2))
    return r;

  if (!PySequence_Check(strlist1)) {
    PyErr_Format(PyExc_TypeError,
                 "%s first argument must be a Sequence", name);
    return r;
  }
  if (!PySequence_Check(strlist2)) {
    PyErr_Format(PyExc_TypeError,
                 "%s second argument must be a Sequence", name);
    return r;
  }

  strseq1 = PySequence_Fast(strlist1, name);
  strseq2 = PySequence_Fast(strlist2, name);

  n1 = PySequence_Fast_GET_SIZE(strseq1);
  n2 = PySequence_Fast_GET_SIZE(strseq2);
  *lensum = n1 + n2;
  if (n1 == 0) {
    Py_DECREF(strseq1);
    Py_DECREF(strseq2);
    return (double)n2;
  }
  if (n2 == 0) {
    Py_DECREF(strseq1);
    Py_DECREF(strseq2);
    return (double)n1;
  }

  stringtype1 = extract_stringlist(strseq1, name, n1, &sizes1, &strings1);
  Py_DECREF(strseq1);
  if (stringtype1 < 0) {
    Py_DECREF(strseq2);
    return r;
  }
  stringtype2 = extract_stringlist(strseq2, name, n2, &sizes2, &strings2);
  Py_DECREF(strseq2);
  if (stringtype2 < 0) {
    free(sizes1);
    free(strings1);
    return r;
  }

  if (stringtype1 != stringtype2) {
    PyErr_Format(PyExc_TypeError,
                  "%s both sequences must consist of items of the same type",
                  name);
  }
  else if (stringtype1 == 0) {
    r = foo.s(n1, sizes1, strings1, n2, sizes2, strings2);
    if (r < 0.0)
      PyErr_NoMemory();
  }
  else if (stringtype1 == 1) {
    r = foo.u(n1, sizes1, strings1, n2, sizes2, strings2);
    if (r < 0.0)
      PyErr_NoMemory();
  }
  else
    PyErr_Format(PyExc_SystemError, "%s internal error", name);

  free(strings1);
  free(strings2);
  free(sizes1);
  free(sizes2);
  return r;
}

static inline LevEditType
string_to_edittype(PyObject *string)
{
  const char *s;
  size_t i, len;

  for (i = 0; i < N_OPCODE_NAMES; i++) {
    if (string == opcode_names[i].pystring)
      return i;
  }

  /* With Python >= 2.2, we shouldn't get here, except when the strings are
   * not Strings but subtypes. */
#ifdef LEV_PYTHON3
  /* For Python 3, the string is an unicode object; use CompareWithAsciiString */
  if (!PyUnicode_Check(string)) {
    return LEV_EDIT_LAST;
  }

  for (i = 0; i < N_OPCODE_NAMES; i++) {
    if (PyUnicode_CompareWithASCIIString(string, opcode_names[i].cstring) == 0) {
      return i;
    }
  }

#else

  if (!PyString_Check(string))
    return LEV_EDIT_LAST;

  s = PyString_AS_STRING(string);
  len = PyString_GET_SIZE(string);
  for (i = 0; i < N_OPCODE_NAMES; i++) {
    if (len == opcode_names[i].len
        && memcmp(s, opcode_names[i].cstring, len) == 0) {
      return i;
    }
  }
#endif

  return LEV_EDIT_LAST;
}

static LevEditOp*
extract_editops(PyObject *list)
{
  LevEditOp *ops;
  size_t i;
  LevEditType type;
  size_t n = PyList_GET_SIZE(list);

  ops = (LevEditOp*)safe_malloc(n, sizeof(LevEditOp));
  if (!ops)
    return (LevEditOp*)PyErr_NoMemory();
  for (i = 0; i < n; i++) {
    PyObject *item;
    PyObject *tuple = PyList_GET_ITEM(list, i);

    if (!PyTuple_Check(tuple) || PyTuple_GET_SIZE(tuple) != 3) {
      free(ops);
      return NULL;
    }
    item = PyTuple_GET_ITEM(tuple, 0);
    if ((type = string_to_edittype(item)) == LEV_EDIT_LAST) {
      free(ops);
      return NULL;
    }
    ops[i].type = type;
    item = PyTuple_GET_ITEM(tuple, 1);
    if (!PyInt_Check(item)) {
      free(ops);
      return NULL;
    }
    ops[i].spos = (size_t)PyInt_AS_LONG(item);
    item = PyTuple_GET_ITEM(tuple, 2);
    if (!PyInt_Check(item)) {
      free(ops);
      return NULL;
    }
    ops[i].dpos = (size_t)PyInt_AS_LONG(item);
  }
  return ops;
}

static LevOpCode*
extract_opcodes(PyObject *list)
{
  LevOpCode *bops;
  size_t i;
  LevEditType type;
  size_t nb = PyList_GET_SIZE(list);

  bops = (LevOpCode*)safe_malloc(nb, sizeof(LevOpCode));
  if (!bops)
    return (LevOpCode*)PyErr_NoMemory();
  for (i = 0; i < nb; i++) {
    PyObject *item;
    PyObject *tuple = PyList_GET_ITEM(list, i);

    if (!PyTuple_Check(tuple) || PyTuple_GET_SIZE(tuple) != 5) {
      free(bops);
      return NULL;
    }

    item = PyTuple_GET_ITEM(tuple, 0);
    if ((type = string_to_edittype(item)) == LEV_EDIT_LAST) {
      free(bops);
      return NULL;
    }
    bops[i].type = type;

    item = PyTuple_GET_ITEM(tuple, 1);
    if (!PyInt_Check(item)) {
      free(bops);
      return NULL;
    }
    bops[i].sbeg = (size_t)PyInt_AS_LONG(item);

    item = PyTuple_GET_ITEM(tuple, 2);
    if (!PyInt_Check(item)) {
      free(bops);
      return NULL;
    }
    bops[i].send = (size_t)PyInt_AS_LONG(item);

    item = PyTuple_GET_ITEM(tuple, 3);
    if (!PyInt_Check(item)) {
      free(bops);
      return NULL;
    }
    bops[i].dbeg = (size_t)PyInt_AS_LONG(item);

    item = PyTuple_GET_ITEM(tuple, 4);
    if (!PyInt_Check(item)) {
      free(bops);
      return NULL;
    }
    bops[i].dend = (size_t)PyInt_AS_LONG(item);
  }
  return bops;
}

static PyObject*
editops_to_tuple_list(size_t n, LevEditOp *ops)
{
  PyObject *list;
  size_t i;

  list = PyList_New(n);
  for (i = 0; i < n; i++, ops++) {
    PyObject *tuple = PyTuple_New(3);
    PyObject *is = opcode_names[ops->type].pystring;
    Py_INCREF(is);
    PyTuple_SET_ITEM(tuple, 0, is);
    PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong((long)ops->spos));
    PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((long)ops->dpos));
    PyList_SET_ITEM(list, i, tuple);
  }

  return list;
}

static PyObject*
matching_blocks_to_tuple_list(size_t len1, size_t len2,
                              size_t nmb, LevMatchingBlock *mblocks)
{
  PyObject *list, *tuple;
  size_t i;

  list = PyList_New(nmb + 1);
  for (i = 0; i < nmb; i++, mblocks++) {
    tuple = PyTuple_New(3);
    PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong((long)mblocks->spos));
    PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong((long)mblocks->dpos));
    PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((long)mblocks->len));
    PyList_SET_ITEM(list, i, tuple);
  }
  tuple = PyTuple_New(3);
  PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong((long)len1));
  PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong((long)len2));
  PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((long)0));
  PyList_SET_ITEM(list, nmb, tuple);

  return list;
}

static size_t
get_length_of_anything(PyObject *object)
{
  if (PyInt_Check(object)) {
    long int len = PyInt_AS_LONG(object);
    if (len < 0)
      len = -1;
    return (size_t)len;
  }
  if (PySequence_Check(object))
    return PySequence_Length(object);
  return (size_t)-1;
}

static PyObject*
editops_py(PyObject *self, PyObject *args)
{
  PyObject *arg1, *arg2, *arg3 = NULL;
  PyObject *oplist;
  size_t len1, len2, n;
  LevEditOp *ops;
  LevOpCode *bops;

  if (!PyArg_UnpackTuple(args, PYARGCFIX("editops"), 2, 3,
                         &arg1, &arg2, &arg3)) {
    return NULL;
  }

  /* convert: we were called (bops, s1, s2) */
  if (arg3) {
    if (!PyList_Check(arg1)) {
      PyErr_Format(PyExc_ValueError,
                  "editops first argument must be a List of edit operations");
      return NULL;
    }
    n = PyList_GET_SIZE(arg1);
    if (!n) {
      Py_INCREF(arg1);
      return arg1;
    }
    len1 = get_length_of_anything(arg2);
    len2 = get_length_of_anything(arg3);
    if (len1 == (size_t)-1 || len2 == (size_t)-1) {
      PyErr_Format(PyExc_ValueError,
                  "editops second and third argument must specify sizes");
      return NULL;
    }

    if ((bops = extract_opcodes(arg1)) != NULL) {
      if (lev_opcodes_check_errors(len1, len2, n, bops)) {
        PyErr_Format(PyExc_ValueError,
                    "editops edit operation list is invalid");
        free(bops);
        return NULL;
      }
      ops = lev_opcodes_to_editops(n, bops, &n, 0); /* XXX: different n's! */
      if (!ops && n) {
        free(bops);
        return PyErr_NoMemory();
      }
      oplist = editops_to_tuple_list(n, ops);
      free(ops);
      free(bops);
      return oplist;
    }
    if ((ops = extract_editops(arg1)) != NULL) {
      if (lev_editops_check_errors(len1, len2, n, ops)) {
        PyErr_Format(PyExc_ValueError,
                    "editops edit operation list is invalid");
        free(ops);
        return NULL;
      }
      free(ops);
      Py_INCREF(arg1);  /* editops -> editops is identity */
      return arg1;
    }
    if (!PyErr_Occurred())
      PyErr_Format(PyExc_TypeError,
                  "editops first argument must be a List of edit operations");
    return NULL;
  }

  /* find editops: we were called (s1, s2) */
  if (PyObject_TypeCheck(arg1, &PyString_Type)
      && PyObject_TypeCheck(arg2, &PyString_Type)) {
    lev_byte *string1, *string2;

    len1 = PyString_GET_SIZE(arg1);
    len2 = PyString_GET_SIZE(arg2);
    string1 = PyString_AS_STRING(arg1);
    string2 = PyString_AS_STRING(arg2);
    ops = lev_editops_find(len1, string1, len2, string2, &n);
  }
  else if (PyObject_TypeCheck(arg1, &PyUnicode_Type)
      && PyObject_TypeCheck(arg2, &PyUnicode_Type)) {
    Py_UNICODE *string1, *string2;

    len1 = PyUnicode_GET_SIZE(arg1);
    len2 = PyUnicode_GET_SIZE(arg2);
    string1 = PyUnicode_AS_UNICODE(arg1);
    string2 = PyUnicode_AS_UNICODE(arg2);
    ops = lev_u_editops_find(len1, string1, len2, string2, &n);
  }
  else {
    PyErr_Format(PyExc_TypeError,
                 "editops expected two Strings or two Unicodes");
    return NULL;
  }
  if (!ops && n)
    return PyErr_NoMemory();
  oplist = editops_to_tuple_list(n, ops);
  free(ops);
  return oplist;
}

static PyObject*
opcodes_to_tuple_list(size_t nb, LevOpCode *bops)
{
  PyObject *list;
  size_t i;

  list = PyList_New(nb);
  for (i = 0; i < nb; i++, bops++) {
    PyObject *tuple = PyTuple_New(5);
    PyObject *is = opcode_names[bops->type].pystring;
    Py_INCREF(is);
    PyTuple_SET_ITEM(tuple, 0, is);
    PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong((long)bops->sbeg));
    PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((long)bops->send));
    PyTuple_SET_ITEM(tuple, 3, PyInt_FromLong((long)bops->dbeg));
    PyTuple_SET_ITEM(tuple, 4, PyInt_FromLong((long)bops->dend));
    PyList_SET_ITEM(list, i, tuple);
  }

  return list;
}

static PyObject*
opcodes_py(PyObject *self, PyObject *args)
{
  PyObject *arg1, *arg2, *arg3 = NULL;
  PyObject *oplist;
  size_t len1, len2, n, nb;
  LevEditOp *ops;
  LevOpCode *bops;

  if (!PyArg_UnpackTuple(args, PYARGCFIX("opcodes"), 2, 3,
                         &arg1, &arg2, &arg3))
    return NULL;

  /* convert: we were called (ops, s1, s2) */
  if (arg3) {
    if (!PyList_Check(arg1)) {
      PyErr_Format(PyExc_TypeError,
                  "opcodes first argument must be a List of edit operations");
      return NULL;
    }
    n = PyList_GET_SIZE(arg1);
    len1 = get_length_of_anything(arg2);
    len2 = get_length_of_anything(arg3);
    if (len1 == (size_t)-1 || len2 == (size_t)-1) {
      PyErr_Format(PyExc_ValueError,
                  "opcodes second and third argument must specify sizes");
      return NULL;
    }

    if ((ops = extract_editops(arg1)) != NULL) {
      if (lev_editops_check_errors(len1, len2, n, ops)) {
        PyErr_Format(PyExc_ValueError,
                    "opcodes edit operation list is invalid");
        free(ops);
        return NULL;
      }
      bops = lev_editops_to_opcodes(n, ops, &n, len1, len2);  /* XXX: n != n */
      if (!bops && n) {
        free(ops);
        return PyErr_NoMemory();
      }
      oplist = opcodes_to_tuple_list(n, bops);
      free(bops);
      free(ops);
      return oplist;
    }
    if ((bops = extract_opcodes(arg1)) != NULL) {
      if (lev_opcodes_check_errors(len1, len2, n, bops)) {
        PyErr_Format(PyExc_ValueError,
                    "opcodes edit operation list is invalid");
        free(bops);
        return NULL;
      }
      free(bops);
      Py_INCREF(arg1);  /* opcodes -> opcodes is identity */
      return arg1;
    }
    if (!PyErr_Occurred())
      PyErr_Format(PyExc_TypeError,
                  "opcodes first argument must be a List of edit operations");
    return NULL;
  }

  /* find opcodes: we were called (s1, s2) */
  if (PyObject_TypeCheck(arg1, &PyString_Type)
      && PyObject_TypeCheck(arg2, &PyString_Type)) {
    lev_byte *string1, *string2;

    len1 = PyString_GET_SIZE(arg1);
    len2 = PyString_GET_SIZE(arg2);
    string1 = PyString_AS_STRING(arg1);
    string2 = PyString_AS_STRING(arg2);
    ops = lev_editops_find(len1, string1, len2, string2, &n);
  }
  else if (PyObject_TypeCheck(arg1, &PyUnicode_Type)
      && PyObject_TypeCheck(arg2, &PyUnicode_Type)) {
    Py_UNICODE *string1, *string2;

    len1 = PyUnicode_GET_SIZE(arg1);
    len2 = PyUnicode_GET_SIZE(arg2);
    string1 = PyUnicode_AS_UNICODE(arg1);
    string2 = PyUnicode_AS_UNICODE(arg2);
    ops = lev_u_editops_find(len1, string1, len2, string2, &n);
  }
  else {
    PyErr_Format(PyExc_TypeError,
                 "opcodes expected two Strings or two Unicodes");
    return NULL;
  }
  if (!ops && n)
    return PyErr_NoMemory();
  bops = lev_editops_to_opcodes(n, ops, &nb, len1, len2);
  free(ops);
  if (!bops && nb)
    return PyErr_NoMemory();
  oplist = opcodes_to_tuple_list(nb, bops);
  free(bops);
  return oplist;
}

static PyObject*
inverse_py(PyObject *self, PyObject *args)
{
  PyObject *list, *result;
  size_t n;
  LevEditOp *ops;
  LevOpCode *bops;

  if (!PyArg_UnpackTuple(args, PYARGCFIX("inverse"), 1, 1, &list)
      || !PyList_Check(list))
    return NULL;

  n = PyList_GET_SIZE(list);
  if (!n) {
    Py_INCREF(list);
    return list;
  }
  if ((ops = extract_editops(list)) != NULL) {
    lev_editops_invert(n, ops);
    result = editops_to_tuple_list(n, ops);
    free(ops);
    return result;
  }
  if ((bops = extract_opcodes(list)) != NULL) {
    lev_opcodes_invert(n, bops);
    result = opcodes_to_tuple_list(n, bops);
    free(bops);
    return result;
  }

  if (!PyErr_Occurred())
    PyErr_Format(PyExc_TypeError,
                "inverse expected a list of edit operations");
  return NULL;
}

static PyObject*
apply_edit_py(PyObject *self, PyObject *args)
{
  PyObject *list, *result, *arg1, *arg2;
  size_t n, len, len1, len2;
  LevEditOp *ops;
  LevOpCode *bops;

  if (!PyArg_UnpackTuple(args, PYARGCFIX("apply_edit"), 3, 3,
                         &list, &arg1, &arg2))
    return NULL;

  if (!PyList_Check(list)) {
    PyErr_Format(PyExc_TypeError,
                 "apply_edit first argument must be a List of edit operations");
    return NULL;
  }
  n = PyList_GET_SIZE(list);

  if (PyObject_TypeCheck(arg1, &PyString_Type)
      && PyObject_TypeCheck(arg2, &PyString_Type)) {
    lev_byte *string1, *string2, *s;

    if (!n) {
      Py_INCREF(arg1);
      return arg1;
    }
    len1 = PyString_GET_SIZE(arg1);
    len2 = PyString_GET_SIZE(arg2);
    string1 = PyString_AS_STRING(arg1);
    string2 = PyString_AS_STRING(arg2);

    if ((ops = extract_editops(list)) != NULL) {
      if (lev_editops_check_errors(len1, len2, n, ops)) {
        PyErr_Format(PyExc_ValueError,
                     "apply_edit edit operations are invalid or inapplicable");
        free(ops);
        return NULL;
      }
      s = lev_editops_apply(len1, string1, len2, string2,
                            n, ops, &len);
      free(ops);
      if (!s && len)
        return PyErr_NoMemory();
      result = PyString_FromStringAndSize(s, len);
      free(s);
      return result;
    }
    if ((bops = extract_opcodes(list)) != NULL) {
      if (lev_opcodes_check_errors(len1, len2, n, bops)) {
        PyErr_Format(PyExc_ValueError,
                     "apply_edit edit operations are invalid or inapplicable");
        free(bops);
        return NULL;
      }
      s = lev_opcodes_apply(len1, string1, len2, string2,
                            n, bops, &len);
      free(bops);
      if (!s && len)
        return PyErr_NoMemory();
      result = PyString_FromStringAndSize(s, len);
      free(s);
      return result;
    }

    if (!PyErr_Occurred())
      PyErr_Format(PyExc_TypeError,
                  "apply_edit first argument must be "
                  "a list of edit operations");
    return NULL;
  }
  if (PyObject_TypeCheck(arg1, &PyUnicode_Type)
      && PyObject_TypeCheck(arg2, &PyUnicode_Type)) {
    Py_UNICODE *string1, *string2, *s;

    if (!n) {
      Py_INCREF(arg1);
      return arg1;
    }
    len1 = PyUnicode_GET_SIZE(arg1);
    len2 = PyUnicode_GET_SIZE(arg2);
    string1 = PyUnicode_AS_UNICODE(arg1);
    string2 = PyUnicode_AS_UNICODE(arg2);

    if ((ops = extract_editops(list)) != NULL) {
      if (lev_editops_check_errors(len1, len2, n, ops)) {
        PyErr_Format(PyExc_ValueError,
                     "apply_edit edit operations are invalid or inapplicable");
        free(ops);
        return NULL;
      }
      s = lev_u_editops_apply(len1, string1, len2, string2,
                              n, ops, &len);
      free(ops);
      if (!s && len)
        return PyErr_NoMemory();
      result = PyUnicode_FromUnicode(s, len);
      free(s);
      return result;
    }
    if ((bops = extract_opcodes(list)) != NULL) {
      if (lev_opcodes_check_errors(len1, len2, n, bops)) {
        PyErr_Format(PyExc_ValueError,
                     "apply_edit edit operations are invalid or inapplicable");
        free(bops);
        return NULL;
      }
      s = lev_u_opcodes_apply(len1, string1, len2, string2,
                              n, bops, &len);
      free(bops);
      if (!s && len)
        return PyErr_NoMemory();
      result = PyUnicode_FromUnicode(s, len);
      free(s);
      return result;
    }

    if (!PyErr_Occurred())
      PyErr_Format(PyExc_TypeError,
                   "apply_edit first argument must be "
                   "a list of edit operations");
    return NULL;
  }

  PyErr_Format(PyExc_TypeError,
               "apply_edit expected two Strings or two Unicodes");
  return NULL;
}

static PyObject*
matching_blocks_py(PyObject *self, PyObject *args)
{
  PyObject *list, *arg1, *arg2, *result;
  size_t n, nmb, len1, len2;
  LevEditOp *ops;
  LevOpCode *bops;
  LevMatchingBlock *mblocks;

  if (!PyArg_UnpackTuple(args, PYARGCFIX("matching_blocks"), 3, 3,
                         &list, &arg1, &arg2)
      || !PyList_Check(list))
    return NULL;

  if (!PyList_Check(list)) {
    PyErr_Format(PyExc_TypeError,
                 "matching_blocks first argument must be "
                 "a List of edit operations");
    return NULL;
  }
  n = PyList_GET_SIZE(list);
  len1 = get_length_of_anything(arg1);
  len2 = get_length_of_anything(arg2);
  if (len1 == (size_t)-1 || len2 == (size_t)-1) {
    PyErr_Format(PyExc_ValueError,
                 "matching_blocks second and third argument "
                 "must specify sizes");
    return NULL;
  }

  if ((ops = extract_editops(list)) != NULL) {
    if (lev_editops_check_errors(len1, len2, n, ops)) {
      PyErr_Format(PyExc_ValueError,
                   "apply_edit edit operations are invalid or inapplicable");
      free(ops);
      return NULL;
    }
    mblocks = lev_editops_matching_blocks(len1, len2, n, ops, &nmb);
    free(ops);
    if (!mblocks && nmb)
      return PyErr_NoMemory();
    result = matching_blocks_to_tuple_list(len1, len2, nmb, mblocks);
    free(mblocks);
    return result;
  }
  if ((bops = extract_opcodes(list)) != NULL) {
    if (lev_opcodes_check_errors(len1, len2, n, bops)) {
      PyErr_Format(PyExc_ValueError,
                   "apply_edit edit operations are invalid or inapplicable");
      free(bops);
      return NULL;
    }
    mblocks = lev_opcodes_matching_blocks(len1, len2, n, bops, &nmb);
    free(bops);
    if (!mblocks && nmb)
      return PyErr_NoMemory();
    result = matching_blocks_to_tuple_list(len1, len2, nmb, mblocks);
    free(mblocks);
    return result;
  }

  if (!PyErr_Occurred())
    PyErr_Format(PyExc_TypeError,
                "inverse expected a list of edit operations");
  return NULL;
}

static PyObject*
subtract_edit_py(PyObject *self, PyObject *args)
{
  PyObject *list, *sub, *result;
  size_t n, ns, nr;
  LevEditOp *ops, *osub, *orem;

  if (!PyArg_UnpackTuple(args, PYARGCFIX("subtract_edit"), 2, 2, &list, &sub)
      || !PyList_Check(list))
    return NULL;

  ns = PyList_GET_SIZE(sub);
  if (!ns) {
    Py_INCREF(list);
    return list;
  }
  n = PyList_GET_SIZE(list);
  if (!n) {
    PyErr_Format(PyExc_ValueError,
                 "subtract_edit subsequence is not a subsequence "
                 "or is invalid");
    return NULL;
  }

  if ((ops = extract_editops(list)) != NULL) {
      if ((osub = extract_editops(sub)) != NULL) {
          orem = lev_editops_subtract(n, ops, ns, osub, &nr);
          free(ops);
          free(osub);

          if (!orem && nr == -1) {
              PyErr_Format(PyExc_ValueError,
                           "subtract_edit subsequence is not a subsequence "
                           "or is invalid");
              return NULL;
          }
          result = editops_to_tuple_list(nr, orem);
          free(orem);

          return result;
      }
      free(ops);
  }

  if (!PyErr_Occurred())
    PyErr_Format(PyExc_TypeError,
                "subtract_edit expected two lists of edit operations");
  return NULL;
}


PY_MOD_INIT_FUNC_DEF(_levenshtein)
{
#ifdef LEV_PYTHON3
  PyObject *module;
#endif
  size_t i;

  PY_INIT_MOD(module, "_levenshtein", Levenshtein_DESC, methods)
  /* create intern strings for edit operation names */
  if (opcode_names[0].pystring)
    abort();
  for (i = 0; i < N_OPCODE_NAMES; i++) {
#ifdef LEV_PYTHON3
    opcode_names[i].pystring
      = PyUnicode_InternFromString(opcode_names[i].cstring);
#else
    opcode_names[i].pystring
      = PyString_InternFromString(opcode_names[i].cstring);
#endif
    opcode_names[i].len = strlen(opcode_names[i].cstring);
  }
  lev_init_rng(0);
#ifdef LEV_PYTHON3
  return module;
#endif
}
/* }}} */
#endif /* not NO_PYTHON */

/****************************************************************************
 *
 * C (i.e. executive) part
 *
 ****************************************************************************/

/****************************************************************************
 *
 * Taus113
 *
 ****************************************************************************/
/* {{{ */

/*
 * Based on Tausworthe random generator implementation rng/taus113.c
 * from the GNU Scientific Library (http://sources.redhat.com/gsl/)
 * which has the notice
 * Copyright (C) 2002 Atakan Gurkan
 * Based on the file taus.c which has the notice
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 James Theiler, Brian Gough
 */

static inline unsigned long
taus113_get(taus113_state_t *state)
{
  unsigned long b1, b2, b3, b4;

  b1 = ((((state->z1 << 6UL) & TAUS113_MASK) ^ state->z1) >> 13UL);
  state->z1 = ((((state->z1 & 4294967294UL) << 18UL) & TAUS113_MASK) ^ b1);

  b2 = ((((state->z2 << 2UL) & TAUS113_MASK) ^ state->z2) >> 27UL);
  state->z2 = ((((state->z2 & 4294967288UL) << 2UL) & TAUS113_MASK) ^ b2);

  b3 = ((((state->z3 << 13UL) & TAUS113_MASK) ^ state->z3) >> 21UL);
  state->z3 = ((((state->z3 & 4294967280UL) << 7UL) & TAUS113_MASK) ^ b3);

  b4 = ((((state->z4 << 3UL) & TAUS113_MASK) ^ state->z4) >> 12UL);
  state->z4 = ((((state->z4 & 4294967168UL) << 13UL) & TAUS113_MASK) ^ b4);

  return (state->z1 ^ state->z2 ^ state->z3 ^ state->z4);

}

static void
taus113_set(taus113_state_t *state,
            unsigned long int s)
{
  if (!s)
    s = 1UL;                    /* default seed is 1 */

  state->z1 = TAUS113_LCG(s);
  if (state->z1 < 2UL)
    state->z1 += 2UL;

  state->z2 = TAUS113_LCG(state->z1);
  if (state->z2 < 8UL)
    state->z2 += 8UL;

  state->z3 = TAUS113_LCG(state->z2);
  if (state->z3 < 16UL)
    state->z3 += 16UL;

  state->z4 = TAUS113_LCG(state->z3);
  if (state->z4 < 128UL)
    state->z4 += 128UL;

  /* Calling RNG ten times to satify recurrence condition */
  taus113_get(state);
  taus113_get(state);
  taus113_get(state);
  taus113_get(state);
  taus113_get(state);
  taus113_get(state);
  taus113_get(state);
  taus113_get(state);
  taus113_get(state);
  taus113_get(state);
}


/**
 * lev_init_rng:
 * @seed: A seed.  If zero, a default value (currently 1) is used instead.
 *
 * Initializes the random generator used by some Levenshtein functions.
 *
 * This does NOT happen automatically when these functions are used.
 **/
_LEV_STATIC_PY void
lev_init_rng(unsigned long int seed)
{
  static taus113_state_t state;

  taus113_set(&state, seed);
}
/* }}} */

/****************************************************************************
 *
 * Basic stuff, Levenshtein distance
 *
 ****************************************************************************/
/* {{{ */

/**
 * lev_edit_distance:
 * @len1: The length of @string1.
 * @string1: A sequence of bytes of length @len1, may contain NUL characters.
 * @len2: The length of @string2.
 * @string2: A sequence of bytes of length @len2, may contain NUL characters.
 * @xcost: If nonzero, the replace operation has weight 2, otherwise all
 *         edit operations have equal weights of 1.
 *
 * Computes Levenshtein edit distance of two strings.
 *
 * Returns: The edit distance.
 **/
_LEV_STATIC_PY size_t
lev_edit_distance(size_t len1, const lev_byte *string1,
                  size_t len2, const lev_byte *string2,
                  int xcost)
{
  size_t i;
  size_t *row;  /* we only need to keep one row of costs */
  size_t *end;
  size_t half;

  /* strip common prefix */
  while (len1 > 0 && len2 > 0 && *string1 == *string2) {
    len1--;
    len2--;
    string1++;
    string2++;
  }

  /* strip common suffix */
  while (len1 > 0 && len2 > 0 && string1[len1-1] == string2[len2-1]) {
    len1--;
    len2--;
  }

  /* catch trivial cases */
  if (len1 == 0)
    return len2;
  if (len2 == 0)
    return len1;

  /* make the inner cycle (i.e. string2) the longer one */
  if (len1 > len2) {
    size_t nx = len1;
    const lev_byte *sx = string1;
    len1 = len2;
    len2 = nx;
    string1 = string2;
    string2 = sx;
  }
  /* check len1 == 1 separately */
  if (len1 == 1) {
    if (xcost)
      return len2 + 1 - 2*(memchr(string2, *string1, len2) != NULL);
    else
      return len2 - (memchr(string2, *string1, len2) != NULL);
  }
  len1++;
  len2++;
  half = len1 >> 1;

  /* initalize first row */
  row = (size_t*)safe_malloc(len2, sizeof(size_t));
  if (!row)
    return (size_t)(-1);
  end = row + len2 - 1;
  for (i = 0; i < len2 - (xcost ? 0 : half); i++)
    row[i] = i;

  /* go through the matrix and compute the costs.  yes, this is an extremely
   * obfuscated version, but also extremely memory-conservative and relatively
   * fast.  */
  if (xcost) {
    for (i = 1; i < len1; i++) {
      size_t *p = row + 1;
      const lev_byte char1 = string1[i - 1];
      const lev_byte *char2p = string2;
      size_t D = i;
      size_t x = i;
      while (p <= end) {
        if (char1 == *(char2p++))
          x = --D;
        else
          x++;
        D = *p;
        D++;
        if (x > D)
          x = D;
        *(p++) = x;
      }
    }
  }
  else {
    /* in this case we don't have to scan two corner triangles (of size len1/2)
     * in the matrix because no best path can go throught them. note this
     * breaks when len1 == len2 == 2 so the memchr() special case above is
     * necessary */
    row[0] = len1 - half - 1;
    for (i = 1; i < len1; i++) {
      size_t *p;
      const lev_byte char1 = string1[i - 1];
      const lev_byte *char2p;
      size_t D, x;
      /* skip the upper triangle */
      if (i >= len1 - half) {
        size_t offset = i - (len1 - half);
        size_t c3;

        char2p = string2 + offset;
        p = row + offset;
        c3 = *(p++) + (char1 != *(char2p++));
        x = *p;
        x++;
        D = x;
        if (x > c3)
          x = c3;
        *(p++) = x;
      }
      else {
        p = row + 1;
        char2p = string2;
        D = x = i;
      }
      /* skip the lower triangle */
      if (i <= half + 1)
        end = row + len2 + i - half - 2;
      /* main */
      while (p <= end) {
        size_t c3 = --D + (char1 != *(char2p++));
        x++;
        if (x > c3)
          x = c3;
        D = *p;
        D++;
        if (x > D)
          x = D;
        *(p++) = x;
      }
      /* lower triangle sentinel */
      if (i <= half) {
        size_t c3 = --D + (char1 != *char2p);
        x++;
        if (x > c3)
          x = c3;
        *p = x;
      }
    }
  }

  i = *end;
  free(row);
  return i;
}

_LEV_STATIC_PY double
lev_edit_distance_sod(size_t len, const lev_byte *string,
                      size_t n, const size_t *lengths,
                      const lev_byte *strings[],
                      const double *weights,
                      int xcost)
{
  size_t i, d;
  double sum = 0.0;

  for (i = 0; i < n; i++) {
    d = lev_edit_distance(len, string, lengths[i], strings[i], xcost);
    if (d == (size_t)-1)
      return -1.0;
    sum += weights[i]*d;
  }
  return sum;
}

/**
 * lev_u_edit_distance:
 * @len1: The length of @string1.
 * @string1: A sequence of Unicode characters of length @len1, may contain NUL
 *           characters.
 * @len2: The length of @string2.
 * @string2: A sequence of Unicode characters of length @len2, may contain NUL
 *           characters.
 * @xcost: If nonzero, the replace operation has weight 2, otherwise all
 *         edit operations have equal weights of 1.
 *
 * Computes Levenshtein edit distance of two Unicode strings.
 *
 * Returns: The edit distance.
 **/
_LEV_STATIC_PY size_t
lev_u_edit_distance(size_t len1, const lev_wchar *string1,
                    size_t len2, const lev_wchar *string2,
                    int xcost)
{
  size_t i;
  size_t *row;  /* we only need to keep one row of costs */
  size_t *end;
  size_t half;

  /* strip common prefix */
  while (len1 > 0 && len2 > 0 && *string1 == *string2) {
    len1--;
    len2--;
    string1++;
    string2++;
  }

  /* strip common suffix */
  while (len1 > 0 && len2 > 0 && string1[len1-1] == string2[len2-1]) {
    len1--;
    len2--;
  }

  /* catch trivial cases */
  if (len1 == 0)
    return len2;
  if (len2 == 0)
    return len1;

  /* make the inner cycle (i.e. string2) the longer one */
  if (len1 > len2) {
    size_t nx = len1;
    const lev_wchar *sx = string1;
    len1 = len2;
    len2 = nx;
    string1 = string2;
    string2 = sx;
  }
  /* check len1 == 1 separately */
  if (len1 == 1) {
    lev_wchar z = *string1;
    const lev_wchar *p = string2;
    for (i = len2; i; i--) {
      if (*(p++) == z)
        return len2 - 1;
    }
    return len2 + (xcost != 0);
  }
  len1++;
  len2++;
  half = len1 >> 1;

  /* initalize first row */
  row = (size_t*)safe_malloc(len2, sizeof(size_t));
  if (!row)
    return (size_t)(-1);
  end = row + len2 - 1;
  for (i = 0; i < len2 - (xcost ? 0 : half); i++)
    row[i] = i;

  /* go through the matrix and compute the costs.  yes, this is an extremely
   * obfuscated version, but also extremely memory-conservative and relatively
   * fast.  */
  if (xcost) {
    for (i = 1; i < len1; i++) {
      size_t *p = row + 1;
      const lev_wchar char1 = string1[i - 1];
      const lev_wchar *char2p = string2;
      size_t D = i - 1;
      size_t x = i;
      while (p <= end) {
        if (char1 == *(char2p++))
          x = D;
        else
          x++;
        D = *p;
        if (x > D + 1)
          x = D + 1;
        *(p++) = x;
      }
    }
  }
  else {
    /* in this case we don't have to scan two corner triangles (of size len1/2)
     * in the matrix because no best path can go throught them. note this
     * breaks when len1 == len2 == 2 so the memchr() special case above is
     * necessary */
    row[0] = len1 - half - 1;
    for (i = 1; i < len1; i++) {
      size_t *p;
      const lev_wchar char1 = string1[i - 1];
      const lev_wchar *char2p;
      size_t D, x;
      /* skip the upper triangle */
      if (i >= len1 - half) {
        size_t offset = i - (len1 - half);
        size_t c3;

        char2p = string2 + offset;
        p = row + offset;
        c3 = *(p++) + (char1 != *(char2p++));
        x = *p;
        x++;
        D = x;
        if (x > c3)
          x = c3;
        *(p++) = x;
      }
      else {
        p = row + 1;
        char2p = string2;
        D = x = i;
      }
      /* skip the lower triangle */
      if (i <= half + 1)
        end = row + len2 + i - half - 2;
      /* main */
      while (p <= end) {
        size_t c3 = --D + (char1 != *(char2p++));
        x++;
        if (x > c3)
          x = c3;
        D = *p;
        D++;
        if (x > D)
          x = D;
        *(p++) = x;
      }
      /* lower triangle sentinel */
      if (i <= half) {
        size_t c3 = --D + (char1 != *char2p);
        x++;
        if (x > c3)
          x = c3;
        *p = x;
      }
    }
  }

  i = *end;
  free(row);
  return i;
}

_LEV_STATIC_PY double
lev_u_edit_distance_sod(size_t len, const lev_wchar *string,
                        size_t n, const size_t *lengths,
                        const lev_wchar *strings[],
                        const double *weights,
                        int xcost)
{
  size_t i, d;
  double sum = 0.0;

  for (i = 0; i < n; i++) {
    d = lev_u_edit_distance(len, string, lengths[i], strings[i], xcost);
    if (d == (size_t)-1)
      return -1.0;
    sum += weights[i]*d;
  }
  return sum;
}
/* }}} */


/****************************************************************************
 *
 * Other simple distances: Hamming, Jaro, Jaro-Winkler
 *
 ****************************************************************************/
/* {{{ */
/**
 * lev_hamming_distance:
 * @len: The length of @string1 and @string2.
 * @string1: A sequence of bytes of length @len1, may contain NUL characters.
 * @string2: A sequence of bytes of length @len2, may contain NUL characters.
 *
 * Computes Hamming distance of two strings.
 *
 * The strings must have the same length.
 *
 * Returns: The Hamming distance.
 **/
_LEV_STATIC_PY size_t
lev_hamming_distance(size_t len,
                     const lev_byte *string1,
                     const lev_byte *string2)
{
  size_t dist, i;

  dist = 0;
  for (i = len; i; i--) {
    if (*(string1++) != *(string2++))
      dist++;
  }

  return dist;
}

/**
 * lev_u_hamming_distance:
 * @len: The length of @string1 and @string2.
 * @string1: A sequence of Unicode characters of length @len1, may contain NUL
 *           characters.
 * @string2: A sequence of Unicode characters of length @len2, may contain NUL
 *           characters.
 *
 * Computes Hamming distance of two strings.
 *
 * The strings must have the same length.
 *
 * Returns: The Hamming distance.
 **/
_LEV_STATIC_PY size_t
lev_u_hamming_distance(size_t len,
                       const lev_wchar *string1,
                       const lev_wchar *string2)
{
  size_t dist, i;

  dist = 0;
  for (i = len; i; i--) {
    if (*(string1++) != *(string2++))
      dist++;
  }

  return dist;
}

/**
 * lev_jaro_ratio:
 * @len1: The length of @string1.
 * @string1: A sequence of bytes of length @len1, may contain NUL characters.
 * @len2: The length of @string2.
 * @string2: A sequence of bytes of length @len2, may contain NUL characters.
 *
 * Computes Jaro string similarity metric of two strings.
 *
 * Returns: The Jaro metric of @string1 and @string2.
 **/
_LEV_STATIC_PY double
lev_jaro_ratio(size_t len1, const lev_byte *string1,
               size_t len2, const lev_byte *string2)
{
  size_t i, j, halflen, trans, match, to;
  size_t *idx;
  double md;

  if (len1 == 0 || len2 == 0) {
    if (len1 == 0 && len2 == 0)
      return 1.0;
    return 0.0;
  }
  /* make len1 always shorter (or equally long) */
  if (len1 > len2) {
    const lev_byte *b;

    b = string1;
    string1 = string2;
    string2 = b;

    i = len1;
    len1 = len2;
    len2 = i;
  }

  halflen = (len1 + 1)/2;
  idx = (size_t*)calloc(len1, sizeof(size_t));
  if (!idx)
    return -1.0;

  /* The literature about Jaro metric is confusing as the method of assigment
   * of common characters is nowhere specified.  There are several possible
   * deterministic mutual assignments of common characters of two strings.
   * We use earliest-position method, which is however suboptimal (e.g., it
   * gives two transpositions in jaro("Jaro", "Joaro") because of assigment
   * of the first `o').  No reasonable algorithm for the optimal one is
   * currently known to me. */
  match = 0;
  /* the part with allowed range overlapping left */
  for (i = 0; i < halflen; i++) {
    for (j = 0; j <= i+halflen; j++) {
      if (string1[j] == string2[i] && !idx[j]) {
        match++;
        idx[j] = match;
        break;
      }
    }
  }
  /* the part with allowed range overlapping right */
  to = len1+halflen < len2 ? len1+halflen : len2;
  for (i = halflen; i < to; i++) {
    for (j = i - halflen; j < len1; j++) {
      if (string1[j] == string2[i] && !idx[j]) {
        match++;
        idx[j] = match;
        break;
      }
    }
  }
  if (!match) {
    free(idx);
    return 0.0;
  }
  /* count transpositions */
  i = 0;
  trans = 0;
  for (j = 0; j < len1; j++) {
    if (idx[j]) {
      i++;
      if (idx[j] != i)
        trans++;
    }
  }
  free(idx);

  md = (double)match;
  return (md/len1 + md/len2 + 1.0 - trans/md/2.0)/3.0;
}

/**
 * lev_u_jaro_ratio:
 * @len1: The length of @string1.
 * @string1: A sequence of Unicode characters of length @len1, may contain NUL
 *           characters.
 * @len2: The length of @string2.
 * @string2: A sequence of Unicode characters of length @len2, may contain NUL
 *           characters.
 *
 * Computes Jaro string similarity metric of two Unicode strings.
 *
 * Returns: The Jaro metric of @string1 and @string2.
 **/
_LEV_STATIC_PY double
lev_u_jaro_ratio(size_t len1, const lev_wchar *string1,
                 size_t len2, const lev_wchar *string2)
{
  size_t i, j, halflen, trans, match, to;
  size_t *idx;
  double md;

  if (len1 == 0 || len2 == 0) {
    if (len1 == 0 && len2 == 0)
      return 1.0;
    return 0.0;
  }
  /* make len1 always shorter (or equally long) */
  if (len1 > len2) {
    const lev_wchar *b;

    b = string1;
    string1 = string2;
    string2 = b;

    i = len1;
    len1 = len2;
    len2 = i;
  }

  halflen = (len1 + 1)/2;
  idx = (size_t*)calloc(len1, sizeof(size_t));
  if (!idx)
    return -1.0;

  match = 0;
  /* the part with allowed range overlapping left */
  for (i = 0; i < halflen; i++) {
    for (j = 0; j <= i+halflen; j++) {
      if (string1[j] == string2[i] && !idx[j]) {
        match++;
        idx[j] = match;
        break;
      }
    }
  }
  /* the part with allowed range overlapping right */
  to = len1+halflen < len2 ? len1+halflen : len2;
  for (i = halflen; i < to; i++) {
    for (j = i - halflen; j < len1; j++) {
      if (string1[j] == string2[i] && !idx[j]) {
        match++;
        idx[j] = match;
        break;
      }
    }
  }
  if (!match) {
    free(idx);
    return 0.0;
  }
  /* count transpositions */
  i = 0;
  trans = 0;
  for (j = 0; j < len1; j++) {
    if (idx[j]) {
      i++;
      if (idx[j] != i)
        trans++;
    }
  }
  free(idx);

  md = (double)match;
  return (md/len1 + md/len2 + 1.0 - trans/md/2.0)/3.0;
}

/**
 * lev_jaro_winkler_ratio:
 * @len1: The length of @string1.
 * @string1: A sequence of bytes of length @len1, may contain NUL characters.
 * @len2: The length of @string2.
 * @string2: A sequence of bytes of length @len2, may contain NUL characters.
 * @pfweight: Prefix weight, i.e., how much weight should be given to a
 *            common prefix.
 *
 * Computes Jaro-Winkler string similarity metric of two strings.
 *
 * The formula is J+@pfweight*P*(1-J), where J is Jaro metric and P is the
 * length of common prefix.
 *
 * Returns: The Jaro-Winkler metric of @string1 and @string2.
 **/
_LEV_STATIC_PY double
lev_jaro_winkler_ratio(size_t len1, const lev_byte *string1,
                       size_t len2, const lev_byte *string2,
                       double pfweight)
{
  double j;
  size_t p, m;

  j = lev_jaro_ratio(len1, string1, len2, string2);
  m = len1 < len2 ? len1 : len2;
  for (p = 0; p < m; p++) {
    if (string1[p] != string2[p])
      break;
  }
  j += (1.0 - j)*p*pfweight;
  return j > 1.0 ? 1.0 : j;
}

/**
 * lev_u_jaro_winkler_ratio:
 * @len1: The length of @string1.
 * @string1: A sequence of Unicode characters of length @len1, may contain NUL
 *           characters.
 * @len2: The length of @string2.
 * @string2: A sequence of Unicode characters of length @len2, may contain NUL
 *           characters.
 * @pfweight: Prefix weight, i.e., how much weight should be given to a
 *            common prefix.
 *
 * Computes Jaro-Winkler string similarity metric of two Unicode strings.
 *
 * The formula is J+@pfweight*P*(1-J), where J is Jaro metric and P is the
 * length of common prefix.
 *
 * Returns: The Jaro-Winkler metric of @string1 and @string2.
 **/
_LEV_STATIC_PY double
lev_u_jaro_winkler_ratio(size_t len1, const lev_wchar *string1,
                         size_t len2, const lev_wchar *string2,
                         double pfweight)
{
  double j;
  size_t p, m;

  j = lev_u_jaro_ratio(len1, string1, len2, string2);
  m = len1 < len2 ? len1 : len2;
  for (p = 0; p < m; p++) {
    if (string1[p] != string2[p])
      break;
  }
  j += (1.0 - j)*p*pfweight;
  return j > 1.0 ? 1.0 : j;
}
/* }}} */

/****************************************************************************
 *
 * Generalized medians, the greedy algorithm, and greedy improvements
 *
 ****************************************************************************/
/* {{{ */

/* compute the sets of symbols each string contains, and the set of symbols
 * in any of them (symset).  meanwhile, count how many different symbols
 * there are (used below for symlist). */
static lev_byte*
make_symlist(size_t n, const size_t *lengths,
             const lev_byte *strings[], size_t *symlistlen)
{
  short int *symset;  /* indexed by ALL symbols, contains 1 for symbols
                         present in the strings, zero for others */
  size_t i, j;
  lev_byte *symlist;

  symset = calloc(0x100, sizeof(short int));
  if (!symset) {
    *symlistlen = (size_t)(-1);
    return NULL;
  }
  *symlistlen = 0;
  for (i = 0; i < n; i++) {
    const lev_byte *stri = strings[i];
    for (j = 0; j < lengths[i]; j++) {
      int c = stri[j];
      if (!symset[c]) {
        (*symlistlen)++;
        symset[c] = 1;
      }
    }
  }
  if (!*symlistlen) {
    free(symset);
    return NULL;
  }

  /* create dense symbol table, so we can easily iterate over only characters
   * present in the strings */
  {
    size_t pos = 0;
    symlist = (lev_byte*)safe_malloc((*symlistlen), sizeof(lev_byte));
    if (!symlist) {
      *symlistlen = (size_t)(-1);
      free(symset);
      return NULL;
    }
    for (j = 0; j < 0x100; j++) {
      if (symset[j])
        symlist[pos++] = (lev_byte)j;
    }
  }
  free(symset);

  return symlist;
}

/**
 * lev_greedy_median:
 * @n: The size of @lengths, @strings, and @weights.
 * @lengths: The lengths of @strings.
 * @strings: An array of strings, that may contain NUL characters.
 * @weights: The string weights (they behave exactly as multiplicities, though
 *           any positive value is allowed, not just integers).
 * @medlength: Where the length of the median should be stored.
 *
 * Finds a generalized median string of @strings using the greedy algorithm.
 *
 * Note it's considerably more efficient to give a string with weight 2
 * than to store two identical strings in @strings (with weights 1).
 *
 * Returns: The generalized median, as a newly allocated string; its length
 *          is stored in @medlength.
 **/
_LEV_STATIC_PY lev_byte*
lev_greedy_median(size_t n, const size_t *lengths,
                  const lev_byte *strings[],
                  const double *weights,
                  size_t *medlength)
{
  size_t i;  /* usually iterates over strings (n) */
  size_t j;  /* usually iterates over characters */
  size_t len;  /* usually iterates over the approximate median string */
  lev_byte *symlist;  /* list of symbols present in the strings,
                              we iterate over it insead of set of all
                              existing symbols */
  size_t symlistlen;  /* length of symlist */
  size_t maxlen;  /* maximum input string length */
  size_t stoplen;  /* maximum tried median string length -- this is slightly
                      higher than maxlen, because the median string may be
                      longer than any of the input strings */
  size_t **rows;  /* Levenshtein matrix rows for each string, we need to keep
                     only one previous row to construct the current one */
  size_t *row;  /* a scratch buffer for new Levenshtein matrix row computation,
                   shared among all strings */
  lev_byte *median;  /* the resulting approximate median string */
  double *mediandist;  /* the total distance of the best median string of
                          given length.  warning!  mediandist[0] is total
                          distance for empty string, while median[] itself
                          is normally zero-based */
  size_t bestlen;  /* the best approximate median string length */

  /* find all symbols */
  symlist = make_symlist(n, lengths, strings, &symlistlen);
  if (!symlist) {
    *medlength = 0;
    if (symlistlen != 0)
      return NULL;
    else
      return calloc(1, sizeof(lev_byte));
  }

  /* allocate and initialize per-string matrix rows and a common work buffer */
  rows = (size_t**)safe_malloc(n, sizeof(size_t*));
  if (!rows) {
    free(symlist);
    return NULL;
  }
  maxlen = 0;
  for (i = 0; i < n; i++) {
    size_t *ri;
    size_t leni = lengths[i];
    if (leni > maxlen)
      maxlen = leni;
    ri = rows[i] = (size_t*)safe_malloc((leni + 1), sizeof(size_t));
    if (!ri) {
      for (j = 0; j < i; j++)
        free(rows[j]);
      free(rows);
      free(symlist);
      return NULL;
    }
    for (j = 0; j <= leni; j++)
      ri[j] = j;
  }
  stoplen = 2*maxlen + 1;
  row = (size_t*)safe_malloc((stoplen + 1), sizeof(size_t));
  if (!row) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(symlist);
    return NULL;
  }

  /* compute final cost of string of length 0 (empty string may be also
   * a valid answer) */
  median = (lev_byte*)safe_malloc(stoplen, sizeof(lev_byte));
  if (!median) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(row);
    free(symlist);
    return NULL;
  }
  mediandist = (double*)safe_malloc((stoplen + 1), sizeof(double));
  if (!mediandist) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(row);
    free(symlist);
    free(median);
    return NULL;
  }
  mediandist[0] = 0.0;
  for (i = 0; i < n; i++)
    mediandist[0] += lengths[i]*weights[i];

  /* build up the approximate median string symbol by symbol
   * XXX: we actually exit on break below, but on the same condition */
  for (len = 1; len <= stoplen; len++) {
    lev_byte symbol;
    double minminsum = LEV_INFINITY;
    row[0] = len;
    /* iterate over all symbols we may want to add */
    for (j = 0; j < symlistlen; j++) {
      double totaldist = 0.0;
      double minsum = 0.0;
      symbol = symlist[j];
      /* sum Levenshtein distances from all the strings, with given weights */
      for (i = 0; i < n; i++) {
        const lev_byte *stri = strings[i];
        size_t *p = rows[i];
        size_t leni = lengths[i];
        size_t *end = rows[i] + leni;
        size_t min = len;
        size_t x = len; /* == row[0] */
        /* compute how another row of Levenshtein matrix would look for median
         * string with this symbol added */
        while (p < end) {
          size_t D = *(p++) + (symbol != *(stri++));
          x++;
          if (x > D)
            x = D;
          if (x > *p + 1)
            x = *p + 1;
          if (x < min)
            min = x;
        }
        minsum += min*weights[i];
        totaldist += x*weights[i];
      }
      /* is this symbol better than all the others? */
      if (minsum < minminsum) {
        minminsum = minsum;
        mediandist[len] = totaldist;
        median[len - 1] = symbol;
      }
    }
    /* stop the iteration if we no longer need to recompute the matrix rows
     * or when we are over maxlen and adding more characters doesn't seem
     * useful */
    if (len == stoplen
        || (len > maxlen && mediandist[len] > mediandist[len - 1])) {
      stoplen = len;
      break;
    }
    /* now the best symbol is known, so recompute all matrix rows for this
     * one */
    symbol = median[len - 1];
    for (i = 0; i < n; i++) {
      const lev_byte *stri = strings[i];
      size_t *oldrow = rows[i];
      size_t leni = lengths[i];
      size_t k;
      /* compute a row of Levenshtein matrix */
      for (k = 1; k <= leni; k++) {
        size_t c1 = oldrow[k] + 1;
        size_t c2 = row[k - 1] + 1;
        size_t c3 = oldrow[k - 1] + (symbol != stri[k - 1]);
        row[k] = c2 > c3 ? c3 : c2;
        if (row[k] > c1)
          row[k] = c1;
      }
      memcpy(oldrow, row, (leni + 1)*sizeof(size_t));
    }
  }

  /* find the string with minimum total distance */
  bestlen = 0;
  for (len = 1; len <= stoplen; len++) {
    if (mediandist[len] < mediandist[bestlen])
      bestlen = len;
  }

  /* clean up */
  for (i = 0; i < n; i++)
    free(rows[i]);
  free(rows);
  free(row);
  free(symlist);
  free(mediandist);

  /* return result */
  {
    lev_byte *result = (lev_byte*)safe_malloc(bestlen, sizeof(lev_byte));
    if (!result) {
      free(median);
      return NULL;
    }
    memcpy(result, median, bestlen*sizeof(lev_byte));
    free(median);
    *medlength = bestlen;
    return result;
  }
}

/*
 * Knowing the distance matrices up to some row, finish the distance
 * computations.
 *
 * string1, len1 are already shortened.
 */
static double
finish_distance_computations(size_t len1, lev_byte *string1,
                             size_t n, const size_t *lengths,
                             const lev_byte **strings,
                             const double *weights, size_t **rows,
                             size_t *row)
{
  size_t *end;
  size_t i, j;
  size_t offset;  /* row[0]; offset + len1 give together real len of string1 */
  double distsum = 0.0;  /* sum of distances */

  /* catch trivia case */
  if (len1 == 0) {
    for (j = 0; j < n; j++)
      distsum += rows[j][lengths[j]]*weights[j];
    return distsum;
  }

  /* iterate through the strings and sum the distances */
  for (j = 0; j < n; j++) {
    size_t *rowi = rows[j];  /* current row */
    size_t leni = lengths[j];  /* current length */
    size_t len = len1;  /* temporary len1 for suffix stripping */
    const lev_byte *stringi = strings[j];  /* current string */

    /* strip common suffix (prefix CAN'T be stripped) */
    while (len && leni && stringi[leni-1] == string1[len-1]) {
      len--;
      leni--;
    }

    /* catch trivial cases */
    if (len == 0) {
      distsum += rowi[leni]*weights[j];
      continue;
    }
    offset = rowi[0];
    if (leni == 0) {
      distsum += (offset + len)*weights[j];
      continue;
    }

    /* complete the matrix */
    memcpy(row, rowi, (leni + 1)*sizeof(size_t));
    end = row + leni;

    for (i = 1; i <= len; i++) {
      size_t *p = row + 1;
      const lev_byte char1 = string1[i - 1];
      const lev_byte *char2p = stringi;
      size_t D, x;

      D = x = i + offset;
      while (p <= end) {
        size_t c3 = --D + (char1 != *(char2p++));
        x++;
        if (x > c3)
          x = c3;
        D = *p;
        D++;
        if (x > D)
          x = D;
        *(p++) = x;
      }
    }
    distsum += weights[j]*(*end);
  }

  return distsum;
}

/**
 * lev_median_improve:
 * @len: The length of @s.
 * @s: The approximate generalized median string to be improved.
 * @n: The size of @lengths, @strings, and @weights.
 * @lengths: The lengths of @strings.
 * @strings: An array of strings, that may contain NUL characters.
 * @weights: The string weights (they behave exactly as multiplicities, though
 *           any positive value is allowed, not just integers).
 * @medlength: Where the new length of the median should be stored.
 *
 * Tries to make @s a better generalized median string of @strings with
 * small perturbations.
 *
 * It never returns a string with larger SOD than @s; in the worst case, a
 * string identical to @s is returned.
 *
 * Returns: The improved generalized median, as a newly allocated string; its
 *          length is stored in @medlength.
 **/
_LEV_STATIC_PY lev_byte*
lev_median_improve(size_t len, const lev_byte *s,
                   size_t n, const size_t *lengths,
                   const lev_byte *strings[],
                   const double *weights,
                   size_t *medlength)
{
  size_t i;  /* usually iterates over strings (n) */
  size_t j;  /* usually iterates over characters */
  size_t pos;  /* the position in the approximate median string we are
                  trying to change */
  lev_byte *symlist;  /* list of symbols present in the strings,
                              we iterate over it insead of set of all
                              existing symbols */
  size_t symlistlen;  /* length of symlist */
  size_t maxlen;  /* maximum input string length */
  size_t stoplen;  /* maximum tried median string length -- this is slightly
                      higher than maxlen, because the median string may be
                      longer than any of the input strings */
  size_t **rows;  /* Levenshtein matrix rows for each string, we need to keep
                     only one previous row to construct the current one */
  size_t *row;  /* a scratch buffer for new Levenshtein matrix row computation,
                   shared among all strings */
  lev_byte *median;  /* the resulting approximate median string */
  size_t medlen;  /* the current approximate median string length */
  double minminsum;  /* the current total distance sum */

  /* find all symbols */
  symlist = make_symlist(n, lengths, strings, &symlistlen);
  if (!symlist) {
    *medlength = 0;
    if (symlistlen != 0)
      return NULL;
    else
      return calloc(1, sizeof(lev_byte));
  }

  /* allocate and initialize per-string matrix rows and a common work buffer */
  rows = (size_t**)safe_malloc(n, sizeof(size_t*));
  if (!rows) {
    free(symlist);
    return NULL;
  }
  maxlen = 0;
  for (i = 0; i < n; i++) {
    size_t *ri;
    size_t leni = lengths[i];
    if (leni > maxlen)
      maxlen = leni;
    ri = rows[i] = (size_t*)safe_malloc((leni + 1), sizeof(size_t));
    if (!ri) {
      for (j = 0; j < i; j++)
        free(rows[j]);
      free(rows);
      free(symlist);
      return NULL;
    }
    for (j = 0; j <= leni; j++)
      ri[j] = j;
  }
  stoplen = 2*maxlen + 1;
  row = (size_t*)safe_malloc((stoplen + 2), sizeof(size_t));
  if (!row) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(symlist);
    return NULL;
  }

  /* initialize median to given string */
  median = (lev_byte*)safe_malloc((stoplen+1), sizeof(lev_byte));
  if (!median) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(row);
    free(symlist);
    return NULL;
  }
  median++;  /* we need -1st element for insertions a pos 0 */
  medlen = len;
  memcpy(median, s, (medlen)*sizeof(lev_byte));
  minminsum = finish_distance_computations(medlen, median,
                                           n, lengths, strings,
                                           weights, rows, row);

  /* sequentially try perturbations on all positions */
  for (pos = 0; pos <= medlen; ) {
    lev_byte orig_symbol, symbol;
    LevEditType operation;
    double sum;

    symbol = median[pos];
    operation = LEV_EDIT_KEEP;
    /* IF pos < medlength: FOREACH symbol: try to replace the symbol
     * at pos, if some lower the total distance, chooste the best */
    if (pos < medlen) {
      orig_symbol = median[pos];
      for (j = 0; j < symlistlen; j++) {
        if (symlist[j] == orig_symbol)
          continue;
        median[pos] = symlist[j];
        sum = finish_distance_computations(medlen - pos, median + pos,
                                           n, lengths, strings,
                                           weights, rows, row);
        if (sum < minminsum) {
          minminsum = sum;
          symbol = symlist[j];
          operation = LEV_EDIT_REPLACE;
        }
      }
      median[pos] = orig_symbol;
    }
    /* FOREACH symbol: try to add it at pos, if some lower the total
     * distance, chooste the best (increase medlength)
     * We simulate insertion by replacing the character at pos-1 */
    orig_symbol = *(median + pos - 1);
    for (j = 0; j < symlistlen; j++) {
      *(median + pos - 1) = symlist[j];
      sum = finish_distance_computations(medlen - pos + 1, median + pos - 1,
                                          n, lengths, strings,
                                         weights, rows, row);
      if (sum < minminsum) {
        minminsum = sum;
        symbol = symlist[j];
        operation = LEV_EDIT_INSERT;
      }
    }
    *(median + pos - 1) = orig_symbol;
    /* IF pos < medlength: try to delete the symbol at pos, if it lowers
     * the total distance remember it (decrease medlength) */
    if (pos < medlen) {
      sum = finish_distance_computations(medlen - pos - 1, median + pos + 1,
                                         n, lengths, strings,
                                         weights, rows, row);
      if (sum < minminsum) {
        minminsum = sum;
        operation = LEV_EDIT_DELETE;
      }
    }
    /* actually perform the operation */
    switch (operation) {
      case LEV_EDIT_REPLACE:
      median[pos] = symbol;
      break;

      case LEV_EDIT_INSERT:
      memmove(median+pos+1, median+pos,
              (medlen - pos)*sizeof(lev_byte));
      median[pos] = symbol;
      medlen++;
      break;

      case LEV_EDIT_DELETE:
      memmove(median+pos, median + pos+1,
              (medlen - pos-1)*sizeof(lev_byte));
      medlen--;
      break;

      default:
      break;
    }
    assert(medlen <= stoplen);
    /* now the result is known, so recompute all matrix rows and move on */
    if (operation != LEV_EDIT_DELETE) {
      symbol = median[pos];
      row[0] = pos + 1;
      for (i = 0; i < n; i++) {
        const lev_byte *stri = strings[i];
        size_t *oldrow = rows[i];
        size_t leni = lengths[i];
        size_t k;
        /* compute a row of Levenshtein matrix */
        for (k = 1; k <= leni; k++) {
          size_t c1 = oldrow[k] + 1;
          size_t c2 = row[k - 1] + 1;
          size_t c3 = oldrow[k - 1] + (symbol != stri[k - 1]);
          row[k] = c2 > c3 ? c3 : c2;
          if (row[k] > c1)
            row[k] = c1;
        }
        memcpy(oldrow, row, (leni + 1)*sizeof(size_t));
      }
      pos++;
    }
  }

  /* clean up */
  for (i = 0; i < n; i++)
    free(rows[i]);
  free(rows);
  free(row);
  free(symlist);

  /* return result */
  {
    lev_byte *result = (lev_byte*)safe_malloc(medlen, sizeof(lev_byte));
    if (!result) {
      free(median);
      return NULL;
    }
    *medlength = medlen;
    memcpy(result, median, medlen*sizeof(lev_byte));
    median--;
    free(median);
    return result;
  }
}

/* used internally in make_usymlist */
typedef struct _HItem HItem;
struct _HItem {
  lev_wchar c;
  HItem *n;
};

/* free usmylist hash
 * this is a separate function because we need it in more than one place */
static void
free_usymlist_hash(HItem *symmap)
{
  size_t j;

  for (j = 0; j < 0x100; j++) {
    HItem *p = symmap + j;
    if (p->n == symmap || p->n == NULL)
      continue;
    p = p->n;
    while (p) {
      HItem *q = p;
      p = p->n;
      free(q);
    }
  }
  free(symmap);
}

/* compute the sets of symbols each string contains, and the set of symbols
 * in any of them (symset).  meanwhile, count how many different symbols
 * there are (used below for symlist). */
static lev_wchar*
make_usymlist(size_t n, const size_t *lengths,
              const lev_wchar *strings[], size_t *symlistlen)
{
  lev_wchar *symlist;
  size_t i, j;
  HItem *symmap;

  j = 0;
  for (i = 0; i < n; i++)
    j += lengths[i];

  *symlistlen = 0;
  if (j == 0)
    return NULL;

  /* find all symbols, use a kind of hash for storage */
  symmap = (HItem*)safe_malloc(0x100, sizeof(HItem));
  if (!symmap) {
    *symlistlen = (size_t)(-1);
    return NULL;
  }
  /* this is an ugly memory allocation avoiding hack: most hash elements
   * will probably contain none or one symbols only so, when p->n is equal
   * to symmap, it means there're no symbols yet, afters insterting the
   * first one, p->n becomes normally NULL and then it behaves like an
   * usual singly linked list */
  for (i = 0; i < 0x100; i++)
    symmap[i].n = symmap;
  for (i = 0; i < n; i++) {
    const lev_wchar *stri = strings[i];
    for (j = 0; j < lengths[i]; j++) {
      int c = stri[j];
      int key = (c + (c >> 7)) & 0xff;
      HItem *p = symmap + key;
      if (p->n == symmap) {
        p->c = c;
        p->n = NULL;
        (*symlistlen)++;
        continue;
      }
      while (p->c != c && p->n != NULL)
        p = p->n;
      if (p->c != c) {
        p->n = (HItem*)malloc(sizeof(HItem));
        if (!p->n) {
          free_usymlist_hash(symmap);
          *symlistlen = (size_t)(-1);
          return NULL;
        }
        p = p->n;
        p->n = NULL;
        p->c = c;
        (*symlistlen)++;
      }
    }
  }
  /* create dense symbol table, so we can easily iterate over only characters
   * present in the strings */
  {
    size_t pos = 0;
    symlist = (lev_wchar*)safe_malloc((*symlistlen), sizeof(lev_wchar));
    if (!symlist) {
      free_usymlist_hash(symmap);
      *symlistlen = (size_t)(-1);
      return NULL;
    }
    for (j = 0; j < 0x100; j++) {
      HItem *p = symmap + j;
      while (p != NULL && p->n != symmap) {
        symlist[pos++] = p->c;
        p = p->n;
      }
    }
  }

  /* free memory */
  free_usymlist_hash(symmap);

  return symlist;
}

/**
 * lev_u_greedy_median:
 * @n: The size of @lengths, @strings, and @weights.
 * @lengths: The lengths of @strings.
 * @strings: An array of strings, that may contain NUL characters.
 * @weights: The string weights (they behave exactly as multiplicities, though
 *           any positive value is allowed, not just integers).
 * @medlength: Where the length of the median should be stored.
 *
 * Finds a generalized median string of @strings using the greedy algorithm.
 *
 * Note it's considerably more efficient to give a string with weight 2
 * than to store two identical strings in @strings (with weights 1).
 *
 * Returns: The generalized median, as a newly allocated string; its length
 *          is stored in @medlength.
 **/
_LEV_STATIC_PY lev_wchar*
lev_u_greedy_median(size_t n, const size_t *lengths,
                    const lev_wchar *strings[],
                    const double *weights,
                    size_t *medlength)
{
  size_t i;  /* usually iterates over strings (n) */
  size_t j;  /* usually iterates over characters */
  size_t len;  /* usually iterates over the approximate median string */
  lev_wchar *symlist;  /* list of symbols present in the strings,
                              we iterate over it insead of set of all
                              existing symbols */
  size_t symlistlen;  /* length of symlistle */
  size_t maxlen;  /* maximum input string length */
  size_t stoplen;  /* maximum tried median string length -- this is slightly
                      higher than maxlen, because the median string may be
                      longer than any of the input strings */
  size_t **rows;  /* Levenshtein matrix rows for each string, we need to keep
                     only one previous row to construct the current one */
  size_t *row;  /* a scratch buffer for new Levenshtein matrix row computation,
                   shared among all strings */
  lev_wchar *median;  /* the resulting approximate median string */
  double *mediandist;  /* the total distance of the best median string of
                          given length.  warning!  mediandist[0] is total
                          distance for empty string, while median[] itself
                          is normally zero-based */
  size_t bestlen;  /* the best approximate median string length */

  /* find all symbols */
  symlist = make_usymlist(n, lengths, strings, &symlistlen);
  if (!symlist) {
    *medlength = 0;
    if (symlistlen != 0)
      return NULL;
    else
      return calloc(1, sizeof(lev_wchar));
  }

  /* allocate and initialize per-string matrix rows and a common work buffer */
  rows = (size_t**)safe_malloc(n, sizeof(size_t*));
  if (!rows) {
    free(symlist);
    return NULL;
  }
  maxlen = 0;
  for (i = 0; i < n; i++) {
    size_t *ri;
    size_t leni = lengths[i];
    if (leni > maxlen)
      maxlen = leni;
    ri = rows[i] = (size_t*)safe_malloc((leni + 1), sizeof(size_t));
    if (!ri) {
      for (j = 0; j < i; j++)
        free(rows[j]);
      free(rows);
      free(symlist);
      return NULL;
    }
    for (j = 0; j <= leni; j++)
      ri[j] = j;
  }
  stoplen = 2*maxlen + 1;
  row = (size_t*)safe_malloc((stoplen + 1), sizeof(size_t));
  if (!row) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(symlist);
    return NULL;
  }

  /* compute final cost of string of length 0 (empty string may be also
   * a valid answer) */
  median = (lev_wchar*)safe_malloc(stoplen, sizeof(lev_wchar));
  if (!median) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(row);
    free(symlist);
    return NULL;
  }
  mediandist = (double*)safe_malloc((stoplen + 1), sizeof(double));
  if (!mediandist) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(row);
    free(symlist);
    free(median);
    return NULL;
  }
  mediandist[0] = 0.0;
  for (i = 0; i < n; i++)
    mediandist[0] += lengths[i]*weights[i];

  /* build up the approximate median string symbol by symbol
   * XXX: we actually exit on break below, but on the same condition */
  for (len = 1; len <= stoplen; len++) {
    lev_wchar symbol;
    double minminsum = LEV_INFINITY;
    row[0] = len;
    /* iterate over all symbols we may want to add */
    for (j = 0; j < symlistlen; j++) {
      double totaldist = 0.0;
      double minsum = 0.0;
      symbol = symlist[j];
      /* sum Levenshtein distances from all the strings, with given weights */
      for (i = 0; i < n; i++) {
        const lev_wchar *stri = strings[i];
        size_t *p = rows[i];
        size_t leni = lengths[i];
        size_t *end = rows[i] + leni;
        size_t min = len;
        size_t x = len; /* == row[0] */
        /* compute how another row of Levenshtein matrix would look for median
         * string with this symbol added */
        while (p < end) {
          size_t D = *(p++) + (symbol != *(stri++));
          x++;
          if (x > D)
            x = D;
          if (x > *p + 1)
            x = *p + 1;
          if (x < min)
            min = x;
        }
        minsum += min*weights[i];
        totaldist += x*weights[i];
      }
      /* is this symbol better than all the others? */
      if (minsum < minminsum) {
        minminsum = minsum;
        mediandist[len] = totaldist;
        median[len - 1] = symbol;
      }
    }
    /* stop the iteration if we no longer need to recompute the matrix rows
     * or when we are over maxlen and adding more characters doesn't seem
     * useful */
    if (len == stoplen
        || (len > maxlen && mediandist[len] > mediandist[len - 1])) {
      stoplen = len;
      break;
    }
    /* now the best symbol is known, so recompute all matrix rows for this
     * one */
    symbol = median[len - 1];
    for (i = 0; i < n; i++) {
      const lev_wchar *stri = strings[i];
      size_t *oldrow = rows[i];
      size_t leni = lengths[i];
      size_t k;
      /* compute a row of Levenshtein matrix */
      for (k = 1; k <= leni; k++) {
        size_t c1 = oldrow[k] + 1;
        size_t c2 = row[k - 1] + 1;
        size_t c3 = oldrow[k - 1] + (symbol != stri[k - 1]);
        row[k] = c2 > c3 ? c3 : c2;
        if (row[k] > c1)
          row[k] = c1;
      }
      memcpy(oldrow, row, (leni + 1)*sizeof(size_t));
    }
  }

  /* find the string with minimum total distance */
  bestlen = 0;
  for (len = 1; len <= stoplen; len++) {
    if (mediandist[len] < mediandist[bestlen])
      bestlen = len;
  }

  /* clean up */
  for (i = 0; i < n; i++)
    free(rows[i]);
  free(rows);
  free(row);
  free(symlist);
  free(mediandist);

  /* return result */
  {
    lev_wchar *result = (lev_wchar*)safe_malloc(bestlen, sizeof(lev_wchar));
    if (!result) {
      free(median);
      return NULL;
    }
    memcpy(result, median, bestlen*sizeof(lev_wchar));
    free(median);
    *medlength = bestlen;
    return result;
  }
}

/*
 * Knowing the distance matrices up to some row, finish the distance
 * computations.
 *
 * string1, len1 are already shortened.
 */
static double
finish_udistance_computations(size_t len1, lev_wchar *string1,
                             size_t n, const size_t *lengths,
                             const lev_wchar **strings,
                             const double *weights, size_t **rows,
                             size_t *row)
{
  size_t *end;
  size_t i, j;
  size_t offset;  /* row[0]; offset + len1 give together real len of string1 */
  double distsum = 0.0;  /* sum of distances */

  /* catch trivia case */
  if (len1 == 0) {
    for (j = 0; j < n; j++)
      distsum += rows[j][lengths[j]]*weights[j];
    return distsum;
  }

  /* iterate through the strings and sum the distances */
  for (j = 0; j < n; j++) {
    size_t *rowi = rows[j];  /* current row */
    size_t leni = lengths[j];  /* current length */
    size_t len = len1;  /* temporary len1 for suffix stripping */
    const lev_wchar *stringi = strings[j];  /* current string */

    /* strip common suffix (prefix CAN'T be stripped) */
    while (len && leni && stringi[leni-1] == string1[len-1]) {
      len--;
      leni--;
    }

    /* catch trivial cases */
    if (len == 0) {
      distsum += rowi[leni]*weights[j];
      continue;
    }
    offset = rowi[0];
    if (leni == 0) {
      distsum += (offset + len)*weights[j];
      continue;
    }

    /* complete the matrix */
    memcpy(row, rowi, (leni + 1)*sizeof(size_t));
    end = row + leni;

    for (i = 1; i <= len; i++) {
      size_t *p = row + 1;
      const lev_wchar char1 = string1[i - 1];
      const lev_wchar *char2p = stringi;
      size_t D, x;

      D = x = i + offset;
      while (p <= end) {
        size_t c3 = --D + (char1 != *(char2p++));
        x++;
        if (x > c3)
          x = c3;
        D = *p;
        D++;
        if (x > D)
          x = D;
        *(p++) = x;
      }
    }
    distsum += weights[j]*(*end);
  }

  return distsum;
}

/**
 * lev_u_median_improve:
 * @len: The length of @s.
 * @s: The approximate generalized median string to be improved.
 * @n: The size of @lengths, @strings, and @weights.
 * @lengths: The lengths of @strings.
 * @strings: An array of strings, that may contain NUL characters.
 * @weights: The string weights (they behave exactly as multiplicities, though
 *           any positive value is allowed, not just integers).
 * @medlength: Where the new length of the median should be stored.
 *
 * Tries to make @s a better generalized median string of @strings with
 * small perturbations.
 *
 * It never returns a string with larger SOD than @s; in the worst case, a
 * string identical to @s is returned.
 *
 * Returns: The improved generalized median, as a newly allocated string; its
 *          length is stored in @medlength.
 **/
_LEV_STATIC_PY lev_wchar*
lev_u_median_improve(size_t len, const lev_wchar *s,
                     size_t n, const size_t *lengths,
                     const lev_wchar *strings[],
                     const double *weights,
                     size_t *medlength)
{
  size_t i;  /* usually iterates over strings (n) */
  size_t j;  /* usually iterates over characters */
  size_t pos;  /* the position in the approximate median string we are
                  trying to change */
  lev_wchar *symlist;  /* list of symbols present in the strings,
                              we iterate over it insead of set of all
                              existing symbols */
  size_t symlistlen;  /* length of symlist */
  size_t maxlen;  /* maximum input string length */
  size_t stoplen;  /* maximum tried median string length -- this is slightly
                      higher than maxlen, because the median string may be
                      longer than any of the input strings */
  size_t **rows;  /* Levenshtein matrix rows for each string, we need to keep
                     only one previous row to construct the current one */
  size_t *row;  /* a scratch buffer for new Levenshtein matrix row computation,
                   shared among all strings */
  lev_wchar *median;  /* the resulting approximate median string */
  size_t medlen;  /* the current approximate median string length */
  double minminsum;  /* the current total distance sum */

  /* find all symbols */
  symlist = make_usymlist(n, lengths, strings, &symlistlen);
  if (!symlist) {
    *medlength = 0;
    if (symlistlen != 0)
      return NULL;
    else
      return calloc(1, sizeof(lev_wchar));
  }

  /* allocate and initialize per-string matrix rows and a common work buffer */
  rows = (size_t**)safe_malloc(n, sizeof(size_t*));
  if (!rows) {
    free(symlist);
    return NULL;
  }
  maxlen = 0;
  for (i = 0; i < n; i++) {
    size_t *ri;
    size_t leni = lengths[i];
    if (leni > maxlen)
      maxlen = leni;
    ri = rows[i] = (size_t*)safe_malloc((leni + 1), sizeof(size_t));
    if (!ri) {
      for (j = 0; j < i; j++)
        free(rows[j]);
      free(rows);
      free(symlist);
      return NULL;
    }
    for (j = 0; j <= leni; j++)
      ri[j] = j;
  }
  stoplen = 2*maxlen + 1;
  row = (size_t*)safe_malloc((stoplen + 2), sizeof(size_t));
  if (!row) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(symlist);
    return NULL;
  }

  /* initialize median to given string */
  median = (lev_wchar*)safe_malloc((stoplen+1), sizeof(lev_wchar));
  if (!median) {
    for (j = 0; j < n; j++)
      free(rows[j]);
    free(rows);
    free(row);
    free(symlist);
    return NULL;
  }
  median++;  /* we need -1st element for insertions a pos 0 */
  medlen = len;
  memcpy(median, s, (medlen)*sizeof(lev_wchar));
  minminsum = finish_udistance_computations(medlen, median,
                                            n, lengths, strings,
                                            weights, rows, row);

  /* sequentially try perturbations on all positions */
  for (pos = 0; pos <= medlen; ) {
    lev_wchar orig_symbol, symbol;
    LevEditType operation;
    double sum;

    symbol = median[pos];
    operation = LEV_EDIT_KEEP;
    /* IF pos < medlength: FOREACH symbol: try to replace the symbol
     * at pos, if some lower the total distance, chooste the best */
    if (pos < medlen) {
      orig_symbol = median[pos];
      for (j = 0; j < symlistlen; j++) {
        if (symlist[j] == orig_symbol)
          continue;
        median[pos] = symlist[j];
        sum = finish_udistance_computations(medlen - pos, median + pos,
                                            n, lengths, strings,
                                            weights, rows, row);
        if (sum < minminsum) {
          minminsum = sum;
          symbol = symlist[j];
          operation = LEV_EDIT_REPLACE;
        }
      }
      median[pos] = orig_symbol;
    }
    /* FOREACH symbol: try to add it at pos, if some lower the total
     * distance, chooste the best (increase medlength)
     * We simulate insertion by replacing the character at pos-1 */
    orig_symbol = *(median + pos - 1);
    for (j = 0; j < symlistlen; j++) {
      *(median + pos - 1) = symlist[j];
      sum = finish_udistance_computations(medlen - pos + 1, median + pos - 1,
                                          n, lengths, strings,
                                          weights, rows, row);
      if (sum < minminsum) {
        minminsum = sum;
        symbol = symlist[j];
        operation = LEV_EDIT_INSERT;
      }
    }
    *(median + pos - 1) = orig_symbol;
    /* IF pos < medlength: try to delete the symbol at pos, if it lowers
     * the total distance remember it (decrease medlength) */
    if (pos < medlen) {
      sum = finish_udistance_computations(medlen - pos - 1, median + pos + 1,
                                          n, lengths, strings,
                                          weights, rows, row);
      if (sum < minminsum) {
        minminsum = sum;
        operation = LEV_EDIT_DELETE;
      }
    }
    /* actually perform the operation */
    switch (operation) {
      case LEV_EDIT_REPLACE:
      median[pos] = symbol;
      break;

      case LEV_EDIT_INSERT:
      memmove(median+pos+1, median+pos,
              (medlen - pos)*sizeof(lev_wchar));
      median[pos] = symbol;
      medlen++;
      break;

      case LEV_EDIT_DELETE:
      memmove(median+pos, median + pos+1,
              (medlen - pos-1)*sizeof(lev_wchar));
      medlen--;
      break;

      default:
      break;
    }
    assert(medlen <= stoplen);
    /* now the result is known, so recompute all matrix rows and move on */
    if (operation != LEV_EDIT_DELETE) {
      symbol = median[pos];
      row[0] = pos + 1;
      for (i = 0; i < n; i++) {
        const lev_wchar *stri = strings[i];
        size_t *oldrow = rows[i];
        size_t leni = lengths[i];
        size_t k;
        /* compute a row of Levenshtein matrix */
        for (k = 1; k <= leni; k++) {
          size_t c1 = oldrow[k] + 1;
          size_t c2 = row[k - 1] + 1;
          size_t c3 = oldrow[k - 1] + (symbol != stri[k - 1]);
          row[k] = c2 > c3 ? c3 : c2;
          if (row[k] > c1)
            row[k] = c1;
        }
        memcpy(oldrow, row, (leni + 1)*sizeof(size_t));
      }
      pos++;
    }
  }

  /* clean up */
  for (i = 0; i < n; i++)
    free(rows[i]);
  free(rows);
  free(row);
  free(symlist);

  /* return result */
  {
    lev_wchar *result = (lev_wchar*)safe_malloc(medlen, sizeof(lev_wchar));
    if (!result) {
      free(median);
      return NULL;
    }
    *medlength = medlen;
    memcpy(result, median, medlen*sizeof(lev_wchar));
    median--;
    free(median);
    return result;
  }
}
/* }}} */

/****************************************************************************
 *
 * Quick (voting) medians
 *
 ****************************************************************************/
/* {{{ */

/* compute the sets of symbols each string contains, and the set of symbols
 * in any of them (symset).  meanwhile, count how many different symbols
 * there are (used below for symlist).
 * the symset is passed as an argument to avoid its allocation and
 * deallocation when it's used in the caller too */
static lev_byte*
make_symlistset(size_t n, const size_t *lengths,
                const lev_byte *strings[], size_t *symlistlen,
                double *symset)
{
  size_t i, j;
  lev_byte *symlist;

  if (!symset) {
    *symlistlen = (size_t)(-1);
    return NULL;
  }
  memset(symset, 0, 0x100*sizeof(double));  /* XXX: needs IEEE doubles?! */
  *symlistlen = 0;
  for (i = 0; i < n; i++) {
    const lev_byte *stri = strings[i];
    for (j = 0; j < lengths[i]; j++) {
      int c = stri[j];
      if (!symset[c]) {
        (*symlistlen)++;
        symset[c] = 1.0;
      }
    }
  }
  if (!*symlistlen)
    return NULL;

  /* create dense symbol table, so we can easily iterate over only characters
   * present in the strings */
  {
    size_t pos = 0;
    symlist = (lev_byte*)safe_malloc((*symlistlen), sizeof(lev_byte));
    if (!symlist) {
      *symlistlen = (size_t)(-1);
      return NULL;
    }
    for (j = 0; j < 0x100; j++) {
      if (symset[j])
        symlist[pos++] = (lev_byte)j;
    }
  }

  return symlist;
}

_LEV_STATIC_PY lev_byte*
lev_quick_median(size_t n,
                 const size_t *lengths,
                 const lev_byte *strings[],
                 const double *weights,
                 size_t *medlength)
{
  size_t symlistlen, len, i, j, k;
  lev_byte *symlist;
  lev_byte *median;  /* the resulting string */
  double *symset;
  double ml, wl;

  /* first check whether the result would be an empty string 
   * and compute resulting string length */
  ml = wl = 0.0;
  for (i = 0; i < n; i++) {
    ml += lengths[i]*weights[i];
    wl += weights[i];
  }
  if (wl == 0.0)
    return (lev_byte*)calloc(1, sizeof(lev_byte));
  ml = floor(ml/wl + 0.499999);
  *medlength = len = ml;
  if (!len)
    return (lev_byte*)calloc(1, sizeof(lev_byte));
  median = (lev_byte*)safe_malloc(len, sizeof(lev_byte));
  if (!median)
    return NULL;

  /* find the symbol set;
   * now an empty symbol set is really a failure */
  symset = (double*)calloc(0x100, sizeof(double));
  if (!symset) {
    free(median);
    return NULL;
  }
  symlist = make_symlistset(n, lengths, strings, &symlistlen, symset);
  if (!symlist) {
    free(median);
    free(symset);
    return NULL;
  }

  for (j = 0; j < len; j++) {
    /* clear the symbol probabilities */
    if (symlistlen < 32) {
      for (i = 0; i < symlistlen; i++)
        symset[symlist[i]] = 0.0;
    }
    else
      memset(symset, 0, 0x100*sizeof(double));

    /* let all strings vote */
    for (i = 0; i < n; i++) {
      const lev_byte *stri = strings[i];
      double weighti = weights[i];
      size_t lengthi = lengths[i];
      double start = lengthi/ml*j;
      double end = start + lengthi/ml;
      size_t istart = floor(start);
      size_t iend = ceil(end);

      /* rounding errors can overflow the buffer */
      if (iend > lengthi)
        iend = lengthi;

      for (k = istart+1; k < iend; k++)
        symset[stri[k]] += weighti;
      symset[stri[istart]] += weighti*(1+istart - start);
      symset[stri[iend-1]] -= weighti*(iend - end);
    }

    /* find the elected symbol */
    k = symlist[0];
    for (i = 1; i < symlistlen; i++) {
      if (symset[symlist[i]] > symset[k])
        k = symlist[i];
    }
    median[j] = k;
  }

  free(symset);
  free(symlist);

  return median;
}

/* used internally in make_usymlistset */
typedef struct _HQItem HQItem;
struct _HQItem {
  lev_wchar c;
  double s;
  HQItem *n;
};

/* free usmylistset hash
 * this is a separate function because we need it in more than one place */
static void
free_usymlistset_hash(HQItem *symmap)
{
  size_t j;

  for (j = 0; j < 0x100; j++) {
    HQItem *p = symmap + j;
    if (p->n == symmap || p->n == NULL)
      continue;
    p = p->n;
    while (p) {
      HQItem *q = p;
      p = p->n;
      free(q);
    }
  }
  free(symmap);
}

/* compute the sets of symbols each string contains, and the set of symbols
 * in any of them (symset).  meanwhile, count how many different symbols
 * there are (used below for symlist).
 * the symset is passed as an argument to avoid its allocation and
 * deallocation when it's used in the caller too */
static lev_wchar*
make_usymlistset(size_t n, const size_t *lengths,
                 const lev_wchar *strings[], size_t *symlistlen,
                 HQItem *symmap)
{
  lev_wchar *symlist;
  size_t i, j;

  j = 0;
  for (i = 0; i < n; i++)
    j += lengths[i];

  *symlistlen = 0;
  if (j == 0)
    return NULL;

  /* this is an ugly memory allocation avoiding hack: most hash elements
   * will probably contain none or one symbols only so, when p->n is equal
   * to symmap, it means there're no symbols yet, afters insterting the
   * first one, p->n becomes normally NULL and then it behaves like an
   * usual singly linked list */
  for (i = 0; i < 0x100; i++)
    symmap[i].n = symmap;
  for (i = 0; i < n; i++) {
    const lev_wchar *stri = strings[i];
    for (j = 0; j < lengths[i]; j++) {
      int c = stri[j];
      int key = (c + (c >> 7)) & 0xff;
      HQItem *p = symmap + key;
      if (p->n == symmap) {
        p->c = c;
        p->n = NULL;
        (*symlistlen)++;
        continue;
      }
      while (p->c != c && p->n != NULL)
        p = p->n;
      if (p->c != c) {
        p->n = (HQItem*)malloc(sizeof(HQItem));
        if (!p->n) {
          *symlistlen = (size_t)(-1);
          return NULL;
        }
        p = p->n;
        p->n = NULL;
        p->c = c;
        (*symlistlen)++;
      }
    }
  }
  /* create dense symbol table, so we can easily iterate over only characters
   * present in the strings */
  {
    size_t pos = 0;
    symlist = (lev_wchar*)safe_malloc((*symlistlen), sizeof(lev_wchar));
    if (!symlist) {
      *symlistlen = (size_t)(-1);
      return NULL;
    }
    for (j = 0; j < 0x100; j++) {
      HQItem *p = symmap + j;
      while (p != NULL && p->n != symmap) {
        symlist[pos++] = p->c;
        p = p->n;
      }
    }
  }

  return symlist;
}

_LEV_STATIC_PY lev_wchar*
lev_u_quick_median(size_t n,
                   const size_t *lengths,
                   const lev_wchar *strings[],
                   const double *weights,
                   size_t *medlength)
{
  size_t symlistlen, len, i, j, k;
  lev_wchar *symlist;
  lev_wchar *median;  /* the resulting string */
  HQItem *symmap;
  double ml, wl;

  /* first check whether the result would be an empty string 
   * and compute resulting string length */
  ml = wl = 0.0;
  for (i = 0; i < n; i++) {
    ml += lengths[i]*weights[i];
    wl += weights[i];
  }
  if (wl == 0.0)
    return (lev_wchar*)calloc(1, sizeof(lev_wchar));
  ml = floor(ml/wl + 0.499999);
  *medlength = len = ml;
  if (!len)
    return (lev_wchar*)calloc(1, sizeof(lev_wchar));
  median = (lev_wchar*)safe_malloc(len, sizeof(lev_wchar));
  if (!median)
    return NULL;

  /* find the symbol set;
   * now an empty symbol set is really a failure */
  symmap = (HQItem*)safe_malloc(0x100, sizeof(HQItem));
  if (!symmap) {
    free(median);
    return NULL;
  }
  symlist = make_usymlistset(n, lengths, strings, &symlistlen, symmap);
  if (!symlist) {
    free(median);
    free_usymlistset_hash(symmap);
    return NULL;
  }

  for (j = 0; j < len; j++) {
    /* clear the symbol probabilities */
    for (i = 0; i < 0x100; i++) {
      HQItem *p = symmap + i;
      if (p->n == symmap)
        continue;
      while (p) {
        p->s = 0.0;
        p = p->n;
      }
    }

    /* let all strings vote */
    for (i = 0; i < n; i++) {
      const lev_wchar *stri = strings[i];
      double weighti = weights[i];
      size_t lengthi = lengths[i];
      double start = lengthi/ml*j;
      double end = start + lengthi/ml;
      size_t istart = floor(start);
      size_t iend = ceil(end);

      /* rounding errors can overflow the buffer */
      if (iend > lengthi)
        iend = lengthi;

      /* the inner part, including the complete last character */
      for (k = istart+1; k < iend; k++) {
        int c = stri[k];
        int key = (c + (c >> 7)) & 0xff;
        HQItem *p = symmap + key;
        while (p->c != c)
          p = p->n;
        p->s += weighti;
      }
      /* the initial fraction */
      {
        int c = stri[istart];
        int key = (c + (c >> 7)) & 0xff;
        HQItem *p = symmap + key;
        while (p->c != c)
          p = p->n;
        p->s += weighti*(1+istart - start);
      }
      /* subtract what we counted from the last character but doesn't
       * actually belong here.
       * this strategy works also when istart+1 == iend (i.e., everything
       * happens inside a one character) */
      {
        int c = stri[iend-1];
        int key = (c + (c >> 7)) & 0xff;
        HQItem *p = symmap + key;
        while (p->c != c)
          p = p->n;
        p->s -= weighti*(iend - end);
      }
    }

    /* find the elected symbol */
    {
      HQItem *max = NULL;

      for (i = 0; i < 0x100; i++) {
        HQItem *p = symmap + i;
        if (p->n == symmap)
          continue;
        while (p) {
          if (!max || p->s > max->s)
            max = p;
          p = p->n;
        }
      }
      median[j] = max->c;
    }
  }

  free_usymlistset_hash(symmap);
  free(symlist);

  return median;
}
/* }}} */

/****************************************************************************
 *
 * Set medians
 *
 ****************************************************************************/
/* {{{ */

/**
 * lev_set_median_index:
 * @n: The size of @lengths, @strings, and @weights.
 * @lengths: The lengths of @strings.
 * @strings: An array of strings, that may contain NUL characters.
 * @weights: The string weights (they behave exactly as multiplicities, though
 *           any positive value is allowed, not just integers).
 *
 * Finds the median string of a string set @strings.
 *
 * Returns: An index in @strings pointing to the set median, -1 in case of
 *          failure.
 **/
_LEV_STATIC_PY size_t
lev_set_median_index(size_t n, const size_t *lengths,
                     const lev_byte *strings[],
                     const double *weights)
{
  size_t minidx = 0;
  double mindist = LEV_INFINITY;
  size_t i;
  long int *distances;

  distances = (long int*)safe_malloc((n*(n - 1)/2), sizeof(long int));
  if (!distances)
    return (size_t)-1;

  memset(distances, 0xff, (n*(n - 1)/2)*sizeof(long int)); /* XXX */
  for (i = 0; i < n; i++) {
    size_t j = 0;
    double dist = 0.0;
    const lev_byte *stri = strings[i];
    size_t leni = lengths[i];
    /* below diagonal */
    while (j < i && dist < mindist) {
      size_t dindex = (i - 1)*(i - 2)/2 + j;
      long int d;
      if (distances[dindex] >= 0)
        d = distances[dindex];
      else {
        d = lev_edit_distance(lengths[j], strings[j], leni, stri, 0);
        if (d < 0) {
          free(distances);
          return (size_t)-1;
        }
      }
      dist += weights[j]*d;
      j++;
    }
    j++;  /* no need to compare item with itself */
    /* above diagonal */
    while (j < n && dist < mindist) {
      size_t dindex = (j - 1)*(j - 2)/2 + i;
      distances[dindex] = lev_edit_distance(lengths[j], strings[j],
                                            leni, stri, 0);
      if (distances[dindex] < 0) {
        free(distances);
        return (size_t)-1;
      }
      dist += weights[j]*distances[dindex];
      j++;
    }

    if (dist < mindist) {
      mindist = dist;
      minidx = i;
    }
  }

  free(distances);
  return minidx;
}

/**
 * lev_u_set_median_index:
 * @n: The size of @lengths, @strings, and @weights.
 * @lengths: The lengths of @strings.
 * @strings: An array of strings, that may contain NUL characters.
 * @weights: The string weights (they behave exactly as multiplicities, though
 *           any positive value is allowed, not just integers).
 *
 * Finds the median string of a string set @strings.
 *
 * Returns: An index in @strings pointing to the set median, -1 in case of
 *          failure.
 **/
_LEV_STATIC_PY size_t
lev_u_set_median_index(size_t n, const size_t *lengths,
                       const lev_wchar *strings[],
                       const double *weights)
{
  size_t minidx = 0;
  double mindist = LEV_INFINITY;
  size_t i;
  long int *distances;

  distances = (long int*)safe_malloc((n*(n - 1)/2), sizeof(long int));
  if (!distances)
    return (size_t)-1;

  memset(distances, 0xff, (n*(n - 1)/2)*sizeof(long int)); /* XXX */
  for (i = 0; i < n; i++) {
    size_t j = 0;
    double dist = 0.0;
    const lev_wchar *stri = strings[i];
    size_t leni = lengths[i];
    /* below diagonal */
    while (j < i && dist < mindist) {
      size_t dindex = (i - 1)*(i - 2)/2 + j;
      long int d;
      if (distances[dindex] >= 0)
        d = distances[dindex];
      else {
        d = lev_u_edit_distance(lengths[j], strings[j], leni, stri, 0);
        if (d < 0) {
          free(distances);
          return (size_t)-1;
        }
      }
      dist += weights[j]*d;
      j++;
    }
    j++;  /* no need to compare item with itself */
    /* above diagonal */
    while (j < n && dist < mindist) {
      size_t dindex = (j - 1)*(j - 2)/2 + i;
      distances[dindex] = lev_u_edit_distance(lengths[j], strings[j],
                                              leni, stri, 0);
      if (distances[dindex] < 0) {
        free(distances);
        return (size_t)-1;
      }
      dist += weights[j]*distances[dindex];
      j++;
    }

    if (dist < mindist) {
      mindist = dist;
      minidx = i;
    }
  }

  free(distances);
  return minidx;
}

/**
 * lev_set_median:
 * @n: The size of @lengths, @strings, and @weights.
 * @lengths: The lengths of @strings.
 * @strings: An array of strings, that may contain NUL characters.
 * @weights: The string weights (they behave exactly as multiplicities, though
 *           any positive value is allowed, not just integers).
 * @medlength: Where the length of the median string should be stored.
 *
 * Finds the median string of a string set @strings.
 *
 * Returns: The set median as a newly allocate string, its length is stored
 *          in @medlength.  %NULL in the case of failure.
 **/
_LEV_STATIC_PY lev_byte*
lev_set_median(size_t n, const size_t *lengths,
               const lev_byte *strings[],
               const double *weights,
               size_t *medlength)
{
  size_t minidx = lev_set_median_index(n, lengths, strings, weights);
  lev_byte *result;

  if (minidx == (size_t)-1)
    return NULL;

  *medlength = lengths[minidx];
  if (!lengths[minidx])
    return (lev_byte*)calloc(1, sizeof(lev_byte));

  result = (lev_byte*)safe_malloc(lengths[minidx], sizeof(lev_byte));
  if (!result)
    return NULL;
  return memcpy(result, strings[minidx], lengths[minidx]*sizeof(lev_byte));
}

/**
 * lev_u_set_median:
 * @n: The size of @lengths, @strings, and @weights.
 * @lengths: The lengths of @strings.
 * @strings: An array of strings, that may contain NUL characters.
 * @weights: The string weights (they behave exactly as multiplicities, though
 *           any positive value is allowed, not just integers).
 * @medlength: Where the length of the median string should be stored.
 *
 * Finds the median string of a string set @strings.
 *
 * Returns: The set median as a newly allocate string, its length is stored
 *          in @medlength.  %NULL in the case of failure.
 **/
_LEV_STATIC_PY lev_wchar*
lev_u_set_median(size_t n, const size_t *lengths,
                 const lev_wchar *strings[],
                 const double *weights,
                 size_t *medlength)
{
  size_t minidx = lev_u_set_median_index(n, lengths, strings, weights);
  lev_wchar *result;

  if (minidx == (size_t)-1)
    return NULL;

  *medlength = lengths[minidx];
  if (!lengths[minidx])
    return (lev_wchar*)calloc(1, sizeof(lev_wchar));

  result = (lev_wchar*)safe_malloc(lengths[minidx], sizeof(lev_wchar));
  if (!result)
    return NULL;
  return memcpy(result, strings[minidx], lengths[minidx]*sizeof(lev_wchar));
}
/* }}} */

/****************************************************************************
 *
 * Set, sequence distances
 *
 ****************************************************************************/
/* {{{ */

/**
 * lev_edit_seq_distance:
 * @n1: The length of @lengths1 and @strings1.
 * @lengths1: The lengths of strings in @strings1.
 * @strings1: An array of strings that may contain NUL characters.
 * @n2: The length of @lengths2 and @strings2.
 * @lengths2: The lengths of strings in @strings2.
 * @strings2: An array of strings that may contain NUL characters.
 *
 * Finds the distance between string sequences @strings1 and @strings2.
 *
 * In other words, this is a double-Levenshtein algorithm.
 *
 * The cost of string replace operation is based on string similarity: it's
 * zero for identical strings and 2 for completely unsimilar strings.
 *
 * Returns: The distance of the two sequences.
 **/
_LEV_STATIC_PY double
lev_edit_seq_distance(size_t n1, const size_t *lengths1,
                      const lev_byte *strings1[],
                      size_t n2, const size_t *lengths2,
                      const lev_byte *strings2[])
{
  size_t i;
  double *row;  /* we only need to keep one row of costs */
  double *end;

  /* strip common prefix */
  while (n1 > 0 && n2 > 0
         && *lengths1 == *lengths2
         && memcmp(*strings1, *strings2,
                   *lengths1*sizeof(lev_byte)) == 0) {
    n1--;
    n2--;
    strings1++;
    strings2++;
    lengths1++;
    lengths2++;
  }

  /* strip common suffix */
  while (n1 > 0 && n2 > 0
         && lengths1[n1-1] == lengths2[n2-1]
         && memcmp(strings1[n1-1], strings2[n2-1],
                   lengths1[n1-1]*sizeof(lev_byte)) == 0) {
    n1--;
    n2--;
  }

  /* catch trivial cases */
  if (n1 == 0)
    return n2;
  if (n2 == 0)
    return n1;

  /* make the inner cycle (i.e. strings2) the longer one */
  if (n1 > n2) {
    size_t nx = n1;
    const size_t *lx = lengths1;
    const lev_byte **sx = strings1;
    n1 = n2;
    n2 = nx;
    lengths1 = lengths2;
    lengths2 = lx;
    strings1 = strings2;
    strings2 = sx;
  }
  n1++;
  n2++;

  /* initalize first row */
  row = (double*)safe_malloc(n2, sizeof(double));
  if (!row)
    return -1.0;
  end = row + n2 - 1;
  for (i = 0; i < n2; i++)
    row[i] = (double)i;

  /* go through the matrix and compute the costs.  yes, this is an extremely
   * obfuscated version, but also extremely memory-conservative and relatively
   * fast.  */
  for (i = 1; i < n1; i++) {
    double *p = row + 1;
    const lev_byte *str1 = strings1[i - 1];
    const size_t len1 = lengths1[i - 1];
    const lev_byte **str2p = strings2;
    const size_t *len2p = lengths2;
    double D = i - 1.0;
    double x = i;
    while (p <= end) {
      size_t l = len1 + *len2p;
      double q;
      if (l == 0)
        q = D;
      else {
        size_t d = lev_edit_distance(len1, str1, *(len2p++), *(str2p++), 1);
        if (d == (size_t)(-1)) {
          free(row);
          return -1.0;
        }
        q = D + 2.0/l*d;
      }
      x += 1.0;
      if (x > q)
        x = q;
      D = *p;
      if (x > D + 1.0)
        x = D + 1.0;
      *(p++) = x;
    }
  }

  {
    double q = *end;
    free(row);
    return q;
  }
}

/**
 * lev_u_edit_seq_distance:
 * @n1: The length of @lengths1 and @strings1.
 * @lengths1: The lengths of strings in @strings1.
 * @strings1: An array of strings that may contain NUL characters.
 * @n2: The length of @lengths2 and @strings2.
 * @lengths2: The lengths of strings in @strings2.
 * @strings2: An array of strings that may contain NUL characters.
 *
 * Finds the distance between string sequences @strings1 and @strings2.
 *
 * In other words, this is a double-Levenshtein algorithm.
 *
 * The cost of string replace operation is based on string similarity: it's
 * zero for identical strings and 2 for completely unsimilar strings.
 *
 * Returns: The distance of the two sequences.
 **/
_LEV_STATIC_PY double
lev_u_edit_seq_distance(size_t n1, const size_t *lengths1,
                        const lev_wchar *strings1[],
                        size_t n2, const size_t *lengths2,
                        const lev_wchar *strings2[])
{
  size_t i;
  double *row;  /* we only need to keep one row of costs */
  double *end;

  /* strip common prefix */
  while (n1 > 0 && n2 > 0
         && *lengths1 == *lengths2
         && memcmp(*strings1, *strings2,
                   *lengths1*sizeof(lev_wchar)) == 0) {
    n1--;
    n2--;
    strings1++;
    strings2++;
    lengths1++;
    lengths2++;
  }

  /* strip common suffix */
  while (n1 > 0 && n2 > 0
         && lengths1[n1-1] == lengths2[n2-1]
         && memcmp(strings1[n1-1], strings2[n2-1],
                   lengths1[n1-1]*sizeof(lev_wchar)) == 0) {
    n1--;
    n2--;
  }

  /* catch trivial cases */
  if (n1 == 0)
    return (double)n2;
  if (n2 == 0)
    return (double)n1;

  /* make the inner cycle (i.e. strings2) the longer one */
  if (n1 > n2) {
    size_t nx = n1;
    const size_t *lx = lengths1;
    const lev_wchar **sx = strings1;
    n1 = n2;
    n2 = nx;
    lengths1 = lengths2;
    lengths2 = lx;
    strings1 = strings2;
    strings2 = sx;
  }
  n1++;
  n2++;

  /* initalize first row */
  row = (double*)safe_malloc(n2, sizeof(double));
  if (!row)
    return -1.0;
  end = row + n2 - 1;
  for (i = 0; i < n2; i++)
    row[i] = (double)i;

  /* go through the matrix and compute the costs.  yes, this is an extremely
   * obfuscated version, but also extremely memory-conservative and relatively
   * fast.  */
  for (i = 1; i < n1; i++) {
    double *p = row + 1;
    const lev_wchar *str1 = strings1[i - 1];
    const size_t len1 = lengths1[i - 1];
    const lev_wchar **str2p = strings2;
    const size_t *len2p = lengths2;
    double D = i - 1.0;
    double x = i;
    while (p <= end) {
      size_t l = len1 + *len2p;
      double q;
      if (l == 0)
        q = D;
      else {
        size_t d = lev_u_edit_distance(len1, str1, *(len2p++), *(str2p++), 1);
        if (d == (size_t)(-1)) {
          free(row);
          return -1.0;
        }
        q = D + 2.0/l*d;
      }
      x += 1.0;
      if (x > q)
        x = q;
      D = *p;
      if (x > D + 1.0)
        x = D + 1.0;
      *(p++) = x;
    }
  }

  {
    double q = *end;
    free(row);
    return q;
  }
}

/**
 * lev_set_distance:
 * @n1: The length of @lengths1 and @strings1.
 * @lengths1: The lengths of strings in @strings1.
 * @strings1: An array of strings that may contain NUL characters.
 * @n2: The length of @lengths2 and @strings2.
 * @lengths2: The lengths of strings in @strings2.
 * @strings2: An array of strings that may contain NUL characters.
 *
 * Finds the distance between string sets @strings1 and @strings2.
 *
 * The difference from lev_edit_seq_distance() is that order doesn't matter.
 * The optimal association of @strings1 and @strings2 is found first and
 * the similarity is computed for that.
 *
 * Uses sequential Munkers-Blackman algorithm.
 *
 * Returns: The distance of the two sets.
 **/
_LEV_STATIC_PY double
lev_set_distance(size_t n1, const size_t *lengths1,
                 const lev_byte *strings1[],
                 size_t n2, const size_t *lengths2,
                 const lev_byte *strings2[])
{
  double *dists;  /* the (modified) distance matrix, indexed [row*n1 + col] */
  double *r;
  size_t i, j;
  size_t *map;
  double sum;

  /* catch trivial cases */
  if (n1 == 0)
    return (double)n2;
  if (n2 == 0)
    return (double)n1;

  /* make the number of columns (n1) smaller than the number of rows */
  if (n1 > n2) {
    size_t nx = n1;
    const size_t *lx = lengths1;
    const lev_byte **sx = strings1;
    n1 = n2;
    n2 = nx;
    lengths1 = lengths2;
    lengths2 = lx;
    strings1 = strings2;
    strings2 = sx;
  }

  /* compute distances from each to each */
  r = dists = (double*)safe_malloc_3(n1, n2, sizeof(double));
  if (!r)
    return -1.0;
  for (i = 0; i < n2; i++) {
    size_t len2 = lengths2[i];
    const lev_byte *str2 = strings2[i];
    const size_t *len1p = lengths1;
    const lev_byte **str1p = strings1;
    for (j = 0; j < n1; j++) {
      size_t l = len2 + *len1p;
      if (l == 0)
        *(r++) = 0.0;
      else {
        size_t d = lev_edit_distance(len2, str2, *(len1p++), *(str1p)++, 1);
        if (d == (size_t)(-1)) {
          free(r);
          return -1.0;
        }
        *(r++) = (double)d/l;
      }
    }
  }

  /* find the optimal mapping between the two sets */
  map = munkers_blackman(n1, n2, dists);
  if (!map)
    return -1.0;

  /* sum the set distance */
  sum = n2 - n1;
  for (j = 0; j < n1; j++) {
    size_t l;
    i = map[j];
    l = lengths1[j] + lengths2[i];
    if (l > 0) {
      size_t d = lev_edit_distance(lengths1[j], strings1[j],
                                   lengths2[i], strings2[i], 1);
      if (d == (size_t)(-1)) {
        free(map);
        return -1.0;
      }
      sum += 2.0*d/l;
    }
  }
  free(map);

  return sum;
}

/**
 * lev_u_set_distance:
 * @n1: The length of @lengths1 and @strings1.
 * @lengths1: The lengths of strings in @strings1.
 * @strings1: An array of strings that may contain NUL characters.
 * @n2: The length of @lengths2 and @strings2.
 * @lengths2: The lengths of strings in @strings2.
 * @strings2: An array of strings that may contain NUL characters.
 *
 * Finds the distance between string sets @strings1 and @strings2.
 *
 * The difference from lev_u_edit_seq_distance() is that order doesn't matter.
 * The optimal association of @strings1 and @strings2 is found first and
 * the similarity is computed for that.
 *
 * Uses sequential Munkers-Blackman algorithm.
 *
 * Returns: The distance of the two sets.
 **/
_LEV_STATIC_PY double
lev_u_set_distance(size_t n1, const size_t *lengths1,
                   const lev_wchar *strings1[],
                   size_t n2, const size_t *lengths2,
                   const lev_wchar *strings2[])
{
  double *dists;  /* the (modified) distance matrix, indexed [row*n1 + col] */
  double *r;
  size_t i, j;
  size_t *map;
  double sum;

  /* catch trivial cases */
  if (n1 == 0)
    return (double)n2;
  if (n2 == 0)
    return (double)n1;

  /* make the number of columns (n1) smaller than the number of rows */
  if (n1 > n2) {
    size_t nx = n1;
    const size_t *lx = lengths1;
    const lev_wchar **sx = strings1;
    n1 = n2;
    n2 = nx;
    lengths1 = lengths2;
    lengths2 = lx;
    strings1 = strings2;
    strings2 = sx;
  }

  /* compute distances from each to each */
  r = dists = (double*)safe_malloc_3(n1, n2, sizeof(double));
  if (!r)
    return -1.0;
  for (i = 0; i < n2; i++) {
    size_t len2 = lengths2[i];
    const lev_wchar *str2 = strings2[i];
    const size_t *len1p = lengths1;
    const lev_wchar **str1p = strings1;
    for (j = 0; j < n1; j++) {
      size_t l = len2 + *len1p;
      if (l == 0)
        *(r++) = 0.0;
      else {
        size_t d = lev_u_edit_distance(len2, str2, *(len1p++), *(str1p)++, 1);
        if (d == (size_t)(-1)) {
          free(r);
          return -1.0;
        }
        *(r++) = (double)d/l;
      }
    }
  }

  /* find the optimal mapping between the two sets */
  map = munkers_blackman(n1, n2, dists);
  if (!map)
    return -1.0;

  /* sum the set distance */
  sum = n2 - n1;
  for (j = 0; j < n1; j++) {
    size_t l;
    i = map[j];
    l = lengths1[j] + lengths2[i];
    if (l > 0) {
      size_t d = lev_u_edit_distance(lengths1[j], strings1[j],
                                     lengths2[i], strings2[i], 1);
      if (d == (size_t)(-1)) {
        free(map);
        return -1.0;
      }
      sum += 2.0*d/l;
    }
  }
  free(map);

  return sum;
}

/*
 * Munkers-Blackman algorithm.
 */
static size_t*
munkers_blackman(size_t n1, size_t n2, double *dists)
{
  size_t i, j;
  size_t *covc, *covr;  /* 1 if column/row is covered */
  /* these contain 1-based indices, so we can use zero as `none'
   * zstarr: column of a z* in given row
   * zstarc: row of a z* in given column
   * zprimer: column of a z' in given row */
  size_t *zstarr, *zstarc, *zprimer;

  /* allocate memory */
  covc = calloc(n1, sizeof(size_t));
  if (!covc)
    return NULL;
  zstarc = calloc(n1, sizeof(size_t));
  if (!zstarc) {
    free(covc);
    return NULL;
  }
  covr = calloc(n2, sizeof(size_t));
  if (!covr) {
    free(zstarc);
    free(covc);
    return NULL;
  }
  zstarr = calloc(n2, sizeof(size_t));
  if (!zstarr) {
    free(covr);
    free(zstarc);
    free(covc);
    return NULL;
  }
  zprimer = calloc(n2, sizeof(size_t));
  if (!zprimer) {
    free(zstarr);
    free(covr);
    free(zstarc);
    free(covc);
    return NULL;
  }

  /* step 0 (subtract minimal distance) and step 1 (find zeroes) */
  for (j = 0; j < n1; j++) {
    size_t minidx = 0;
    double *col = dists + j;
    double min = *col;
    double *p = col + n1;
    for (i = 1; i < n2; i++) {
      if (min > *p) {
        minidx = i;
        min = *p;
      }
      p += n1;
    }
    /* subtract */
    p = col;
    for (i = 0; i < n2; i++) {
      *p -= min;
      if (*p < LEV_EPSILON)
        *p = 0.0;
      p += n1;
    }
    /* star the zero, if possible */
    if (!zstarc[j] && !zstarr[minidx]) {
      zstarc[j] = minidx + 1;
      zstarr[minidx] = j + 1;
    }
    else {
      /* otherwise try to find some other */
      p = col;
      for (i = 0; i < n2; i++) {
        if (i != minidx && *p == 0.0
            && !zstarc[j] && !zstarr[i]) {
          zstarc[j] = i + 1;
          zstarr[i] = j + 1;
          break;
        }
        p += n1;
      }
    }
  }

  /* main */
  while (1) {
    /* step 2 (cover columns containing z*) */
    {
      size_t nc = 0;
      for (j = 0; j < n1; j++) {
        if (zstarc[j]) {
          covc[j] = 1;
          nc++;
        }
      }
      if (nc == n1)
        break;
    }

    /* step 3 (find uncovered zeroes) */
    while (1) {
      step_3:
      /* search uncovered matrix entries */
      for (j = 0; j < n1; j++) {
        double *p = dists + j;
        if (covc[j])
          continue;
        for (i = 0; i < n2; i++) {
          if (!covr[i] && *p == 0.0) {
            /* when a zero is found, prime it */
            zprimer[i] = j + 1;
            if (zstarr[i]) {
              /* if there's a z* in the same row,
               * uncover the column, cover the row and redo */
              covr[i] = 1;
              covc[zstarr[i] - 1] = 0;
              goto step_3;
            }
            /* if there's no z*,
             * we are at the end of our path an can convert z'
             * to z* */
            goto step_4;
          }
          p += n1;
        }
      }

      /* step 5 (new zero manufacturer)
       * we can't get here, unless no zero is found at all */
      {
        /* find the smallest uncovered entry */
        double min = LEV_INFINITY;
        for (j = 0; j < n1; j++) {
          double *p = dists + j;
          if (covc[j])
            continue;
          for (i = 0; i < n2; i++) {
            if (!covr[i] && min > *p) {
              min = *p;
            }
            p += n1;
          }
        }
        /* add it to all covered rows */
        for (i = 0; i < n2; i++) {
          double *p = dists + i*n1;
          if (!covr[i])
            continue;
          for (j = 0; j < n1; j++)
            *(p++) += min;
        }
        /* subtract if from all uncovered columns */
        for (j = 0; j < n1; j++) {
          double *p = dists + j;
          if (covc[j])
            continue;
          for (i = 0; i < n2; i++) {
            *p -= min;
            if (*p < LEV_EPSILON)
              *p = 0.0;
            p += n1;
          }
        }
      }
    }

    /* step 4 (increment the number of z*)
     * i is the row number (we get it from step 3) */
    step_4:
    i++;
    do {
      size_t x = i;

      i--;
      j = zprimer[i] - 1;  /* move to z' in the same row */
      zstarr[i] = j + 1;   /* mark it as z* in row buffer */
      i = zstarc[j];       /* move to z* in the same column */
      zstarc[j] = x;       /* mark the z' as being new z* */
    } while (i);
    memset(zprimer, 0, n2*sizeof(size_t));
    memset(covr, 0, n2*sizeof(size_t));
    memset(covc, 0, n1*sizeof(size_t));
  }

  free(dists);
  free(covc);
  free(covr);
  /*free(zstarc);  this is the result */
  free(zstarr);
  free(zprimer);

  for (j = 0; j < n1; j++)
    zstarc[j]--;
  return zstarc;
}
/* }}} */

/****************************************************************************
 *
 * Editops and other difflib-like stuff.
 *
 ****************************************************************************/
/* {{{ */

/**
 * lev_editops_check_errors:
 * @len1: The length of an eventual @ops source string.
 * @len2: The length of an eventual @ops destination string.
 * @n: The length of @ops.
 * @ops: An array of elementary edit operations.
 *
 * Checks whether @ops is consistent and applicable as a partial edit from a
 * string of length @len1 to a string of length @len2.
 *
 * Returns: Zero if @ops seems OK, a nonzero error code otherwise.
 **/
_LEV_STATIC_PY int
lev_editops_check_errors(size_t len1, size_t len2,
                         size_t n, const LevEditOp *ops)
{
  const LevEditOp *o;
  size_t i;

  if (!n)
    return LEV_EDIT_ERR_OK;

  /* check bounds */
  o = ops;
  for (i = n; i; i--, o++) {
    if (o->type >= LEV_EDIT_LAST)
      return LEV_EDIT_ERR_TYPE;
    if (o->spos > len1 || o->dpos > len2)
      return LEV_EDIT_ERR_OUT;
    if (o->spos == len1 && o->type != LEV_EDIT_INSERT)
      return LEV_EDIT_ERR_OUT;
    if (o->dpos == len2 && o->type != LEV_EDIT_DELETE)
      return LEV_EDIT_ERR_OUT;
  }

  /* check ordering */
  o = ops + 1;
  for (i = n - 1; i; i--, o++, ops++) {
    if (o->spos < ops->spos || o->dpos < ops->dpos)
      return LEV_EDIT_ERR_ORDER;
  }

  return LEV_EDIT_ERR_OK;
}

/**
 * lev_opcodes_check_errors:
 * @len1: The length of an eventual @ops source string.
 * @len2: The length of an eventual @ops destination string.
 * @nb: The length of @bops.
 * @bops: An array of difflib block edit operation codes.
 *
 * Checks whether @bops is consistent and applicable as an edit from a
 * string of length @len1 to a string of length @len2.
 *
 * Returns: Zero if @bops seems OK, a nonzero error code otherwise.
 **/
_LEV_STATIC_PY int
lev_opcodes_check_errors(size_t len1, size_t len2,
                         size_t nb, const LevOpCode *bops)
{
  const LevOpCode *b;
  size_t i;

  if (!nb)
    return 1;

  /* check completenes */
  if (bops->sbeg || bops->dbeg
      || bops[nb - 1].send != len1 || bops[nb - 1].dend != len2)
    return LEV_EDIT_ERR_SPAN;

  /* check bounds and block consistency */
  b = bops;
  for (i = nb; i; i--, b++) {
    if (b->send > len1 || b->dend > len2)
      return LEV_EDIT_ERR_OUT;
    switch (b->type) {
      case LEV_EDIT_KEEP:
      case LEV_EDIT_REPLACE:
      if (b->dend - b->dbeg != b->send - b->sbeg || b->dend == b->dbeg)
        return LEV_EDIT_ERR_BLOCK;
      break;

      case LEV_EDIT_INSERT:
      if (b->dend - b->dbeg == 0 || b->send - b->sbeg != 0)
        return LEV_EDIT_ERR_BLOCK;
      break;

      case LEV_EDIT_DELETE:
      if (b->send - b->sbeg == 0 || b->dend - b->dbeg != 0)
        return LEV_EDIT_ERR_BLOCK;
      break;

      default:
      return LEV_EDIT_ERR_TYPE;
      break;
    }
  }

  /* check ordering */
  b = bops + 1;
  for (i = nb - 1; i; i--, b++, bops++) {
    if (b->sbeg != bops->send || b->dbeg != bops->dend)
      return LEV_EDIT_ERR_ORDER;
  }

  return LEV_EDIT_ERR_OK;
}

/**
 * lev_editops_invert:
 * @n: The length of @ops.
 * @ops: An array of elementary edit operations.
 *
 * Inverts the sense of @ops.  It is modified in place.
 *
 * In other words, @ops becomes a valid partial edit for the original source
 * and destination strings with their roles exchanged.
 **/
_LEV_STATIC_PY void
lev_editops_invert(size_t n, LevEditOp *ops)
{
  size_t i;

  for (i = n; i; i--, ops++) {
    size_t z;

    z = ops->dpos;
    ops->dpos = ops->spos;
    ops->spos = z;
    if (ops->type & 2)
      ops->type ^= 1;
  }
}

/**
 * lev_editops_apply:
 * @len1: The length of @string1.
 * @string1: A string of length @len1, may contain NUL characters.
 * @len2: The length of @string2.
 * @string2: A string of length @len2, may contain NUL characters.
 * @n: The size of @ops.
 * @ops: An array of elementary edit operations.
 * @len: Where the size of the resulting string should be stored.
 *
 * Applies a partial edit @ops from @string1 to @string2.
 *
 * NB: @ops is not checked for applicability.
 *
 * Returns: The result of the partial edit as a newly allocated string, its
 *          length is stored in @len.
 **/
_LEV_STATIC_PY lev_byte*
lev_editops_apply(size_t len1, const lev_byte *string1,
                  __attribute__((unused)) size_t len2, const lev_byte *string2,
                  size_t n, const LevEditOp *ops,
                  size_t *len)
{
  lev_byte *dst, *dpos;  /* destination string */
  const lev_byte *spos;  /* source string position */
  size_t i, j;

  /* this looks too complex for such a simple task, but note ops is not
   * a complete edit sequence, we have to be able to apply anything anywhere */
  dpos = dst = (lev_byte*)safe_malloc((n + len1), sizeof(lev_byte));
  if (!dst) {
    *len = (size_t)(-1);
    return NULL;
  }
  spos = string1;
  for (i = n; i; i--, ops++) {
    /* XXX: this fine with gcc internal memcpy, but when memcpy is
     * actually a function, it may be pretty slow */
    j = ops->spos - (spos - string1) + (ops->type == LEV_EDIT_KEEP);
    if (j) {
      memcpy(dpos, spos, j*sizeof(lev_byte));
      spos += j;
      dpos += j;
    }
    switch (ops->type) {
      case LEV_EDIT_DELETE:
      spos++;
      break;

      case LEV_EDIT_REPLACE:
      spos++;
      case LEV_EDIT_INSERT:
      *(dpos++) = string2[ops->dpos];
      break;

      default:
      break;
    }
  }
  j = len1 - (spos - string1);
  if (j) {
    memcpy(dpos, spos, j*sizeof(lev_byte));
    spos += j;
    dpos += j;
  }

  *len = dpos - dst;
  /* possible realloc failure is detected with *len != 0 */
  return realloc(dst, *len*sizeof(lev_byte));
}

/**
 * lev_u_editops_apply:
 * @len1: The length of @string1.
 * @string1: A string of length @len1, may contain NUL characters.
 * @len2: The length of @string2.
 * @string2: A string of length @len2, may contain NUL characters.
 * @n: The size of @ops.
 * @ops: An array of elementary edit operations.
 * @len: Where the size of the resulting string should be stored.
 *
 * Applies a partial edit @ops from @string1 to @string2.
 *
 * NB: @ops is not checked for applicability.
 *
 * Returns: The result of the partial edit as a newly allocated string, its
 *          length is stored in @len.
 **/
_LEV_STATIC_PY lev_wchar*
lev_u_editops_apply(size_t len1, const lev_wchar *string1,
                    __attribute__((unused)) size_t len2,
                    const lev_wchar *string2,
                    size_t n, const LevEditOp *ops,
                    size_t *len)
{
  lev_wchar *dst, *dpos;  /* destination string */
  const lev_wchar *spos;  /* source string position */
  size_t i, j;

  /* this looks too complex for such a simple task, but note ops is not
   * a complete edit sequence, we have to be able to apply anything anywhere */
  dpos = dst = (lev_wchar*)safe_malloc((n + len1), sizeof(lev_wchar));
  if (!dst) {
    *len = (size_t)(-1);
    return NULL;
  }
  spos = string1;
  for (i = n; i; i--, ops++) {
    /* XXX: this is fine with gcc internal memcpy, but when memcpy is
     * actually a function, it may be pretty slow */
    j = ops->spos - (spos - string1) + (ops->type == LEV_EDIT_KEEP);
    if (j) {
      memcpy(dpos, spos, j*sizeof(lev_wchar));
      spos += j;
      dpos += j;
    }
    switch (ops->type) {
      case LEV_EDIT_DELETE:
      spos++;
      break;

      case LEV_EDIT_REPLACE:
      spos++;
      case LEV_EDIT_INSERT:
      *(dpos++) = string2[ops->dpos];
      break;

      default:
      break;
    }
  }
  j = len1 - (spos - string1);
  if (j) {
    memcpy(dpos, spos, j*sizeof(lev_wchar));
    spos += j;
    dpos += j;
  }

  *len = dpos - dst;
  /* possible realloc failure is detected with *len != 0 */
  return realloc(dst, *len*sizeof(lev_wchar));
}

/**
 * editops_from_cost_matrix:
 * @len1: The length of @string1.
 * @string1: A string of length @len1, may contain NUL characters.
 * @o1: The offset where the matrix starts from the start of @string1.
 * @len2: The length of @string2.
 * @string2: A string of length @len2, may contain NUL characters.
 * @o2: The offset where the matrix starts from the start of @string2.
 * @matrix: The cost matrix.
 * @n: Where the number of edit operations should be stored.
 *
 * Reconstructs the optimal edit sequence from the cost matrix @matrix.
 *
 * The matrix is freed.
 *
 * Returns: The optimal edit sequence, as a newly allocated array of
 *          elementary edit operations, it length is stored in @n.
 **/
static LevEditOp*
editops_from_cost_matrix(size_t len1, const lev_byte *string1, size_t off1,
                         size_t len2, const lev_byte *string2, size_t off2,
                         size_t *matrix, size_t *n)
{
  size_t *p;
  size_t i, j, pos;
  LevEditOp *ops;
  int dir = 0;

  pos = *n = matrix[len1*len2 - 1];
  if (!*n) {
    free(matrix);
    return NULL;
  }
  ops = (LevEditOp*)safe_malloc((*n), sizeof(LevEditOp));
  if (!ops) {
    free(matrix);
    *n = (size_t)(-1);
    return NULL;
  }
  i = len1 - 1;
  j = len2 - 1;
  p = matrix + len1*len2 - 1;
  while (i || j) {
    /* prefer contiuning in the same direction */
    if (dir < 0 && j && *p == *(p - 1) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_INSERT;
      ops[pos].spos = i + off1;
      ops[pos].dpos = --j + off2;
      p--;
      continue;
    }
    if (dir > 0 && i && *p == *(p - len2) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_DELETE;
      ops[pos].spos = --i + off1;
      ops[pos].dpos = j + off2;
      p -= len2;
      continue;
    }
    if (i && j && *p == *(p - len2 - 1)
        && string1[i - 1] == string2[j - 1]) {
      /* don't be stupid like difflib, don't store LEV_EDIT_KEEP */
      i--;
      j--;
      p -= len2 + 1;
      dir = 0;
      continue;
    }
    if (i && j && *p == *(p - len2 - 1) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_REPLACE;
      ops[pos].spos = --i + off1;
      ops[pos].dpos = --j + off2;
      p -= len2 + 1;
      dir = 0;
      continue;
    }
    /* we cant't turn directly from -1 to 1, in this case it would be better
     * to go diagonally, but check it (dir == 0) */
    if (dir == 0 && j && *p == *(p - 1) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_INSERT;
      ops[pos].spos = i + off1;
      ops[pos].dpos = --j + off2;
      p--;
      dir = -1;
      continue;
    }
    if (dir == 0 && i && *p == *(p - len2) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_DELETE;
      ops[pos].spos = --i + off1;
      ops[pos].dpos = j + off2;
      p -= len2;
      dir = 1;
      continue;
    }
    /* coredump right now, later might be too late ;-) */
    assert("lost in the cost matrix" == NULL);
  }
  free(matrix);

  return ops;
}

/**
 * lev_editops_find:
 * @len1: The length of @string1.
 * @string1: A string of length @len1, may contain NUL characters.
 * @len2: The length of @string2.
 * @string2: A string of length @len2, may contain NUL characters.
 * @n: Where the number of edit operations should be stored.
 *
 * Find an optimal edit sequence from @string1 to @string2.
 *
 * When there's more than one optimal sequence, a one is arbitrarily (though
 * deterministically) chosen.
 *
 * Returns: The optimal edit sequence, as a newly allocated array of
 *          elementary edit operations, it length is stored in @n.
 *          It is normalized, i.e., keep operations are not included.
 **/
_LEV_STATIC_PY LevEditOp*
lev_editops_find(size_t len1, const lev_byte *string1,
                 size_t len2, const lev_byte *string2,
                 size_t *n)
{
  size_t len1o, len2o;
  size_t i;
  size_t *matrix; /* cost matrix */

  /* strip common prefix */
  len1o = 0;
  while (len1 > 0 && len2 > 0 && *string1 == *string2) {
    len1--;
    len2--;
    string1++;
    string2++;
    len1o++;
  }
  len2o = len1o;

  /* strip common suffix */
  while (len1 > 0 && len2 > 0 && string1[len1-1] == string2[len2-1]) {
    len1--;
    len2--;
  }
  len1++;
  len2++;

  /* initialize first row and column */
  matrix = (size_t*)safe_malloc_3(len1, len2, sizeof(size_t));
  if (!matrix) {
    *n = (size_t)(-1);
    return NULL;
  }
  for (i = 0; i < len2; i++)
    matrix[i] = i;
  for (i = 1; i < len1; i++)
    matrix[len2*i] = i;

  /* find the costs and fill the matrix */
  for (i = 1; i < len1; i++) {
    size_t *prev = matrix + (i - 1)*len2;
    size_t *p = matrix + i*len2;
    size_t *end = p + len2 - 1;
    const lev_byte char1 = string1[i - 1];
    const lev_byte *char2p = string2;
    size_t x = i;
    p++;
    while (p <= end) {
      size_t c3 = *(prev++) + (char1 != *(char2p++));
      x++;
      if (x > c3)
        x = c3;
      c3 = *prev + 1;
      if (x > c3)
        x = c3;
      *(p++) = x;
    }
  }

  /* find the way back */
  return editops_from_cost_matrix(len1, string1, len1o,
                                  len2, string2, len2o,
                                  matrix, n);
}

/**
 * ueditops_from_cost_matrix:
 * @len1: The length of @string1.
 * @string1: A string of length @len1, may contain NUL characters.
 * @o1: The offset where the matrix starts from the start of @string1.
 * @len2: The length of @string2.
 * @string2: A string of length @len2, may contain NUL characters.
 * @o2: The offset where the matrix starts from the start of @string2.
 * @matrix: The cost matrix.
 * @n: Where the number of edit operations should be stored.
 *
 * Reconstructs the optimal edit sequence from the cost matrix @matrix.
 *
 * The matrix is freed.
 *
 * Returns: The optimal edit sequence, as a newly allocated array of
 *          elementary edit operations, it length is stored in @n.
 **/
static LevEditOp*
ueditops_from_cost_matrix(size_t len1, const lev_wchar *string1, size_t o1,
                          size_t len2, const lev_wchar *string2, size_t o2,
                          size_t *matrix, size_t *n)
{
  size_t *p;
  size_t i, j, pos;
  LevEditOp *ops;
  int dir = 0;

  pos = *n = matrix[len1*len2 - 1];
  if (!*n) {
    free(matrix);
    return NULL;
  }
  ops = (LevEditOp*)safe_malloc((*n), sizeof(LevEditOp));
  if (!ops) {
    free(matrix);
    *n = (size_t)(-1);
    return NULL;
  }
  i = len1 - 1;
  j = len2 - 1;
  p = matrix + len1*len2 - 1;
  while (i || j) {
    /* prefer contiuning in the same direction */
    if (dir < 0 && j && *p == *(p - 1) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_INSERT;
      ops[pos].spos = i + o1;
      ops[pos].dpos = --j + o2;
      p--;
      continue;
    }
    if (dir > 0 && i && *p == *(p - len2) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_DELETE;
      ops[pos].spos = --i + o1;
      ops[pos].dpos = j + o2;
      p -= len2;
      continue;
    }
    if (i && j && *p == *(p - len2 - 1)
        && string1[i - 1] == string2[j - 1]) {
      /* don't be stupid like difflib, don't store LEV_EDIT_KEEP */
      i--;
      j--;
      p -= len2 + 1;
      dir = 0;
      continue;
    }
    if (i && j && *p == *(p - len2 - 1) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_REPLACE;
      ops[pos].spos = --i + o1;
      ops[pos].dpos = --j + o2;
      p -= len2 + 1;
      dir = 0;
      continue;
    }
    /* we cant't turn directly from -1 to 1, in this case it would be better
     * to go diagonally, but check it (dir == 0) */
    if (dir == 0 && j && *p == *(p - 1) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_INSERT;
      ops[pos].spos = i + o1;
      ops[pos].dpos = --j + o2;
      p--;
      dir = -1;
      continue;
    }
    if (dir == 0 && i && *p == *(p - len2) + 1) {
      pos--;
      ops[pos].type = LEV_EDIT_DELETE;
      ops[pos].spos = --i + o1;
      ops[pos].dpos = j + o2;
      p -= len2;
      dir = 1;
      continue;
    }
    /* coredump right now, later might be too late ;-) */
    assert("lost in the cost matrix" == NULL);
  }
  free(matrix);

  return ops;
}

/**
 * lev_u_editops_find:
 * @len1: The length of @string1.
 * @string1: A string of length @len1, may contain NUL characters.
 * @len2: The length of @string2.
 * @string2: A string of length @len2, may contain NUL characters.
 * @n: Where the number of edit operations should be stored.
 *
 * Find an optimal edit sequence from @string1 to @string2.
 *
 * When there's more than one optimal sequence, a one is arbitrarily (though
 * deterministically) chosen.
 *
 * Returns: The optimal edit sequence, as a newly allocated array of
 *          elementary edit operations, it length is stored in @n.
 *          It is normalized, i.e., keep operations are not included.
 **/
_LEV_STATIC_PY LevEditOp*
lev_u_editops_find(size_t len1, const lev_wchar *string1,
                   size_t len2, const lev_wchar *string2,
                   size_t *n)
{
  size_t len1o, len2o;
  size_t i;
  size_t *matrix; /* cost matrix */

  /* strip common prefix */
  len1o = 0;
  while (len1 > 0 && len2 > 0 && *string1 == *string2) {
    len1--;
    len2--;
    string1++;
    string2++;
    len1o++;
  }
  len2o = len1o;

  /* strip common suffix */
  while (len1 > 0 && len2 > 0 && string1[len1-1] == string2[len2-1]) {
    len1--;
    len2--;
  }
  len1++;
  len2++;

  /* initalize first row and column */
  matrix = (size_t*)safe_malloc_3(len1, len2, sizeof(size_t));
  if (!matrix) {
    *n = (size_t)(-1);
    return NULL;
  }
  for (i = 0; i < len2; i++)
    matrix[i] = i;
  for (i = 1; i < len1; i++)
    matrix[len2*i] = i;

  /* find the costs and fill the matrix */
  for (i = 1; i < len1; i++) {
    size_t *prev = matrix + (i - 1)*len2;
    size_t *p = matrix + i*len2;
    size_t *end = p + len2 - 1;
    const lev_wchar char1 = string1[i - 1];
    const lev_wchar *char2p = string2;
    size_t x = i;
    p++;
    while (p <= end) {
      size_t c3 = *(prev++) + (char1 != *(char2p++));
      x++;
      if (x > c3)
        x = c3;
      c3 = *prev + 1;
      if (x > c3)
        x = c3;
      *(p++) = x;
    }
  }

  /* find the way back */
  return ueditops_from_cost_matrix(len1, string1, len1o,
                                   len2, string2, len2o,
                                   matrix, n);
}

/**
 * lev_opcodes_to_editops:
 * @nb: The length of @bops.
 * @bops: An array of difflib block edit operation codes.
 * @n: Where the number of edit operations should be stored.
 * @keepkeep: If nonzero, keep operations will be included.  Otherwise the
 *            result will be normalized, i.e. without any keep operations.
 *
 * Converts difflib block operation codes to elementary edit operations.
 *
 * Returns: The converted edit operatiosn, as a newly allocated array; its
 *          size is stored in @n.
 **/
_LEV_STATIC_PY LevEditOp*
lev_opcodes_to_editops(size_t nb, const LevOpCode *bops,
                       size_t *n, int keepkeep)
{
  size_t i;
  const LevOpCode *b;
  LevEditOp *ops, *o;

  /* compute the number of atomic operations */
  *n = 0;
  if (!nb)
    return NULL;

  b = bops;
  if (keepkeep) {
    for (i = nb; i; i--, b++) {
      size_t sd = b->send - b->sbeg;
      size_t dd = b->dend - b->dbeg;
      *n += (sd > dd ? sd : dd);
    }
  }
  else {
    for (i = nb; i; i--, b++) {
      size_t sd = b->send - b->sbeg;
      size_t dd = b->dend - b->dbeg;
      *n += (b->type != LEV_EDIT_KEEP ? (sd > dd ? sd : dd) : 0);
    }
  }

  /* convert */
  o = ops = (LevEditOp*)safe_malloc((*n), sizeof(LevEditOp));
  if (!ops) {
    *n = (size_t)(-1);
    return NULL;
  }
  b = bops;
  for (i = nb; i; i--, b++) {
    size_t j;

    switch (b->type) {
      case LEV_EDIT_KEEP:
      if (keepkeep) {
        for (j = 0; j < b->send - b->sbeg; j++, o++) {
          o->type = LEV_EDIT_KEEP;
          o->spos = b->sbeg + j;
          o->dpos = b->dbeg + j;
        }
      }
      break;

      case LEV_EDIT_REPLACE:
      for (j = 0; j < b->send - b->sbeg; j++, o++) {
        o->type = LEV_EDIT_REPLACE;
        o->spos = b->sbeg + j;
        o->dpos = b->dbeg + j;
      }
      break;

      case LEV_EDIT_DELETE:
      for (j = 0; j < b->send - b->sbeg; j++, o++) {
        o->type = LEV_EDIT_DELETE;
        o->spos = b->sbeg + j;
        o->dpos = b->dbeg;
      }
      break;

      case LEV_EDIT_INSERT:
      for (j = 0; j < b->dend - b->dbeg; j++, o++) {
        o->type = LEV_EDIT_INSERT;
        o->spos = b->sbeg;
        o->dpos = b->dbeg + j;
      }
      break;

      default:
      break;
    }
  }
  assert((size_t)(o - ops) == *n);

  return ops;
}

/**
 * lev_editops_to_opcodes:
 * @n: The size of @ops.
 * @ops: An array of elementary edit operations.
 * @nb: Where the number of difflib block operation codes should be stored.
 * @len1: The length of the source string.
 * @len2: The length of the destination string.
 *
 * Converts elementary edit operations to difflib block operation codes.
 *
 * Note the string lengths are necessary since difflib doesn't allow omitting
 * keep operations.
 *
 * Returns: The converted block operation codes, as a newly allocated array;
 *          its length is stored in @nb.
 **/
_LEV_STATIC_PY LevOpCode*
lev_editops_to_opcodes(size_t n, const LevEditOp *ops, size_t *nb,
                       size_t len1, size_t len2)
{
  size_t nbl, i, spos, dpos;
  const LevEditOp *o;
  LevOpCode *bops, *b;
  LevEditType type;

  /* compute the number of blocks */
  nbl = 0;
  o = ops;
  spos = dpos = 0;
  type = LEV_EDIT_KEEP;
  for (i = n; i; ) {
    /* simply pretend there are no keep blocks */
    while (o->type == LEV_EDIT_KEEP && --i)
      o++;
    if (!i)
      break;
    if (spos < o->spos || dpos < o->dpos) {
      nbl++;
      spos = o->spos;
      dpos = o->dpos;
    }
    nbl++;
    type = o->type;
    switch (type) {
      case LEV_EDIT_REPLACE:
      do {
        spos++;
        dpos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      case LEV_EDIT_DELETE:
      do {
        spos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      case LEV_EDIT_INSERT:
      do {
        dpos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      default:
      break;
    }
  }
  if (spos < len1 || dpos < len2)
    nbl++;

  /* convert */
  b = bops = (LevOpCode*)safe_malloc(nbl, sizeof(LevOpCode));
  if (!bops) {
    *nb = (size_t)(-1);
    return NULL;
  }
  o = ops;
  spos = dpos = 0;
  type = LEV_EDIT_KEEP;
  for (i = n; i; ) {
    /* simply pretend there are no keep blocks */
    while (o->type == LEV_EDIT_KEEP && --i)
      o++;
    if (!i)
      break;
    b->sbeg = spos;
    b->dbeg = dpos;
    if (spos < o->spos || dpos < o->dpos) {
      b->type = LEV_EDIT_KEEP;
      spos = b->send = o->spos;
      dpos = b->dend = o->dpos;
      b++;
      b->sbeg = spos;
      b->dbeg = dpos;
    }
    type = o->type;
    switch (type) {
      case LEV_EDIT_REPLACE:
      do {
        spos++;
        dpos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      case LEV_EDIT_DELETE:
      do {
        spos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      case LEV_EDIT_INSERT:
      do {
        dpos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      default:
      break;
    }
    b->type = type;
    b->send = spos;
    b->dend = dpos;
    b++;
  }
  if (spos < len1 || dpos < len2) {
    assert(len1 - spos == len2 - dpos);
    b->type = LEV_EDIT_KEEP;
    b->sbeg = spos;
    b->dbeg = dpos;
    b->send = len1;
    b->dend = len2;
    b++;
  }
  assert((size_t)(b - bops) == nbl);

  *nb = nbl;
  return bops;
}

/**
 * lev_opcodes_apply:
 * @len1: The length of the source string.
 * @string1: A string of length @len1, may contain NUL characters.
 * @len2: The length of the destination string.
 * @string2: A string of length @len2, may contain NUL characters.
 * @nb: The length of @bops.
 * @bops: An array of difflib block edit operation codes.
 * @len: Where the size of the resulting string should be stored.
 *
 * Applies a sequence of difflib block operations to a string.
 *
 * NB: @bops is not checked for applicability.
 *
 * Returns: The result of the edit as a newly allocated string, its length
 *          is stored in @len.
 **/
_LEV_STATIC_PY lev_byte*
lev_opcodes_apply(size_t len1, const lev_byte *string1,
                  size_t len2, const lev_byte *string2,
                  size_t nb, const LevOpCode *bops,
                  size_t *len)
{
  lev_byte *dst, *dpos;  /* destination string */
  const lev_byte *spos;  /* source string position */
  size_t i;

  /* this looks too complex for such a simple task, but note ops is not
   * a complete edit sequence, we have to be able to apply anything anywhere */
  dpos = dst = (lev_byte*)safe_malloc((len1 + len2), sizeof(lev_byte));
  if (!dst) {
    *len = (size_t)(-1);
    return NULL;
  }
  spos = string1;
  for (i = nb; i; i--, bops++) {
    switch (bops->type) {
      case LEV_EDIT_INSERT:
      case LEV_EDIT_REPLACE:
      memcpy(dpos, string2 + bops->dbeg,
             (bops->dend - bops->dbeg)*sizeof(lev_byte));
      break;

      case LEV_EDIT_KEEP:
      memcpy(dpos, string1 + bops->sbeg,
             (bops->send - bops->sbeg)*sizeof(lev_byte));
      break;

      default:
      break;
    }
    spos += bops->send - bops->sbeg;
    dpos += bops->dend - bops->dbeg;
  }

  *len = dpos - dst;
  /* possible realloc failure is detected with *len != 0 */
  return realloc(dst, *len*sizeof(lev_byte));
}

/**
 * lev_u_opcodes_apply:
 * @len1: The length of the source string.
 * @string1: A string of length @len1, may contain NUL characters.
 * @len2: The length of the destination string.
 * @string2: A string of length @len2, may contain NUL characters.
 * @nb: The length of @bops.
 * @bops: An array of difflib block edit operation codes.
 * @len: Where the size of the resulting string should be stored.
 *
 * Applies a sequence of difflib block operations to a string.
 *
 * NB: @bops is not checked for applicability.
 *
 * Returns: The result of the edit as a newly allocated string, its length
 *          is stored in @len.
 **/
_LEV_STATIC_PY lev_wchar*
lev_u_opcodes_apply(size_t len1, const lev_wchar *string1,
                    size_t len2, const lev_wchar *string2,
                    size_t nb, const LevOpCode *bops,
                    size_t *len)
{
  lev_wchar *dst, *dpos;  /* destination string */
  const lev_wchar *spos;  /* source string position */
  size_t i;

  /* this looks too complex for such a simple task, but note ops is not
   * a complete edit sequence, we have to be able to apply anything anywhere */
  dpos = dst = (lev_wchar*)safe_malloc((len1 + len2), sizeof(lev_wchar));
  if (!dst) {
    *len = (size_t)(-1);
    return NULL;
  }
  spos = string1;
  for (i = nb; i; i--, bops++) {
    switch (bops->type) {
      case LEV_EDIT_INSERT:
      case LEV_EDIT_REPLACE:
      memcpy(dpos, string2 + bops->dbeg,
             (bops->dend - bops->dbeg)*sizeof(lev_wchar));
      break;

      case LEV_EDIT_KEEP:
      memcpy(dpos, string1 + bops->sbeg,
             (bops->send - bops->sbeg)*sizeof(lev_wchar));
      break;

      default:
      break;
    }
    spos += bops->send - bops->sbeg;
    dpos += bops->dend - bops->dbeg;
  }

  *len = dpos - dst;
  /* possible realloc failure is detected with *len != 0 */
  return realloc(dst, *len*sizeof(lev_wchar));
}

/**
 * lev_opcodes_invert:
 * @nb: The length of @ops.
 * @bops: An array of difflib block edit operation codes.
 *
 * Inverts the sense of @ops.  It is modified in place.
 *
 * In other words, @ops becomes a partial edit for the original source
 * and destination strings with their roles exchanged.
 **/
_LEV_STATIC_PY void
lev_opcodes_invert(size_t nb, LevOpCode *bops)
{
  size_t i;

  for (i = nb; i; i--, bops++) {
    size_t z;

    z = bops->dbeg;
    bops->dbeg = bops->sbeg;
    bops->sbeg = z;
    z = bops->dend;
    bops->dend = bops->send;
    bops->send = z;
    if (bops->type & 2)
      bops->type ^= 1;
  }
}

/**
 * lev_editops_matching_blocks:
 * @len1: The length of the source string.
 * @len2: The length of the destination string.
 * @n: The size of @ops.
 * @ops: An array of elementary edit operations.
 * @nmblocks: Where the number of matching block should be stored.
 *
 * Computes the matching block corresponding to an optimal edit @ops.
 *
 * Returns: The matching blocks as a newly allocated array, it length is
 *          stored in @nmblocks.
 **/
_LEV_STATIC_PY LevMatchingBlock*
lev_editops_matching_blocks(size_t len1,
                            size_t len2,
                            size_t n,
                            const LevEditOp *ops,
                            size_t *nmblocks)
{
  size_t nmb, i, spos, dpos;
  LevEditType type;
  const LevEditOp *o;
  LevMatchingBlock *mblocks, *mb;

  /* compute the number of matching blocks */
  nmb = 0;
  o = ops;
  spos = dpos = 0;
  type = LEV_EDIT_KEEP;
  for (i = n; i; ) {
    /* simply pretend there are no keep blocks */
    while (o->type == LEV_EDIT_KEEP && --i)
      o++;
    if (!i)
      break;
    if (spos < o->spos || dpos < o->dpos) {
      nmb++;
      spos = o->spos;
      dpos = o->dpos;
    }
    type = o->type;
    switch (type) {
      case LEV_EDIT_REPLACE:
      do {
        spos++;
        dpos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      case LEV_EDIT_DELETE:
      do {
        spos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      case LEV_EDIT_INSERT:
      do {
        dpos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      default:
      break;
    }
  }
  if (spos < len1 || dpos < len2)
    nmb++;

  /* fill the info */
  mb = mblocks = (LevMatchingBlock*)safe_malloc(nmb, sizeof(LevOpCode));
  if (!mblocks) {
    *nmblocks = (size_t)(-1);
    return NULL;
  }
  o = ops;
  spos = dpos = 0;
  type = LEV_EDIT_KEEP;
  for (i = n; i; ) {
    /* simply pretend there are no keep blocks */
    while (o->type == LEV_EDIT_KEEP && --i)
      o++;
    if (!i)
      break;
    if (spos < o->spos || dpos < o->dpos) {
      mb->spos = spos;
      mb->dpos = dpos;
      mb->len = o->spos - spos;
      spos = o->spos;
      dpos = o->dpos;
      mb++;
    }
    type = o->type;
    switch (type) {
      case LEV_EDIT_REPLACE:
      do {
        spos++;
        dpos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      case LEV_EDIT_DELETE:
      do {
        spos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      case LEV_EDIT_INSERT:
      do {
        dpos++;
        i--;
        o++;
      } while (i && o->type == type && spos == o->spos && dpos == o->dpos);
      break;

      default:
      break;
    }
  }
  if (spos < len1 || dpos < len2) {
    assert(len1 - spos == len2 - dpos);
    mb->spos = spos;
    mb->dpos = dpos;
    mb->len = len1 - spos;
    mb++;
  }
  assert((size_t)(mb - mblocks) == nmb);

  *nmblocks = nmb;
  return mblocks;
}

/**
 * lev_opcodes_matching_blocks:
 * @len1: The length of the source string.
 * @len2: The length of the destination string.
 * @nb: The size of @bops.
 * @bops: An array of difflib block edit operation codes.
 * @nmblocks: Where the number of matching block should be stored.
 *
 * Computes the matching block corresponding to an optimal edit @bops.
 *
 * Returns: The matching blocks as a newly allocated array, it length is
 *          stored in @nmblocks.
 **/
_LEV_STATIC_PY LevMatchingBlock*
lev_opcodes_matching_blocks(size_t len1,
                            __attribute__((unused)) size_t len2,
                            size_t nb,
                            const LevOpCode *bops,
                            size_t *nmblocks)
{
  size_t nmb, i;
  const LevOpCode *b;
  LevMatchingBlock *mblocks, *mb;

  /* compute the number of matching blocks */
  nmb = 0;
  b = bops;
  for (i = nb; i; i--, b++) {
    if (b->type == LEV_EDIT_KEEP) {
      nmb++;
      /* adjacent KEEP blocks -- we never produce it, but... */
      while (i && b->type == LEV_EDIT_KEEP) {
        i--;
        b++;
      }
      if (!i)
        break;
    }
  }

  /* convert */
  mb = mblocks = (LevMatchingBlock*)safe_malloc(nmb, sizeof(LevOpCode));
  if (!mblocks) {
    *nmblocks = (size_t)(-1);
    return NULL;
  }
  b = bops;
  for (i = nb; i; i--, b++) {
    if (b->type == LEV_EDIT_KEEP) {
      mb->spos = b->sbeg;
      mb->dpos = b->dbeg;
      /* adjacent KEEP blocks -- we never produce it, but... */
      while (i && b->type == LEV_EDIT_KEEP) {
        i--;
        b++;
      }
      if (!i) {
        mb->len = len1 - mb->spos;
        mb++;
        break;
      }
      mb->len = b->sbeg - mb->spos;
      mb++;
    }
  }
  assert((size_t)(mb - mblocks) == nmb);

  *nmblocks = nmb;
  return mblocks;
}

/**
 * lev_editops_total_cost:
 * @n: Tne size of @ops.
 * @ops: An array of elementary edit operations.
 *
 * Computes the total cost of operations in @ops.
 *
 * The costs of elementary operations are all 1.
 *
 * Returns: The total cost.
 **/
_LEV_STATIC_PY size_t
lev_editops_total_cost(size_t n,
                       const LevEditOp *ops)
{
  size_t i, sum = 0;

  for (i = n; i; i--, ops++)
    sum += !!ops->type;

  return sum;
}

/**
 * lev_editops_normalize:
 * @n: The size of @ops.
 * @ops: An array of elementary edit operations.
 * @nnorm: Where to store then length of the normalized array.
 *
 * Normalizes a list of edit operations to contain no keep operations.
 *
 * Note it returns %NULL for an empty array.
 *
 * Returns: A newly allocated array of normalized edit operations, its length
 *          is stored to @nnorm.
 **/
_LEV_STATIC_PY LevEditOp*
lev_editops_normalize(size_t n,
                      const LevEditOp *ops,
                      size_t *nnorm)
{
  size_t nx, i;
  const LevEditOp *o;
  LevEditOp *opsnorm, *on;

  if (!n || !ops) {
    *nnorm = 0;
    return NULL;
  }

  nx = 0;
  o = ops;
  for (i = n; i; i--, o++)
    nx += (o->type == LEV_EDIT_KEEP);

  *nnorm = n - nx;
  if (!*nnorm)
    return NULL;

  opsnorm = on = (LevEditOp*)safe_malloc((n - nx), sizeof(LevEditOp));
  o = ops;
  for (i = n; i; i--, o++) {
    if (o->type == LEV_EDIT_KEEP)
      continue;
    memcpy(on++, o, sizeof(LevEditOp));
  }

  return opsnorm;
}

/**
 * lev_opcodes_total_cost:
 * @nb: Tne size of @bops.
 * @bops: An array of difflib block operation codes.
 *
 * Computes the total cost of operations in @bops.
 *
 * The costs of elementary operations are all 1.
 *
 * Returns: The total cost.
 **/
_LEV_STATIC_PY size_t
lev_opcodes_total_cost(size_t nb,
                       const LevOpCode *bops)
{
  size_t i, sum = 0;

  for (i = nb; i; i--, bops++) {
    switch (bops->type) {
      case LEV_EDIT_REPLACE:
      case LEV_EDIT_DELETE:
      sum += (bops->send - bops->sbeg);
      break;

      case LEV_EDIT_INSERT:
      sum += (bops->dend - bops->dbeg);
      break;

      default:
      break;
    }
  }

  return sum;
}

/**
 * lev_editops_subtract:
 * @n: The size of @ops.
 * @ops: An array of elementary edit operations.
 * @ns: The size of @sub.
 * @sub: A subsequence (ordered subset) of @ops
 * @nrem: Where to store then length of the remainder array.
 *
 * Subtracts a subsequence of elementary edit operations from a sequence.
 *
 * The remainder is a sequence that, applied to result of application of @sub,
 * gives the same final result as application of @ops to original string.
 *
 * Returns: A newly allocated array of normalized edit operations, its length
 *          is stored to @nrem.  It is always normalized, i.e, without any
 *          keep operations.  On failure, %NULL is returned.
 **/
_LEV_STATIC_PY LevEditOp*
lev_editops_subtract(size_t n,
                     const LevEditOp *ops,
                     size_t ns,
                     const LevEditOp *sub,
                     size_t *nrem)
{
    static const int shifts[] = { 0, 0, 1, -1 };
    LevEditOp *rem;
    size_t i, j, nr, nn;
    int shift;

    /* compute remainder size */
    *nrem = -1;
    nr = nn = 0;
    for (i = 0; i < n; i++) {
        if (ops[i].type != LEV_EDIT_KEEP)
            nr++;
    }
    for (i = 0; i < ns; i++) {
        if (sub[i].type != LEV_EDIT_KEEP)
            nn++;
    }
    if (nn > nr)
        return NULL;
    nr -= nn;


    /* subtract */
    /* we could simply return NULL when nr == 0, but then it would be possible
     * to subtract *any* sequence of the right length to get an empty sequence
     * -- clrealy incorrectly; so we have to scan the list to check */
    rem = nr ? (LevEditOp*)safe_malloc(nr, sizeof(LevEditOp)) : NULL;
    j = nn = shift = 0;
    for (i = 0; i < ns; i++) {
        while ((ops[j].spos != sub[i].spos
                || ops[j].dpos != sub[i].dpos
                || ops[j].type != sub[i].type)
               && j < n) {
            if (ops[j].type != LEV_EDIT_KEEP) {
                rem[nn] = ops[j];
                rem[nn].spos += shift;
                nn++;
            }
            j++;
        }
        if (j == n) {
            free(rem);
            return NULL;
        }

        shift += shifts[sub[i].type];
        j++;
    }

    while (j < n) {
        if (ops[j].type != LEV_EDIT_KEEP) {
            rem[nn] = ops[j];
            rem[nn].spos += shift;
            nn++;
        }
        j++;
    }
    assert(nn == nr);

    *nrem = nr;
    return rem;
}
/* }}} */

/****************************************************************************
 *
 * Weighted mean strings
 *
 ****************************************************************************/

/*
_LEV_STATIC_PY lev_byte*
lev_weighted_average(size_t len1,
                     const lev_byte* string1,
                     size_t len2,
                     const lev_byte* string2,
                     size_t n,
                     const LevEditOp *ops,
                     LevAveragingType avtype,
                     size_t total_cost,
                     double alpha,
                     size_t *len)
{
  return NULL;
}
*/

