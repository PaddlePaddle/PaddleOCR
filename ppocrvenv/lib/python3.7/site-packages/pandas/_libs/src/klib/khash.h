/* The MIT License

   Copyright (c) 2008, 2009, 2011 by Attractive Chaos <attractor@live.co.uk>

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/*
  An example:

#include "khash.h"
KHASH_MAP_INIT_INT(32, char)
int main() {
	int ret, is_missing;
	khiter_t k;
	khash_t(32) *h = kh_init(32);
	k = kh_put(32, h, 5, &ret);
	if (!ret) kh_del(32, h, k);
	kh_value(h, k) = 10;
	k = kh_get(32, h, 10);
	is_missing = (k == kh_end(h));
	k = kh_get(32, h, 5);
	kh_del(32, h, k);
	for (k = kh_begin(h); k != kh_end(h); ++k)
		if (kh_exist(h, k)) kh_value(h, k) = 1;
	kh_destroy(32, h);
	return 0;
}
*/

/*
  2011-09-16 (0.2.6):

	* The capacity is a power of 2. This seems to dramatically improve the
	  speed for simple keys. Thank Zilong Tan for the suggestion. Reference:

	   - https://github.com/stefanocasazza/ULib
	   - https://nothings.org/computer/judy/

	* Allow to optionally use linear probing which usually has better
	  performance for random input. Double hashing is still the default as it
	  is more robust to certain non-random input.

	* Added Wang's integer hash function (not used by default). This hash
	  function is more robust to certain non-random input.

  2011-02-14 (0.2.5):

    * Allow to declare global functions.

  2009-09-26 (0.2.4):

    * Improve portability

  2008-09-19 (0.2.3):

	* Corrected the example
	* Improved interfaces

  2008-09-11 (0.2.2):

	* Improved speed a little in kh_put()

  2008-09-10 (0.2.1):

	* Added kh_clear()
	* Fixed a compiling error

  2008-09-02 (0.2.0):

	* Changed to token concatenation which increases flexibility.

  2008-08-31 (0.1.2):

	* Fixed a bug in kh_get(), which has not been tested previously.

  2008-08-31 (0.1.1):

	* Added destructor
*/


#ifndef __AC_KHASH_H
#define __AC_KHASH_H

/*!
  @header

  Generic hash table library.
 */

#define AC_VERSION_KHASH_H "0.2.6"

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "../inline_helper.h"


// hooks for memory allocator, C-runtime allocator used per default
#ifndef KHASH_MALLOC
#define KHASH_MALLOC malloc
#endif

#ifndef KHASH_REALLOC
#define KHASH_REALLOC realloc
#endif

#ifndef KHASH_CALLOC
#define KHASH_CALLOC calloc
#endif

#ifndef KHASH_FREE
#define KHASH_FREE free
#endif


#if UINT_MAX == 0xffffffffu
typedef unsigned int khuint32_t;
typedef signed int khint32_t;
#elif ULONG_MAX == 0xffffffffu
typedef unsigned long khuint32_t;
typedef signed long khint32_t;
#endif

#if ULONG_MAX == ULLONG_MAX
typedef unsigned long khuint64_t;
typedef signed long khint64_t;
#else
typedef unsigned long long khuint64_t;
typedef signed long long khint64_t;
#endif

#if UINT_MAX == 0xffffu
typedef unsigned int khuint16_t;
typedef signed int khint16_t;
#elif USHRT_MAX == 0xffffu
typedef unsigned short khuint16_t;
typedef signed short khint16_t;
#endif

#if UCHAR_MAX == 0xffu
typedef unsigned char khuint8_t;
typedef signed char khint8_t;
#endif

typedef double khfloat64_t;
typedef float khfloat32_t;

typedef khuint32_t khuint_t;
typedef khuint_t khiter_t;

#define __ac_isempty(flag, i) ((flag[i>>5]>>(i&0x1fU))&1)
#define __ac_isdel(flag, i) (0)
#define __ac_iseither(flag, i) __ac_isempty(flag, i)
#define __ac_set_isdel_false(flag, i) (0)
#define __ac_set_isempty_false(flag, i) (flag[i>>5]&=~(1ul<<(i&0x1fU)))
#define __ac_set_isempty_true(flag, i) (flag[i>>5]|=(1ul<<(i&0x1fU)))
#define __ac_set_isboth_false(flag, i) __ac_set_isempty_false(flag, i)
#define __ac_set_isdel_true(flag, i) ((void)0)


// specializations of https://github.com/aappleby/smhasher/blob/master/src/MurmurHash2.cpp
khuint32_t PANDAS_INLINE murmur2_32to32(khuint32_t k){
    const khuint32_t SEED = 0xc70f6907UL;
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.
    const khuint32_t M_32 = 0x5bd1e995;
    const int R_32 = 24;

    // Initialize the hash to a 'random' value
    khuint32_t h = SEED ^ 4;

    //handle 4 bytes:
    k *= M_32;
    k ^= k >> R_32;
    k *= M_32;

    h *= M_32;
    h ^= k;

    // Do a few final mixes of the hash to ensure the "last few
    // bytes" are well-incorporated. (Really needed here?)
    h ^= h >> 13;
    h *= M_32;
    h ^= h >> 15;
    return h;
}

// it is possible to have a special x64-version, which would need less operations, but
// using 32bit version always has also some benifits:
//    - one code for 32bit and 64bit builds
//    - the same case for 32bit and 64bit builds
//    - no performance difference could be measured compared to a possible x64-version

khuint32_t PANDAS_INLINE murmur2_32_32to32(khuint32_t k1, khuint32_t k2){
    const khuint32_t SEED = 0xc70f6907UL;
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.
    const khuint32_t M_32 = 0x5bd1e995;
    const int R_32 = 24;

    // Initialize the hash to a 'random' value
    khuint32_t h = SEED ^ 4;

    //handle first 4 bytes:
    k1 *= M_32;
    k1 ^= k1 >> R_32;
    k1 *= M_32;

    h *= M_32;
    h ^= k1;

    //handle second 4 bytes:
    k2 *= M_32;
    k2 ^= k2 >> R_32;
    k2 *= M_32;

    h *= M_32;
    h ^= k2;

    // Do a few final mixes of the hash to ensure the "last few
    // bytes" are well-incorporated.
    h ^= h >> 13;
    h *= M_32;
    h ^= h >> 15;
    return h;
}

khuint32_t PANDAS_INLINE murmur2_64to32(khuint64_t k){
    khuint32_t k1 = (khuint32_t)k;
    khuint32_t k2 = (khuint32_t)(k >> 32);

    return murmur2_32_32to32(k1, k2);
}


#ifdef KHASH_LINEAR
#define __ac_inc(k, m) 1
#else
#define __ac_inc(k, m) (murmur2_32to32(k) | 1) & (m)
#endif

#define __ac_fsize(m) ((m) < 32? 1 : (m)>>5)

#ifndef kroundup32
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))
#endif

static const double __ac_HASH_UPPER = 0.77;

#define KHASH_DECLARE(name, khkey_t, khval_t)		 					\
	typedef struct {													\
		khuint_t n_buckets, size, n_occupied, upper_bound;				\
		khuint32_t *flags;												\
		khkey_t *keys;													\
		khval_t *vals;													\
	} kh_##name##_t;													\
	extern kh_##name##_t *kh_init_##name();								\
	extern void kh_destroy_##name(kh_##name##_t *h);					\
	extern void kh_clear_##name(kh_##name##_t *h);						\
	extern khuint_t kh_get_##name(const kh_##name##_t *h, khkey_t key); 	\
	extern void kh_resize_##name(kh_##name##_t *h, khuint_t new_n_buckets); \
	extern khuint_t kh_put_##name(kh_##name##_t *h, khkey_t key, int *ret); \
	extern void kh_del_##name(kh_##name##_t *h, khuint_t x);

#define KHASH_INIT2(name, SCOPE, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal) \
	typedef struct {													\
		khuint_t n_buckets, size, n_occupied, upper_bound;				\
		khuint32_t *flags;												\
		khkey_t *keys;													\
		khval_t *vals;													\
	} kh_##name##_t;													\
	SCOPE kh_##name##_t *kh_init_##name(void) {								\
		return (kh_##name##_t*)KHASH_CALLOC(1, sizeof(kh_##name##_t));		\
	}																	\
	SCOPE void kh_destroy_##name(kh_##name##_t *h)						\
	{																	\
		if (h) {														\
			KHASH_FREE(h->keys); KHASH_FREE(h->flags);								\
			KHASH_FREE(h->vals);												\
			KHASH_FREE(h);													\
		}																\
	}																	\
	SCOPE void kh_clear_##name(kh_##name##_t *h)						\
	{																	\
		if (h && h->flags) {											\
			memset(h->flags, 0xaa, __ac_fsize(h->n_buckets) * sizeof(khuint32_t)); \
			h->size = h->n_occupied = 0;								\
		}																\
	}																	\
	SCOPE khuint_t kh_get_##name(const kh_##name##_t *h, khkey_t key) 	\
	{																	\
		if (h->n_buckets) {												\
			khuint_t inc, k, i, last, mask;								\
			mask = h->n_buckets - 1;									\
			k = __hash_func(key); i = k & mask;							\
			inc = __ac_inc(k, mask); last = i; /* inc==1 for linear probing */ \
			while (!__ac_isempty(h->flags, i) && (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) { \
				i = (i + inc) & mask; 									\
				if (i == last) return h->n_buckets;						\
			}															\
			return __ac_iseither(h->flags, i)? h->n_buckets : i;		\
		} else return 0;												\
	}																	\
	SCOPE void kh_resize_##name(kh_##name##_t *h, khuint_t new_n_buckets) \
	{ /* This function uses 0.25*n_bucktes bytes of working space instead of [sizeof(key_t+val_t)+.25]*n_buckets. */ \
		khuint32_t *new_flags = 0;										\
		khuint_t j = 1;													\
		{																\
			kroundup32(new_n_buckets); 									\
			if (new_n_buckets < 4) new_n_buckets = 4;					\
			if (h->size >= (khuint_t)(new_n_buckets * __ac_HASH_UPPER + 0.5)) j = 0;	/* requested size is too small */ \
			else { /* hash table size to be changed (shrink or expand); rehash */ \
				new_flags = (khuint32_t*)KHASH_MALLOC(__ac_fsize(new_n_buckets) * sizeof(khuint32_t));	\
				memset(new_flags, 0xff, __ac_fsize(new_n_buckets) * sizeof(khuint32_t)); \
				if (h->n_buckets < new_n_buckets) {	/* expand */		\
					h->keys = (khkey_t*)KHASH_REALLOC(h->keys, new_n_buckets * sizeof(khkey_t)); \
					if (kh_is_map) h->vals = (khval_t*)KHASH_REALLOC(h->vals, new_n_buckets * sizeof(khval_t)); \
				} /* otherwise shrink */								\
			}															\
		}																\
		if (j) { /* rehashing is needed */								\
			for (j = 0; j != h->n_buckets; ++j) {						\
				if (__ac_iseither(h->flags, j) == 0) {					\
					khkey_t key = h->keys[j];							\
					khval_t val;										\
					khuint_t new_mask;									\
					new_mask = new_n_buckets - 1; 						\
					if (kh_is_map) val = h->vals[j];					\
					__ac_set_isempty_true(h->flags, j);					\
					while (1) { /* kick-out process; sort of like in Cuckoo hashing */ \
						khuint_t inc, k, i;								\
						k = __hash_func(key);							\
						i = k & new_mask;								\
						inc = __ac_inc(k, new_mask);					\
						while (!__ac_isempty(new_flags, i)) i = (i + inc) & new_mask; \
						__ac_set_isempty_false(new_flags, i);			\
						if (i < h->n_buckets && __ac_iseither(h->flags, i) == 0) { /* kick out the existing element */ \
							{ khkey_t tmp = h->keys[i]; h->keys[i] = key; key = tmp; } \
							if (kh_is_map) { khval_t tmp = h->vals[i]; h->vals[i] = val; val = tmp; } \
							__ac_set_isempty_true(h->flags, i); /* mark it as deleted in the old hash table */ \
						} else { /* write the element and jump out of the loop */ \
							h->keys[i] = key;							\
							if (kh_is_map) h->vals[i] = val;			\
							break;										\
						}												\
					}													\
				}														\
			}															\
			if (h->n_buckets > new_n_buckets) { /* shrink the hash table */ \
				h->keys = (khkey_t*)KHASH_REALLOC(h->keys, new_n_buckets * sizeof(khkey_t)); \
				if (kh_is_map) h->vals = (khval_t*)KHASH_REALLOC(h->vals, new_n_buckets * sizeof(khval_t)); \
			}															\
			KHASH_FREE(h->flags); /* free the working space */				\
			h->flags = new_flags;										\
			h->n_buckets = new_n_buckets;								\
			h->n_occupied = h->size;									\
			h->upper_bound = (khuint_t)(h->n_buckets * __ac_HASH_UPPER + 0.5); \
		}																\
	}																	\
	SCOPE khuint_t kh_put_##name(kh_##name##_t *h, khkey_t key, int *ret) \
	{																	\
		khuint_t x;														\
		if (h->n_occupied >= h->upper_bound) { /* update the hash table */ \
			if (h->n_buckets > (h->size<<1)) kh_resize_##name(h, h->n_buckets - 1); /* clear "deleted" elements */ \
			else kh_resize_##name(h, h->n_buckets + 1); /* expand the hash table */ \
		} /* TODO: to implement automatically shrinking; resize() already support shrinking */ \
		{																\
			khuint_t inc, k, i, site, last, mask = h->n_buckets - 1;		\
			x = site = h->n_buckets; k = __hash_func(key); i = k & mask; \
			if (__ac_isempty(h->flags, i)) x = i; /* for speed up */	\
			else {														\
				inc = __ac_inc(k, mask); last = i;						\
				while (!__ac_isempty(h->flags, i) && (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) { \
					if (__ac_isdel(h->flags, i)) site = i;				\
					i = (i + inc) & mask; 								\
					if (i == last) { x = site; break; }					\
				}														\
				if (x == h->n_buckets) {								\
					if (__ac_isempty(h->flags, i) && site != h->n_buckets) x = site; \
					else x = i;											\
				}														\
			}															\
		}																\
		if (__ac_isempty(h->flags, x)) { /* not present at all */		\
			h->keys[x] = key;											\
			__ac_set_isboth_false(h->flags, x);							\
			++h->size; ++h->n_occupied;									\
			*ret = 1;													\
		} else if (__ac_isdel(h->flags, x)) { /* deleted */				\
			h->keys[x] = key;											\
			__ac_set_isboth_false(h->flags, x);							\
			++h->size;													\
			*ret = 2;													\
		} else *ret = 0; /* Don't touch h->keys[x] if present and not deleted */ \
		return x;														\
	}																	\
	SCOPE void kh_del_##name(kh_##name##_t *h, khuint_t x)				\
	{																	\
		if (x != h->n_buckets && !__ac_iseither(h->flags, x)) {			\
			__ac_set_isdel_true(h->flags, x);							\
			--h->size;													\
		}																\
	}

#define KHASH_INIT(name, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal) \
	KHASH_INIT2(name, PANDAS_INLINE, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal)

/* --- BEGIN OF HASH FUNCTIONS --- */

/*! @function
  @abstract     Integer hash function
  @param  key   The integer [khuint32_t]
  @return       The hash value [khuint_t]
 */
#define kh_int_hash_func(key) (khuint32_t)(key)
/*! @function
  @abstract     Integer comparison function
 */
#define kh_int_hash_equal(a, b) ((a) == (b))
/*! @function
  @abstract     64-bit integer hash function
  @param  key   The integer [khuint64_t]
  @return       The hash value [khuint_t]
 */
PANDAS_INLINE khuint_t kh_int64_hash_func(khuint64_t key)
{
    return (khuint_t)((key)>>33^(key)^(key)<<11);
}
/*! @function
  @abstract     64-bit integer comparison function
 */
#define kh_int64_hash_equal(a, b) ((a) == (b))

/*! @function
  @abstract     const char* hash function
  @param  s     Pointer to a null terminated string
  @return       The hash value
 */
PANDAS_INLINE khuint_t __ac_X31_hash_string(const char *s)
{
	khuint_t h = *s;
	if (h) for (++s ; *s; ++s) h = (h << 5) - h + *s;
	return h;
}
/*! @function
  @abstract     Another interface to const char* hash function
  @param  key   Pointer to a null terminated string [const char*]
  @return       The hash value [khuint_t]
 */
#define kh_str_hash_func(key) __ac_X31_hash_string(key)
/*! @function
  @abstract     Const char* comparison function
 */
#define kh_str_hash_equal(a, b) (strcmp(a, b) == 0)

PANDAS_INLINE khuint_t __ac_Wang_hash(khuint_t key)
{
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}
#define kh_int_hash_func2(k) __ac_Wang_hash((khuint_t)key)

/* --- END OF HASH FUNCTIONS --- */

/* Other convenient macros... */

/*!
  @abstract Type of the hash table.
  @param  name  Name of the hash table [symbol]
 */
#define khash_t(name) kh_##name##_t

/*! @function
  @abstract     Initiate a hash table.
  @param  name  Name of the hash table [symbol]
  @return       Pointer to the hash table [khash_t(name)*]
 */
#define kh_init(name) kh_init_##name(void)

/*! @function
  @abstract     Destroy a hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
 */
#define kh_destroy(name, h) kh_destroy_##name(h)

/*! @function
  @abstract     Reset a hash table without deallocating memory.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
 */
#define kh_clear(name, h) kh_clear_##name(h)

/*! @function
  @abstract     Resize a hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  s     New size [khuint_t]
 */
#define kh_resize(name, h, s) kh_resize_##name(h, s)

/*! @function
  @abstract     Insert a key to the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Key [type of keys]
  @param  r     Extra return code: 0 if the key is present in the hash table;
                1 if the bucket is empty (never used); 2 if the element in
				the bucket has been deleted [int*]
  @return       Iterator to the inserted element [khuint_t]
 */
#define kh_put(name, h, k, r) kh_put_##name(h, k, r)

/*! @function
  @abstract     Retrieve a key from the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Key [type of keys]
  @return       Iterator to the found element, or kh_end(h) is the element is absent [khuint_t]
 */
#define kh_get(name, h, k) kh_get_##name(h, k)

/*! @function
  @abstract     Remove a key from the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Iterator to the element to be deleted [khuint_t]
 */
#define kh_del(name, h, k) kh_del_##name(h, k)

/*! @function
  @abstract     Test whether a bucket contains data.
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khuint_t]
  @return       1 if containing data; 0 otherwise [int]
 */
#define kh_exist(h, x) (!__ac_iseither((h)->flags, (x)))

/*! @function
  @abstract     Get key given an iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khuint_t]
  @return       Key [type of keys]
 */
#define kh_key(h, x) ((h)->keys[x])

/*! @function
  @abstract     Get value given an iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khuint_t]
  @return       Value [type of values]
  @discussion   For hash sets, calling this results in segfault.
 */
#define kh_val(h, x) ((h)->vals[x])

/*! @function
  @abstract     Alias of kh_val()
 */
#define kh_value(h, x) ((h)->vals[x])

/*! @function
  @abstract     Get the start iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       The start iterator [khuint_t]
 */
#define kh_begin(h) (khuint_t)(0)

/*! @function
  @abstract     Get the end iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       The end iterator [khuint_t]
 */
#define kh_end(h) ((h)->n_buckets)

/*! @function
  @abstract     Get the number of elements in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       Number of elements in the hash table [khuint_t]
 */
#define kh_size(h) ((h)->size)

/*! @function
  @abstract     Get the number of buckets in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       Number of buckets in the hash table [khuint_t]
 */
#define kh_n_buckets(h) ((h)->n_buckets)

/* More convenient interfaces */

/*! @function
  @abstract     Instantiate a hash set containing integer keys
  @param  name  Name of the hash table [symbol]
 */
#define KHASH_SET_INIT_INT(name)										\
	KHASH_INIT(name, khint32_t, char, 0, kh_int_hash_func, kh_int_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing integer keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_INT(name, khval_t)								\
	KHASH_INIT(name, khint32_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

#define KHASH_MAP_INIT_UINT(name, khval_t)								\
	KHASH_INIT(name, khuint32_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing 64-bit integer keys
  @param  name  Name of the hash table [symbol]
 */
#define KHASH_SET_INIT_UINT64(name)										\
	KHASH_INIT(name, khuint64_t, char, 0, kh_int64_hash_func, kh_int64_hash_equal)

#define KHASH_SET_INIT_INT64(name)										\
	KHASH_INIT(name, khint64_t, char, 0, kh_int64_hash_func, kh_int64_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing 64-bit integer keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_UINT64(name, khval_t)								\
	KHASH_INIT(name, khuint64_t, khval_t, 1, kh_int64_hash_func, kh_int64_hash_equal)

#define KHASH_MAP_INIT_INT64(name, khval_t)								\
	KHASH_INIT(name, khint64_t, khval_t, 1, kh_int64_hash_func, kh_int64_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing 16bit-integer keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_INT16(name, khval_t)								\
	KHASH_INIT(name, khint16_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

#define KHASH_MAP_INIT_UINT16(name, khval_t)								\
	KHASH_INIT(name, khuint16_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing 8bit-integer keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_INT8(name, khval_t)								\
	KHASH_INIT(name, khint8_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

#define KHASH_MAP_INIT_UINT8(name, khval_t)								\
	KHASH_INIT(name, khuint8_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)



typedef const char *kh_cstr_t;
/*! @function
  @abstract     Instantiate a hash map containing const char* keys
  @param  name  Name of the hash table [symbol]
 */
#define KHASH_SET_INIT_STR(name)										\
	KHASH_INIT(name, kh_cstr_t, char, 0, kh_str_hash_func, kh_str_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing const char* keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_STR(name, khval_t)								\
	KHASH_INIT(name, kh_cstr_t, khval_t, 1, kh_str_hash_func, kh_str_hash_equal)


#define kh_exist_str(h, k) (kh_exist(h, k))
#define kh_exist_float64(h, k) (kh_exist(h, k))
#define kh_exist_uint64(h, k) (kh_exist(h, k))
#define kh_exist_int64(h, k) (kh_exist(h, k))
#define kh_exist_float32(h, k) (kh_exist(h, k))
#define kh_exist_int32(h, k) (kh_exist(h, k))
#define kh_exist_uint32(h, k) (kh_exist(h, k))
#define kh_exist_int16(h, k) (kh_exist(h, k))
#define kh_exist_uint16(h, k) (kh_exist(h, k))
#define kh_exist_int8(h, k) (kh_exist(h, k))
#define kh_exist_uint8(h, k) (kh_exist(h, k))

KHASH_MAP_INIT_STR(str, size_t)
KHASH_MAP_INIT_INT(int32, size_t)
KHASH_MAP_INIT_UINT(uint32, size_t)
KHASH_MAP_INIT_INT64(int64, size_t)
KHASH_MAP_INIT_UINT64(uint64, size_t)
KHASH_MAP_INIT_INT16(int16, size_t)
KHASH_MAP_INIT_UINT16(uint16, size_t)
KHASH_MAP_INIT_INT8(int8, size_t)
KHASH_MAP_INIT_UINT8(uint8, size_t)


#endif /* __AC_KHASH_H */
