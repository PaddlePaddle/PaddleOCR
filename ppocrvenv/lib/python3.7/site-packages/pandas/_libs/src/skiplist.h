/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Flexibly-sized, index-able skiplist data structure for maintaining a sorted
list of values

Port of Wes McKinney's Cython version of Raymond Hettinger's original pure
Python recipe (https://rhettinger.wordpress.com/2010/02/06/lost-knowledge/)
*/

#ifndef PANDAS__LIBS_SRC_SKIPLIST_H_
#define PANDAS__LIBS_SRC_SKIPLIST_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "inline_helper.h"

PANDAS_INLINE float __skiplist_nanf(void) {
    const union {
        int __i;
        float __f;
    } __bint = {0x7fc00000UL};
    return __bint.__f;
}
#define PANDAS_NAN ((double)__skiplist_nanf())

PANDAS_INLINE double Log2(double val) { return log(val) / log(2.); }

typedef struct node_t node_t;

struct node_t {
    node_t **next;
    int *width;
    double value;
    int is_nil;
    int levels;
    int ref_count;
};

typedef struct {
    node_t *head;
    node_t **tmp_chain;
    int *tmp_steps;
    int size;
    int maxlevels;
} skiplist_t;

PANDAS_INLINE double urand(void) {
    return ((double)rand() + 1) / ((double)RAND_MAX + 2);
}

PANDAS_INLINE int int_min(int a, int b) { return a < b ? a : b; }

PANDAS_INLINE node_t *node_init(double value, int levels) {
    node_t *result;
    result = (node_t *)malloc(sizeof(node_t));
    if (result) {
        result->value = value;
        result->levels = levels;
        result->is_nil = 0;
        result->ref_count = 0;
        result->next = (node_t **)malloc(levels * sizeof(node_t *));
        result->width = (int *)malloc(levels * sizeof(int));
        if (!(result->next && result->width) && (levels != 0)) {
            free(result->next);
            free(result->width);
            free(result);
            return NULL;
        }
    }
    return result;
}

// do this ourselves
PANDAS_INLINE void node_incref(node_t *node) { ++(node->ref_count); }

PANDAS_INLINE void node_decref(node_t *node) { --(node->ref_count); }

static void node_destroy(node_t *node) {
    int i;
    if (node) {
        if (node->ref_count <= 1) {
            for (i = 0; i < node->levels; ++i) {
                node_destroy(node->next[i]);
            }
            free(node->next);
            free(node->width);
            // printf("Reference count was 1, freeing\n");
            free(node);
        } else {
            node_decref(node);
        }
        // pretty sure that freeing the struct above will be enough
    }
}

PANDAS_INLINE void skiplist_destroy(skiplist_t *skp) {
    if (skp) {
        node_destroy(skp->head);
        free(skp->tmp_steps);
        free(skp->tmp_chain);
        free(skp);
    }
}

PANDAS_INLINE skiplist_t *skiplist_init(int expected_size) {
    skiplist_t *result;
    node_t *NIL, *head;
    int maxlevels, i;

    maxlevels = 1 + Log2((double)expected_size);
    result = (skiplist_t *)malloc(sizeof(skiplist_t));
    if (!result) {
        return NULL;
    }
    result->tmp_chain = (node_t **)malloc(maxlevels * sizeof(node_t *));
    result->tmp_steps = (int *)malloc(maxlevels * sizeof(int));
    result->maxlevels = maxlevels;
    result->size = 0;

    head = result->head = node_init(PANDAS_NAN, maxlevels);
    NIL = node_init(0.0, 0);

    if (!(result->tmp_chain && result->tmp_steps && result->head && NIL)) {
        skiplist_destroy(result);
        node_destroy(NIL);
        return NULL;
    }

    node_incref(head);

    NIL->is_nil = 1;

    for (i = 0; i < maxlevels; ++i) {
        head->next[i] = NIL;
        head->width[i] = 1;
        node_incref(NIL);
    }

    return result;
}

// 1 if left < right, 0 if left == right, -1 if left > right
PANDAS_INLINE int _node_cmp(node_t *node, double value) {
    if (node->is_nil || node->value > value) {
        return -1;
    } else if (node->value < value) {
        return 1;
    } else {
        return 0;
    }
}

PANDAS_INLINE double skiplist_get(skiplist_t *skp, int i, int *ret) {
    node_t *node;
    int level;

    if (i < 0 || i >= skp->size) {
        *ret = 0;
        return 0;
    }

    node = skp->head;
    ++i;
    for (level = skp->maxlevels - 1; level >= 0; --level) {
        while (node->width[level] <= i) {
            i -= node->width[level];
            node = node->next[level];
        }
    }

    *ret = 1;
    return node->value;
}

PANDAS_INLINE int skiplist_insert(skiplist_t *skp, double value) {
    node_t *node, *prevnode, *newnode, *next_at_level;
    int *steps_at_level;
    int size, steps, level;
    node_t **chain;

    chain = skp->tmp_chain;

    steps_at_level = skp->tmp_steps;
    memset(steps_at_level, 0, skp->maxlevels * sizeof(int));

    node = skp->head;

    for (level = skp->maxlevels - 1; level >= 0; --level) {
        next_at_level = node->next[level];
        while (_node_cmp(next_at_level, value) >= 0) {
            steps_at_level[level] += node->width[level];
            node = next_at_level;
            next_at_level = node->next[level];
        }
        chain[level] = node;
    }

    size = int_min(skp->maxlevels, 1 - ((int)Log2(urand())));

    newnode = node_init(value, size);
    if (!newnode) {
        return -1;
    }
    steps = 0;

    for (level = 0; level < size; ++level) {
        prevnode = chain[level];
        newnode->next[level] = prevnode->next[level];

        prevnode->next[level] = newnode;
        node_incref(newnode);  // increment the reference count

        newnode->width[level] = prevnode->width[level] - steps;
        prevnode->width[level] = steps + 1;

        steps += steps_at_level[level];
    }

    for (level = size; level < skp->maxlevels; ++level) {
        chain[level]->width[level] += 1;
    }

    ++(skp->size);

    return 1;
}

PANDAS_INLINE int skiplist_remove(skiplist_t *skp, double value) {
    int level, size;
    node_t *node, *prevnode, *tmpnode, *next_at_level;
    node_t **chain;

    chain = skp->tmp_chain;
    node = skp->head;

    for (level = skp->maxlevels - 1; level >= 0; --level) {
        next_at_level = node->next[level];
        while (_node_cmp(next_at_level, value) > 0) {
            node = next_at_level;
            next_at_level = node->next[level];
        }
        chain[level] = node;
    }

    if (value != chain[0]->next[0]->value) {
        return 0;
    }

    size = chain[0]->next[0]->levels;

    for (level = 0; level < size; ++level) {
        prevnode = chain[level];

        tmpnode = prevnode->next[level];

        prevnode->width[level] += tmpnode->width[level] - 1;
        prevnode->next[level] = tmpnode->next[level];

        tmpnode->next[level] = NULL;
        node_destroy(tmpnode);  // decrement refcount or free
    }

    for (level = size; level < skp->maxlevels; ++level) {
        --(chain[level]->width[level]);
    }

    --(skp->size);
    return 1;
}

#endif  // PANDAS__LIBS_SRC_SKIPLIST_H_
