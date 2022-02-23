# -*- coding: utf-8 -*-

def fast_comp(seq1, seq2, transpositions=False):
	"""Compute the distance between the two sequences `seq1` and `seq2` up to a
	maximum of 2 included, and return it. If the edit distance between the two
	sequences is higher than that, -1 is returned.
	
	If `transpositions` is `True`, transpositions will be taken into account for
	the computation of the distance. This can make a difference, e.g.:

		>>> fast_comp("abc", "bac", transpositions=False)
		2
		>>> fast_comp("abc", "bac", transpositions=True)
		1
	
	This is faster than `levenshtein` by an order of magnitude, but on the
	other hand is of limited use.

	The algorithm comes from `http://writingarchives.sakura.ne.jp/fastcomp`.
	I've added transpositions support to the original code.
	"""
	replace, insert, delete = "r", "i", "d"

	L1, L2  = len(seq1), len(seq2)
	if L1 < L2:
		L1, L2 = L2, L1
		seq1, seq2 = seq2, seq1

	ldiff = L1 - L2
	if ldiff == 0:
		models = (insert+delete, delete+insert, replace+replace)
	elif ldiff == 1:
		models = (delete+replace, replace+delete)
	elif ldiff == 2:
		models = (delete+delete,)
	else:
		return -1

	res = 3
	for model in models:
		i = j = c = 0
		while (i < L1) and (j < L2):
			if seq1[i] != seq2[j]:
				c = c+1
				if 2 < c:
					break
            
				if transpositions and ldiff != 2 \
            	and i < L1 - 1 and j < L2 - 1 \
            	and seq1[i+1] == seq2[j] and seq1[i] == seq2[j+1]:
					i, j = i+2, j+2
				else:
					cmd = model[c-1]
					if cmd == delete:
						i = i+1
					elif cmd == insert:
						j = j+1
					else:
						assert cmd == replace
						i,j = i+1, j+1
			else:
				i,j = i+1, j+1

		if 2 < c:
			continue
		elif i < L1:
			if L1-i <= model[c:].count(delete):
				c = c + (L1-i)
			else:
				continue
		elif j < L2:
			if L2-j <= model[c:].count(insert):
				c = c + (L2-j)
			else:
				continue

		if c < res:
			res = c

	if res == 3:
		res = -1
	return res
