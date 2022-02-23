# -*- coding: utf-8 -*-

def hamming(seq1, seq2, normalized=False):
	"""Compute the Hamming distance between the two sequences `seq1` and `seq2`.
	The Hamming distance is the number of differing items in two ordered
	sequences of the same length. If the sequences submitted do not have the
	same length, an error will be raised.
	
	If `normalized` evaluates to `False`, the return value will be an integer
	between 0 and the length of the sequences provided, edge values included;
	otherwise, it will be a float between 0 and 1 included, where 0 means
	equal, and 1 totally different. Normalized hamming distance is computed as:
	
		0.0                         if len(seq1) == 0
		hamming_dist / len(seq1)    otherwise
	"""
	L = len(seq1)
	if L != len(seq2):
		raise ValueError("expected two strings of the same length")
	if L == 0:
		return 0.0 if normalized else 0  # equal
	dist = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
	if normalized:
		return dist / float(L)
	return dist

def jaccard(seq1, seq2):
	"""Compute the Jaccard distance between the two sequences `seq1` and `seq2`.
	They should contain hashable items.
	
	The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
	"""
	set1, set2 = set(seq1), set(seq2)
	return 1 - len(set1 & set2) / float(len(set1 | set2))


def sorensen(seq1, seq2):
	"""Compute the Sorensen distance between the two sequences `seq1` and `seq2`.
	They should contain hashable items.
	
	The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
	"""
	set1, set2 = set(seq1), set(seq2)
	return 1 - (2 * len(set1 & set2) / float(len(set1) + len(set2)))
