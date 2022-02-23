# -*- coding: utf-8 -*-

from array import array


def levenshtein(seq1, seq2, max_dist=-1, normalized=False):
	"""Compute the absolute Levenshtein distance between the two sequences
	`seq1` and `seq2`.
	
	The Levenshtein distance is the minimum number of edit operations necessary
	for transforming one sequence into the other. The edit operations allowed are:
	
		* deletion:     ABC -> BC, AC, AB
		* insertion:    ABC -> ABCD, EABC, AEBC..
		* substitution: ABC -> ABE, ADC, FBC..
	
	The `max_dist` parameter controls at which moment we should stop computing the
	distance between the provided sequences. If it is a negative integer, the
	distance will be computed until the sequences are exhausted; otherwise, the
	computation will stop at the moment the calculated distance is higher than
	`max_dist`, and then return -1. For example:
	
		>>> levenshtein("abc", "abcd", max_dist=1)  # dist = 1
		1
		>>> levenshtein("abc", "abcde", max_dist=1) # dist = 2
		-1
	
	This can be a time saver if you're not interested in the exact distance, but
	only need to check if the distance between the given sequences is below a
	given threshold.
	
	The `normalized` parameter is here for backward compatibility; providing
	it will result in a call to `nlevenshtein`, which should be used directly
	instead. 
	"""
	if normalized:
		return nlevenshtein(seq1, seq2, method=1)
		
	if seq1 == seq2:
		return 0
	
	len1, len2 = len(seq1), len(seq2)
	if max_dist >= 0 and abs(len1 - len2) > max_dist:
		return -1
	if len1 == 0:
		return len2
	if len2 == 0:
		return len1
	if len1 < len2:
		len1, len2 = len2, len1
		seq1, seq2 = seq2, seq1
	
	column = array('L', range(len2 + 1))
	
	for x in range(1, len1 + 1):
		column[0] = x
		last = x - 1
		for y in range(1, len2 + 1):
			old = column[y]
			cost = int(seq1[x - 1] != seq2[y - 1])
			column[y] = min(column[y] + 1, column[y - 1] + 1, last + cost)
			last = old
		if max_dist >= 0 and min(column) > max_dist:
			return -1
	
	if max_dist >= 0 and column[len2] > max_dist:
		# stay consistent, even if we have the exact distance
		return -1
	return column[len2]


def nlevenshtein(seq1, seq2, method=1):
	"""Compute the normalized Levenshtein distance between `seq1` and `seq2`.
	
	Two normalization methods are provided. For both of them, the normalized
	distance will be a float between 0 and 1, where 0 means equal and 1
	completely different. The computation obeys the following patterns:
	
		0.0                       if seq1 == seq2
		1.0                       if len(seq1) == 0 or len(seq2) == 0
		edit distance / factor    otherwise
	
	The `method` parameter specifies which normalization factor should be used.
	It can have the value 1 or 2, which correspond to the following:
	
		1: the length of the shortest alignment between the sequences
		   (that is, the length of the longest sequence)
		2: the length of the longest alignment between the sequences
	
	Which normalization factor should be chosen is a matter of taste. The first
	one is cheap to compute. The second one is more costly, but it accounts
	better than the first one for parallelisms of symbols between the sequences.
		
	For the rationale behind the use of the second method, see:
	Heeringa, "Measuring Dialect Pronunciation Differences using Levenshtein
	Distance", 2004, p. 130 sq, which is available online at:
	http://www.let.rug.nl/~heeringa/dialectology/thesis/thesis.pdf
	"""
	
	if seq1 == seq2:
		return 0.0
	len1, len2 = len(seq1), len(seq2)
	if len1 == 0 or len2 == 0:
		return 1.0
	if len1 < len2: # minimize the arrays size
		len1, len2 = len2, len1
		seq1, seq2 = seq2, seq1
	
	if method == 1:
		return levenshtein(seq1, seq2) / float(len1)
	if method != 2:
		raise ValueError("expected either 1 or 2 for `method` parameter")
	
	column = array('L', range(len2 + 1))
	length = array('L', range(len2 + 1))
	
	for x in range(1, len1 + 1):
	
		column[0] = length[0] = x
		last = llast = x - 1
		
		for y in range(1, len2 + 1):
		
			# dist
			old = column[y]
			ic = column[y - 1] + 1
			dc = column[y] + 1
			rc = last + (seq1[x - 1] != seq2[y - 1])
			column[y] = min(ic, dc, rc)
			last = old
			
			# length
			lold = length[y]
			lic = length[y - 1] + 1 if ic == column[y] else 0
			ldc = length[y] + 1 if dc == column[y] else 0
			lrc = llast + 1 if rc == column[y] else 0
			length[y] = max(ldc, lic, lrc)
			llast = lold
	
	return column[y] / float(length[y])
