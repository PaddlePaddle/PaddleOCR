from ._pyimports import levenshtein, fast_comp

def ilevenshtein(seq1, seqs, max_dist=-1):
	"""Compute the Levenshtein distance between the sequence `seq1` and the series
	of	sequences `seqs`.
	
		`seq1`: the reference sequence
		`seqs`: a series of sequences (can be a generator)
		`max_dist`: if provided and > 0, only the sequences which distance from
		the reference sequence is lower or equal to this value will be returned.
	
	The return value is a series of pairs (distance, sequence).
	
	The sequence objects in `seqs` are expected to be of the same kind than
	the reference sequence in the C implementation; the same holds true for
	`ifast_comp`.
	"""
	for seq2 in seqs:
		dist = levenshtein(seq1, seq2, max_dist=max_dist)
		if dist != -1:
			yield dist, seq2


def ifast_comp(seq1, seqs, transpositions=False):
	"""Return an iterator over all the sequences in `seqs` which distance from
	`seq1` is lower or equal to 2. The sequences which distance from the
	reference sequence is higher than that are dropped.
	
		`seq1`: the reference sequence.
		`seqs`: a series of sequences (can be a generator)
		`transpositions` has the same sense than in `fast_comp`.
	
	The return value is a series of pairs (distance, sequence).
	
	You might want to call `sorted()` on the iterator to get the results in a
	significant order:
	
		>>> g = ifast_comp("foo", ["fo", "bar", "foob", "foo", "foobaz"])
		>>> sorted(g)
		[(0, 'foo'), (1, 'fo'), (1, 'foob')]
	"""
	for seq2 in seqs:
		dist = fast_comp(seq1, seq2, transpositions)
		if dist != -1:
			yield dist, seq2
