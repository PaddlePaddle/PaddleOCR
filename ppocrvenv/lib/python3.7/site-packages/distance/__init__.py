"Utilities for comparing sequences"

__all__ = ["hamming", "levenshtein", "nlevenshtein", "jaccard", "sorensen",
	"fast_comp", "lcsubstrings", "ilevenshtein", "ifast_comp"]

try:
	from .cdistance import *
except ImportError:
	from ._pyimports import *

from ._pyimports import jaccard, sorensen

def quick_levenshtein(str1, str2):
	return fast_comp(str1, str2, transpositions=False)

def iquick_levenshtein(str1, strs):
	return ifast_comp(str1, str2, transpositions=False)
