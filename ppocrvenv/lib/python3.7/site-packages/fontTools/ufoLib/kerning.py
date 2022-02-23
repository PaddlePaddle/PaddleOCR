

def lookupKerningValue(pair, kerning, groups, fallback=0, glyphToFirstGroup=None, glyphToSecondGroup=None):
	"""
	Note: This expects kerning to be a flat dictionary
	of kerning pairs, not the nested structure used
	in kerning.plist.

	>>> groups = {
	...     "public.kern1.O" : ["O", "D", "Q"],
	...     "public.kern2.E" : ["E", "F"]
	... }
	>>> kerning = {
	...     ("public.kern1.O", "public.kern2.E") : -100,
	...     ("public.kern1.O", "F") : -200,
	...     ("D", "F") : -300
	... }
	>>> lookupKerningValue(("D", "F"), kerning, groups)
	-300
	>>> lookupKerningValue(("O", "F"), kerning, groups)
	-200
	>>> lookupKerningValue(("O", "E"), kerning, groups)
	-100
	>>> lookupKerningValue(("O", "O"), kerning, groups)
	0
	>>> lookupKerningValue(("E", "E"), kerning, groups)
	0
	>>> lookupKerningValue(("E", "O"), kerning, groups)
	0
	>>> lookupKerningValue(("X", "X"), kerning, groups)
	0
	>>> lookupKerningValue(("public.kern1.O", "public.kern2.E"),
	...     kerning, groups)
	-100
	>>> lookupKerningValue(("public.kern1.O", "F"), kerning, groups)
	-200
	>>> lookupKerningValue(("O", "public.kern2.E"), kerning, groups)
	-100
	>>> lookupKerningValue(("public.kern1.X", "public.kern2.X"), kerning, groups)
	0
	"""
	# quickly check to see if the pair is in the kerning dictionary
	if pair in kerning:
		return kerning[pair]
	# create glyph to group mapping
	if glyphToFirstGroup is not None:
		assert glyphToSecondGroup is not None
	if glyphToSecondGroup is not None:
		assert glyphToFirstGroup is not None
	if glyphToFirstGroup is None:
		glyphToFirstGroup = {}
		glyphToSecondGroup = {}
		for group, groupMembers in groups.items():
			if group.startswith("public.kern1."):
				for glyph in groupMembers:
					glyphToFirstGroup[glyph] = group
			elif group.startswith("public.kern2."):
				for glyph in groupMembers:
					glyphToSecondGroup[glyph] = group
	# get group names and make sure first and second are glyph names
	first, second = pair
	firstGroup = secondGroup = None
	if first.startswith("public.kern1."):
		firstGroup = first
		first = None
	else:
		firstGroup = glyphToFirstGroup.get(first)
	if second.startswith("public.kern2."):
		secondGroup = second
		second = None
	else:
		secondGroup = glyphToSecondGroup.get(second)
	# make an ordered list of pairs to look up
	pairs = [
		(first, second),
		(first, secondGroup),
		(firstGroup, second),
		(firstGroup, secondGroup)
	]
	# look up the pairs and return any matches
	for pair in pairs:
		if pair in kerning:
			return kerning[pair]
	# use the fallback value
	return fallback

if __name__ == "__main__":
	import doctest
	doctest.testmod()
