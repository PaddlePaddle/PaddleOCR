"""
User name to file name conversion.
This was taken from the UFO 3 spec.
"""

# Restrictions are taken mostly from
# https://docs.microsoft.com/en-gb/windows/win32/fileio/naming-a-file#naming-conventions.
#
# 1. Integer value zero, sometimes referred to as the ASCII NUL character.
# 2. Characters whose integer representations are in the range 1 to 31,
#    inclusive.
# 3. Various characters that (mostly) Windows and POSIX-y filesystems don't
#    allow, plus "(" and ")", as per the specification.
illegalCharacters = {
    "\x00",
    "\x01",
    "\x02",
    "\x03",
    "\x04",
    "\x05",
    "\x06",
    "\x07",
    "\x08",
    "\t",
    "\n",
    "\x0b",
    "\x0c",
    "\r",
    "\x0e",
    "\x0f",
    "\x10",
    "\x11",
    "\x12",
    "\x13",
    "\x14",
    "\x15",
    "\x16",
    "\x17",
    "\x18",
    "\x19",
    "\x1a",
    "\x1b",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
    '"',
    "*",
    "+",
    "/",
    ":",
    "<",
    ">",
    "?",
    "[",
    "\\",
    "]",
    "(",
    ")",
    "|",
    "\x7f",
}
reservedFileNames = {
    "aux",
    "clock$",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}
maxFileNameLength = 255


class NameTranslationError(Exception):
	pass


def userNameToFileName(userName: str, existing=(), prefix="", suffix=""):
	"""
	`existing` should be a set-like object.

	>>> userNameToFileName("a") == "a"
	True
	>>> userNameToFileName("A") == "A_"
	True
	>>> userNameToFileName("AE") == "A_E_"
	True
	>>> userNameToFileName("Ae") == "A_e"
	True
	>>> userNameToFileName("ae") == "ae"
	True
	>>> userNameToFileName("aE") == "aE_"
	True
	>>> userNameToFileName("a.alt") == "a.alt"
	True
	>>> userNameToFileName("A.alt") == "A_.alt"
	True
	>>> userNameToFileName("A.Alt") == "A_.A_lt"
	True
	>>> userNameToFileName("A.aLt") == "A_.aL_t"
	True
	>>> userNameToFileName(u"A.alT") == "A_.alT_"
	True
	>>> userNameToFileName("T_H") == "T__H_"
	True
	>>> userNameToFileName("T_h") == "T__h"
	True
	>>> userNameToFileName("t_h") == "t_h"
	True
	>>> userNameToFileName("F_F_I") == "F__F__I_"
	True
	>>> userNameToFileName("f_f_i") == "f_f_i"
	True
	>>> userNameToFileName("Aacute_V.swash") == "A_acute_V_.swash"
	True
	>>> userNameToFileName(".notdef") == "_notdef"
	True
	>>> userNameToFileName("con") == "_con"
	True
	>>> userNameToFileName("CON") == "C_O_N_"
	True
	>>> userNameToFileName("con.alt") == "_con.alt"
	True
	>>> userNameToFileName("alt.con") == "alt._con"
	True
	"""
	# the incoming name must be a string
	if not isinstance(userName, str):
		raise ValueError("The value for userName must be a string.")
	# establish the prefix and suffix lengths
	prefixLength = len(prefix)
	suffixLength = len(suffix)
	# replace an initial period with an _
	# if no prefix is to be added
	if not prefix and userName[0] == ".":
		userName = "_" + userName[1:]
	# filter the user name
	filteredUserName = []
	for character in userName:
		# replace illegal characters with _
		if character in illegalCharacters:
			character = "_"
		# add _ to all non-lower characters
		elif character != character.lower():
			character += "_"
		filteredUserName.append(character)
	userName = "".join(filteredUserName)
	# clip to 255
	sliceLength = maxFileNameLength - prefixLength - suffixLength
	userName = userName[:sliceLength]
	# test for illegal files names
	parts = []
	for part in userName.split("."):
		if part.lower() in reservedFileNames:
			part = "_" + part
		parts.append(part)
	userName = ".".join(parts)
	# test for clash
	fullName = prefix + userName + suffix
	if fullName.lower() in existing:
		fullName = handleClash1(userName, existing, prefix, suffix)
	# finished
	return fullName

def handleClash1(userName, existing=[], prefix="", suffix=""):
	"""
	existing should be a case-insensitive list
	of all existing file names.

	>>> prefix = ("0" * 5) + "."
	>>> suffix = "." + ("0" * 10)
	>>> existing = ["a" * 5]

	>>> e = list(existing)
	>>> handleClash1(userName="A" * 5, existing=e,
	...		prefix=prefix, suffix=suffix) == (
	... 	'00000.AAAAA000000000000001.0000000000')
	True

	>>> e = list(existing)
	>>> e.append(prefix + "aaaaa" + "1".zfill(15) + suffix)
	>>> handleClash1(userName="A" * 5, existing=e,
	...		prefix=prefix, suffix=suffix) == (
	... 	'00000.AAAAA000000000000002.0000000000')
	True

	>>> e = list(existing)
	>>> e.append(prefix + "AAAAA" + "2".zfill(15) + suffix)
	>>> handleClash1(userName="A" * 5, existing=e,
	...		prefix=prefix, suffix=suffix) == (
	... 	'00000.AAAAA000000000000001.0000000000')
	True
	"""
	# if the prefix length + user name length + suffix length + 15 is at
	# or past the maximum length, silce 15 characters off of the user name
	prefixLength = len(prefix)
	suffixLength = len(suffix)
	if prefixLength + len(userName) + suffixLength + 15 > maxFileNameLength:
		l = (prefixLength + len(userName) + suffixLength + 15)
		sliceLength = maxFileNameLength - l
		userName = userName[:sliceLength]
	finalName = None
	# try to add numbers to create a unique name
	counter = 1
	while finalName is None:
		name = userName + str(counter).zfill(15)
		fullName = prefix + name + suffix
		if fullName.lower() not in existing:
			finalName = fullName
			break
		else:
			counter += 1
		if counter >= 999999999999999:
			break
	# if there is a clash, go to the next fallback
	if finalName is None:
		finalName = handleClash2(existing, prefix, suffix)
	# finished
	return finalName

def handleClash2(existing=[], prefix="", suffix=""):
	"""
	existing should be a case-insensitive list
	of all existing file names.

	>>> prefix = ("0" * 5) + "."
	>>> suffix = "." + ("0" * 10)
	>>> existing = [prefix + str(i) + suffix for i in range(100)]

	>>> e = list(existing)
	>>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
	... 	'00000.100.0000000000')
	True

	>>> e = list(existing)
	>>> e.remove(prefix + "1" + suffix)
	>>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
	... 	'00000.1.0000000000')
	True

	>>> e = list(existing)
	>>> e.remove(prefix + "2" + suffix)
	>>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
	... 	'00000.2.0000000000')
	True
	"""
	# calculate the longest possible string
	maxLength = maxFileNameLength - len(prefix) - len(suffix)
	maxValue = int("9" * maxLength)
	# try to find a number
	finalName = None
	counter = 1
	while finalName is None:
		fullName = prefix + str(counter) + suffix
		if fullName.lower() not in existing:
			finalName = fullName
			break
		else:
			counter += 1
		if counter >= maxValue:
			break
	# raise an error if nothing has been found
	if finalName is None:
		raise NameTranslationError("No unique name could be found.")
	# finished
	return finalName

if __name__ == "__main__":
	import doctest
	doctest.testmod()
