"""fontTools.t1Lib.py -- Tools for PostScript Type 1 fonts (Python2 only)

Functions for reading and writing raw Type 1 data:

read(path)
	reads any Type 1 font file, returns the raw data and a type indicator:
	'LWFN', 'PFB' or 'OTHER', depending on the format of the file pointed
	to by 'path'.
	Raises an error when the file does not contain valid Type 1 data.

write(path, data, kind='OTHER', dohex=False)
	writes raw Type 1 data to the file pointed to by 'path'.
	'kind' can be one of 'LWFN', 'PFB' or 'OTHER'; it defaults to 'OTHER'.
	'dohex' is a flag which determines whether the eexec encrypted
	part should be written as hexadecimal or binary, but only if kind
	is 'OTHER'.
"""
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin
import os
import re

__author__ = "jvr"
__version__ = "1.0b2"
DEBUG = 0


try:
	try:
		from Carbon import Res
	except ImportError:
		import Res  # MacPython < 2.2
except ImportError:
	haveMacSupport = 0
else:
	haveMacSupport = 1


class T1Error(Exception): pass


class T1Font(object):

	"""Type 1 font class.

	Uses a minimal interpeter that supports just about enough PS to parse
	Type 1 fonts.
	"""

	def __init__(self, path, encoding="ascii", kind=None):
		if kind is None:
			self.data, _ = read(path)
		elif kind == "LWFN":
			self.data = readLWFN(path)
		elif kind == "PFB":
			self.data = readPFB(path)
		elif kind == "OTHER":
			self.data = readOther(path)
		else:
			raise ValueError(kind)
		self.encoding = encoding

	def saveAs(self, path, type, dohex=False):
		write(path, self.getData(), type, dohex)

	def getData(self):
		# XXX Todo: if the data has been converted to Python object,
		# recreate the PS stream
		return self.data

	def getGlyphSet(self):
		"""Return a generic GlyphSet, which is a dict-like object
		mapping glyph names to glyph objects. The returned glyph objects
		have a .draw() method that supports the Pen protocol, and will
		have an attribute named 'width', but only *after* the .draw() method
		has been called.

		In the case of Type 1, the GlyphSet is simply the CharStrings dict.
		"""
		return self["CharStrings"]

	def __getitem__(self, key):
		if not hasattr(self, "font"):
			self.parse()
		return self.font[key]

	def parse(self):
		from fontTools.misc import psLib
		from fontTools.misc import psCharStrings
		self.font = psLib.suckfont(self.data, self.encoding)
		charStrings = self.font["CharStrings"]
		lenIV = self.font["Private"].get("lenIV", 4)
		assert lenIV >= 0
		subrs = self.font["Private"]["Subrs"]
		for glyphName, charString in charStrings.items():
			charString, R = eexec.decrypt(charString, 4330)
			charStrings[glyphName] = psCharStrings.T1CharString(charString[lenIV:],
					subrs=subrs)
		for i in range(len(subrs)):
			charString, R = eexec.decrypt(subrs[i], 4330)
			subrs[i] = psCharStrings.T1CharString(charString[lenIV:], subrs=subrs)
		del self.data


# low level T1 data read and write functions

def read(path, onlyHeader=False):
	"""reads any Type 1 font file, returns raw data"""
	_, ext = os.path.splitext(path)
	ext = ext.lower()
	creator, typ = getMacCreatorAndType(path)
	if typ == 'LWFN':
		return readLWFN(path, onlyHeader), 'LWFN'
	if ext == '.pfb':
		return readPFB(path, onlyHeader), 'PFB'
	else:
		return readOther(path), 'OTHER'

def write(path, data, kind='OTHER', dohex=False):
	assertType1(data)
	kind = kind.upper()
	try:
		os.remove(path)
	except os.error:
		pass
	err = 1
	try:
		if kind == 'LWFN':
			writeLWFN(path, data)
		elif kind == 'PFB':
			writePFB(path, data)
		else:
			writeOther(path, data, dohex)
		err = 0
	finally:
		if err and not DEBUG:
			try:
				os.remove(path)
			except os.error:
				pass


# -- internal --

LWFNCHUNKSIZE = 2000
HEXLINELENGTH = 80


def readLWFN(path, onlyHeader=False):
	"""reads an LWFN font file, returns raw data"""
	from fontTools.misc.macRes import ResourceReader
	reader = ResourceReader(path)
	try:
		data = []
		for res in reader.get('POST', []):
			code = byteord(res.data[0])
			if byteord(res.data[1]) != 0:
				raise T1Error('corrupt LWFN file')
			if code in [1, 2]:
				if onlyHeader and code == 2:
					break
				data.append(res.data[2:])
			elif code in [3, 5]:
				break
			elif code == 4:
				with open(path, "rb") as f:
					data.append(f.read())
			elif code == 0:
				pass # comment, ignore
			else:
				raise T1Error('bad chunk code: ' + repr(code))
	finally:
		reader.close()
	data = bytesjoin(data)
	assertType1(data)
	return data

def readPFB(path, onlyHeader=False):
	"""reads a PFB font file, returns raw data"""
	data = []
	with open(path, "rb") as f:
		while True:
			if f.read(1) != bytechr(128):
				raise T1Error('corrupt PFB file')
			code = byteord(f.read(1))
			if code in [1, 2]:
				chunklen = stringToLong(f.read(4))
				chunk = f.read(chunklen)
				assert len(chunk) == chunklen
				data.append(chunk)
			elif code == 3:
				break
			else:
				raise T1Error('bad chunk code: ' + repr(code))
			if onlyHeader:
				break
	data = bytesjoin(data)
	assertType1(data)
	return data

def readOther(path):
	"""reads any (font) file, returns raw data"""
	with open(path, "rb") as f:
		data = f.read()
	assertType1(data)
	chunks = findEncryptedChunks(data)
	data = []
	for isEncrypted, chunk in chunks:
		if isEncrypted and isHex(chunk[:4]):
			data.append(deHexString(chunk))
		else:
			data.append(chunk)
	return bytesjoin(data)

# file writing tools

def writeLWFN(path, data):
	# Res.FSpCreateResFile was deprecated in OS X 10.5
	Res.FSpCreateResFile(path, "just", "LWFN", 0)
	resRef = Res.FSOpenResFile(path, 2)  # write-only
	try:
		Res.UseResFile(resRef)
		resID = 501
		chunks = findEncryptedChunks(data)
		for isEncrypted, chunk in chunks:
			if isEncrypted:
				code = 2
			else:
				code = 1
			while chunk:
				res = Res.Resource(bytechr(code) + '\0' + chunk[:LWFNCHUNKSIZE - 2])
				res.AddResource('POST', resID, '')
				chunk = chunk[LWFNCHUNKSIZE - 2:]
				resID = resID + 1
		res = Res.Resource(bytechr(5) + '\0')
		res.AddResource('POST', resID, '')
	finally:
		Res.CloseResFile(resRef)

def writePFB(path, data):
	chunks = findEncryptedChunks(data)
	with open(path, "wb") as f:
		for isEncrypted, chunk in chunks:
			if isEncrypted:
				code = 2
			else:
				code = 1
			f.write(bytechr(128) + bytechr(code))
			f.write(longToString(len(chunk)))
			f.write(chunk)
		f.write(bytechr(128) + bytechr(3))

def writeOther(path, data, dohex=False):
	chunks = findEncryptedChunks(data)
	with open(path, "wb") as f:
		hexlinelen = HEXLINELENGTH // 2
		for isEncrypted, chunk in chunks:
			if isEncrypted:
				code = 2
			else:
				code = 1
			if code == 2 and dohex:
				while chunk:
					f.write(eexec.hexString(chunk[:hexlinelen]))
					f.write(b'\r')
					chunk = chunk[hexlinelen:]
			else:
				f.write(chunk)


# decryption tools

EEXECBEGIN = b"currentfile eexec"
# The spec allows for 512 ASCII zeros interrupted by arbitrary whitespace to
# follow eexec
EEXECEND = re.compile(b'(0[ \t\r\n]*){512}', flags=re.M)
EEXECINTERNALEND = b"currentfile closefile"
EEXECBEGINMARKER = b"%-- eexec start\r"
EEXECENDMARKER = b"%-- eexec end\r"

_ishexRE = re.compile(b'[0-9A-Fa-f]*$')

def isHex(text):
	return _ishexRE.match(text) is not None


def decryptType1(data):
	chunks = findEncryptedChunks(data)
	data = []
	for isEncrypted, chunk in chunks:
		if isEncrypted:
			if isHex(chunk[:4]):
				chunk = deHexString(chunk)
			decrypted, R = eexec.decrypt(chunk, 55665)
			decrypted = decrypted[4:]
			if decrypted[-len(EEXECINTERNALEND)-1:-1] != EEXECINTERNALEND \
					and decrypted[-len(EEXECINTERNALEND)-2:-2] != EEXECINTERNALEND:
				raise T1Error("invalid end of eexec part")
			decrypted = decrypted[:-len(EEXECINTERNALEND)-2] + b'\r'
			data.append(EEXECBEGINMARKER + decrypted + EEXECENDMARKER)
		else:
			if chunk[-len(EEXECBEGIN)-1:-1] == EEXECBEGIN:
				data.append(chunk[:-len(EEXECBEGIN)-1])
			else:
				data.append(chunk)
	return bytesjoin(data)

def findEncryptedChunks(data):
	chunks = []
	while True:
		eBegin = data.find(EEXECBEGIN)
		if eBegin < 0:
			break
		eBegin = eBegin + len(EEXECBEGIN) + 1
		endMatch = EEXECEND.search(data, eBegin)
		if endMatch is None:
			raise T1Error("can't find end of eexec part")
		eEnd = endMatch.start()
		cypherText = data[eBegin:eEnd + 2]
		if isHex(cypherText[:4]):
			cypherText = deHexString(cypherText)
		plainText, R = eexec.decrypt(cypherText, 55665)
		eEndLocal = plainText.find(EEXECINTERNALEND)
		if eEndLocal < 0:
			raise T1Error("can't find end of eexec part")
		chunks.append((0, data[:eBegin]))
		chunks.append((1, cypherText[:eEndLocal + len(EEXECINTERNALEND) + 1]))
		data = data[eEnd:]
	chunks.append((0, data))
	return chunks

def deHexString(hexstring):
	return eexec.deHexString(bytesjoin(hexstring.split()))


# Type 1 assertion

_fontType1RE = re.compile(br"/FontType\s+1\s+def")

def assertType1(data):
	for head in [b'%!PS-AdobeFont', b'%!FontType1']:
		if data[:len(head)] == head:
			break
	else:
		raise T1Error("not a PostScript font")
	if not _fontType1RE.search(data):
		raise T1Error("not a Type 1 font")
	if data.find(b"currentfile eexec") < 0:
		raise T1Error("not an encrypted Type 1 font")
	# XXX what else?
	return data


# pfb helpers

def longToString(long):
	s = b""
	for i in range(4):
		s += bytechr((long & (0xff << (i * 8))) >> i * 8)
	return s

def stringToLong(s):
	if len(s) != 4:
		raise ValueError('string must be 4 bytes long')
	l = 0
	for i in range(4):
		l += byteord(s[i]) << (i * 8)
	return l
