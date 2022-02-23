"""Compiles/decompiles SVG table.

https://docs.microsoft.com/en-us/typography/opentype/spec/svg

The XML format is:

.. code-block:: xml

	<SVG>
		<svgDoc endGlyphID="1" startGlyphID="1">
			<![CDATA[ <complete SVG doc> ]]
		</svgDoc>
	...
		<svgDoc endGlyphID="n" startGlyphID="m">
			<![CDATA[ <complete SVG doc> ]]
		</svgDoc>
	</SVG>
"""

from fontTools.misc.textTools import bytesjoin, strjoin, tobytes, tostr
from fontTools.misc import sstruct
from . import DefaultTable
from io import BytesIO
import struct
import logging


log = logging.getLogger(__name__)


SVG_format_0 = """
	>   # big endian
	version:                  H
	offsetToSVGDocIndex:      L
	reserved:                 L
"""

SVG_format_0Size = sstruct.calcsize(SVG_format_0)

doc_index_entry_format_0 = """
	>   # big endian
	startGlyphID:             H
	endGlyphID:               H
	svgDocOffset:             L
	svgDocLength:             L
"""

doc_index_entry_format_0Size = sstruct.calcsize(doc_index_entry_format_0)



class table_S_V_G_(DefaultTable.DefaultTable):

	def decompile(self, data, ttFont):
		self.docList = []
		# Version 0 is the standardized version of the table; and current.
		# https://www.microsoft.com/typography/otspec/svg.htm
		sstruct.unpack(SVG_format_0, data[:SVG_format_0Size], self)
		if self.version != 0:
			log.warning(
				"Unknown SVG table version '%s'. Decompiling as version 0.", self.version)
		# read in SVG Documents Index
		# data starts with the first entry of the entry list.
		pos = subTableStart = self.offsetToSVGDocIndex
		self.numEntries = struct.unpack(">H", data[pos:pos+2])[0]
		pos += 2
		if self.numEntries > 0:
			data2 = data[pos:]
			entries = []
			for i in range(self.numEntries):
				docIndexEntry, data2 = sstruct.unpack2(doc_index_entry_format_0, data2, DocumentIndexEntry())
				entries.append(docIndexEntry)

			for entry in entries:
				start = entry.svgDocOffset + subTableStart
				end = start + entry.svgDocLength
				doc = data[start:end]
				if doc.startswith(b"\x1f\x8b"):
					import gzip
					bytesIO = BytesIO(doc)
					with gzip.GzipFile(None, "r", fileobj=bytesIO) as gunzipper:
						doc = gunzipper.read()
					self.compressed = True
					del bytesIO
				doc = tostr(doc, "utf_8")
				self.docList.append( [doc, entry.startGlyphID, entry.endGlyphID] )

	def compile(self, ttFont):
		version = 0
		offsetToSVGDocIndex = SVG_format_0Size # I start the SVGDocIndex right after the header.
		# get SGVDoc info.
		docList = []
		entryList = []
		numEntries = len(self.docList)
		datum = struct.pack(">H",numEntries)
		entryList.append(datum)
		curOffset = len(datum) + doc_index_entry_format_0Size*numEntries
		seenDocs = {}
		for doc, startGlyphID, endGlyphID in self.docList:
			docBytes = tobytes(doc, encoding="utf_8")
			if getattr(self, "compressed", False) and not docBytes.startswith(b"\x1f\x8b"):
				import gzip
				bytesIO = BytesIO()
				with gzip.GzipFile(None, "w", fileobj=bytesIO) as gzipper:
					gzipper.write(docBytes)
				gzipped = bytesIO.getvalue()
				if len(gzipped) < len(docBytes):
					docBytes = gzipped
				del gzipped, bytesIO
			docLength = len(docBytes)
			if docBytes in seenDocs:
				docOffset = seenDocs[docBytes]
			else:
				docOffset = curOffset
				curOffset += docLength
				seenDocs[docBytes] = docOffset
				docList.append(docBytes)
			entry = struct.pack(">HHLL", startGlyphID, endGlyphID, docOffset, docLength)
			entryList.append(entry)
		entryList.extend(docList)
		svgDocData = bytesjoin(entryList)

		reserved = 0
		header = struct.pack(">HLL", version, offsetToSVGDocIndex, reserved)
		data = [header, svgDocData]
		data = bytesjoin(data)
		return data

	def toXML(self, writer, ttFont):
		for doc, startGID, endGID in self.docList:
			writer.begintag("svgDoc", startGlyphID=startGID, endGlyphID=endGID)
			writer.newline()
			writer.writecdata(doc)
			writer.newline()
			writer.endtag("svgDoc")
			writer.newline()

	def fromXML(self, name, attrs, content, ttFont):
		if name == "svgDoc":
			if not hasattr(self, "docList"):
				self.docList = []
			doc = strjoin(content)
			doc = doc.strip()
			startGID = int(attrs["startGlyphID"])
			endGID = int(attrs["endGlyphID"])
			self.docList.append( [doc, startGID, endGID] )
		else:
			log.warning("Unknown %s %s", name, content)


class DocumentIndexEntry(object):
	def __init__(self):
		self.startGlyphID = None # USHORT
		self.endGlyphID = None # USHORT
		self.svgDocOffset = None # ULONG
		self.svgDocLength = None # ULONG

	def __repr__(self):
		return "startGlyphID: %s, endGlyphID: %s, svgDocOffset: %s, svgDocLength: %s" % (self.startGlyphID, self.endGlyphID, self.svgDocOffset, self.svgDocLength)
