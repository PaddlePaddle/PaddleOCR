from fontTools.misc.roundTools import noRound, otRound
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (buildVarRegionList, buildVarStore,
				      buildVarRegion, buildVarData)
from functools import partial
from collections import defaultdict


def _getLocationKey(loc):
	return tuple(sorted(loc.items(), key=lambda kv: kv[0]))


class OnlineVarStoreBuilder(object):

	def __init__(self, axisTags):
		self._axisTags = axisTags
		self._regionMap = {}
		self._regionList = buildVarRegionList([], axisTags)
		self._store = buildVarStore(self._regionList, [])
		self._data = None
		self._model = None
		self._supports = None
		self._varDataIndices = {}
		self._varDataCaches = {}
		self._cache = {}

	def setModel(self, model):
		self.setSupports(model.supports)
		self._model = model

	def setSupports(self, supports):
		self._model = None
		self._supports = list(supports)
		if not self._supports[0]:
			del self._supports[0] # Drop base master support
		self._cache = {}
		self._data = None

	def finish(self, optimize=True):
		self._regionList.RegionCount = len(self._regionList.Region)
		self._store.VarDataCount = len(self._store.VarData)
		for data in self._store.VarData:
			data.ItemCount = len(data.Item)
			data.calculateNumShorts(optimize=optimize)
		return self._store

	def _add_VarData(self):
		regionMap = self._regionMap
		regionList = self._regionList

		regions = self._supports
		regionIndices = []
		for region in regions:
			key = _getLocationKey(region)
			idx = regionMap.get(key)
			if idx is None:
				varRegion = buildVarRegion(region, self._axisTags)
				idx = regionMap[key] = len(regionList.Region)
				regionList.Region.append(varRegion)
			regionIndices.append(idx)

		# Check if we have one already...
		key = tuple(regionIndices)
		varDataIdx = self._varDataIndices.get(key)
		if varDataIdx is not None:
			self._outer = varDataIdx
			self._data = self._store.VarData[varDataIdx]
			self._cache = self._varDataCaches[key]
			if len(self._data.Item) == 0xFFFF:
				# This is full.  Need new one.
				varDataIdx = None

		if varDataIdx is None:
			self._data = buildVarData(regionIndices, [], optimize=False)
			self._outer = len(self._store.VarData)
			self._store.VarData.append(self._data)
			self._varDataIndices[key] = self._outer
			if key not in self._varDataCaches:
				self._varDataCaches[key] = {}
			self._cache = self._varDataCaches[key]


	def storeMasters(self, master_values):
		deltas = self._model.getDeltas(master_values, round=round)
		base = deltas.pop(0)
		return base, self.storeDeltas(deltas, round=noRound)

	def storeDeltas(self, deltas, *, round=round):
		deltas = [round(d) for d in deltas]
		if len(deltas) == len(self._supports) + 1:
			deltas = tuple(deltas[1:])
		else:
			assert len(deltas) == len(self._supports)
			deltas = tuple(deltas)

		varIdx = self._cache.get(deltas)
		if varIdx is not None:
			return varIdx

		if not self._data:
			self._add_VarData()
		inner = len(self._data.Item)
		if inner == 0xFFFF:
			# Full array. Start new one.
			self._add_VarData()
			return self.storeDeltas(deltas)
		self._data.addItem(deltas, round=noRound)

		varIdx = (self._outer << 16) + inner
		self._cache[deltas] = varIdx
		return varIdx

def VarData_addItem(self, deltas, *, round=round):
	deltas = [round(d) for d in deltas]

	countUs = self.VarRegionCount
	countThem = len(deltas)
	if countUs + 1 == countThem:
		deltas = tuple(deltas[1:])
	else:
		assert countUs == countThem, (countUs, countThem)
		deltas = tuple(deltas)
	self.Item.append(list(deltas))
	self.ItemCount = len(self.Item)

ot.VarData.addItem = VarData_addItem

def VarRegion_get_support(self, fvar_axes):
	return {
		fvar_axes[i].axisTag: (reg.StartCoord,reg.PeakCoord,reg.EndCoord)
		for i, reg in enumerate(self.VarRegionAxis)
		if reg.PeakCoord != 0
	}

ot.VarRegion.get_support = VarRegion_get_support

class VarStoreInstancer(object):

	def __init__(self, varstore, fvar_axes, location={}):
		self.fvar_axes = fvar_axes
		assert varstore is None or varstore.Format == 1
		self._varData = varstore.VarData if varstore else []
		self._regions = varstore.VarRegionList.Region if varstore else []
		self.setLocation(location)

	def setLocation(self, location):
		self.location = dict(location)
		self._clearCaches()

	def _clearCaches(self):
		self._scalars = {}

	def _getScalar(self, regionIdx):
		scalar = self._scalars.get(regionIdx)
		if scalar is None:
			support = self._regions[regionIdx].get_support(self.fvar_axes)
			scalar = supportScalar(self.location, support)
			self._scalars[regionIdx] = scalar
		return scalar

	@staticmethod
	def interpolateFromDeltasAndScalars(deltas, scalars):
		delta = 0.
		for d,s in zip(deltas, scalars):
			if not s: continue
			delta += d * s
		return delta

	def __getitem__(self, varidx):
		major, minor = varidx >> 16, varidx & 0xFFFF
		varData = self._varData
		scalars = [self._getScalar(ri) for ri in varData[major].VarRegionIndex]
		deltas = varData[major].Item[minor]
		return self.interpolateFromDeltasAndScalars(deltas, scalars)

	def interpolateFromDeltas(self, varDataIndex, deltas):
		varData = self._varData
		scalars = [self._getScalar(ri) for ri in
					varData[varDataIndex].VarRegionIndex]
		return self.interpolateFromDeltasAndScalars(deltas, scalars)


#
# Optimizations
#
# retainFirstMap - If true, major 0 mappings are retained. Deltas for unused indices are zeroed
# advIdxes - Set of major 0 indices for advance deltas to be listed first. Other major 0 indices follow.

def VarStore_subset_varidxes(self, varIdxes, optimize=True, retainFirstMap=False, advIdxes=set()):

	# Sort out used varIdxes by major/minor.
	used = {}
	for varIdx in varIdxes:
		major = varIdx >> 16
		minor = varIdx & 0xFFFF
		d = used.get(major)
		if d is None:
			d = used[major] = set()
		d.add(minor)
	del varIdxes

	#
	# Subset VarData
	#

	varData = self.VarData
	newVarData = []
	varDataMap = {}
	for major,data in enumerate(varData):
		usedMinors = used.get(major)
		if usedMinors is None:
			continue
		newMajor = len(newVarData)
		newVarData.append(data)

		items = data.Item
		newItems = []
		if major == 0 and retainFirstMap:
			for minor in range(len(items)):
				newItems.append(items[minor] if minor in usedMinors else [0] * len(items[minor]))
				varDataMap[minor] = minor
		else:
			if major == 0:
				minors = sorted(advIdxes) + sorted(usedMinors - advIdxes)
			else:
				minors = sorted(usedMinors)
			for minor in minors:
				newMinor = len(newItems)
				newItems.append(items[minor])
				varDataMap[(major<<16)+minor] = (newMajor<<16)+newMinor

		data.Item = newItems
		data.ItemCount = len(data.Item)

		data.calculateNumShorts(optimize=optimize)

	self.VarData = newVarData
	self.VarDataCount = len(self.VarData)

	self.prune_regions()

	return varDataMap

ot.VarStore.subset_varidxes = VarStore_subset_varidxes

def VarStore_prune_regions(self):
	"""Remove unused VarRegions."""
	#
	# Subset VarRegionList
	#

	# Collect.
	usedRegions = set()
	for data in self.VarData:
		usedRegions.update(data.VarRegionIndex)
	# Subset.
	regionList = self.VarRegionList
	regions = regionList.Region
	newRegions = []
	regionMap = {}
	for i in sorted(usedRegions):
		regionMap[i] = len(newRegions)
		newRegions.append(regions[i])
	regionList.Region = newRegions
	regionList.RegionCount = len(regionList.Region)
	# Map.
	for data in self.VarData:
		data.VarRegionIndex = [regionMap[i] for i in data.VarRegionIndex]

ot.VarStore.prune_regions = VarStore_prune_regions


def _visit(self, func):
	"""Recurse down from self, if type of an object is ot.Device,
	call func() on it.  Works on otData-style classes."""

	if type(self) == ot.Device:
		func(self)

	elif isinstance(self, list):
		for that in self:
			_visit(that, func)

	elif hasattr(self, 'getConverters') and not hasattr(self, 'postRead'):
		for conv in self.getConverters():
			that = getattr(self, conv.name, None)
			if that is not None:
				_visit(that, func)

	elif isinstance(self, ot.ValueRecord):
		for that in self.__dict__.values():
			_visit(that, func)

def _Device_recordVarIdx(self, s):
	"""Add VarIdx in this Device table (if any) to the set s."""
	if self.DeltaFormat == 0x8000:
		s.add((self.StartSize<<16)+self.EndSize)

def Object_collect_device_varidxes(self, varidxes):
	adder = partial(_Device_recordVarIdx, s=varidxes)
	_visit(self, adder)

ot.GDEF.collect_device_varidxes = Object_collect_device_varidxes
ot.GPOS.collect_device_varidxes = Object_collect_device_varidxes

def _Device_mapVarIdx(self, mapping, done):
	"""Map VarIdx in this Device table (if any) through mapping."""
	if id(self) in done:
		return
	done.add(id(self))
	if self.DeltaFormat == 0x8000:
		varIdx = mapping[(self.StartSize<<16)+self.EndSize]
		self.StartSize = varIdx >> 16
		self.EndSize = varIdx & 0xFFFF

def Object_remap_device_varidxes(self, varidxes_map):
	mapper = partial(_Device_mapVarIdx, mapping=varidxes_map, done=set())
	_visit(self, mapper)

ot.GDEF.remap_device_varidxes = Object_remap_device_varidxes
ot.GPOS.remap_device_varidxes = Object_remap_device_varidxes


class _Encoding(object):

	def __init__(self, chars):
		self.chars = chars
		self.width = self._popcount(chars)
		self.overhead = self._characteristic_overhead(chars)
		self.items = set()

	def append(self, row):
		self.items.add(row)

	def extend(self, lst):
		self.items.update(lst)

	def get_room(self):
		"""Maximum number of bytes that can be added to characteristic
		while still being beneficial to merge it into another one."""
		count = len(self.items)
		return max(0, (self.overhead - 1) // count - self.width)
	room = property(get_room)

	@property
	def gain(self):
		"""Maximum possible byte gain from merging this into another
		characteristic."""
		count = len(self.items)
		return max(0, self.overhead - count * (self.width + 1))

	def sort_key(self):
		return self.width, self.chars

	def __len__(self):
		return len(self.items)

	def can_encode(self, chars):
		return not (chars & ~self.chars)

	def __sub__(self, other):
		return self._popcount(self.chars & ~other.chars)

	@staticmethod
	def _popcount(n):
		# Apparently this is the fastest native way to do it...
		# https://stackoverflow.com/a/9831671
		return bin(n).count('1')

	@staticmethod
	def _characteristic_overhead(chars):
		"""Returns overhead in bytes of encoding this characteristic
		as a VarData."""
		c = 6
		while chars:
			if chars & 0b1111:
				c += 2
			chars >>= 4
		return c

	def _find_yourself_best_new_encoding(self, done_by_width):
		self.best_new_encoding = None
		for new_width in range(self.width+1, self.width+self.room+1):
			for new_encoding in done_by_width[new_width]:
				if new_encoding.can_encode(self.chars):
					break
			else:
				new_encoding = None
			self.best_new_encoding = new_encoding


class _EncodingDict(dict):

	def __missing__(self, chars):
		r = self[chars] = _Encoding(chars)
		return r

	def add_row(self, row):
		chars = self._row_characteristics(row)
		self[chars].append(row)

	@staticmethod
	def _row_characteristics(row):
		"""Returns encoding characteristics for a row."""
		longWords = False

		chars = 0
		i = 1
		for v in row:
			if v:
				chars += i
			if not (-128 <= v <= 127):
				chars += i * 0b0010
			if not (-32768 <= v <= 32767):
				longWords = True
				break
			i <<= 4

		if longWords:
			# Redo; only allow 2byte/4byte encoding
			chars = 0
			i = 1
			for v in row:
				if v:
					chars += i * 0b0011
				if not (-32768 <= v <= 32767):
					chars += i * 0b1100
				i <<= 4

		return chars


def VarStore_optimize(self):
	"""Optimize storage. Returns mapping from old VarIdxes to new ones."""

	# TODO
	# Check that no two VarRegions are the same; if they are, fold them.

	n = len(self.VarRegionList.Region) # Number of columns
	zeroes = [0] * n

	front_mapping = {} # Map from old VarIdxes to full row tuples

	encodings = _EncodingDict()

	# Collect all items into a set of full rows (with lots of zeroes.)
	for major,data in enumerate(self.VarData):
		regionIndices = data.VarRegionIndex

		for minor,item in enumerate(data.Item):

			row = list(zeroes)
			for regionIdx,v in zip(regionIndices, item):
				row[regionIdx] += v
			row = tuple(row)

			encodings.add_row(row)
			front_mapping[(major<<16)+minor] = row

	# Separate encodings that have no gain (are decided) and those having
	# possible gain (possibly to be merged into others.)
	encodings = sorted(encodings.values(), key=_Encoding.__len__, reverse=True)
	done_by_width = defaultdict(list)
	todo = []
	for encoding in encodings:
		if not encoding.gain:
			done_by_width[encoding.width].append(encoding)
		else:
			todo.append(encoding)

	# For each encoding that is possibly to be merged, find the best match
	# in the decided encodings, and record that.
	todo.sort(key=_Encoding.get_room)
	for encoding in todo:
		encoding._find_yourself_best_new_encoding(done_by_width)

	# Walk through todo encodings, for each, see if merging it with
	# another todo encoding gains more than each of them merging with
	# their best decided encoding. If yes, merge them and add resulting
	# encoding back to todo queue.  If not, move the enconding to decided
	# list.  Repeat till done.
	while todo:
		encoding = todo.pop()
		best_idx = None
		best_gain = 0
		for i,other_encoding in enumerate(todo):
			combined_chars = other_encoding.chars | encoding.chars
			combined_width = _Encoding._popcount(combined_chars)
			combined_overhead = _Encoding._characteristic_overhead(combined_chars)
			combined_gain = (
					+ encoding.overhead
					+ other_encoding.overhead
					- combined_overhead
					- (combined_width - encoding.width) * len(encoding)
					- (combined_width - other_encoding.width) * len(other_encoding)
					)
			this_gain = 0 if encoding.best_new_encoding is None else (
						+ encoding.overhead
						- (encoding.best_new_encoding.width - encoding.width) * len(encoding)
					)
			other_gain = 0 if other_encoding.best_new_encoding is None else (
						+ other_encoding.overhead
						- (other_encoding.best_new_encoding.width - other_encoding.width) * len(other_encoding)
					)
			separate_gain = this_gain + other_gain

			if combined_gain > separate_gain:
				best_idx = i
				best_gain = combined_gain - separate_gain

		if best_idx is None:
			# Encoding is decided as is
			done_by_width[encoding.width].append(encoding)
		else:
			other_encoding = todo[best_idx]
			combined_chars = other_encoding.chars | encoding.chars
			combined_encoding = _Encoding(combined_chars)
			combined_encoding.extend(encoding.items)
			combined_encoding.extend(other_encoding.items)
			combined_encoding._find_yourself_best_new_encoding(done_by_width)
			del todo[best_idx]
			todo.append(combined_encoding)

	# Assemble final store.
	back_mapping = {} # Mapping from full rows to new VarIdxes
	encodings = sum(done_by_width.values(), [])
	encodings.sort(key=_Encoding.sort_key)
	self.VarData = []
	for major,encoding in enumerate(encodings):
		data = ot.VarData()
		self.VarData.append(data)
		data.VarRegionIndex = range(n)
		data.VarRegionCount = len(data.VarRegionIndex)
		data.Item = sorted(encoding.items)
		for minor,item in enumerate(data.Item):
			back_mapping[item] = (major<<16)+minor

	# Compile final mapping.
	varidx_map = {}
	for k,v in front_mapping.items():
		varidx_map[k] = back_mapping[v]

	# Remove unused regions.
	self.prune_regions()

	# Recalculate things and go home.
	self.VarRegionList.RegionCount = len(self.VarRegionList.Region)
	self.VarDataCount = len(self.VarData)
	for data in self.VarData:
		data.ItemCount = len(data.Item)
		data.optimize()

	return varidx_map

ot.VarStore.optimize = VarStore_optimize


def main(args=None):
	"""Optimize a font's GDEF variation store"""
	from argparse import ArgumentParser
	from fontTools import configLogger
	from fontTools.ttLib import TTFont
	from fontTools.ttLib.tables.otBase import OTTableWriter

	parser = ArgumentParser(prog='varLib.varStore', description= main.__doc__)
	parser.add_argument('fontfile')
	parser.add_argument('outfile', nargs='?')
	options = parser.parse_args(args)

	# TODO: allow user to configure logging via command-line options
	configLogger(level="INFO")

	fontfile = options.fontfile
	outfile = options.outfile

	font = TTFont(fontfile)
	gdef = font['GDEF']
	store = gdef.table.VarStore

	writer = OTTableWriter()
	store.compile(writer, font)
	size = len(writer.getAllData())
	print("Before: %7d bytes" % size)

	varidx_map = store.optimize()

	gdef.table.remap_device_varidxes(varidx_map)
	if 'GPOS' in font:
		font['GPOS'].table.remap_device_varidxes(varidx_map)

	writer = OTTableWriter()
	store.compile(writer, font)
	size = len(writer.getAllData())
	print("After:  %7d bytes" % size)

	if outfile is not None:
		font.save(outfile)


if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1:
		sys.exit(main())
	import doctest
	sys.exit(doctest.testmod().failed)
