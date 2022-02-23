"""
Module for dealing with 'gvar'-style font variations, also known as run-time
interpolation.

The ideas here are very similar to MutatorMath.  There is even code to read
MutatorMath .designspace files in the varLib.designspace module.

For now, if you run this file on a designspace file, it tries to find
ttf-interpolatable files for the masters and build a variable-font from
them.  Such ttf-interpolatable and designspace files can be generated from
a Glyphs source, eg., using noto-source as an example:

	$ fontmake -o ttf-interpolatable -g NotoSansArabic-MM.glyphs

Then you can make a variable-font this way:

	$ fonttools varLib master_ufo/NotoSansArabic.designspace

API *will* change in near future.
"""
from fontTools.misc.vector import Vector
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.textTools import Tag, tostr
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis, NamedInstance
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.ttLib.tables.ttProgram import Program
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.merger import VariationMerger
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta_optimize
from fontTools.varLib.featureVars import addFeatureVariations
from fontTools.designspaceLib import DesignSpaceDocument
from functools import partial
from collections import OrderedDict, namedtuple
import os.path
import logging
from copy import deepcopy
from pprint import pformat
from .errors import VarLibError, VarLibValidationError

log = logging.getLogger("fontTools.varLib")

# This is a lib key for the designspace document. The value should be
# an OpenType feature tag, to be used as the FeatureVariations feature.
# If present, the DesignSpace <rules processing="..."> flag is ignored.
FEAVAR_FEATURETAG_LIB_KEY = "com.github.fonttools.varLib.featureVarsFeatureTag"

#
# Creation routines
#

def _add_fvar(font, axes, instances):
	"""
	Add 'fvar' table to font.

	axes is an ordered dictionary of DesignspaceAxis objects.

	instances is list of dictionary objects with 'location', 'stylename',
	and possibly 'postscriptfontname' entries.
	"""

	assert axes
	assert isinstance(axes, OrderedDict)

	log.info("Generating fvar")

	fvar = newTable('fvar')
	nameTable = font['name']

	for a in axes.values():
		axis = Axis()
		axis.axisTag = Tag(a.tag)
		# TODO Skip axes that have no variation.
		axis.minValue, axis.defaultValue, axis.maxValue = a.minimum, a.default, a.maximum
		axis.axisNameID = nameTable.addMultilingualName(a.labelNames, font, minNameID=256)
		axis.flags = int(a.hidden)
		fvar.axes.append(axis)

	for instance in instances:
		coordinates = instance.location

		if "en" not in instance.localisedStyleName:
			if not instance.styleName:
				raise VarLibValidationError(
					f"Instance at location '{coordinates}' must have a default English "
					"style name ('stylename' attribute on the instance element or a "
					"stylename element with an 'xml:lang=\"en\"' attribute)."
				)
			localisedStyleName = dict(instance.localisedStyleName)
			localisedStyleName["en"] = tostr(instance.styleName)
		else:
			localisedStyleName = instance.localisedStyleName

		psname = instance.postScriptFontName

		inst = NamedInstance()
		inst.subfamilyNameID = nameTable.addMultilingualName(localisedStyleName)
		if psname is not None:
			psname = tostr(psname)
			inst.postscriptNameID = nameTable.addName(psname)
		inst.coordinates = {axes[k].tag:axes[k].map_backward(v) for k,v in coordinates.items()}
		#inst.coordinates = {axes[k].tag:v for k,v in coordinates.items()}
		fvar.instances.append(inst)

	assert "fvar" not in font
	font['fvar'] = fvar

	return fvar

def _add_avar(font, axes):
	"""
	Add 'avar' table to font.

	axes is an ordered dictionary of AxisDescriptor objects.
	"""

	assert axes
	assert isinstance(axes, OrderedDict)

	log.info("Generating avar")

	avar = newTable('avar')

	interesting = False
	for axis in axes.values():
		# Currently, some rasterizers require that the default value maps
		# (-1 to -1, 0 to 0, and 1 to 1) be present for all the segment
		# maps, even when the default normalization mapping for the axis
		# was not modified.
		# https://github.com/googlei18n/fontmake/issues/295
		# https://github.com/fonttools/fonttools/issues/1011
		# TODO(anthrotype) revert this (and 19c4b37) when issue is fixed
		curve = avar.segments[axis.tag] = {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0}
		if not axis.map:
			continue

		items = sorted(axis.map)
		keys = [item[0] for item in items]
		vals = [item[1] for item in items]

		# Current avar requirements.  We don't have to enforce
		# these on the designer and can deduce some ourselves,
		# but for now just enforce them.
		if axis.minimum != min(keys):
			raise VarLibValidationError(
				f"Axis '{axis.name}': there must be a mapping for the axis minimum "
				f"value {axis.minimum} and it must be the lowest input mapping value."
			)
		if axis.maximum != max(keys):
			raise VarLibValidationError(
				f"Axis '{axis.name}': there must be a mapping for the axis maximum "
				f"value {axis.maximum} and it must be the highest input mapping value."
			)
		if axis.default not in keys:
			raise VarLibValidationError(
				f"Axis '{axis.name}': there must be a mapping for the axis default "
				f"value {axis.default}."
			)
		# No duplicate input values (output values can be >= their preceeding value).
		if len(set(keys)) != len(keys):
			raise VarLibValidationError(
				f"Axis '{axis.name}': All axis mapping input='...' values must be "
				"unique, but we found duplicates."
			)
		# Ascending values
		if sorted(vals) != vals:
			raise VarLibValidationError(
				f"Axis '{axis.name}': mapping output values must be in ascending order."
			)

		keys_triple = (axis.minimum, axis.default, axis.maximum)
		vals_triple = tuple(axis.map_forward(v) for v in keys_triple)

		keys = [models.normalizeValue(v, keys_triple) for v in keys]
		vals = [models.normalizeValue(v, vals_triple) for v in vals]

		if all(k == v for k, v in zip(keys, vals)):
			continue
		interesting = True

		curve.update(zip(keys, vals))

		assert 0.0 in curve and curve[0.0] == 0.0
		assert -1.0 not in curve or curve[-1.0] == -1.0
		assert +1.0 not in curve or curve[+1.0] == +1.0
		# curve.update({-1.0: -1.0, 0.0: 0.0, 1.0: 1.0})

	assert "avar" not in font
	if not interesting:
		log.info("No need for avar")
		avar = None
	else:
		font['avar'] = avar

	return avar

def _add_stat(font, axes):
	# for now we just get the axis tags and nameIDs from the fvar,
	# so we can reuse the same nameIDs which were defined in there.
	# TODO make use of 'axes' once it adds style attributes info:
	# https://github.com/LettError/designSpaceDocument/issues/8

	if "STAT" in font:
		return

	from ..otlLib.builder import buildStatTable
	fvarTable = font['fvar']
	axes = [dict(tag=a.axisTag, name=a.axisNameID) for a in fvarTable.axes]
	buildStatTable(font, axes)

_MasterData = namedtuple('_MasterData', ['glyf', 'hMetrics', 'vMetrics'])

def _add_gvar(font, masterModel, master_ttfs, tolerance=0.5, optimize=True):
	if tolerance < 0:
		raise ValueError("`tolerance` must be a positive number.")

	log.info("Generating gvar")
	assert "gvar" not in font
	gvar = font["gvar"] = newTable('gvar')
	glyf = font['glyf']
	defaultMasterIndex = masterModel.reverseMapping[0]

	master_datas = [_MasterData(m['glyf'],
				    m['hmtx'].metrics,
				    getattr(m.get('vmtx'), 'metrics', None))
			for m in master_ttfs]

	for glyph in font.getGlyphOrder():

		isComposite = glyf[glyph].isComposite()

		allData = [
			m.glyf._getCoordinatesAndControls(glyph, m.hMetrics, m.vMetrics)
			for m in master_datas
		]

		if allData[defaultMasterIndex][1].numberOfContours != 0:
			# If the default master is not empty, interpret empty non-default masters
			# as missing glyphs from a sparse master
			allData = [
				d if d is not None and d[1].numberOfContours != 0 else None
				for d in allData
			]

		model, allData = masterModel.getSubModel(allData)

		allCoords = [d[0] for d in allData]
		allControls = [d[1] for d in allData]
		control = allControls[0]
		if not models.allEqual(allControls):
			log.warning("glyph %s has incompatible masters; skipping" % glyph)
			continue
		del allControls

		# Update gvar
		gvar.variations[glyph] = []
		deltas = model.getDeltas(allCoords, round=partial(GlyphCoordinates.__round__, round=round))
		supports = model.supports
		assert len(deltas) == len(supports)

		# Prepare for IUP optimization
		origCoords = deltas[0]
		endPts = control.endPts

		for i,(delta,support) in enumerate(zip(deltas[1:], supports[1:])):
			if all(v == 0 for v in delta.array) and not isComposite:
				continue
			var = TupleVariation(support, delta)
			if optimize:
				delta_opt = iup_delta_optimize(delta, origCoords, endPts, tolerance=tolerance)

				if None in delta_opt:
					"""In composite glyphs, there should be one 0 entry
					to make sure the gvar entry is written to the font.

					This is to work around an issue with macOS 10.14 and can be
					removed once the behaviour of macOS is changed.

					https://github.com/fonttools/fonttools/issues/1381
					"""
					if all(d is None for d in delta_opt):
						delta_opt = [(0, 0)] + [None] * (len(delta_opt) - 1)
					# Use "optimized" version only if smaller...
					var_opt = TupleVariation(support, delta_opt)

					axis_tags = sorted(support.keys()) # Shouldn't matter that this is different from fvar...?
					tupleData, auxData = var.compile(axis_tags)
					unoptimized_len = len(tupleData) + len(auxData)
					tupleData, auxData = var_opt.compile(axis_tags)
					optimized_len = len(tupleData) + len(auxData)

					if optimized_len < unoptimized_len:
						var = var_opt

			gvar.variations[glyph].append(var)


def _remove_TTHinting(font):
	for tag in ("cvar", "cvt ", "fpgm", "prep"):
		if tag in font:
			del font[tag]
	maxp = font['maxp']
	for attr in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements", "maxSizeOfInstructions"):
		setattr(maxp, attr, 0)
	maxp.maxZones = 1
	font["glyf"].removeHinting()
	# TODO: Modify gasp table to deactivate gridfitting for all ranges?

def _merge_TTHinting(font, masterModel, master_ttfs):

	log.info("Merging TT hinting")
	assert "cvar" not in font

	# Check that the existing hinting is compatible

	# fpgm and prep table

	for tag in ("fpgm", "prep"):
		all_pgms = [m[tag].program for m in master_ttfs if tag in m]
		if not all_pgms:
			continue
		font_pgm = getattr(font.get(tag), 'program', None)
		if any(pgm != font_pgm for pgm in all_pgms):
			log.warning("Masters have incompatible %s tables, hinting is discarded." % tag)
			_remove_TTHinting(font)
			return

	# glyf table

	font_glyf = font['glyf']
	master_glyfs = [m['glyf'] for m in master_ttfs]
	for name, glyph in font_glyf.glyphs.items():
		all_pgms = [
			getattr(glyf.get(name), 'program', None)
			for glyf in master_glyfs
		]
		if not any(all_pgms):
			continue
		glyph.expand(font_glyf)
		font_pgm = getattr(glyph, 'program', None)
		if any(pgm != font_pgm for pgm in all_pgms if pgm):
			log.warning("Masters have incompatible glyph programs in glyph '%s', hinting is discarded." % name)
			# TODO Only drop hinting from this glyph.
			_remove_TTHinting(font)
			return

	# cvt table

	all_cvs = [Vector(m["cvt "].values) if 'cvt ' in m else None
		   for m in master_ttfs]

	nonNone_cvs = models.nonNone(all_cvs)
	if not nonNone_cvs:
		# There is no cvt table to make a cvar table from, we're done here.
		return

	if not models.allEqual(len(c) for c in nonNone_cvs):
		log.warning("Masters have incompatible cvt tables, hinting is discarded.")
		_remove_TTHinting(font)
		return

	variations = []
	deltas, supports = masterModel.getDeltasAndSupports(all_cvs, round=round) # builtin round calls into Vector.__round__, which uses builtin round as we like
	for i,(delta,support) in enumerate(zip(deltas[1:], supports[1:])):
		if all(v == 0 for v in delta):
			continue
		var = TupleVariation(support, delta)
		variations.append(var)

	# We can build the cvar table now.
	if variations:
		cvar = font["cvar"] = newTable('cvar')
		cvar.version = 1
		cvar.variations = variations


_MetricsFields = namedtuple('_MetricsFields',
	['tableTag', 'metricsTag', 'sb1', 'sb2', 'advMapping', 'vOrigMapping'])

HVAR_FIELDS = _MetricsFields(tableTag='HVAR', metricsTag='hmtx', sb1='LsbMap',
	sb2='RsbMap', advMapping='AdvWidthMap', vOrigMapping=None)

VVAR_FIELDS = _MetricsFields(tableTag='VVAR', metricsTag='vmtx', sb1='TsbMap',
	sb2='BsbMap', advMapping='AdvHeightMap', vOrigMapping='VOrgMap')

def _add_HVAR(font, masterModel, master_ttfs, axisTags):
	_add_VHVAR(font, masterModel, master_ttfs, axisTags, HVAR_FIELDS)

def _add_VVAR(font, masterModel, master_ttfs, axisTags):
	_add_VHVAR(font, masterModel, master_ttfs, axisTags, VVAR_FIELDS)

def _add_VHVAR(font, masterModel, master_ttfs, axisTags, tableFields):

	tableTag = tableFields.tableTag
	assert tableTag not in font
	log.info("Generating " + tableTag)
	VHVAR = newTable(tableTag)
	tableClass = getattr(ot, tableTag)
	vhvar = VHVAR.table = tableClass()
	vhvar.Version = 0x00010000

	glyphOrder = font.getGlyphOrder()

	# Build list of source font advance widths for each glyph
	metricsTag = tableFields.metricsTag
	advMetricses = [m[metricsTag].metrics for m in master_ttfs]

	# Build list of source font vertical origin coords for each glyph
	if tableTag == 'VVAR' and 'VORG' in master_ttfs[0]:
		vOrigMetricses = [m['VORG'].VOriginRecords for m in master_ttfs]
		defaultYOrigs = [m['VORG'].defaultVertOriginY for m in master_ttfs]
		vOrigMetricses = list(zip(vOrigMetricses, defaultYOrigs))
	else:
		vOrigMetricses = None

	metricsStore, advanceMapping, vOrigMapping = _get_advance_metrics(font,
		masterModel, master_ttfs, axisTags, glyphOrder, advMetricses,
		vOrigMetricses)

	vhvar.VarStore = metricsStore
	if advanceMapping is None:
		setattr(vhvar, tableFields.advMapping, None)
	else:
		setattr(vhvar, tableFields.advMapping, advanceMapping)
	if vOrigMapping is not None:
		setattr(vhvar, tableFields.vOrigMapping, vOrigMapping)
	setattr(vhvar, tableFields.sb1, None)
	setattr(vhvar, tableFields.sb2, None)

	font[tableTag] = VHVAR
	return

def _get_advance_metrics(font, masterModel, master_ttfs,
		axisTags, glyphOrder, advMetricses, vOrigMetricses=None):

	vhAdvanceDeltasAndSupports = {}
	vOrigDeltasAndSupports = {}
	for glyph in glyphOrder:
		vhAdvances = [metrics[glyph][0] if glyph in metrics else None for metrics in advMetricses]
		vhAdvanceDeltasAndSupports[glyph] = masterModel.getDeltasAndSupports(vhAdvances, round=round)

	singleModel = models.allEqual(id(v[1]) for v in vhAdvanceDeltasAndSupports.values())

	if vOrigMetricses:
		singleModel = False
		for glyph in glyphOrder:
			# We need to supply a vOrigs tuple with non-None default values
			# for each glyph. vOrigMetricses contains values only for those
			# glyphs which have a non-default vOrig.
			vOrigs = [metrics[glyph] if glyph in metrics else defaultVOrig
				for metrics, defaultVOrig in vOrigMetricses]
			vOrigDeltasAndSupports[glyph] = masterModel.getDeltasAndSupports(vOrigs, round=round)

	directStore = None
	if singleModel:
		# Build direct mapping
		supports = next(iter(vhAdvanceDeltasAndSupports.values()))[1][1:]
		varTupleList = builder.buildVarRegionList(supports, axisTags)
		varTupleIndexes = list(range(len(supports)))
		varData = builder.buildVarData(varTupleIndexes, [], optimize=False)
		for glyphName in glyphOrder:
			varData.addItem(vhAdvanceDeltasAndSupports[glyphName][0], round=noRound)
		varData.optimize()
		directStore = builder.buildVarStore(varTupleList, [varData])

	# Build optimized indirect mapping
	storeBuilder = varStore.OnlineVarStoreBuilder(axisTags)
	advMapping = {}
	for glyphName in glyphOrder:
		deltas, supports = vhAdvanceDeltasAndSupports[glyphName]
		storeBuilder.setSupports(supports)
		advMapping[glyphName] = storeBuilder.storeDeltas(deltas, round=noRound)

	if vOrigMetricses:
		vOrigMap = {}
		for glyphName in glyphOrder:
			deltas, supports = vOrigDeltasAndSupports[glyphName]
			storeBuilder.setSupports(supports)
			vOrigMap[glyphName] = storeBuilder.storeDeltas(deltas, round=noRound)

	indirectStore = storeBuilder.finish()
	mapping2 = indirectStore.optimize()
	advMapping = [mapping2[advMapping[g]] for g in glyphOrder]
	advanceMapping = builder.buildVarIdxMap(advMapping, glyphOrder)

	if vOrigMetricses:
		vOrigMap = [mapping2[vOrigMap[g]] for g in glyphOrder]

	useDirect = False
	vOrigMapping = None
	if directStore:
		# Compile both, see which is more compact

		writer = OTTableWriter()
		directStore.compile(writer, font)
		directSize = len(writer.getAllData())

		writer = OTTableWriter()
		indirectStore.compile(writer, font)
		advanceMapping.compile(writer, font)
		indirectSize = len(writer.getAllData())

		useDirect = directSize < indirectSize

	if useDirect:
		metricsStore = directStore
		advanceMapping = None
	else:
		metricsStore = indirectStore
		if vOrigMetricses:
			vOrigMapping = builder.buildVarIdxMap(vOrigMap, glyphOrder)

	return metricsStore, advanceMapping, vOrigMapping

def _add_MVAR(font, masterModel, master_ttfs, axisTags):

	log.info("Generating MVAR")

	store_builder = varStore.OnlineVarStoreBuilder(axisTags)

	records = []
	lastTableTag = None
	fontTable = None
	tables = None
	# HACK: we need to special-case post.underlineThickness and .underlinePosition
	# and unilaterally/arbitrarily define a sentinel value to distinguish the case
	# when a post table is present in a given master simply because that's where
	# the glyph names in TrueType must be stored, but the underline values are not
	# meant to be used for building MVAR's deltas. The value of -0x8000 (-36768)
	# the minimum FWord (int16) value, was chosen for its unlikelyhood to appear
	# in real-world underline position/thickness values.
	specialTags = {"unds": -0x8000, "undo": -0x8000}

	for tag, (tableTag, itemName) in sorted(MVAR_ENTRIES.items(), key=lambda kv: kv[1]):
		# For each tag, fetch the associated table from all fonts (or not when we are
		# still looking at a tag from the same tables) and set up the variation model
		# for them.
		if tableTag != lastTableTag:
			tables = fontTable = None
			if tableTag in font:
				fontTable = font[tableTag]
				tables = []
				for master in master_ttfs:
					if tableTag not in master or (
						tag in specialTags
						and getattr(master[tableTag], itemName) == specialTags[tag]
					):
						tables.append(None)
					else:
						tables.append(master[tableTag])
				model, tables = masterModel.getSubModel(tables)
				store_builder.setModel(model)
			lastTableTag = tableTag

		if tables is None:  # Tag not applicable to the master font.
			continue

		# TODO support gasp entries

		master_values = [getattr(table, itemName) for table in tables]
		if models.allEqual(master_values):
			base, varIdx = master_values[0], None
		else:
			base, varIdx = store_builder.storeMasters(master_values)
		setattr(fontTable, itemName, base)

		if varIdx is None:
			continue
		log.info('	%s: %s.%s	%s', tag, tableTag, itemName, master_values)
		rec = ot.MetricsValueRecord()
		rec.ValueTag = tag
		rec.VarIdx = varIdx
		records.append(rec)

	assert "MVAR" not in font
	if records:
		store = store_builder.finish()
		# Optimize
		mapping = store.optimize()
		for rec in records:
			rec.VarIdx = mapping[rec.VarIdx]

		MVAR = font["MVAR"] = newTable('MVAR')
		mvar = MVAR.table = ot.MVAR()
		mvar.Version = 0x00010000
		mvar.Reserved = 0
		mvar.VarStore = store
		# XXX these should not be hard-coded but computed automatically
		mvar.ValueRecordSize = 8
		mvar.ValueRecordCount = len(records)
		mvar.ValueRecord = sorted(records, key=lambda r: r.ValueTag)


def _add_BASE(font, masterModel, master_ttfs, axisTags):

	log.info("Generating BASE")

	merger = VariationMerger(masterModel, axisTags, font)
	merger.mergeTables(font, master_ttfs, ['BASE'])
	store = merger.store_builder.finish()

	if not store.VarData:
		return
	base = font['BASE'].table
	assert base.Version == 0x00010000
	base.Version = 0x00010001
	base.VarStore = store


def _merge_OTL(font, model, master_fonts, axisTags):

	log.info("Merging OpenType Layout tables")
	merger = VariationMerger(model, axisTags, font)

	merger.mergeTables(font, master_fonts, ['GSUB', 'GDEF', 'GPOS'])
	store = merger.store_builder.finish()
	if not store.VarData:
		return
	try:
		GDEF = font['GDEF'].table
		assert GDEF.Version <= 0x00010002
	except KeyError:
		font['GDEF'] = newTable('GDEF')
		GDEFTable = font["GDEF"] = newTable('GDEF')
		GDEF = GDEFTable.table = ot.GDEF()
		GDEF.GlyphClassDef = None
		GDEF.AttachList = None
		GDEF.LigCaretList = None
		GDEF.MarkAttachClassDef = None
		GDEF.MarkGlyphSetsDef = None

	GDEF.Version = 0x00010003
	GDEF.VarStore = store

	# Optimize
	varidx_map = store.optimize()
	GDEF.remap_device_varidxes(varidx_map)
	if 'GPOS' in font:
		font['GPOS'].table.remap_device_varidxes(varidx_map)


def _add_GSUB_feature_variations(font, axes, internal_axis_supports, rules, featureTag):

	def normalize(name, value):
		return models.normalizeLocation(
			{name: value}, internal_axis_supports
		)[name]

	log.info("Generating GSUB FeatureVariations")

	axis_tags = {name: axis.tag for name, axis in axes.items()}

	conditional_subs = []
	for rule in rules:

		region = []
		for conditions in rule.conditionSets:
			space = {}
			for condition in conditions:
				axis_name = condition["name"]
				if condition["minimum"] is not None:
					minimum = normalize(axis_name, condition["minimum"])
				else:
					minimum = -1.0
				if condition["maximum"] is not None:
					maximum = normalize(axis_name, condition["maximum"])
				else:
					maximum = 1.0
				tag = axis_tags[axis_name]
				space[tag] = (minimum, maximum)
			region.append(space)

		subs = {k: v for k, v in rule.subs}

		conditional_subs.append((region, subs))

	addFeatureVariations(font, conditional_subs, featureTag)


_DesignSpaceData = namedtuple(
	"_DesignSpaceData",
	[
		"axes",
		"internal_axis_supports",
		"base_idx",
		"normalized_master_locs",
		"masters",
		"instances",
		"rules",
		"rulesProcessingLast",
		"lib",
	],
)


def _add_CFF2(varFont, model, master_fonts):
	from .cff import merge_region_fonts
	glyphOrder = varFont.getGlyphOrder()
	if "CFF2" not in varFont:
		from .cff import convertCFFtoCFF2
		convertCFFtoCFF2(varFont)
	ordered_fonts_list = model.reorderMasters(master_fonts, model.reverseMapping)
	# re-ordering the master list simplifies building the CFF2 data item lists.
	merge_region_fonts(varFont, model, ordered_fonts_list, glyphOrder)


def load_designspace(designspace):
	# TODO: remove this and always assume 'designspace' is a DesignSpaceDocument,
	# never a file path, as that's already handled by caller
	if hasattr(designspace, "sources"):  # Assume a DesignspaceDocument
		ds = designspace
	else:  # Assume a file path
		ds = DesignSpaceDocument.fromfile(designspace)

	masters = ds.sources
	if not masters:
		raise VarLibValidationError("Designspace must have at least one source.")
	instances = ds.instances

	# TODO: Use fontTools.designspaceLib.tagForAxisName instead.
	standard_axis_map = OrderedDict([
		('weight',  ('wght', {'en': u'Weight'})),
		('width',   ('wdth', {'en': u'Width'})),
		('slant',   ('slnt', {'en': u'Slant'})),
		('optical', ('opsz', {'en': u'Optical Size'})),
		('italic',  ('ital', {'en': u'Italic'})),
		])

	# Setup axes
	if not ds.axes:
		raise VarLibValidationError(f"Designspace must have at least one axis.")

	axes = OrderedDict()
	for axis_index, axis in enumerate(ds.axes):
		axis_name = axis.name
		if not axis_name:
			if not axis.tag:
				raise VarLibValidationError(f"Axis at index {axis_index} needs a tag.")
			axis_name = axis.name = axis.tag

		if axis_name in standard_axis_map:
			if axis.tag is None:
				axis.tag = standard_axis_map[axis_name][0]
			if not axis.labelNames:
				axis.labelNames.update(standard_axis_map[axis_name][1])
		else:
			if not axis.tag:
				raise VarLibValidationError(f"Axis at index {axis_index} needs a tag.")
			if not axis.labelNames:
				axis.labelNames["en"] = tostr(axis_name)

		axes[axis_name] = axis
	log.info("Axes:\n%s", pformat([axis.asdict() for axis in axes.values()]))

	# Check all master and instance locations are valid and fill in defaults
	for obj in masters+instances:
		obj_name = obj.name or obj.styleName or ''
		loc = obj.location
		if loc is None:
			raise VarLibValidationError(
				f"Source or instance '{obj_name}' has no location."
			)
		for axis_name in loc.keys():
			if axis_name not in axes:
				raise VarLibValidationError(
					f"Location axis '{axis_name}' unknown for '{obj_name}'."
				)
		for axis_name,axis in axes.items():
			if axis_name not in loc:
				# NOTE: `axis.default` is always user-space, but `obj.location` always design-space.
				loc[axis_name] = axis.map_forward(axis.default)
			else:
				v = axis.map_backward(loc[axis_name])
				if not (axis.minimum <= v <= axis.maximum):
					raise VarLibValidationError(
						f"Source or instance '{obj_name}' has out-of-range location "
						f"for axis '{axis_name}': is mapped to {v} but must be in "
						f"mapped range [{axis.minimum}..{axis.maximum}] (NOTE: all "
						"values are in user-space)."
					)

	# Normalize master locations

	internal_master_locs = [o.location for o in masters]
	log.info("Internal master locations:\n%s", pformat(internal_master_locs))

	# TODO This mapping should ideally be moved closer to logic in _add_fvar/avar
	internal_axis_supports = {}
	for axis in axes.values():
		triple = (axis.minimum, axis.default, axis.maximum)
		internal_axis_supports[axis.name] = [axis.map_forward(v) for v in triple]
	log.info("Internal axis supports:\n%s", pformat(internal_axis_supports))

	normalized_master_locs = [models.normalizeLocation(m, internal_axis_supports) for m in internal_master_locs]
	log.info("Normalized master locations:\n%s", pformat(normalized_master_locs))

	# Find base master
	base_idx = None
	for i,m in enumerate(normalized_master_locs):
		if all(v == 0 for v in m.values()):
			if base_idx is not None:
				raise VarLibValidationError(
					"More than one base master found in Designspace."
				)
			base_idx = i
	if base_idx is None:
		raise VarLibValidationError(
			"Base master not found; no master at default location?"
		)
	log.info("Index of base master: %s", base_idx)

	return _DesignSpaceData(
		axes,
		internal_axis_supports,
		base_idx,
		normalized_master_locs,
		masters,
		instances,
		ds.rules,
		ds.rulesProcessingLast,
		ds.lib,
	)


# https://docs.microsoft.com/en-us/typography/opentype/spec/os2#uswidthclass
WDTH_VALUE_TO_OS2_WIDTH_CLASS = {
	50: 1,
	62.5: 2,
	75: 3,
	87.5: 4,
	100: 5,
	112.5: 6,
	125: 7,
	150: 8,
	200: 9,
}


def set_default_weight_width_slant(font, location):
	if "OS/2" in font:
		if "wght" in location:
			weight_class = otRound(max(1, min(location["wght"], 1000)))
			if font["OS/2"].usWeightClass != weight_class:
				log.info("Setting OS/2.usWeightClass = %s", weight_class)
				font["OS/2"].usWeightClass = weight_class

		if "wdth" in location:
			# map 'wdth' axis (50..200) to OS/2.usWidthClass (1..9), rounding to closest
			widthValue = min(max(location["wdth"], 50), 200)
			widthClass = otRound(
				models.piecewiseLinearMap(widthValue, WDTH_VALUE_TO_OS2_WIDTH_CLASS)
			)
			if font["OS/2"].usWidthClass != widthClass:
				log.info("Setting OS/2.usWidthClass = %s", widthClass)
				font["OS/2"].usWidthClass = widthClass

	if "slnt" in location and "post" in font:
		italicAngle = max(-90, min(location["slnt"], 90))
		if font["post"].italicAngle != italicAngle:
			log.info("Setting post.italicAngle = %s", italicAngle)
			font["post"].italicAngle = italicAngle


def build(designspace, master_finder=lambda s:s, exclude=[], optimize=True):
	"""
	Build variation font from a designspace file.

	If master_finder is set, it should be a callable that takes master
	filename as found in designspace file and map it to master font
	binary as to be opened (eg. .ttf or .otf).
	"""
	if hasattr(designspace, "sources"):  # Assume a DesignspaceDocument
		pass
	else:  # Assume a file path
		designspace = DesignSpaceDocument.fromfile(designspace)

	ds = load_designspace(designspace)
	log.info("Building variable font")

	log.info("Loading master fonts")
	master_fonts = load_masters(designspace, master_finder)

	# TODO: 'master_ttfs' is unused except for return value, remove later
	master_ttfs = []
	for master in master_fonts:
		try:
			master_ttfs.append(master.reader.file.name)
		except AttributeError:
			master_ttfs.append(None)  # in-memory fonts have no path

	# Copy the base master to work from it
	vf = deepcopy(master_fonts[ds.base_idx])

	# TODO append masters as named-instances as well; needs .designspace change.
	fvar = _add_fvar(vf, ds.axes, ds.instances)
	if 'STAT' not in exclude:
		_add_stat(vf, ds.axes)
	if 'avar' not in exclude:
		_add_avar(vf, ds.axes)

	# Map from axis names to axis tags...
	normalized_master_locs = [
		{ds.axes[k].tag: v for k,v in loc.items()} for loc in ds.normalized_master_locs
	]
	# From here on, we use fvar axes only
	axisTags = [axis.axisTag for axis in fvar.axes]

	# Assume single-model for now.
	model = models.VariationModel(normalized_master_locs, axisOrder=axisTags)
	assert 0 == model.mapping[ds.base_idx]

	log.info("Building variations tables")
	if 'BASE' not in exclude and 'BASE' in vf:
		_add_BASE(vf, model, master_fonts, axisTags)
	if 'MVAR' not in exclude:
		_add_MVAR(vf, model, master_fonts, axisTags)
	if 'HVAR' not in exclude:
		_add_HVAR(vf, model, master_fonts, axisTags)
	if 'VVAR' not in exclude and 'vmtx' in vf:
		_add_VVAR(vf, model, master_fonts, axisTags)
	if 'GDEF' not in exclude or 'GPOS' not in exclude:
		_merge_OTL(vf, model, master_fonts, axisTags)
	if 'gvar' not in exclude and 'glyf' in vf:
		_add_gvar(vf, model, master_fonts, optimize=optimize)
	if 'cvar' not in exclude and 'glyf' in vf:
		_merge_TTHinting(vf, model, master_fonts)
	if 'GSUB' not in exclude and ds.rules:
		featureTag = ds.lib.get(
			FEAVAR_FEATURETAG_LIB_KEY,
			"rclt" if ds.rulesProcessingLast else "rvrn"
		)
		_add_GSUB_feature_variations(vf, ds.axes, ds.internal_axis_supports, ds.rules, featureTag)
	if 'CFF2' not in exclude and ('CFF ' in vf or 'CFF2' in vf):
		_add_CFF2(vf, model, master_fonts)
		if "post" in vf:
			# set 'post' to format 2 to keep the glyph names dropped from CFF2
			post = vf["post"]
			if post.formatType != 2.0:
				post.formatType = 2.0
				post.extraNames = []
				post.mapping = {}

	set_default_weight_width_slant(
		vf, location={axis.axisTag: axis.defaultValue for axis in vf["fvar"].axes}
	)

	for tag in exclude:
		if tag in vf:
			del vf[tag]

	# TODO: Only return vf for 4.0+, the rest is unused.
	return vf, model, master_ttfs


def _open_font(path, master_finder=lambda s: s):
	# load TTFont masters from given 'path': this can be either a .TTX or an
	# OpenType binary font; or if neither of these, try use the 'master_finder'
	# callable to resolve the path to a valid .TTX or OpenType font binary.
	from fontTools.ttx import guessFileType

	master_path = os.path.normpath(path)
	tp = guessFileType(master_path)
	if tp is None:
		# not an OpenType binary/ttx, fall back to the master finder.
		master_path = master_finder(master_path)
		tp = guessFileType(master_path)
	if tp in ("TTX", "OTX"):
		font = TTFont()
		font.importXML(master_path)
	elif tp in ("TTF", "OTF", "WOFF", "WOFF2"):
		font = TTFont(master_path)
	else:
		raise VarLibValidationError("Invalid master path: %r" % master_path)
	return font


def load_masters(designspace, master_finder=lambda s: s):
	"""Ensure that all SourceDescriptor.font attributes have an appropriate TTFont
	object loaded, or else open TTFont objects from the SourceDescriptor.path
	attributes.

	The paths can point to either an OpenType font, a TTX file, or a UFO. In the
	latter case, use the provided master_finder callable to map from UFO paths to
	the respective master font binaries (e.g. .ttf, .otf or .ttx).

	Return list of master TTFont objects in the same order they are listed in the
	DesignSpaceDocument.
	"""
	for master in designspace.sources:
		# If a SourceDescriptor has a layer name, demand that the compiled TTFont
		# be supplied by the caller. This spares us from modifying MasterFinder.
		if master.layerName and master.font is None:
			raise VarLibValidationError(
				f"Designspace source '{master.name or '<Unknown>'}' specified a "
				"layer name but lacks the required TTFont object in the 'font' "
				"attribute."
			)

	return designspace.loadSourceFonts(_open_font, master_finder=master_finder)


class MasterFinder(object):

	def __init__(self, template):
		self.template = template

	def __call__(self, src_path):
		fullname = os.path.abspath(src_path)
		dirname, basename = os.path.split(fullname)
		stem, ext = os.path.splitext(basename)
		path = self.template.format(
			fullname=fullname,
			dirname=dirname,
			basename=basename,
			stem=stem,
			ext=ext,
		)
		return os.path.normpath(path)


def main(args=None):
	"""Build a variable font from a designspace file and masters"""
	from argparse import ArgumentParser
	from fontTools import configLogger

	parser = ArgumentParser(prog='varLib', description = main.__doc__)
	parser.add_argument('designspace')
	parser.add_argument(
		'-o',
		metavar='OUTPUTFILE',
		dest='outfile',
		default=None,
		help='output file'
	)
	parser.add_argument(
		'-x',
		metavar='TAG',
		dest='exclude',
		action='append',
		default=[],
		help='exclude table'
	)
	parser.add_argument(
		'--disable-iup',
		dest='optimize',
		action='store_false',
		help='do not perform IUP optimization'
	)
	parser.add_argument(
		'--master-finder',
		default='master_ttf_interpolatable/{stem}.ttf',
		help=(
			'templated string used for finding binary font '
			'files given the source file names defined in the '
			'designspace document. The following special strings '
			'are defined: {fullname} is the absolute source file '
			'name; {basename} is the file name without its '
			'directory; {stem} is the basename without the file '
			'extension; {ext} is the source file extension; '
			'{dirname} is the directory of the absolute file '
			'name. The default value is "%(default)s".'
		)
	)
	logging_group = parser.add_mutually_exclusive_group(required=False)
	logging_group.add_argument(
		"-v", "--verbose",
                action="store_true",
                help="Run more verbosely.")
	logging_group.add_argument(
		"-q", "--quiet",
                action="store_true",
                help="Turn verbosity off.")
	options = parser.parse_args(args)

	configLogger(level=(
		"DEBUG" if options.verbose else
		"ERROR" if options.quiet else
		"INFO"))

	designspace_filename = options.designspace
	finder = MasterFinder(options.master_finder)

	vf, _, _ = build(
		designspace_filename,
		finder,
		exclude=options.exclude,
		optimize=options.optimize
	)

	outfile = options.outfile
	if outfile is None:
		ext = "otf" if vf.sfntVersion == "OTTO" else "ttf"
		outfile = os.path.splitext(designspace_filename)[0] + '-VF.' + ext

	log.info("Saving variation font %s", outfile)
	vf.save(outfile)


if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1:
		sys.exit(main())
	import doctest
	sys.exit(doctest.testmod().failed)
