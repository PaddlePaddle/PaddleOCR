"""Pen calculating area, center of mass, variance and standard-deviation,
covariance and correlation, and slant, of glyph shapes."""
import math
from fontTools.pens.momentsPen import MomentsPen

__all__ = ["StatisticsPen"]


class StatisticsPen(MomentsPen):

	"""Pen calculating area, center of mass, variance and
	standard-deviation, covariance and correlation, and slant,
	of glyph shapes.

	Note that all the calculated values are 'signed'. Ie. if the
	glyph shape is self-intersecting, the values are not correct
	(but well-defined). As such, area will be negative if contour
	directions are clockwise.  Moreover, variance might be negative
	if the shapes are self-intersecting in certain ways."""

	def __init__(self, glyphset=None):
		MomentsPen.__init__(self, glyphset=glyphset)
		self.__zero()

	def _closePath(self):
		MomentsPen._closePath(self)
		self.__update()

	def __zero(self):
		self.meanX = 0
		self.meanY = 0
		self.varianceX = 0
		self.varianceY = 0
		self.stddevX = 0
		self.stddevY = 0
		self.covariance = 0
		self.correlation = 0
		self.slant = 0

	def __update(self):

		area = self.area
		if not area:
			self.__zero()
			return

		# Center of mass
		# https://en.wikipedia.org/wiki/Center_of_mass#A_continuous_volume
		self.meanX = meanX = self.momentX / area
		self.meanY = meanY = self.momentY / area

		#  Var(X) = E[X^2] - E[X]^2
		self.varianceX = varianceX = self.momentXX / area - meanX**2
		self.varianceY = varianceY = self.momentYY / area - meanY**2

		self.stddevX = stddevX = math.copysign(abs(varianceX)**.5, varianceX)
		self.stddevY = stddevY = math.copysign(abs(varianceY)**.5, varianceY)

		#  Covariance(X,Y) = ( E[X.Y] - E[X]E[Y] )
		self.covariance = covariance = self.momentXY / area - meanX*meanY

		#  Correlation(X,Y) = Covariance(X,Y) / ( stddev(X) * stddev(Y) )
		# https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
		correlation = covariance / (stddevX * stddevY)
		self.correlation = correlation if abs(correlation) > 1e-3 else 0

		slant = covariance / varianceY
		self.slant = slant if abs(slant) > 1e-3 else 0


def _test(glyphset, upem, glyphs):
	from fontTools.pens.transformPen import TransformPen
	from fontTools.misc.transform import Scale

	print('upem', upem)

	for glyph_name in glyphs:
		print()
		print("glyph:", glyph_name)
		glyph = glyphset[glyph_name]
		pen = StatisticsPen(glyphset=glyphset)
		transformer = TransformPen(pen, Scale(1./upem))
		glyph.draw(transformer)
		for item in ['area', 'momentX', 'momentY', 'momentXX', 'momentYY', 'momentXY', 'meanX', 'meanY', 'varianceX', 'varianceY', 'stddevX', 'stddevY', 'covariance', 'correlation', 'slant']:
			if item[0] == '_': continue
			print ("%s: %g" % (item, getattr(pen, item)))

def main(args):
	if not args:
		return
	filename, glyphs = args[0], args[1:]
	if not glyphs:
		glyphs = ['e', 'o', 'I', 'slash', 'E', 'zero', 'eight', 'minus', 'equal']
	from fontTools.ttLib import TTFont
	font = TTFont(filename)
	_test(font.getGlyphSet(), font['head'].unitsPerEm, glyphs)

if __name__ == '__main__':
	import sys
	main(sys.argv[1:])
