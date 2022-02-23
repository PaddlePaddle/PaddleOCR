from . import DefaultTable
from . import ttProgram

class table__f_p_g_m(DefaultTable.DefaultTable):

	def decompile(self, data, ttFont):
		program = ttProgram.Program()
		program.fromBytecode(data)
		self.program = program

	def compile(self, ttFont):
		return self.program.getBytecode()

	def toXML(self, writer, ttFont):
		self.program.toXML(writer, ttFont)

	def fromXML(self, name, attrs, content, ttFont):
		program = ttProgram.Program()
		program.fromXML(name, attrs, content, ttFont)
		self.program = program

	def __bool__(self):
		"""
		>>> fpgm = table__f_p_g_m()
		>>> bool(fpgm)
		False
		>>> p = ttProgram.Program()
		>>> fpgm.program = p
		>>> bool(fpgm)
		False
		>>> bc = bytearray([0])
		>>> p.fromBytecode(bc)
		>>> bool(fpgm)
		True
		>>> p.bytecode.pop()
		0
		>>> bool(fpgm)
		False
		"""
		return hasattr(self, 'program') and bool(self.program)

	__nonzero__ = __bool__


if __name__ == "__main__":
	import sys
	import doctest
	sys.exit(doctest.testmod().failed)
