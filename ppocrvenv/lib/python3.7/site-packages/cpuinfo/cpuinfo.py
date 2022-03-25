#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2014-2021 Matthew Brennan Jones <matthew.brennan.jones@gmail.com>
# Py-cpuinfo gets CPU info with pure Python 2 & 3
# It uses the MIT License
# It is hosted at: https://github.com/workhorsy/py-cpuinfo
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

CPUINFO_VERSION = (8, 0, 0)
CPUINFO_VERSION_STRING = '.'.join([str(n) for n in CPUINFO_VERSION])

import os, sys
import platform
import multiprocessing
import ctypes


IS_PY2 = sys.version_info[0] == 2
CAN_CALL_CPUID_IN_SUBPROCESS = True

g_trace = None


class Trace(object):
	def __init__(self, is_active, is_stored_in_string):
		self._is_active = is_active
		if not self._is_active:
			return

		from datetime import datetime

		if IS_PY2:
			from cStringIO import StringIO
		else:
			from io import StringIO

		if is_stored_in_string:
			self._output = StringIO()
		else:
			date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
			self._output = open('cpuinfo_trace_{0}.trace'.format(date), 'w')

		self._stdout = StringIO()
		self._stderr = StringIO()
		self._err = None

	def header(self, msg):
		if not self._is_active: return

		from inspect import stack
		frame = stack()[1]
		file = frame[1]
		line = frame[2]
		self._output.write("{0} ({1} {2})\n".format(msg, file, line))
		self._output.flush()

	def success(self):
		if not self._is_active: return

		from inspect import stack
		frame = stack()[1]
		file = frame[1]
		line = frame[2]

		self._output.write("Success ... ({0} {1})\n\n".format(file, line))
		self._output.flush()

	def fail(self, msg):
		if not self._is_active: return

		from inspect import stack
		frame = stack()[1]
		file = frame[1]
		line = frame[2]

		if isinstance(msg, str):
			msg = ''.join(['\t' + line for line in msg.split('\n')]) + '\n'

			self._output.write(msg)
			self._output.write("Failed ... ({0} {1})\n\n".format(file, line))
			self._output.flush()
		elif isinstance(msg, Exception):
			from traceback import format_exc
			err_string = format_exc()
			self._output.write("\tFailed ... ({0} {1})\n".format(file, line))
			self._output.write(''.join(['\t\t{0}\n'.format(n) for n in err_string.split('\n')]) + '\n')
			self._output.flush()

	def command_header(self, msg):
		if not self._is_active: return

		from inspect import stack
		frame = stack()[3]
		file = frame[1]
		line = frame[2]
		self._output.write("\t{0} ({1} {2})\n".format(msg, file, line))
		self._output.flush()

	def command_output(self, msg, output):
		if not self._is_active: return

		self._output.write("\t\t{0}\n".format(msg))
		self._output.write(''.join(['\t\t\t{0}\n'.format(n) for n in output.split('\n')]) + '\n')
		self._output.flush()

	def keys(self, keys, info, new_info):
		if not self._is_active: return

		from inspect import stack
		frame = stack()[2]
		file = frame[1]
		line = frame[2]

		# List updated keys
		self._output.write("\tChanged keys ({0} {1})\n".format(file, line))
		changed_keys = [key for key in keys if key in info and key in new_info and info[key] != new_info[key]]
		if changed_keys:
			for key in changed_keys:
				self._output.write('\t\t{0}: {1} to {2}\n'.format(key, info[key], new_info[key]))
		else:
			self._output.write('\t\tNone\n')

		# List new keys
		self._output.write("\tNew keys ({0} {1})\n".format(file, line))
		new_keys = [key for key in keys if key in new_info and key not in info]
		if new_keys:
			for key in new_keys:
				self._output.write('\t\t{0}: {1}\n'.format(key, new_info[key]))
		else:
			self._output.write('\t\tNone\n')

		self._output.write('\n')
		self._output.flush()

	def write(self, msg):
		if not self._is_active: return

		self._output.write(msg + '\n')
		self._output.flush()

	def to_dict(self, info, is_fail):
		return {
		'output' : self._output.getvalue(),
		'stdout' : self._stdout.getvalue(),
		'stderr' : self._stderr.getvalue(),
		'info' : info,
		'err' : self._err,
		'is_fail' : is_fail
		}

class DataSource(object):
	bits = platform.architecture()[0]
	cpu_count = multiprocessing.cpu_count()
	is_windows = platform.system().lower() == 'windows'
	arch_string_raw = platform.machine()
	uname_string_raw = platform.uname()[5]
	can_cpuid = True

	@staticmethod
	def has_proc_cpuinfo():
		return os.path.exists('/proc/cpuinfo')

	@staticmethod
	def has_dmesg():
		return len(_program_paths('dmesg')) > 0

	@staticmethod
	def has_var_run_dmesg_boot():
		uname = platform.system().strip().strip('"').strip("'").strip().lower()
		return 'linux' in uname and os.path.exists('/var/run/dmesg.boot')

	@staticmethod
	def has_cpufreq_info():
		return len(_program_paths('cpufreq-info')) > 0

	@staticmethod
	def has_sestatus():
		return len(_program_paths('sestatus')) > 0

	@staticmethod
	def has_sysctl():
		return len(_program_paths('sysctl')) > 0

	@staticmethod
	def has_isainfo():
		return len(_program_paths('isainfo')) > 0

	@staticmethod
	def has_kstat():
		return len(_program_paths('kstat')) > 0

	@staticmethod
	def has_sysinfo():
		uname = platform.system().strip().strip('"').strip("'").strip().lower()
		is_beos = 'beos' in uname or 'haiku' in uname
		return is_beos and len(_program_paths('sysinfo')) > 0

	@staticmethod
	def has_lscpu():
		return len(_program_paths('lscpu')) > 0

	@staticmethod
	def has_ibm_pa_features():
		return len(_program_paths('lsprop')) > 0

	@staticmethod
	def has_wmic():
		returncode, output = _run_and_get_stdout(['wmic', 'os', 'get', 'Version'])
		return returncode == 0 and len(output) > 0

	@staticmethod
	def cat_proc_cpuinfo():
		return _run_and_get_stdout(['cat', '/proc/cpuinfo'])

	@staticmethod
	def cpufreq_info():
		return _run_and_get_stdout(['cpufreq-info'])

	@staticmethod
	def sestatus_b():
		return _run_and_get_stdout(['sestatus', '-b'])

	@staticmethod
	def dmesg_a():
		return _run_and_get_stdout(['dmesg', '-a'])

	@staticmethod
	def cat_var_run_dmesg_boot():
		return _run_and_get_stdout(['cat', '/var/run/dmesg.boot'])

	@staticmethod
	def sysctl_machdep_cpu_hw_cpufrequency():
		return _run_and_get_stdout(['sysctl', 'machdep.cpu', 'hw.cpufrequency'])

	@staticmethod
	def isainfo_vb():
		return _run_and_get_stdout(['isainfo', '-vb'])

	@staticmethod
	def kstat_m_cpu_info():
		return _run_and_get_stdout(['kstat', '-m', 'cpu_info'])

	@staticmethod
	def sysinfo_cpu():
		return _run_and_get_stdout(['sysinfo', '-cpu'])

	@staticmethod
	def lscpu():
		return _run_and_get_stdout(['lscpu'])

	@staticmethod
	def ibm_pa_features():
		import glob

		ibm_features = glob.glob('/proc/device-tree/cpus/*/ibm,pa-features')
		if ibm_features:
			return _run_and_get_stdout(['lsprop', ibm_features[0]])

	@staticmethod
	def wmic_cpu():
		return _run_and_get_stdout(['wmic', 'cpu', 'get', 'Name,CurrentClockSpeed,L2CacheSize,L3CacheSize,Description,Caption,Manufacturer', '/format:list'])

	@staticmethod
	def winreg_processor_brand():
		processor_brand = _read_windows_registry_key(r"Hardware\Description\System\CentralProcessor\0", "ProcessorNameString")
		return processor_brand.strip()

	@staticmethod
	def winreg_vendor_id_raw():
		vendor_id_raw = _read_windows_registry_key(r"Hardware\Description\System\CentralProcessor\0", "VendorIdentifier")
		return vendor_id_raw

	@staticmethod
	def winreg_arch_string_raw():
		arch_string_raw = _read_windows_registry_key(r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", "PROCESSOR_ARCHITECTURE")
		return arch_string_raw

	@staticmethod
	def winreg_hz_actual():
		hz_actual = _read_windows_registry_key(r"Hardware\Description\System\CentralProcessor\0", "~Mhz")
		hz_actual = _to_decimal_string(hz_actual)
		return hz_actual

	@staticmethod
	def winreg_feature_bits():
		feature_bits = _read_windows_registry_key(r"Hardware\Description\System\CentralProcessor\0", "FeatureSet")
		return feature_bits


def _program_paths(program_name):
	paths = []
	exts = filter(None, os.environ.get('PATHEXT', '').split(os.pathsep))
	path = os.environ['PATH']
	for p in os.environ['PATH'].split(os.pathsep):
		p = os.path.join(p, program_name)
		if os.access(p, os.X_OK):
			paths.append(p)
		for e in exts:
			pext = p + e
			if os.access(pext, os.X_OK):
				paths.append(pext)
	return paths

def _run_and_get_stdout(command, pipe_command=None):
	from subprocess import Popen, PIPE

	p1, p2, stdout_output, stderr_output = None, None, None, None

	g_trace.command_header('Running command "' + ' '.join(command) + '" ...')

	# Run the command normally
	if not pipe_command:
		p1 = Popen(command, stdout=PIPE, stderr=PIPE, stdin=PIPE)
	# Run the command and pipe it into another command
	else:
		p2 = Popen(command, stdout=PIPE, stderr=PIPE, stdin=PIPE)
		p1 = Popen(pipe_command, stdin=p2.stdout, stdout=PIPE, stderr=PIPE)
		p2.stdout.close()

	# Get the stdout and stderr
	stdout_output, stderr_output = p1.communicate()
	if not IS_PY2:
		stdout_output = stdout_output.decode(encoding='UTF-8')
		stderr_output = stderr_output.decode(encoding='UTF-8')

	# Send the result to the logger
	g_trace.command_output('return code:', str(p1.returncode))
	g_trace.command_output('stdout:', stdout_output)

	# Return the return code and stdout
	return p1.returncode, stdout_output

def _read_windows_registry_key(key_name, field_name):
	g_trace.command_header('Reading Registry key "{0}" field "{1}" ...'.format(key_name, field_name))

	try:
		import _winreg as winreg
	except ImportError as err:
		try:
			import winreg
		except ImportError as err:
			pass

	key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_name)
	value = winreg.QueryValueEx(key, field_name)[0]
	winreg.CloseKey(key)
	g_trace.command_output('value:', str(value))
	return value

# Make sure we are running on a supported system
def _check_arch():
	arch, bits = _parse_arch(DataSource.arch_string_raw)
	if not arch in ['X86_32', 'X86_64', 'ARM_7', 'ARM_8',
	               'PPC_64', 'S390X', 'MIPS_32', 'MIPS_64']:
		raise Exception("py-cpuinfo currently only works on X86 "
		                "and some ARM/PPC/S390X/MIPS CPUs.")

def _obj_to_b64(thing):
	import pickle
	import base64

	a = thing
	b = pickle.dumps(a)
	c = base64.b64encode(b)
	d = c.decode('utf8')
	return d

def _b64_to_obj(thing):
	import pickle
	import base64

	try:
		a = base64.b64decode(thing)
		b = pickle.loads(a)
		return b
	except:
		return {}

def _utf_to_str(input):
	if IS_PY2 and isinstance(input, unicode):
		return input.encode('utf-8')
	elif isinstance(input, list):
		return [_utf_to_str(element) for element in input]
	elif isinstance(input, dict):
		return {_utf_to_str(key): _utf_to_str(value)
			for key, value in input.items()}
	else:
		return input

def _copy_new_fields(info, new_info):
	keys = [
		'vendor_id_raw', 'hardware_raw', 'brand_raw', 'hz_advertised_friendly', 'hz_actual_friendly',
		'hz_advertised', 'hz_actual', 'arch', 'bits', 'count',
		'arch_string_raw', 'uname_string_raw',
		'l2_cache_size', 'l2_cache_line_size', 'l2_cache_associativity',
		'stepping', 'model', 'family',
		'processor_type', 'flags',
		'l3_cache_size', 'l1_data_cache_size', 'l1_instruction_cache_size'
	]

	g_trace.keys(keys, info, new_info)

	# Update the keys with new values
	for key in keys:
		if new_info.get(key, None) and not info.get(key, None):
			info[key] = new_info[key]
		elif key == 'flags' and new_info.get('flags'):
			for f in new_info['flags']:
				if f not in info['flags']: info['flags'].append(f)
			info['flags'].sort()

def _get_field_actual(cant_be_number, raw_string, field_names):
	for line in raw_string.splitlines():
		for field_name in field_names:
			field_name = field_name.lower()
			if ':' in line:
				left, right = line.split(':', 1)
				left = left.strip().lower()
				right = right.strip()
				if left == field_name and len(right) > 0:
					if cant_be_number:
						if not right.isdigit():
							return right
					else:
						return right

	return None

def _get_field(cant_be_number, raw_string, convert_to, default_value, *field_names):
	retval = _get_field_actual(cant_be_number, raw_string, field_names)

	# Convert the return value
	if retval and convert_to:
		try:
			retval = convert_to(retval)
		except:
			retval = default_value

	# Return the default if there is no return value
	if retval is None:
		retval = default_value

	return retval

def _to_decimal_string(ticks):
	try:
		# Convert to string
		ticks = '{0}'.format(ticks)
		# Sometimes ',' is used as a decimal separator
		ticks = ticks.replace(',', '.')

		# Strip off non numbers and decimal places
		ticks = "".join(n for n in ticks if n.isdigit() or n=='.').strip()
		if ticks == '':
			ticks = '0'

		# Add decimal if missing
		if '.' not in ticks:
			ticks = '{0}.0'.format(ticks)

		# Remove trailing zeros
		ticks = ticks.rstrip('0')

		# Add one trailing zero for empty right side
		if ticks.endswith('.'):
			ticks = '{0}0'.format(ticks)

		# Make sure the number can be converted to a float
		ticks = float(ticks)
		ticks = '{0}'.format(ticks)
		return ticks
	except:
		return '0.0'

def _hz_short_to_full(ticks, scale):
	try:
		# Make sure the number can be converted to a float
		ticks = float(ticks)
		ticks = '{0}'.format(ticks)

		# Scale the numbers
		hz = ticks.lstrip('0')
		old_index = hz.index('.')
		hz = hz.replace('.', '')
		hz = hz.ljust(scale + old_index+1, '0')
		new_index = old_index + scale
		hz = '{0}.{1}'.format(hz[:new_index], hz[new_index:])
		left, right = hz.split('.')
		left, right = int(left), int(right)
		return (left, right)
	except:
		return (0, 0)

def _hz_friendly_to_full(hz_string):
	try:
		hz_string = hz_string.strip().lower()
		hz, scale = (None, None)

		if hz_string.endswith('ghz'):
			scale = 9
		elif hz_string.endswith('mhz'):
			scale = 6
		elif hz_string.endswith('hz'):
			scale = 0

		hz = "".join(n for n in hz_string if n.isdigit() or n=='.').strip()
		if not '.' in hz:
			hz += '.0'

		hz, scale = _hz_short_to_full(hz, scale)

		return (hz, scale)
	except:
		return (0, 0)

def _hz_short_to_friendly(ticks, scale):
	try:
		# Get the raw Hz as a string
		left, right = _hz_short_to_full(ticks, scale)
		result = '{0}.{1}'.format(left, right)

		# Get the location of the dot, and remove said dot
		dot_index = result.index('.')
		result = result.replace('.', '')

		# Get the Hz symbol and scale
		symbol = "Hz"
		scale = 0
		if dot_index > 9:
			symbol = "GHz"
			scale = 9
		elif dot_index > 6:
			symbol = "MHz"
			scale = 6
		elif dot_index > 3:
			symbol = "KHz"
			scale = 3

		# Get the Hz with the dot at the new scaled point
		result = '{0}.{1}'.format(result[:-scale-1], result[-scale-1:])

		# Format the ticks to have 4 numbers after the decimal
		# and remove any superfluous zeroes.
		result = '{0:.4f} {1}'.format(float(result), symbol)
		result = result.rstrip('0')
		return result
	except:
		return '0.0000 Hz'

def _to_friendly_bytes(input):
	import re

	if not input:
		return input
	input = "{0}".format(input)

	formats = {
		r"^[0-9]+B$" : 'B',
		r"^[0-9]+K$" : 'KB',
		r"^[0-9]+M$" : 'MB',
		r"^[0-9]+G$" : 'GB'
	}

	for pattern, friendly_size in formats.items():
		if re.match(pattern, input):
			return "{0} {1}".format(input[ : -1].strip(), friendly_size)

	return input

def _friendly_bytes_to_int(friendly_bytes):
	input = friendly_bytes.lower()

	formats = {
		'gb' : 1024 * 1024 * 1024,
		'mb' : 1024 * 1024,
		'kb' : 1024,

		'g' : 1024 * 1024 * 1024,
		'm' : 1024 * 1024,
		'k' : 1024,
		'b' : 1,
	}

	try:
		for pattern, multiplier in formats.items():
			if input.endswith(pattern):
				return int(input.split(pattern)[0].strip()) * multiplier

	except Exception as err:
		pass

	return friendly_bytes

def _parse_cpu_brand_string(cpu_string):
	# Just return 0 if the processor brand does not have the Hz
	if not 'hz' in cpu_string.lower():
		return ('0.0', 0)

	hz = cpu_string.lower()
	scale = 0

	if hz.endswith('mhz'):
		scale = 6
	elif hz.endswith('ghz'):
		scale = 9
	if '@' in hz:
		hz = hz.split('@')[1]
	else:
		hz = hz.rsplit(None, 1)[1]

	hz = hz.rstrip('mhz').rstrip('ghz').strip()
	hz = _to_decimal_string(hz)

	return (hz, scale)

def _parse_cpu_brand_string_dx(cpu_string):
	import re

	# Find all the strings inside brackets ()
	starts = [m.start() for m in re.finditer(r"\(", cpu_string)]
	ends = [m.start() for m in re.finditer(r"\)", cpu_string)]
	insides = {k: v for k, v in zip(starts, ends)}
	insides = [cpu_string[start+1 : end] for start, end in insides.items()]

	# Find all the fields
	vendor_id, stepping, model, family = (None, None, None, None)
	for inside in insides:
		for pair in inside.split(','):
			pair = [n.strip() for n in pair.split(':')]
			if len(pair) > 1:
				name, value = pair[0], pair[1]
				if name == 'origin':
					vendor_id = value.strip('"')
				elif name == 'stepping':
					stepping = int(value.lstrip('0x'), 16)
				elif name == 'model':
					model = int(value.lstrip('0x'), 16)
				elif name in ['fam', 'family']:
					family = int(value.lstrip('0x'), 16)

	# Find the Processor Brand
	# Strip off extra strings in brackets at end
	brand = cpu_string.strip()
	is_working = True
	while is_working:
		is_working = False
		for inside in insides:
			full = "({0})".format(inside)
			if brand.endswith(full):
				brand = brand[ :-len(full)].strip()
				is_working = True

	# Find the Hz in the brand string
	hz_brand, scale = _parse_cpu_brand_string(brand)

	# Find Hz inside brackets () after the brand string
	if hz_brand == '0.0':
		for inside in insides:
			hz = inside
			for entry in ['GHz', 'MHz', 'Hz']:
				if entry in hz:
					hz = "CPU @ " + hz[ : hz.find(entry) + len(entry)]
					hz_brand, scale = _parse_cpu_brand_string(hz)
					break

	return (hz_brand, scale, brand, vendor_id, stepping, model, family)

def _parse_dmesg_output(output):
	try:
		# Get all the dmesg lines that might contain a CPU string
		lines = output.split(' CPU0:')[1:] + \
				output.split(' CPU1:')[1:] + \
				output.split(' CPU:')[1:] + \
				output.split('\nCPU0:')[1:] + \
				output.split('\nCPU1:')[1:] + \
				output.split('\nCPU:')[1:]
		lines = [l.split('\n')[0].strip() for l in lines]

		# Convert the lines to CPU strings
		cpu_strings = [_parse_cpu_brand_string_dx(l) for l in lines]

		# Find the CPU string that has the most fields
		best_string = None
		highest_count = 0
		for cpu_string in cpu_strings:
			count = sum([n is not None for n in cpu_string])
			if count > highest_count:
				highest_count = count
				best_string = cpu_string

		# If no CPU string was found, return {}
		if not best_string:
			return {}

		hz_actual, scale, processor_brand, vendor_id, stepping, model, family = best_string

		# Origin
		if '  Origin=' in output:
			fields = output[output.find('  Origin=') : ].split('\n')[0]
			fields = fields.strip().split()
			fields = [n.strip().split('=') for n in fields]
			fields = [{n[0].strip().lower() : n[1].strip()} for n in fields]

			for field in fields:
				name = list(field.keys())[0]
				value = list(field.values())[0]

				if name == 'origin':
					vendor_id = value.strip('"')
				elif name == 'stepping':
					stepping = int(value.lstrip('0x'), 16)
				elif name == 'model':
					model = int(value.lstrip('0x'), 16)
				elif name in ['fam', 'family']:
					family = int(value.lstrip('0x'), 16)

		# Features
		flag_lines = []
		for category in ['  Features=', '  Features2=', '  AMD Features=', '  AMD Features2=']:
			if category in output:
				flag_lines.append(output.split(category)[1].split('\n')[0])

		flags = []
		for line in flag_lines:
			line = line.split('<')[1].split('>')[0].lower()
			for flag in line.split(','):
				flags.append(flag)
		flags.sort()

		# Convert from GHz/MHz string to Hz
		hz_advertised, scale = _parse_cpu_brand_string(processor_brand)

		# If advertised hz not found, use the actual hz
		if hz_advertised == '0.0':
			scale = 6
			hz_advertised = _to_decimal_string(hz_actual)

		info = {
		'vendor_id_raw' : vendor_id,
		'brand_raw' : processor_brand,

		'stepping' : stepping,
		'model' : model,
		'family' : family,
		'flags' : flags
		}

		if hz_advertised and hz_advertised != '0.0':
			info['hz_advertised_friendly'] = _hz_short_to_friendly(hz_advertised, scale)
			info['hz_actual_friendly'] = _hz_short_to_friendly(hz_actual, scale)

		if hz_advertised and hz_advertised != '0.0':
			info['hz_advertised'] = _hz_short_to_full(hz_advertised, scale)
			info['hz_actual'] = _hz_short_to_full(hz_actual, scale)

		return {k: v for k, v in info.items() if v}
	except Exception as err:
		g_trace.fail(err)
		#raise
		pass

	return {}

def _parse_arch(arch_string_raw):
	import re

	arch, bits = None, None
	arch_string_raw = arch_string_raw.lower()

	# X86
	if re.match(r'^i\d86$|^x86$|^x86_32$|^i86pc$|^ia32$|^ia-32$|^bepc$', arch_string_raw):
		arch = 'X86_32'
		bits = 32
	elif re.match(r'^x64$|^x86_64$|^x86_64t$|^i686-64$|^amd64$|^ia64$|^ia-64$', arch_string_raw):
		arch = 'X86_64'
		bits = 64
	# ARM
	elif re.match(r'^armv8-a|aarch64|arm64$', arch_string_raw):
		arch = 'ARM_8'
		bits = 64
	elif re.match(r'^armv7$|^armv7[a-z]$|^armv7-[a-z]$|^armv6[a-z]$', arch_string_raw):
		arch = 'ARM_7'
		bits = 32
	elif re.match(r'^armv8$|^armv8[a-z]$|^armv8-[a-z]$', arch_string_raw):
		arch = 'ARM_8'
		bits = 32
	# PPC
	elif re.match(r'^ppc32$|^prep$|^pmac$|^powermac$', arch_string_raw):
		arch = 'PPC_32'
		bits = 32
	elif re.match(r'^powerpc$|^ppc64$|^ppc64le$', arch_string_raw):
		arch = 'PPC_64'
		bits = 64
	# SPARC
	elif re.match(r'^sparc32$|^sparc$', arch_string_raw):
		arch = 'SPARC_32'
		bits = 32
	elif re.match(r'^sparc64$|^sun4u$|^sun4v$', arch_string_raw):
		arch = 'SPARC_64'
		bits = 64
	# S390X
	elif re.match(r'^s390x$', arch_string_raw):
		arch = 'S390X'
		bits = 64
	elif arch_string_raw == 'mips':
		arch = 'MIPS_32'
		bits = 32
	elif arch_string_raw == 'mips64':
		arch = 'MIPS_64'
		bits = 64

	return (arch, bits)

def _is_bit_set(reg, bit):
	mask = 1 << bit
	is_set = reg & mask > 0
	return is_set


def _is_selinux_enforcing(trace):
	# Just return if the SE Linux Status Tool is not installed
	if not DataSource.has_sestatus():
		trace.fail('Failed to find sestatus.')
		return False

	# Run the sestatus, and just return if it failed to run
	returncode, output = DataSource.sestatus_b()
	if returncode != 0:
		trace.fail('Failed to run sestatus. Skipping ...')
		return False

	# Figure out if explicitly in enforcing mode
	for line in output.splitlines():
		line = line.strip().lower()
		if line.startswith("current mode:"):
			if line.endswith("enforcing"):
				return True
			else:
				return False

	# Figure out if we can execute heap and execute memory
	can_selinux_exec_heap = False
	can_selinux_exec_memory = False
	for line in output.splitlines():
		line = line.strip().lower()
		if line.startswith("allow_execheap") and line.endswith("on"):
			can_selinux_exec_heap = True
		elif line.startswith("allow_execmem") and line.endswith("on"):
			can_selinux_exec_memory = True

	trace.command_output('can_selinux_exec_heap:', can_selinux_exec_heap)
	trace.command_output('can_selinux_exec_memory:', can_selinux_exec_memory)

	return (not can_selinux_exec_heap or not can_selinux_exec_memory)

def _filter_dict_keys_with_empty_values(info):
	# Filter out None, 0, "", (), {}, []
	info = {k: v for k, v in info.items() if v}

	# Filter out (0, 0)
	info = {k: v for k, v in info.items() if v != (0, 0)}

	# Filter out strings that start with "0.0"
	info = {k: v for k, v in info.items() if not (type(v) == str and v.startswith('0.0'))}

	return info


class ASM(object):
	def __init__(self, restype=None, argtypes=(), machine_code=[]):
		self.restype = restype
		self.argtypes = argtypes
		self.machine_code = machine_code
		self.prochandle = None
		self.mm = None
		self.func = None
		self.address = None
		self.size = 0

	def compile(self):
		machine_code = bytes.join(b'', self.machine_code)
		self.size = ctypes.c_size_t(len(machine_code))

		if DataSource.is_windows:
			# Allocate a memory segment the size of the machine code, and make it executable
			size = len(machine_code)
			# Alloc at least 1 page to ensure we own all pages that we want to change protection on
			if size < 0x1000: size = 0x1000
			MEM_COMMIT = ctypes.c_ulong(0x1000)
			PAGE_READWRITE = ctypes.c_ulong(0x4)
			pfnVirtualAlloc = ctypes.windll.kernel32.VirtualAlloc
			pfnVirtualAlloc.restype = ctypes.c_void_p
			self.address = pfnVirtualAlloc(None, ctypes.c_size_t(size), MEM_COMMIT, PAGE_READWRITE)
			if not self.address:
				raise Exception("Failed to VirtualAlloc")

			# Copy the machine code into the memory segment
			memmove = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)(ctypes._memmove_addr)
			if memmove(self.address, machine_code, size) < 0:
				raise Exception("Failed to memmove")

			# Enable execute permissions
			PAGE_EXECUTE = ctypes.c_ulong(0x10)
			old_protect = ctypes.c_ulong(0)
			pfnVirtualProtect = ctypes.windll.kernel32.VirtualProtect
			res = pfnVirtualProtect(ctypes.c_void_p(self.address), ctypes.c_size_t(size), PAGE_EXECUTE, ctypes.byref(old_protect))
			if not res:
				raise Exception("Failed VirtualProtect")

			# Flush Instruction Cache
			# First, get process Handle
			if not self.prochandle:
				pfnGetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
				pfnGetCurrentProcess.restype = ctypes.c_void_p
				self.prochandle = ctypes.c_void_p(pfnGetCurrentProcess())
			# Actually flush cache
			res = ctypes.windll.kernel32.FlushInstructionCache(self.prochandle, ctypes.c_void_p(self.address), ctypes.c_size_t(size))
			if not res:
				raise Exception("Failed FlushInstructionCache")
		else:
			from mmap import mmap, MAP_PRIVATE, MAP_ANONYMOUS, PROT_WRITE, PROT_READ, PROT_EXEC

			# Allocate a private and executable memory segment the size of the machine code
			machine_code = bytes.join(b'', self.machine_code)
			self.size = len(machine_code)
			self.mm = mmap(-1, self.size, flags=MAP_PRIVATE | MAP_ANONYMOUS, prot=PROT_WRITE | PROT_READ | PROT_EXEC)

			# Copy the machine code into the memory segment
			self.mm.write(machine_code)
			self.address = ctypes.addressof(ctypes.c_int.from_buffer(self.mm))

		# Cast the memory segment into a function
		functype = ctypes.CFUNCTYPE(self.restype, *self.argtypes)
		self.func = functype(self.address)

	def run(self):
		# Call the machine code like a function
		retval = self.func()

		return retval

	def free(self):
		# Free the function memory segment
		if DataSource.is_windows:
			MEM_RELEASE = ctypes.c_ulong(0x8000)
			ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.address), ctypes.c_size_t(0), MEM_RELEASE)
		else:
			self.mm.close()

		self.prochandle = None
		self.mm = None
		self.func = None
		self.address = None
		self.size = 0


class CPUID(object):
	def __init__(self, trace=None):
		if trace == None:
			trace = Trace(False, False)

		# Figure out if SE Linux is on and in enforcing mode
		self.is_selinux_enforcing = _is_selinux_enforcing(trace)

	def _asm_func(self, restype=None, argtypes=(), machine_code=[]):
		asm = ASM(restype, argtypes, machine_code)
		asm.compile()
		return asm

	def _run_asm(self, *machine_code):
		asm = ASM(ctypes.c_uint32, (), machine_code)
		asm.compile()
		retval = asm.run()
		asm.free()
		return retval

	# http://en.wikipedia.org/wiki/CPUID#EAX.3D0:_Get_vendor_ID
	def get_vendor_id(self):
		# EBX
		ebx = self._run_asm(
			b"\x31\xC0",        # xor eax,eax
			b"\x0F\xA2"         # cpuid
			b"\x89\xD8"         # mov ax,bx
			b"\xC3"             # ret
		)

		# ECX
		ecx = self._run_asm(
			b"\x31\xC0",        # xor eax,eax
			b"\x0f\xa2"         # cpuid
			b"\x89\xC8"         # mov ax,cx
			b"\xC3"             # ret
		)

		# EDX
		edx = self._run_asm(
			b"\x31\xC0",        # xor eax,eax
			b"\x0f\xa2"         # cpuid
			b"\x89\xD0"         # mov ax,dx
			b"\xC3"             # ret
		)

		# Each 4bits is a ascii letter in the name
		vendor_id = []
		for reg in [ebx, edx, ecx]:
			for n in [0, 8, 16, 24]:
				vendor_id.append(chr((reg >> n) & 0xFF))
		vendor_id = ''.join(vendor_id)

		return vendor_id

	# http://en.wikipedia.org/wiki/CPUID#EAX.3D1:_Processor_Info_and_Feature_Bits
	def get_info(self):
		# EAX
		eax = self._run_asm(
			b"\xB8\x01\x00\x00\x00",   # mov eax,0x1"
			b"\x0f\xa2"                # cpuid
			b"\xC3"                    # ret
		)

		# Get the CPU info
		stepping_id = (eax >> 0) & 0xF # 4 bits
		model = (eax >> 4) & 0xF # 4 bits
		family_id = (eax >> 8) & 0xF # 4 bits
		processor_type = (eax >> 12) & 0x3 # 2 bits
		extended_model_id = (eax >> 16) & 0xF # 4 bits
		extended_family_id = (eax >> 20) & 0xFF # 8 bits
		family = 0

		if family_id in [15]:
			family = extended_family_id + family_id
		else:
			family = family_id

		if family_id in [6, 15]:
			model = (extended_model_id << 4) + model

		return {
			'stepping' : stepping_id,
			'model' : model,
			'family' : family,
			'processor_type' : processor_type
		}

	# http://en.wikipedia.org/wiki/CPUID#EAX.3D80000000h:_Get_Highest_Extended_Function_Supported
	def get_max_extension_support(self):
		# Check for extension support
		max_extension_support = self._run_asm(
			b"\xB8\x00\x00\x00\x80" # mov ax,0x80000000
			b"\x0f\xa2"             # cpuid
			b"\xC3"                 # ret
		)

		return max_extension_support

	# http://en.wikipedia.org/wiki/CPUID#EAX.3D1:_Processor_Info_and_Feature_Bits
	def get_flags(self, max_extension_support):
		# EDX
		edx = self._run_asm(
			b"\xB8\x01\x00\x00\x00",   # mov eax,0x1"
			b"\x0f\xa2"                # cpuid
			b"\x89\xD0"                # mov ax,dx
			b"\xC3"                    # ret
		)

		# ECX
		ecx = self._run_asm(
			b"\xB8\x01\x00\x00\x00",   # mov eax,0x1"
			b"\x0f\xa2"                # cpuid
			b"\x89\xC8"                # mov ax,cx
			b"\xC3"                    # ret
		)

		# Get the CPU flags
		flags = {
			'fpu' : _is_bit_set(edx, 0),
			'vme' : _is_bit_set(edx, 1),
			'de' : _is_bit_set(edx, 2),
			'pse' : _is_bit_set(edx, 3),
			'tsc' : _is_bit_set(edx, 4),
			'msr' : _is_bit_set(edx, 5),
			'pae' : _is_bit_set(edx, 6),
			'mce' : _is_bit_set(edx, 7),
			'cx8' : _is_bit_set(edx, 8),
			'apic' : _is_bit_set(edx, 9),
			#'reserved1' : _is_bit_set(edx, 10),
			'sep' : _is_bit_set(edx, 11),
			'mtrr' : _is_bit_set(edx, 12),
			'pge' : _is_bit_set(edx, 13),
			'mca' : _is_bit_set(edx, 14),
			'cmov' : _is_bit_set(edx, 15),
			'pat' : _is_bit_set(edx, 16),
			'pse36' : _is_bit_set(edx, 17),
			'pn' : _is_bit_set(edx, 18),
			'clflush' : _is_bit_set(edx, 19),
			#'reserved2' : _is_bit_set(edx, 20),
			'dts' : _is_bit_set(edx, 21),
			'acpi' : _is_bit_set(edx, 22),
			'mmx' : _is_bit_set(edx, 23),
			'fxsr' : _is_bit_set(edx, 24),
			'sse' : _is_bit_set(edx, 25),
			'sse2' : _is_bit_set(edx, 26),
			'ss' : _is_bit_set(edx, 27),
			'ht' : _is_bit_set(edx, 28),
			'tm' : _is_bit_set(edx, 29),
			'ia64' : _is_bit_set(edx, 30),
			'pbe' : _is_bit_set(edx, 31),

			'pni' : _is_bit_set(ecx, 0),
			'pclmulqdq' : _is_bit_set(ecx, 1),
			'dtes64' : _is_bit_set(ecx, 2),
			'monitor' : _is_bit_set(ecx, 3),
			'ds_cpl' : _is_bit_set(ecx, 4),
			'vmx' : _is_bit_set(ecx, 5),
			'smx' : _is_bit_set(ecx, 6),
			'est' : _is_bit_set(ecx, 7),
			'tm2' : _is_bit_set(ecx, 8),
			'ssse3' : _is_bit_set(ecx, 9),
			'cid' : _is_bit_set(ecx, 10),
			#'reserved3' : _is_bit_set(ecx, 11),
			'fma' : _is_bit_set(ecx, 12),
			'cx16' : _is_bit_set(ecx, 13),
			'xtpr' : _is_bit_set(ecx, 14),
			'pdcm' : _is_bit_set(ecx, 15),
			#'reserved4' : _is_bit_set(ecx, 16),
			'pcid' : _is_bit_set(ecx, 17),
			'dca' : _is_bit_set(ecx, 18),
			'sse4_1' : _is_bit_set(ecx, 19),
			'sse4_2' : _is_bit_set(ecx, 20),
			'x2apic' : _is_bit_set(ecx, 21),
			'movbe' : _is_bit_set(ecx, 22),
			'popcnt' : _is_bit_set(ecx, 23),
			'tscdeadline' : _is_bit_set(ecx, 24),
			'aes' : _is_bit_set(ecx, 25),
			'xsave' : _is_bit_set(ecx, 26),
			'osxsave' : _is_bit_set(ecx, 27),
			'avx' : _is_bit_set(ecx, 28),
			'f16c' : _is_bit_set(ecx, 29),
			'rdrnd' : _is_bit_set(ecx, 30),
			'hypervisor' : _is_bit_set(ecx, 31)
		}

		# Get a list of only the flags that are true
		flags = [k for k, v in flags.items() if v]

		# http://en.wikipedia.org/wiki/CPUID#EAX.3D7.2C_ECX.3D0:_Extended_Features
		if max_extension_support >= 7:
			# EBX
			ebx = self._run_asm(
				b"\x31\xC9",            # xor ecx,ecx
				b"\xB8\x07\x00\x00\x00" # mov eax,7
				b"\x0f\xa2"         # cpuid
				b"\x89\xD8"         # mov ax,bx
				b"\xC3"             # ret
			)

			# ECX
			ecx = self._run_asm(
				b"\x31\xC9",            # xor ecx,ecx
				b"\xB8\x07\x00\x00\x00" # mov eax,7
				b"\x0f\xa2"         # cpuid
				b"\x89\xC8"         # mov ax,cx
				b"\xC3"             # ret
			)

			# Get the extended CPU flags
			extended_flags = {
				#'fsgsbase' : _is_bit_set(ebx, 0),
				#'IA32_TSC_ADJUST' : _is_bit_set(ebx, 1),
				'sgx' : _is_bit_set(ebx, 2),
				'bmi1' : _is_bit_set(ebx, 3),
				'hle' : _is_bit_set(ebx, 4),
				'avx2' : _is_bit_set(ebx, 5),
				#'reserved' : _is_bit_set(ebx, 6),
				'smep' : _is_bit_set(ebx, 7),
				'bmi2' : _is_bit_set(ebx, 8),
				'erms' : _is_bit_set(ebx, 9),
				'invpcid' : _is_bit_set(ebx, 10),
				'rtm' : _is_bit_set(ebx, 11),
				'pqm' : _is_bit_set(ebx, 12),
				#'FPU CS and FPU DS deprecated' : _is_bit_set(ebx, 13),
				'mpx' : _is_bit_set(ebx, 14),
				'pqe' : _is_bit_set(ebx, 15),
				'avx512f' : _is_bit_set(ebx, 16),
				'avx512dq' : _is_bit_set(ebx, 17),
				'rdseed' : _is_bit_set(ebx, 18),
				'adx' : _is_bit_set(ebx, 19),
				'smap' : _is_bit_set(ebx, 20),
				'avx512ifma' : _is_bit_set(ebx, 21),
				'pcommit' : _is_bit_set(ebx, 22),
				'clflushopt' : _is_bit_set(ebx, 23),
				'clwb' : _is_bit_set(ebx, 24),
				'intel_pt' : _is_bit_set(ebx, 25),
				'avx512pf' : _is_bit_set(ebx, 26),
				'avx512er' : _is_bit_set(ebx, 27),
				'avx512cd' : _is_bit_set(ebx, 28),
				'sha' : _is_bit_set(ebx, 29),
				'avx512bw' : _is_bit_set(ebx, 30),
				'avx512vl' : _is_bit_set(ebx, 31),

				'prefetchwt1' : _is_bit_set(ecx, 0),
				'avx512vbmi' : _is_bit_set(ecx, 1),
				'umip' : _is_bit_set(ecx, 2),
				'pku' : _is_bit_set(ecx, 3),
				'ospke' : _is_bit_set(ecx, 4),
				#'reserved' : _is_bit_set(ecx, 5),
				'avx512vbmi2' : _is_bit_set(ecx, 6),
				#'reserved' : _is_bit_set(ecx, 7),
				'gfni' : _is_bit_set(ecx, 8),
				'vaes' : _is_bit_set(ecx, 9),
				'vpclmulqdq' : _is_bit_set(ecx, 10),
				'avx512vnni' : _is_bit_set(ecx, 11),
				'avx512bitalg' : _is_bit_set(ecx, 12),
				#'reserved' : _is_bit_set(ecx, 13),
				'avx512vpopcntdq' : _is_bit_set(ecx, 14),
				#'reserved' : _is_bit_set(ecx, 15),
				#'reserved' : _is_bit_set(ecx, 16),
				#'mpx0' : _is_bit_set(ecx, 17),
				#'mpx1' : _is_bit_set(ecx, 18),
				#'mpx2' : _is_bit_set(ecx, 19),
				#'mpx3' : _is_bit_set(ecx, 20),
				#'mpx4' : _is_bit_set(ecx, 21),
				'rdpid' : _is_bit_set(ecx, 22),
				#'reserved' : _is_bit_set(ecx, 23),
				#'reserved' : _is_bit_set(ecx, 24),
				#'reserved' : _is_bit_set(ecx, 25),
				#'reserved' : _is_bit_set(ecx, 26),
				#'reserved' : _is_bit_set(ecx, 27),
				#'reserved' : _is_bit_set(ecx, 28),
				#'reserved' : _is_bit_set(ecx, 29),
				'sgx_lc' : _is_bit_set(ecx, 30),
				#'reserved' : _is_bit_set(ecx, 31)
			}

			# Get a list of only the flags that are true
			extended_flags = [k for k, v in extended_flags.items() if v]
			flags += extended_flags

		# http://en.wikipedia.org/wiki/CPUID#EAX.3D80000001h:_Extended_Processor_Info_and_Feature_Bits
		if max_extension_support >= 0x80000001:
			# EBX
			ebx = self._run_asm(
				b"\xB8\x01\x00\x00\x80" # mov ax,0x80000001
				b"\x0f\xa2"         # cpuid
				b"\x89\xD8"         # mov ax,bx
				b"\xC3"             # ret
			)

			# ECX
			ecx = self._run_asm(
				b"\xB8\x01\x00\x00\x80" # mov ax,0x80000001
				b"\x0f\xa2"         # cpuid
				b"\x89\xC8"         # mov ax,cx
				b"\xC3"             # ret
			)

			# Get the extended CPU flags
			extended_flags = {
				'fpu' : _is_bit_set(ebx, 0),
				'vme' : _is_bit_set(ebx, 1),
				'de' : _is_bit_set(ebx, 2),
				'pse' : _is_bit_set(ebx, 3),
				'tsc' : _is_bit_set(ebx, 4),
				'msr' : _is_bit_set(ebx, 5),
				'pae' : _is_bit_set(ebx, 6),
				'mce' : _is_bit_set(ebx, 7),
				'cx8' : _is_bit_set(ebx, 8),
				'apic' : _is_bit_set(ebx, 9),
				#'reserved' : _is_bit_set(ebx, 10),
				'syscall' : _is_bit_set(ebx, 11),
				'mtrr' : _is_bit_set(ebx, 12),
				'pge' : _is_bit_set(ebx, 13),
				'mca' : _is_bit_set(ebx, 14),
				'cmov' : _is_bit_set(ebx, 15),
				'pat' : _is_bit_set(ebx, 16),
				'pse36' : _is_bit_set(ebx, 17),
				#'reserved' : _is_bit_set(ebx, 18),
				'mp' : _is_bit_set(ebx, 19),
				'nx' : _is_bit_set(ebx, 20),
				#'reserved' : _is_bit_set(ebx, 21),
				'mmxext' : _is_bit_set(ebx, 22),
				'mmx' : _is_bit_set(ebx, 23),
				'fxsr' : _is_bit_set(ebx, 24),
				'fxsr_opt' : _is_bit_set(ebx, 25),
				'pdpe1gp' : _is_bit_set(ebx, 26),
				'rdtscp' : _is_bit_set(ebx, 27),
				#'reserved' : _is_bit_set(ebx, 28),
				'lm' : _is_bit_set(ebx, 29),
				'3dnowext' : _is_bit_set(ebx, 30),
				'3dnow' : _is_bit_set(ebx, 31),

				'lahf_lm' : _is_bit_set(ecx, 0),
				'cmp_legacy' : _is_bit_set(ecx, 1),
				'svm' : _is_bit_set(ecx, 2),
				'extapic' : _is_bit_set(ecx, 3),
				'cr8_legacy' : _is_bit_set(ecx, 4),
				'abm' : _is_bit_set(ecx, 5),
				'sse4a' : _is_bit_set(ecx, 6),
				'misalignsse' : _is_bit_set(ecx, 7),
				'3dnowprefetch' : _is_bit_set(ecx, 8),
				'osvw' : _is_bit_set(ecx, 9),
				'ibs' : _is_bit_set(ecx, 10),
				'xop' : _is_bit_set(ecx, 11),
				'skinit' : _is_bit_set(ecx, 12),
				'wdt' : _is_bit_set(ecx, 13),
				#'reserved' : _is_bit_set(ecx, 14),
				'lwp' : _is_bit_set(ecx, 15),
				'fma4' : _is_bit_set(ecx, 16),
				'tce' : _is_bit_set(ecx, 17),
				#'reserved' : _is_bit_set(ecx, 18),
				'nodeid_msr' : _is_bit_set(ecx, 19),
				#'reserved' : _is_bit_set(ecx, 20),
				'tbm' : _is_bit_set(ecx, 21),
				'topoext' : _is_bit_set(ecx, 22),
				'perfctr_core' : _is_bit_set(ecx, 23),
				'perfctr_nb' : _is_bit_set(ecx, 24),
				#'reserved' : _is_bit_set(ecx, 25),
				'dbx' : _is_bit_set(ecx, 26),
				'perftsc' : _is_bit_set(ecx, 27),
				'pci_l2i' : _is_bit_set(ecx, 28),
				#'reserved' : _is_bit_set(ecx, 29),
				#'reserved' : _is_bit_set(ecx, 30),
				#'reserved' : _is_bit_set(ecx, 31)
			}

			# Get a list of only the flags that are true
			extended_flags = [k for k, v in extended_flags.items() if v]
			flags += extended_flags

		flags.sort()
		return flags

	# http://en.wikipedia.org/wiki/CPUID#EAX.3D80000002h.2C80000003h.2C80000004h:_Processor_Brand_String
	def get_processor_brand(self, max_extension_support):
		processor_brand = ""

		# Processor brand string
		if max_extension_support >= 0x80000004:
			instructions = [
				b"\xB8\x02\x00\x00\x80", # mov ax,0x80000002
				b"\xB8\x03\x00\x00\x80", # mov ax,0x80000003
				b"\xB8\x04\x00\x00\x80"  # mov ax,0x80000004
			]
			for instruction in instructions:
				# EAX
				eax = self._run_asm(
					instruction,  # mov ax,0x8000000?
					b"\x0f\xa2"   # cpuid
					b"\x89\xC0"   # mov ax,ax
					b"\xC3"       # ret
				)

				# EBX
				ebx = self._run_asm(
					instruction,  # mov ax,0x8000000?
					b"\x0f\xa2"   # cpuid
					b"\x89\xD8"   # mov ax,bx
					b"\xC3"       # ret
				)

				# ECX
				ecx = self._run_asm(
					instruction,  # mov ax,0x8000000?
					b"\x0f\xa2"   # cpuid
					b"\x89\xC8"   # mov ax,cx
					b"\xC3"       # ret
				)

				# EDX
				edx = self._run_asm(
					instruction,  # mov ax,0x8000000?
					b"\x0f\xa2"   # cpuid
					b"\x89\xD0"   # mov ax,dx
					b"\xC3"       # ret
				)

				# Combine each of the 4 bytes in each register into the string
				for reg in [eax, ebx, ecx, edx]:
					for n in [0, 8, 16, 24]:
						processor_brand += chr((reg >> n) & 0xFF)

		# Strip off any trailing NULL terminators and white space
		processor_brand = processor_brand.strip("\0").strip()

		return processor_brand

	# http://en.wikipedia.org/wiki/CPUID#EAX.3D80000006h:_Extended_L2_Cache_Features
	def get_cache(self, max_extension_support):
		cache_info = {}

		# Just return if the cache feature is not supported
		if max_extension_support < 0x80000006:
			return cache_info

		# ECX
		ecx = self._run_asm(
			b"\xB8\x06\x00\x00\x80"  # mov ax,0x80000006
			b"\x0f\xa2"              # cpuid
			b"\x89\xC8"              # mov ax,cx
			b"\xC3"                   # ret
		)

		cache_info = {
			'size_b' : (ecx & 0xFF) * 1024,
			'associativity' : (ecx >> 12) & 0xF,
			'line_size_b' : (ecx >> 16) & 0xFFFF
		}

		return cache_info

	def get_ticks_func(self):
		retval = None

		if DataSource.bits == '32bit':
			# Works on x86_32
			restype = None
			argtypes = (ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint))
			get_ticks_x86_32 = self._asm_func(restype, argtypes,
				[
				b"\x55",         # push bp
				b"\x89\xE5",     # mov bp,sp
				b"\x31\xC0",     # xor ax,ax
				b"\x0F\xA2",     # cpuid
				b"\x0F\x31",     # rdtsc
				b"\x8B\x5D\x08", # mov bx,[di+0x8]
				b"\x8B\x4D\x0C", # mov cx,[di+0xc]
				b"\x89\x13",     # mov [bp+di],dx
				b"\x89\x01",     # mov [bx+di],ax
				b"\x5D",         # pop bp
				b"\xC3"          # ret
				]
			)

			# Monkey patch func to combine high and low args into one return
			old_func = get_ticks_x86_32.func
			def new_func():
				# Pass two uint32s into function
				high = ctypes.c_uint32(0)
				low = ctypes.c_uint32(0)
				old_func(ctypes.byref(high), ctypes.byref(low))

				# Shift the two uint32s into one uint64
				retval = ((high.value << 32) & 0xFFFFFFFF00000000) | low.value
				return retval
			get_ticks_x86_32.func = new_func

			retval = get_ticks_x86_32
		elif DataSource.bits == '64bit':
			# Works on x86_64
			restype = ctypes.c_uint64
			argtypes = ()
			get_ticks_x86_64 = self._asm_func(restype, argtypes,
				[
				b"\x48",         # dec ax
				b"\x31\xC0",     # xor ax,ax
				b"\x0F\xA2",     # cpuid
				b"\x0F\x31",     # rdtsc
				b"\x48",         # dec ax
				b"\xC1\xE2\x20", # shl dx,byte 0x20
				b"\x48",         # dec ax
				b"\x09\xD0",     # or ax,dx
				b"\xC3",         # ret
				]
			)

			retval = get_ticks_x86_64
		return retval

	def get_raw_hz(self):
		from time import sleep

		ticks_fn = self.get_ticks_func()

		start = ticks_fn.func()
		sleep(1)
		end = ticks_fn.func()

		ticks = (end - start)
		ticks_fn.free()

		return ticks

def _get_cpu_info_from_cpuid_actual():
	'''
	Warning! This function has the potential to crash the Python runtime.
	Do not call it directly. Use the _get_cpu_info_from_cpuid function instead.
	It will safely call this function in another process.
	'''

	if IS_PY2:
		from cStringIO import StringIO
	else:
		from io import StringIO

	trace = Trace(True, True)
	info = {}

	# Pipe stdout and stderr to strings
	sys.stdout = trace._stdout
	sys.stderr = trace._stderr

	try:
		# Get the CPU arch and bits
		arch, bits = _parse_arch(DataSource.arch_string_raw)

		# Return none if this is not an X86 CPU
		if not arch in ['X86_32', 'X86_64']:
			trace.fail('Not running on X86_32 or X86_64. Skipping ...')
			return trace.to_dict(info, True)

		# Return none if SE Linux is in enforcing mode
		cpuid = CPUID(trace)
		if cpuid.is_selinux_enforcing:
			trace.fail('SELinux is enforcing. Skipping ...')
			return trace.to_dict(info, True)

		# Get the cpu info from the CPUID register
		max_extension_support = cpuid.get_max_extension_support()
		cache_info = cpuid.get_cache(max_extension_support)
		info = cpuid.get_info()

		processor_brand = cpuid.get_processor_brand(max_extension_support)

		# Get the Hz and scale
		hz_actual = cpuid.get_raw_hz()
		hz_actual = _to_decimal_string(hz_actual)

		# Get the Hz and scale
		hz_advertised, scale = _parse_cpu_brand_string(processor_brand)
		info = {
		'vendor_id_raw' : cpuid.get_vendor_id(),
		'hardware_raw' : '',
		'brand_raw' : processor_brand,

		'hz_advertised_friendly' : _hz_short_to_friendly(hz_advertised, scale),
		'hz_actual_friendly' : _hz_short_to_friendly(hz_actual, 0),
		'hz_advertised' : _hz_short_to_full(hz_advertised, scale),
		'hz_actual' : _hz_short_to_full(hz_actual, 0),

		'l2_cache_size' : cache_info['size_b'],
		'l2_cache_line_size' : cache_info['line_size_b'],
		'l2_cache_associativity' : cache_info['associativity'],

		'stepping' : info['stepping'],
		'model' : info['model'],
		'family' : info['family'],
		'processor_type' : info['processor_type'],
		'flags' : cpuid.get_flags(max_extension_support)
		}

		info = _filter_dict_keys_with_empty_values(info)
		trace.success()
	except Exception as err:
		from traceback import format_exc
		err_string = format_exc()
		trace._err = ''.join(['\t\t{0}\n'.format(n) for n in err_string.split('\n')]) + '\n'
		return trace.to_dict(info, True)

	return trace.to_dict(info, False)

def _get_cpu_info_from_cpuid_subprocess_wrapper(queue):
	orig_stdout = sys.stdout
	orig_stderr = sys.stderr

	output = _get_cpu_info_from_cpuid_actual()

	sys.stdout = orig_stdout
	sys.stderr = orig_stderr

	queue.put(_obj_to_b64(output))

def _get_cpu_info_from_cpuid():
	'''
	Returns the CPU info gathered by querying the X86 cpuid register in a new process.
	Returns {} on non X86 cpus.
	Returns {} if SELinux is in enforcing mode.
	'''

	g_trace.header('Tying to get info from CPUID ...')

	from multiprocessing import Process, Queue

	# Return {} if can't cpuid
	if not DataSource.can_cpuid:
		g_trace.fail('Can\'t CPUID. Skipping ...')
		return {}

	# Get the CPU arch and bits
	arch, bits = _parse_arch(DataSource.arch_string_raw)

	# Return {} if this is not an X86 CPU
	if not arch in ['X86_32', 'X86_64']:
		g_trace.fail('Not running on X86_32 or X86_64. Skipping ...')
		return {}

	try:
		if CAN_CALL_CPUID_IN_SUBPROCESS:
			# Start running the function in a subprocess
			queue = Queue()
			p = Process(target=_get_cpu_info_from_cpuid_subprocess_wrapper, args=(queue,))
			p.start()

			# Wait for the process to end, while it is still alive
			while p.is_alive():
				p.join(0)

			# Return {} if it failed
			if p.exitcode != 0:
				g_trace.fail('Failed to run CPUID in process. Skipping ...')
				return {}

			# Return {} if no results
			if queue.empty():
				g_trace.fail('Failed to get anything from CPUID process. Skipping ...')
				return {}
			# Return the result, only if there is something to read
			else:
				output = _b64_to_obj(queue.get())
				import pprint
				pp = pprint.PrettyPrinter(indent=4)
				#pp.pprint(output)

				if 'output' in output and output['output']:
					g_trace.write(output['output'])

				if 'stdout' in output and output['stdout']:
					sys.stdout.write('{0}\n'.format(output['stdout']))
					sys.stdout.flush()

				if 'stderr' in output and output['stderr']:
					sys.stderr.write('{0}\n'.format(output['stderr']))
					sys.stderr.flush()

				if 'is_fail' not in output:
					g_trace.fail('Failed to get is_fail from CPUID process. Skipping ...')
					return {}

				# Fail if there was an exception
				if 'err' in output and output['err']:
					g_trace.fail('Failed to run CPUID in process. Skipping ...')
					g_trace.write(output['err'])
					g_trace.write('Failed ...')
					return {}

				if 'is_fail' in output and output['is_fail']:
					g_trace.write('Failed ...')
					return {}

				if 'info' not in output or not output['info']:
					g_trace.fail('Failed to get return info from CPUID process. Skipping ...')
					return {}

				return output['info']
		else:
			# FIXME: This should write the values like in the above call to actual
			orig_stdout = sys.stdout
			orig_stderr = sys.stderr

			output = _get_cpu_info_from_cpuid_actual()

			sys.stdout = orig_stdout
			sys.stderr = orig_stderr

			g_trace.success()
			return output['info']
	except Exception as err:
		g_trace.fail(err)
		pass

	# Return {} if everything failed
	return {}

def _get_cpu_info_from_proc_cpuinfo():
	'''
	Returns the CPU info gathered from /proc/cpuinfo.
	Returns {} if /proc/cpuinfo is not found.
	'''

	g_trace.header('Tying to get info from /proc/cpuinfo ...')

	try:
		# Just return {} if there is no cpuinfo
		if not DataSource.has_proc_cpuinfo():
			g_trace.fail('Failed to find /proc/cpuinfo. Skipping ...')
			return {}

		returncode, output = DataSource.cat_proc_cpuinfo()
		if returncode != 0:
			g_trace.fail('Failed to run cat /proc/cpuinfo. Skipping ...')
			return {}

		# Various fields
		vendor_id = _get_field(False, output, None, '', 'vendor_id', 'vendor id', 'vendor')
		processor_brand = _get_field(True, output, None, None, 'model name','cpu', 'processor')
		cache_size = _get_field(False, output, None, '', 'cache size')
		stepping = _get_field(False, output, int, 0, 'stepping')
		model = _get_field(False, output, int, 0, 'model')
		family = _get_field(False, output, int, 0, 'cpu family')
		hardware = _get_field(False, output, None, '', 'Hardware')

		# Flags
		flags = _get_field(False, output, None, None, 'flags', 'Features', 'ASEs implemented')
		if flags:
			flags = flags.split()
			flags.sort()

		# Check for other cache format
		if not cache_size:
			try:
				for i in range(0, 10):
					name = "cache{0}".format(i)
					value = _get_field(False, output, None, None, name)
					if value:
						value = [entry.split('=') for entry in value.split(' ')]
						value = dict(value)
						if 'level' in value and value['level'] == '3' and 'size' in value:
							cache_size = value['size']
							break
			except Exception:
				pass

		# Convert from MHz string to Hz
		hz_actual = _get_field(False, output, None, '', 'cpu MHz', 'cpu speed', 'clock', 'cpu MHz dynamic', 'cpu MHz static')
		hz_actual = hz_actual.lower().rstrip('mhz').strip()
		hz_actual = _to_decimal_string(hz_actual)

		# Convert from GHz/MHz string to Hz
		hz_advertised, scale = (None, 0)
		try:
			hz_advertised, scale = _parse_cpu_brand_string(processor_brand)
		except Exception:
			pass

		info = {
		'hardware_raw' : hardware,
		'brand_raw' : processor_brand,

		'l3_cache_size' : _friendly_bytes_to_int(cache_size),
		'flags' : flags,
		'vendor_id_raw' : vendor_id,
		'stepping' : stepping,
		'model' : model,
		'family' : family,
		}

		# Make the Hz the same for actual and advertised if missing any
		if not hz_advertised or hz_advertised == '0.0':
			hz_advertised = hz_actual
			scale = 6
		elif not hz_actual or hz_actual == '0.0':
			hz_actual = hz_advertised

		# Add the Hz if there is one
		if _hz_short_to_full(hz_advertised, scale) > (0, 0):
			info['hz_advertised_friendly'] = _hz_short_to_friendly(hz_advertised, scale)
			info['hz_advertised'] = _hz_short_to_full(hz_advertised, scale)
		if _hz_short_to_full(hz_actual, scale) > (0, 0):
			info['hz_actual_friendly'] = _hz_short_to_friendly(hz_actual, 6)
			info['hz_actual'] = _hz_short_to_full(hz_actual, 6)

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		#raise # NOTE: To have this throw on error, uncomment this line
		return {}

def _get_cpu_info_from_cpufreq_info():
	'''
	Returns the CPU info gathered from cpufreq-info.
	Returns {} if cpufreq-info is not found.
	'''

	g_trace.header('Tying to get info from cpufreq-info ...')

	try:
		hz_brand, scale = '0.0', 0

		if not DataSource.has_cpufreq_info():
			g_trace.fail('Failed to find cpufreq-info. Skipping ...')
			return {}

		returncode, output = DataSource.cpufreq_info()
		if returncode != 0:
			g_trace.fail('Failed to run cpufreq-info. Skipping ...')
			return {}

		hz_brand = output.split('current CPU frequency is')[1].split('\n')[0]
		i = hz_brand.find('Hz')
		assert(i != -1)
		hz_brand = hz_brand[0 : i+2].strip().lower()

		if hz_brand.endswith('mhz'):
			scale = 6
		elif hz_brand.endswith('ghz'):
			scale = 9
		hz_brand = hz_brand.rstrip('mhz').rstrip('ghz').strip()
		hz_brand = _to_decimal_string(hz_brand)

		info = {
			'hz_advertised_friendly' : _hz_short_to_friendly(hz_brand, scale),
			'hz_actual_friendly' : _hz_short_to_friendly(hz_brand, scale),
			'hz_advertised' : _hz_short_to_full(hz_brand, scale),
			'hz_actual' : _hz_short_to_full(hz_brand, scale),
		}

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		#raise # NOTE: To have this throw on error, uncomment this line
		return {}

def _get_cpu_info_from_lscpu():
	'''
	Returns the CPU info gathered from lscpu.
	Returns {} if lscpu is not found.
	'''

	g_trace.header('Tying to get info from lscpu ...')

	try:
		if not DataSource.has_lscpu():
			g_trace.fail('Failed to find lscpu. Skipping ...')
			return {}

		returncode, output = DataSource.lscpu()
		if returncode != 0:
			g_trace.fail('Failed to run lscpu. Skipping ...')
			return {}

		info = {}

		new_hz = _get_field(False, output, None, None, 'CPU max MHz', 'CPU MHz')
		if new_hz:
			new_hz = _to_decimal_string(new_hz)
			scale = 6
			info['hz_advertised_friendly'] = _hz_short_to_friendly(new_hz, scale)
			info['hz_actual_friendly'] = _hz_short_to_friendly(new_hz, scale)
			info['hz_advertised'] = _hz_short_to_full(new_hz, scale)
			info['hz_actual'] = _hz_short_to_full(new_hz, scale)

		new_hz = _get_field(False, output, None, None, 'CPU dynamic MHz', 'CPU static MHz')
		if new_hz:
			new_hz = _to_decimal_string(new_hz)
			scale = 6
			info['hz_advertised_friendly'] = _hz_short_to_friendly(new_hz, scale)
			info['hz_actual_friendly'] = _hz_short_to_friendly(new_hz, scale)
			info['hz_advertised'] = _hz_short_to_full(new_hz, scale)
			info['hz_actual'] = _hz_short_to_full(new_hz, scale)

		vendor_id = _get_field(False, output, None, None, 'Vendor ID')
		if vendor_id:
			info['vendor_id_raw'] = vendor_id

		brand = _get_field(False, output, None, None, 'Model name')
		if brand:
			info['brand_raw'] = brand
		else:
			brand = _get_field(False, output, None, None, 'Model')
			if brand and not brand.isdigit():
				info['brand_raw'] = brand

		family = _get_field(False, output, None, None, 'CPU family')
		if family and family.isdigit():
			info['family'] = int(family)

		stepping = _get_field(False, output, None, None, 'Stepping')
		if stepping and stepping.isdigit():
			info['stepping'] = int(stepping)

		model = _get_field(False, output, None, None, 'Model')
		if model and model.isdigit():
			info['model'] = int(model)

		l1_data_cache_size = _get_field(False, output, None, None, 'L1d cache')
		if l1_data_cache_size:
			info['l1_data_cache_size'] = _friendly_bytes_to_int(l1_data_cache_size)

		l1_instruction_cache_size = _get_field(False, output, None, None, 'L1i cache')
		if l1_instruction_cache_size:
			info['l1_instruction_cache_size'] = _friendly_bytes_to_int(l1_instruction_cache_size)

		l2_cache_size = _get_field(False, output, None, None, 'L2 cache', 'L2d cache')
		if l2_cache_size:
			info['l2_cache_size'] = _friendly_bytes_to_int(l2_cache_size)

		l3_cache_size = _get_field(False, output, None, None, 'L3 cache')
		if l3_cache_size:
			info['l3_cache_size'] = _friendly_bytes_to_int(l3_cache_size)

		# Flags
		flags = _get_field(False, output, None, None, 'flags', 'Features', 'ASEs implemented')
		if flags:
			flags = flags.split()
			flags.sort()
			info['flags'] = flags

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		#raise # NOTE: To have this throw on error, uncomment this line
		return {}

def _get_cpu_info_from_dmesg():
	'''
	Returns the CPU info gathered from dmesg.
	Returns {} if dmesg is not found or does not have the desired info.
	'''

	g_trace.header('Tying to get info from the dmesg ...')

	# Just return {} if this arch has an unreliable dmesg log
	arch, bits = _parse_arch(DataSource.arch_string_raw)
	if arch in ['S390X']:
		g_trace.fail('Running on S390X. Skipping ...')
		return {}

	# Just return {} if there is no dmesg
	if not DataSource.has_dmesg():
		g_trace.fail('Failed to find dmesg. Skipping ...')
		return {}

	# If dmesg fails return {}
	returncode, output = DataSource.dmesg_a()
	if output == None or returncode != 0:
		g_trace.fail('Failed to run \"dmesg -a\". Skipping ...')
		return {}

	info = _parse_dmesg_output(output)
	g_trace.success()
	return info


# https://openpowerfoundation.org/wp-content/uploads/2016/05/LoPAPR_DRAFT_v11_24March2016_cmt1.pdf
# page 767
def _get_cpu_info_from_ibm_pa_features():
	'''
	Returns the CPU info gathered from lsprop /proc/device-tree/cpus/*/ibm,pa-features
	Returns {} if lsprop is not found or ibm,pa-features does not have the desired info.
	'''

	g_trace.header('Tying to get info from lsprop ...')

	try:
		# Just return {} if there is no lsprop
		if not DataSource.has_ibm_pa_features():
			g_trace.fail('Failed to find lsprop. Skipping ...')
			return {}

		# If ibm,pa-features fails return {}
		returncode, output = DataSource.ibm_pa_features()
		if output == None or returncode != 0:
			g_trace.fail('Failed to glob /proc/device-tree/cpus/*/ibm,pa-features. Skipping ...')
			return {}

		# Filter out invalid characters from output
		value = output.split("ibm,pa-features")[1].lower()
		value = [s for s in value if s in list('0123456789abcfed')]
		value = ''.join(value)

		# Get data converted to Uint32 chunks
		left = int(value[0 : 8], 16)
		right = int(value[8 : 16], 16)

		# Get the CPU flags
		flags = {
			# Byte 0
			'mmu' : _is_bit_set(left, 0),
			'fpu' : _is_bit_set(left, 1),
			'slb' : _is_bit_set(left, 2),
			'run' : _is_bit_set(left, 3),
			#'reserved' : _is_bit_set(left, 4),
			'dabr' : _is_bit_set(left, 5),
			'ne' : _is_bit_set(left, 6),
			'wtr' : _is_bit_set(left, 7),

			# Byte 1
			'mcr' : _is_bit_set(left, 8),
			'dsisr' : _is_bit_set(left, 9),
			'lp' : _is_bit_set(left, 10),
			'ri' : _is_bit_set(left, 11),
			'dabrx' : _is_bit_set(left, 12),
			'sprg3' : _is_bit_set(left, 13),
			'rislb' : _is_bit_set(left, 14),
			'pp' : _is_bit_set(left, 15),

			# Byte 2
			'vpm' : _is_bit_set(left, 16),
			'dss_2.05' : _is_bit_set(left, 17),
			#'reserved' : _is_bit_set(left, 18),
			'dar' : _is_bit_set(left, 19),
			#'reserved' : _is_bit_set(left, 20),
			'ppr' : _is_bit_set(left, 21),
			'dss_2.02' : _is_bit_set(left, 22),
			'dss_2.06' : _is_bit_set(left, 23),

			# Byte 3
			'lsd_in_dscr' : _is_bit_set(left, 24),
			'ugr_in_dscr' : _is_bit_set(left, 25),
			#'reserved' : _is_bit_set(left, 26),
			#'reserved' : _is_bit_set(left, 27),
			#'reserved' : _is_bit_set(left, 28),
			#'reserved' : _is_bit_set(left, 29),
			#'reserved' : _is_bit_set(left, 30),
			#'reserved' : _is_bit_set(left, 31),

			# Byte 4
			'sso_2.06' : _is_bit_set(right, 0),
			#'reserved' : _is_bit_set(right, 1),
			#'reserved' : _is_bit_set(right, 2),
			#'reserved' : _is_bit_set(right, 3),
			#'reserved' : _is_bit_set(right, 4),
			#'reserved' : _is_bit_set(right, 5),
			#'reserved' : _is_bit_set(right, 6),
			#'reserved' : _is_bit_set(right, 7),

			# Byte 5
			'le' : _is_bit_set(right, 8),
			'cfar' : _is_bit_set(right, 9),
			'eb' : _is_bit_set(right, 10),
			'lsq_2.07' : _is_bit_set(right, 11),
			#'reserved' : _is_bit_set(right, 12),
			#'reserved' : _is_bit_set(right, 13),
			#'reserved' : _is_bit_set(right, 14),
			#'reserved' : _is_bit_set(right, 15),

			# Byte 6
			'dss_2.07' : _is_bit_set(right, 16),
			#'reserved' : _is_bit_set(right, 17),
			#'reserved' : _is_bit_set(right, 18),
			#'reserved' : _is_bit_set(right, 19),
			#'reserved' : _is_bit_set(right, 20),
			#'reserved' : _is_bit_set(right, 21),
			#'reserved' : _is_bit_set(right, 22),
			#'reserved' : _is_bit_set(right, 23),

			# Byte 7
			#'reserved' : _is_bit_set(right, 24),
			#'reserved' : _is_bit_set(right, 25),
			#'reserved' : _is_bit_set(right, 26),
			#'reserved' : _is_bit_set(right, 27),
			#'reserved' : _is_bit_set(right, 28),
			#'reserved' : _is_bit_set(right, 29),
			#'reserved' : _is_bit_set(right, 30),
			#'reserved' : _is_bit_set(right, 31),
		}

		# Get a list of only the flags that are true
		flags = [k for k, v in flags.items() if v]
		flags.sort()

		info = {
			'flags' : flags
		}
		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		return {}


def _get_cpu_info_from_cat_var_run_dmesg_boot():
	'''
	Returns the CPU info gathered from /var/run/dmesg.boot.
	Returns {} if dmesg is not found or does not have the desired info.
	'''

	g_trace.header('Tying to get info from the /var/run/dmesg.boot log ...')

	# Just return {} if there is no /var/run/dmesg.boot
	if not DataSource.has_var_run_dmesg_boot():
		g_trace.fail('Failed to find /var/run/dmesg.boot file. Skipping ...')
		return {}

	# If dmesg.boot fails return {}
	returncode, output = DataSource.cat_var_run_dmesg_boot()
	if output == None or returncode != 0:
		g_trace.fail('Failed to run \"cat /var/run/dmesg.boot\". Skipping ...')
		return {}

	info = _parse_dmesg_output(output)
	g_trace.success()
	return info


def _get_cpu_info_from_sysctl():
	'''
	Returns the CPU info gathered from sysctl.
	Returns {} if sysctl is not found.
	'''

	g_trace.header('Tying to get info from sysctl ...')

	try:
		# Just return {} if there is no sysctl
		if not DataSource.has_sysctl():
			g_trace.fail('Failed to find sysctl. Skipping ...')
			return {}

		# If sysctl fails return {}
		returncode, output = DataSource.sysctl_machdep_cpu_hw_cpufrequency()
		if output == None or returncode != 0:
			g_trace.fail('Failed to run \"sysctl machdep.cpu hw.cpufrequency\". Skipping ...')
			return {}

		# Various fields
		vendor_id = _get_field(False, output, None, None, 'machdep.cpu.vendor')
		processor_brand = _get_field(True, output, None, None, 'machdep.cpu.brand_string')
		cache_size = _get_field(False, output, int, 0, 'machdep.cpu.cache.size')
		stepping = _get_field(False, output, int, 0, 'machdep.cpu.stepping')
		model = _get_field(False, output, int, 0, 'machdep.cpu.model')
		family = _get_field(False, output, int, 0, 'machdep.cpu.family')

		# Flags
		flags = _get_field(False, output, None, '', 'machdep.cpu.features').lower().split()
		flags.extend(_get_field(False, output, None, '', 'machdep.cpu.leaf7_features').lower().split())
		flags.extend(_get_field(False, output, None, '', 'machdep.cpu.extfeatures').lower().split())
		flags.sort()

		# Convert from GHz/MHz string to Hz
		hz_advertised, scale = _parse_cpu_brand_string(processor_brand)
		hz_actual = _get_field(False, output, None, None, 'hw.cpufrequency')
		hz_actual = _to_decimal_string(hz_actual)

		info = {
		'vendor_id_raw' : vendor_id,
		'brand_raw' : processor_brand,

		'hz_advertised_friendly' : _hz_short_to_friendly(hz_advertised, scale),
		'hz_actual_friendly' : _hz_short_to_friendly(hz_actual, 0),
		'hz_advertised' : _hz_short_to_full(hz_advertised, scale),
		'hz_actual' : _hz_short_to_full(hz_actual, 0),

		'l2_cache_size' : int(cache_size) * 1024,

		'stepping' : stepping,
		'model' : model,
		'family' : family,
		'flags' : flags
		}

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		return {}


def _get_cpu_info_from_sysinfo():
	'''
	Returns the CPU info gathered from sysinfo.
	Returns {} if sysinfo is not found.
	'''

	info = _get_cpu_info_from_sysinfo_v1()
	info.update(_get_cpu_info_from_sysinfo_v2())
	return info

def _get_cpu_info_from_sysinfo_v1():
	'''
	Returns the CPU info gathered from sysinfo.
	Returns {} if sysinfo is not found.
	'''

	g_trace.header('Tying to get info from sysinfo version 1 ...')

	try:
		# Just return {} if there is no sysinfo
		if not DataSource.has_sysinfo():
			g_trace.fail('Failed to find sysinfo. Skipping ...')
			return {}

		# If sysinfo fails return {}
		returncode, output = DataSource.sysinfo_cpu()
		if output == None or returncode != 0:
			g_trace.fail('Failed to run \"sysinfo -cpu\". Skipping ...')
			return {}

		# Various fields
		vendor_id = '' #_get_field(False, output, None, None, 'CPU #0: ')
		processor_brand = output.split('CPU #0: "')[1].split('"\n')[0].strip()
		cache_size = '' #_get_field(False, output, None, None, 'machdep.cpu.cache.size')
		stepping = int(output.split(', stepping ')[1].split(',')[0].strip())
		model = int(output.split(', model ')[1].split(',')[0].strip())
		family = int(output.split(', family ')[1].split(',')[0].strip())

		# Flags
		flags = []
		for line in output.split('\n'):
			if line.startswith('\t\t'):
				for flag in line.strip().lower().split():
					flags.append(flag)
		flags.sort()

		# Convert from GHz/MHz string to Hz
		hz_advertised, scale = _parse_cpu_brand_string(processor_brand)
		hz_actual = hz_advertised

		info = {
		'vendor_id_raw' : vendor_id,
		'brand_raw' : processor_brand,

		'hz_advertised_friendly' : _hz_short_to_friendly(hz_advertised, scale),
		'hz_actual_friendly' : _hz_short_to_friendly(hz_actual, scale),
		'hz_advertised' : _hz_short_to_full(hz_advertised, scale),
		'hz_actual' : _hz_short_to_full(hz_actual, scale),

		'l2_cache_size' : _to_friendly_bytes(cache_size),

		'stepping' : stepping,
		'model' : model,
		'family' : family,
		'flags' : flags
		}

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		#raise # NOTE: To have this throw on error, uncomment this line
		return {}

def _get_cpu_info_from_sysinfo_v2():
	'''
	Returns the CPU info gathered from sysinfo.
	Returns {} if sysinfo is not found.
	'''

	g_trace.header('Tying to get info from sysinfo version 2 ...')

	try:
		# Just return {} if there is no sysinfo
		if not DataSource.has_sysinfo():
			g_trace.fail('Failed to find sysinfo. Skipping ...')
			return {}

		# If sysinfo fails return {}
		returncode, output = DataSource.sysinfo_cpu()
		if output == None or returncode != 0:
			g_trace.fail('Failed to run \"sysinfo -cpu\". Skipping ...')
			return {}

		# Various fields
		vendor_id = '' #_get_field(False, output, None, None, 'CPU #0: ')
		processor_brand = output.split('CPU #0: "')[1].split('"\n')[0].strip()
		cache_size = '' #_get_field(False, output, None, None, 'machdep.cpu.cache.size')
		signature = output.split('Signature:')[1].split('\n')[0].strip()
		#
		stepping = int(signature.split('stepping ')[1].split(',')[0].strip())
		model = int(signature.split('model ')[1].split(',')[0].strip())
		family = int(signature.split('family ')[1].split(',')[0].strip())

		# Flags
		def get_subsection_flags(output):
			retval = []
			for line in output.split('\n')[1:]:
				if not line.startswith('                ') and not line.startswith('		'): break
				for entry in line.strip().lower().split(' '):
					retval.append(entry)
			return retval

		flags = get_subsection_flags(output.split('Features: ')[1]) + \
				get_subsection_flags(output.split('Extended Features (0x00000001): ')[1]) + \
				get_subsection_flags(output.split('Extended Features (0x80000001): ')[1])
		flags.sort()

		# Convert from GHz/MHz string to Hz
		lines = [n for n in output.split('\n') if n]
		raw_hz = lines[0].split('running at ')[1].strip().lower()
		hz_advertised = raw_hz.rstrip('mhz').rstrip('ghz').strip()
		hz_advertised = _to_decimal_string(hz_advertised)
		hz_actual = hz_advertised

		scale = 0
		if raw_hz.endswith('mhz'):
			scale = 6
		elif raw_hz.endswith('ghz'):
			scale = 9

		info = {
		'vendor_id_raw' : vendor_id,
		'brand_raw' : processor_brand,

		'hz_advertised_friendly' : _hz_short_to_friendly(hz_advertised, scale),
		'hz_actual_friendly' : _hz_short_to_friendly(hz_actual, scale),
		'hz_advertised' : _hz_short_to_full(hz_advertised, scale),
		'hz_actual' : _hz_short_to_full(hz_actual, scale),

		'l2_cache_size' : _to_friendly_bytes(cache_size),

		'stepping' : stepping,
		'model' : model,
		'family' : family,
		'flags' : flags
		}

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		#raise # NOTE: To have this throw on error, uncomment this line
		return {}

def _get_cpu_info_from_wmic():
	'''
	Returns the CPU info gathered from WMI.
	Returns {} if not on Windows, or wmic is not installed.
	'''
	g_trace.header('Tying to get info from wmic ...')

	try:
		# Just return {} if not Windows or there is no wmic
		if not DataSource.is_windows or not DataSource.has_wmic():
			g_trace.fail('Failed to find WMIC, or not on Windows. Skipping ...')
			return {}

		returncode, output = DataSource.wmic_cpu()
		if output == None or returncode != 0:
			g_trace.fail('Failed to run wmic. Skipping ...')
			return {}

		# Break the list into key values pairs
		value = output.split("\n")
		value = [s.rstrip().split('=') for s in value if '=' in s]
		value = {k: v for k, v in value if v}

		# Get the advertised MHz
		processor_brand = value.get('Name')
		hz_advertised, scale_advertised = _parse_cpu_brand_string(processor_brand)

		# Get the actual MHz
		hz_actual = value.get('CurrentClockSpeed')
		scale_actual = 6
		if hz_actual:
			hz_actual = _to_decimal_string(hz_actual)

		# Get cache sizes
		l2_cache_size = value.get('L2CacheSize') # NOTE: L2CacheSize is in kilobytes
		if l2_cache_size:
			l2_cache_size = int(l2_cache_size) * 1024

		l3_cache_size = value.get('L3CacheSize') # NOTE: L3CacheSize is in kilobytes
		if l3_cache_size:
			l3_cache_size = int(l3_cache_size) * 1024

		# Get family, model, and stepping
		family, model, stepping = '', '', ''
		description = value.get('Description') or value.get('Caption')
		entries = description.split(' ')

		if 'Family' in entries and entries.index('Family') < len(entries)-1:
			i = entries.index('Family')
			family = int(entries[i + 1])

		if 'Model' in entries and entries.index('Model') < len(entries)-1:
			i = entries.index('Model')
			model = int(entries[i + 1])

		if 'Stepping' in entries and entries.index('Stepping') < len(entries)-1:
			i = entries.index('Stepping')
			stepping = int(entries[i + 1])

		info = {
			'vendor_id_raw' : value.get('Manufacturer'),
			'brand_raw' : processor_brand,

			'hz_advertised_friendly' : _hz_short_to_friendly(hz_advertised, scale_advertised),
			'hz_actual_friendly' : _hz_short_to_friendly(hz_actual, scale_actual),
			'hz_advertised' : _hz_short_to_full(hz_advertised, scale_advertised),
			'hz_actual' : _hz_short_to_full(hz_actual, scale_actual),

			'l2_cache_size' : l2_cache_size,
			'l3_cache_size' : l3_cache_size,

			'stepping' : stepping,
			'model' : model,
			'family' : family,
		}

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		#raise # NOTE: To have this throw on error, uncomment this line
		return {}

def _get_cpu_info_from_registry():
	'''
	Returns the CPU info gathered from the Windows Registry.
	Returns {} if not on Windows.
	'''

	g_trace.header('Tying to get info from Windows registry ...')

	try:
		# Just return {} if not on Windows
		if not DataSource.is_windows:
			g_trace.fail('Not running on Windows. Skipping ...')
			return {}

		# Get the CPU name
		processor_brand = DataSource.winreg_processor_brand().strip()

		# Get the CPU vendor id
		vendor_id = DataSource.winreg_vendor_id_raw()

		# Get the CPU arch and bits
		arch_string_raw = DataSource.winreg_arch_string_raw()
		arch, bits = _parse_arch(arch_string_raw)

		# Get the actual CPU Hz
		hz_actual = DataSource.winreg_hz_actual()
		hz_actual = _to_decimal_string(hz_actual)

		# Get the advertised CPU Hz
		hz_advertised, scale = _parse_cpu_brand_string(processor_brand)

		# If advertised hz not found, use the actual hz
		if hz_advertised == '0.0':
			scale = 6
			hz_advertised = _to_decimal_string(hz_actual)

		# Get the CPU features
		feature_bits = DataSource.winreg_feature_bits()

		def is_set(bit):
			mask = 0x80000000 >> bit
			retval = mask & feature_bits > 0
			return retval

		# http://en.wikipedia.org/wiki/CPUID
		# http://unix.stackexchange.com/questions/43539/what-do-the-flags-in-proc-cpuinfo-mean
		# http://www.lohninger.com/helpcsuite/public_constants_cpuid.htm
		flags = {
			'fpu' : is_set(0), # Floating Point Unit
			'vme' : is_set(1), # V86 Mode Extensions
			'de' : is_set(2), # Debug Extensions - I/O breakpoints supported
			'pse' : is_set(3), # Page Size Extensions (4 MB pages supported)
			'tsc' : is_set(4), # Time Stamp Counter and RDTSC instruction are available
			'msr' : is_set(5), # Model Specific Registers
			'pae' : is_set(6), # Physical Address Extensions (36 bit address, 2MB pages)
			'mce' : is_set(7), # Machine Check Exception supported
			'cx8' : is_set(8), # Compare Exchange Eight Byte instruction available
			'apic' : is_set(9), # Local APIC present (multiprocessor operation support)
			'sepamd' : is_set(10), # Fast system calls (AMD only)
			'sep' : is_set(11), # Fast system calls
			'mtrr' : is_set(12), # Memory Type Range Registers
			'pge' : is_set(13), # Page Global Enable
			'mca' : is_set(14), # Machine Check Architecture
			'cmov' : is_set(15), # Conditional MOVe instructions
			'pat' : is_set(16), # Page Attribute Table
			'pse36' : is_set(17), # 36 bit Page Size Extensions
			'serial' : is_set(18), # Processor Serial Number
			'clflush' : is_set(19), # Cache Flush
			#'reserved1' : is_set(20), # reserved
			'dts' : is_set(21), # Debug Trace Store
			'acpi' : is_set(22), # ACPI support
			'mmx' : is_set(23), # MultiMedia Extensions
			'fxsr' : is_set(24), # FXSAVE and FXRSTOR instructions
			'sse' : is_set(25), # SSE instructions
			'sse2' : is_set(26), # SSE2 (WNI) instructions
			'ss' : is_set(27), # self snoop
			#'reserved2' : is_set(28), # reserved
			'tm' : is_set(29), # Automatic clock control
			'ia64' : is_set(30), # IA64 instructions
			'3dnow' : is_set(31) # 3DNow! instructions available
		}

		# Get a list of only the flags that are true
		flags = [k for k, v in flags.items() if v]
		flags.sort()

		info = {
		'vendor_id_raw' : vendor_id,
		'brand_raw' : processor_brand,

		'hz_advertised_friendly' : _hz_short_to_friendly(hz_advertised, scale),
		'hz_actual_friendly' : _hz_short_to_friendly(hz_actual, 6),
		'hz_advertised' : _hz_short_to_full(hz_advertised, scale),
		'hz_actual' : _hz_short_to_full(hz_actual, 6),

		'flags' : flags
		}

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		return {}

def _get_cpu_info_from_kstat():
	'''
	Returns the CPU info gathered from isainfo and kstat.
	Returns {} if isainfo or kstat are not found.
	'''

	g_trace.header('Tying to get info from kstat ...')

	try:
		# Just return {} if there is no isainfo or kstat
		if not DataSource.has_isainfo() or not DataSource.has_kstat():
			g_trace.fail('Failed to find isinfo or kstat. Skipping ...')
			return {}

		# If isainfo fails return {}
		returncode, flag_output = DataSource.isainfo_vb()
		if flag_output == None or returncode != 0:
			g_trace.fail('Failed to run \"isainfo -vb\". Skipping ...')
			return {}

		# If kstat fails return {}
		returncode, kstat = DataSource.kstat_m_cpu_info()
		if kstat == None or returncode != 0:
			g_trace.fail('Failed to run \"kstat -m cpu_info\". Skipping ...')
			return {}

		# Various fields
		vendor_id = kstat.split('\tvendor_id ')[1].split('\n')[0].strip()
		processor_brand = kstat.split('\tbrand ')[1].split('\n')[0].strip()
		stepping = int(kstat.split('\tstepping ')[1].split('\n')[0].strip())
		model = int(kstat.split('\tmodel ')[1].split('\n')[0].strip())
		family = int(kstat.split('\tfamily ')[1].split('\n')[0].strip())

		# Flags
		flags = flag_output.strip().split('\n')[-1].strip().lower().split()
		flags.sort()

		# Convert from GHz/MHz string to Hz
		scale = 6
		hz_advertised = kstat.split('\tclock_MHz ')[1].split('\n')[0].strip()
		hz_advertised = _to_decimal_string(hz_advertised)

		# Convert from GHz/MHz string to Hz
		hz_actual = kstat.split('\tcurrent_clock_Hz ')[1].split('\n')[0].strip()
		hz_actual = _to_decimal_string(hz_actual)

		info = {
		'vendor_id_raw' : vendor_id,
		'brand_raw' : processor_brand,

		'hz_advertised_friendly' : _hz_short_to_friendly(hz_advertised, scale),
		'hz_actual_friendly' : _hz_short_to_friendly(hz_actual, 0),
		'hz_advertised' : _hz_short_to_full(hz_advertised, scale),
		'hz_actual' : _hz_short_to_full(hz_actual, 0),

		'stepping' : stepping,
		'model' : model,
		'family' : family,
		'flags' : flags
		}

		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		return {}

def _get_cpu_info_from_platform_uname():

	g_trace.header('Tying to get info from platform.uname ...')

	try:
		uname = DataSource.uname_string_raw.split(',')[0]

		family, model, stepping = (None, None, None)
		entries = uname.split(' ')

		if 'Family' in entries and entries.index('Family') < len(entries)-1:
			i = entries.index('Family')
			family = int(entries[i + 1])

		if 'Model' in entries and entries.index('Model') < len(entries)-1:
			i = entries.index('Model')
			model = int(entries[i + 1])

		if 'Stepping' in entries and entries.index('Stepping') < len(entries)-1:
			i = entries.index('Stepping')
			stepping = int(entries[i + 1])

		info = {
			'family' : family,
			'model' : model,
			'stepping' : stepping
		}
		info = _filter_dict_keys_with_empty_values(info)
		g_trace.success()
		return info
	except Exception as err:
		g_trace.fail(err)
		return {}

def _get_cpu_info_internal():
	'''
	Returns the CPU info by using the best sources of information for your OS.
	Returns {} if nothing is found.
	'''

	g_trace.write('!' * 80)

	# Get the CPU arch and bits
	arch, bits = _parse_arch(DataSource.arch_string_raw)

	friendly_maxsize = { 2**31-1: '32 bit', 2**63-1: '64 bit' }.get(sys.maxsize) or 'unknown bits'
	friendly_version = "{0}.{1}.{2}.{3}.{4}".format(*sys.version_info)
	PYTHON_VERSION = "{0} ({1})".format(friendly_version, friendly_maxsize)

	info = {
		'python_version' : PYTHON_VERSION,
		'cpuinfo_version' : CPUINFO_VERSION,
		'cpuinfo_version_string' : CPUINFO_VERSION_STRING,
		'arch' : arch,
		'bits' : bits,
		'count' : DataSource.cpu_count,
		'arch_string_raw' : DataSource.arch_string_raw,
	}

	g_trace.write("python_version: {0}".format(info['python_version']))
	g_trace.write("cpuinfo_version: {0}".format(info['cpuinfo_version']))
	g_trace.write("arch: {0}".format(info['arch']))
	g_trace.write("bits: {0}".format(info['bits']))
	g_trace.write("count: {0}".format(info['count']))
	g_trace.write("arch_string_raw: {0}".format(info['arch_string_raw']))

	# Try the Windows wmic
	_copy_new_fields(info, _get_cpu_info_from_wmic())

	# Try the Windows registry
	_copy_new_fields(info, _get_cpu_info_from_registry())

	# Try /proc/cpuinfo
	_copy_new_fields(info, _get_cpu_info_from_proc_cpuinfo())

	# Try cpufreq-info
	_copy_new_fields(info, _get_cpu_info_from_cpufreq_info())

	# Try LSCPU
	_copy_new_fields(info, _get_cpu_info_from_lscpu())

	# Try sysctl
	_copy_new_fields(info, _get_cpu_info_from_sysctl())

	# Try kstat
	_copy_new_fields(info, _get_cpu_info_from_kstat())

	# Try dmesg
	_copy_new_fields(info, _get_cpu_info_from_dmesg())

	# Try /var/run/dmesg.boot
	_copy_new_fields(info, _get_cpu_info_from_cat_var_run_dmesg_boot())

	# Try lsprop ibm,pa-features
	_copy_new_fields(info, _get_cpu_info_from_ibm_pa_features())

	# Try sysinfo
	_copy_new_fields(info, _get_cpu_info_from_sysinfo())

	# Try querying the CPU cpuid register
	# FIXME: This should print stdout and stderr to trace log
	_copy_new_fields(info, _get_cpu_info_from_cpuid())

	# Try platform.uname
	_copy_new_fields(info, _get_cpu_info_from_platform_uname())

	g_trace.write('!' * 80)

	return info

def get_cpu_info_json():
	'''
	Returns the CPU info by using the best sources of information for your OS.
	Returns the result in a json string
	'''

	import json

	output = None

	# If running under pyinstaller, run normally
	if getattr(sys, 'frozen', False):
		info = _get_cpu_info_internal()
		output = json.dumps(info)
		output = "{0}".format(output)
	# if not running under pyinstaller, run in another process.
	# This is done because multiprocesing has a design flaw that
	# causes non main programs to run multiple times on Windows.
	else:
		from subprocess import Popen, PIPE

		command = [sys.executable, __file__, '--json']
		p1 = Popen(command, stdout=PIPE, stderr=PIPE, stdin=PIPE)
		output = p1.communicate()[0]

		if p1.returncode != 0:
			return "{}"

		if not IS_PY2:
			output = output.decode(encoding='UTF-8')

	return output

def get_cpu_info():
	'''
	Returns the CPU info by using the best sources of information for your OS.
	Returns the result in a dict
	'''

	import json

	output = get_cpu_info_json()

	# Convert JSON to Python with non unicode strings
	output = json.loads(output, object_hook = _utf_to_str)

	return output

def main():
	from argparse import ArgumentParser
	import json

	# Parse args
	parser = ArgumentParser(description='Gets CPU info with pure Python 2 & 3')
	parser.add_argument('--json', action='store_true', help='Return the info in JSON format')
	parser.add_argument('--version', action='store_true', help='Return the version of py-cpuinfo')
	parser.add_argument('--trace', action='store_true', help='Traces code paths used to find CPU info to file')
	args = parser.parse_args()

	global g_trace
	g_trace = Trace(args.trace, False)

	try:
		_check_arch()
	except Exception as err:
		sys.stderr.write(str(err) + "\n")
		sys.exit(1)

	info = _get_cpu_info_internal()

	if not info:
		sys.stderr.write("Failed to find cpu info\n")
		sys.exit(1)

	if args.json:
		print(json.dumps(info))
	elif args.version:
		print(CPUINFO_VERSION_STRING)
	else:
		print('Python Version: {0}'.format(info.get('python_version', '')))
		print('Cpuinfo Version: {0}'.format(info.get('cpuinfo_version_string', '')))
		print('Vendor ID Raw: {0}'.format(info.get('vendor_id_raw', '')))
		print('Hardware Raw: {0}'.format(info.get('hardware_raw', '')))
		print('Brand Raw: {0}'.format(info.get('brand_raw', '')))
		print('Hz Advertised Friendly: {0}'.format(info.get('hz_advertised_friendly', '')))
		print('Hz Actual Friendly: {0}'.format(info.get('hz_actual_friendly', '')))
		print('Hz Advertised: {0}'.format(info.get('hz_advertised', '')))
		print('Hz Actual: {0}'.format(info.get('hz_actual', '')))
		print('Arch: {0}'.format(info.get('arch', '')))
		print('Bits: {0}'.format(info.get('bits', '')))
		print('Count: {0}'.format(info.get('count', '')))
		print('Arch String Raw: {0}'.format(info.get('arch_string_raw', '')))
		print('L1 Data Cache Size: {0}'.format(info.get('l1_data_cache_size', '')))
		print('L1 Instruction Cache Size: {0}'.format(info.get('l1_instruction_cache_size', '')))
		print('L2 Cache Size: {0}'.format(info.get('l2_cache_size', '')))
		print('L2 Cache Line Size: {0}'.format(info.get('l2_cache_line_size', '')))
		print('L2 Cache Associativity: {0}'.format(info.get('l2_cache_associativity', '')))
		print('L3 Cache Size: {0}'.format(info.get('l3_cache_size', '')))
		print('Stepping: {0}'.format(info.get('stepping', '')))
		print('Model: {0}'.format(info.get('model', '')))
		print('Family: {0}'.format(info.get('family', '')))
		print('Processor Type: {0}'.format(info.get('processor_type', '')))
		print('Flags: {0}'.format(', '.join(info.get('flags', ''))))


if __name__ == '__main__':
	main()
else:
	g_trace = Trace(False, False)
	_check_arch()
