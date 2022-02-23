#
#   Module parse tree node
#

from __future__ import absolute_import

import cython
cython.declare(Naming=object, Options=object, PyrexTypes=object, TypeSlots=object,
               error=object, warning=object, py_object_type=object, UtilityCode=object,
               EncodedString=object, re=object)

from collections import defaultdict
import json
import operator
import os
import re

from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran

from .Errors import error, warning
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version
from .Code import UtilityCode, IncludeCode
from .StringEncoding import EncodedString
from .Pythran import has_np_pythran

def check_c_declarations_pxd(module_node):
    module_node.scope.check_c_classes_pxd()
    return module_node


def check_c_declarations(module_node):
    module_node.scope.check_c_classes()
    module_node.scope.check_c_functions()
    return module_node


def generate_c_code_config(env, options):
    if Options.annotate or options.annotate:
        emit_linenums = False
    else:
        emit_linenums = options.emit_linenums

    return Code.CCodeConfig(
        emit_linenums=emit_linenums,
        emit_code_comments=env.directives['emit_code_comments'],
        c_line_in_traceback=options.c_line_in_traceback)


class ModuleNode(Nodes.Node, Nodes.BlockNode):
    #  doc       string or None
    #  body      StatListNode
    #
    #  referenced_modules   [ModuleScope]
    #  full_module_name     string
    #
    #  scope                The module scope.
    #  compilation_source   A CompilationSource (see Main)
    #  directives           Top-level compiler directives

    child_attrs = ["body"]
    directives = None

    def merge_in(self, tree, scope, merge_scope=False):
        # Merges in the contents of another tree, and possibly scope. With the
        # current implementation below, this must be done right prior
        # to code generation.
        #
        # Note: This way of doing it seems strange -- I believe the
        # right concept is to split ModuleNode into a ModuleNode and a
        # CodeGenerator, and tell that CodeGenerator to generate code
        # from multiple sources.
        assert isinstance(self.body, Nodes.StatListNode)
        if isinstance(tree, Nodes.StatListNode):
            self.body.stats.extend(tree.stats)
        else:
            self.body.stats.append(tree)

        self.scope.utility_code_list.extend(scope.utility_code_list)

        for inc in scope.c_includes.values():
            self.scope.process_include(inc)

        def extend_if_not_in(L1, L2):
            for x in L2:
                if x not in L1:
                    L1.append(x)

        extend_if_not_in(self.scope.included_files, scope.included_files)

        if merge_scope:
            # Ensure that we don't generate import code for these entries!
            for entry in scope.c_class_entries:
                entry.type.module_name = self.full_module_name
                entry.type.scope.directives["internal"] = True

            self.scope.merge_in(scope)

    def analyse_declarations(self, env):
        if has_np_pythran(env):
            Pythran.include_pythran_generic(env)
        if self.directives:
            env.old_style_globals = self.directives['old_style_globals']
        if not Options.docstrings:
            env.doc = self.doc = None
        elif Options.embed_pos_in_docstring:
            env.doc = EncodedString(u'File: %s (starting at line %s)' % Nodes.relative_position(self.pos))
            if self.doc is not None:
                env.doc = EncodedString(env.doc + u'\n' + self.doc)
                env.doc.encoding = self.doc.encoding
        else:
            env.doc = self.doc
        env.directives = self.directives

        self.body.analyse_declarations(env)

    def prepare_utility_code(self):
        # prepare any utility code that must be created before code generation
        # specifically: CythonUtilityCode
        env = self.scope
        if env.has_import_star:
            self.create_import_star_conversion_utility_code(env)
        for name, entry in sorted(env.entries.items()):
            if (entry.create_wrapper and entry.scope is env
                and entry.is_type and entry.type.is_enum):
                    entry.type.create_type_wrapper(env)

    def process_implementation(self, options, result):
        env = self.scope
        env.return_type = PyrexTypes.c_void_type
        self.referenced_modules = []
        self.find_referenced_modules(env, self.referenced_modules, {})
        self.sort_cdef_classes(env)
        self.generate_c_code(env, options, result)
        self.generate_h_code(env, options, result)
        self.generate_api_code(env, options, result)

    def has_imported_c_functions(self):
        for module in self.referenced_modules:
            for entry in module.cfunc_entries:
                if entry.defined_in_pxd:
                    return 1
        return 0

    def generate_h_code(self, env, options, result):
        def h_entries(entries, api=0, pxd=0):
            return [entry for entry in entries
                    if ((entry.visibility == 'public') or
                        (api and entry.api) or
                        (pxd and entry.defined_in_pxd))]
        h_types = h_entries(env.type_entries, api=1)
        h_vars = h_entries(env.var_entries)
        h_funcs = h_entries(env.cfunc_entries)
        h_extension_types = h_entries(env.c_class_entries)
        if h_types or  h_vars or h_funcs or h_extension_types:
            result.h_file = replace_suffix(result.c_file, ".h")
            h_code = Code.CCodeWriter()
            c_code_config = generate_c_code_config(env, options)
            Code.GlobalState(h_code, self, c_code_config)
            if options.generate_pxi:
                result.i_file = replace_suffix(result.c_file, ".pxi")
                i_code = Code.PyrexCodeWriter(result.i_file)
            else:
                i_code = None

            h_code.put_generated_by()
            h_guard = Naming.h_guard_prefix + self.api_name(env)
            h_code.put_h_guard(h_guard)
            h_code.putln("")
            h_code.putln('#include "Python.h"')
            self.generate_type_header_code(h_types, h_code)
            if options.capi_reexport_cincludes:
                self.generate_includes(env, [], h_code)
            h_code.putln("")
            api_guard = Naming.api_guard_prefix + self.api_name(env)
            h_code.putln("#ifndef %s" % api_guard)
            h_code.putln("")
            self.generate_extern_c_macro_definition(h_code)
            h_code.putln("")
            self.generate_dl_import_macro(h_code)
            if h_extension_types:
                h_code.putln("")
                for entry in h_extension_types:
                    self.generate_cclass_header_code(entry.type, h_code)
                    if i_code:
                        self.generate_cclass_include_code(entry.type, i_code)
            if h_funcs:
                h_code.putln("")
                for entry in h_funcs:
                    self.generate_public_declaration(entry, h_code, i_code)
            if h_vars:
                h_code.putln("")
                for entry in h_vars:
                    self.generate_public_declaration(entry, h_code, i_code)
            h_code.putln("")
            h_code.putln("#endif /* !%s */" % api_guard)
            h_code.putln("")
            h_code.putln("/* WARNING: the interface of the module init function changed in CPython 3.5. */")
            h_code.putln("/* It now returns a PyModuleDef instance instead of a PyModule instance. */")
            h_code.putln("")
            h_code.putln("#if PY_MAJOR_VERSION < 3")
            h_code.putln("PyMODINIT_FUNC init%s(void);" % env.module_name)
            h_code.putln("#else")
            h_code.putln("PyMODINIT_FUNC %s(void);" % self.mod_init_func_cname('PyInit', env))
            h_code.putln("#endif")
            h_code.putln("")
            h_code.putln("#endif /* !%s */" % h_guard)

            f = open_new_file(result.h_file)
            try:
                h_code.copyto(f)
            finally:
                f.close()

    def generate_public_declaration(self, entry, h_code, i_code):
        h_code.putln("%s %s;" % (
            Naming.extern_c_macro,
            entry.type.declaration_code(entry.cname)))
        if i_code:
            i_code.putln("cdef extern %s" % (
                entry.type.declaration_code(entry.cname, pyrex=1)))

    def api_name(self, env):
        return env.qualified_name.replace(".", "__")

    def generate_api_code(self, env, options, result):
        def api_entries(entries, pxd=0):
            return [entry for entry in entries
                    if entry.api or (pxd and entry.defined_in_pxd)]
        api_vars = api_entries(env.var_entries)
        api_funcs = api_entries(env.cfunc_entries)
        api_extension_types = api_entries(env.c_class_entries)
        if api_vars or api_funcs or api_extension_types:
            result.api_file = replace_suffix(result.c_file, "_api.h")
            h_code = Code.CCodeWriter()
            c_code_config = generate_c_code_config(env, options)
            Code.GlobalState(h_code, self, c_code_config)
            h_code.put_generated_by()
            api_guard = Naming.api_guard_prefix + self.api_name(env)
            h_code.put_h_guard(api_guard)
            # Work around https://bugs.python.org/issue4709
            h_code.putln('#ifdef __MINGW64__')
            h_code.putln('#define MS_WIN64')
            h_code.putln('#endif')

            h_code.putln('#include "Python.h"')
            if result.h_file:
                h_code.putln('#include "%s"' % os.path.basename(result.h_file))
            if api_extension_types:
                h_code.putln("")
                for entry in api_extension_types:
                    type = entry.type
                    h_code.putln("static PyTypeObject *%s = 0;" % type.typeptr_cname)
                    h_code.putln("#define %s (*%s)" % (
                        type.typeobj_cname, type.typeptr_cname))
            if api_funcs:
                h_code.putln("")
                for entry in api_funcs:
                    type = CPtrType(entry.type)
                    cname = env.mangle(Naming.func_prefix_api, entry.name)
                    h_code.putln("static %s = 0;" % type.declaration_code(cname))
                    h_code.putln("#define %s %s" % (entry.name, cname))
            if api_vars:
                h_code.putln("")
                for entry in api_vars:
                    type = CPtrType(entry.type)
                    cname = env.mangle(Naming.varptr_prefix_api, entry.name)
                    h_code.putln("static %s = 0;" %  type.declaration_code(cname))
                    h_code.putln("#define %s (*%s)" % (entry.name, cname))
            h_code.put(UtilityCode.load_as_string("PyIdentifierFromString", "ImportExport.c")[0])
            if api_vars:
                h_code.put(UtilityCode.load_as_string("VoidPtrImport", "ImportExport.c")[1])
            if api_funcs:
                h_code.put(UtilityCode.load_as_string("FunctionImport", "ImportExport.c")[1])
            if api_extension_types:
                h_code.put(UtilityCode.load_as_string("TypeImport", "ImportExport.c")[0])
                h_code.put(UtilityCode.load_as_string("TypeImport", "ImportExport.c")[1])
            h_code.putln("")
            h_code.putln("static int import_%s(void) {" % self.api_name(env))
            h_code.putln("PyObject *module = 0;")
            h_code.putln('module = PyImport_ImportModule("%s");' % env.qualified_name)
            h_code.putln("if (!module) goto bad;")
            for entry in api_funcs:
                cname = env.mangle(Naming.func_prefix_api, entry.name)
                sig = entry.type.signature_string()
                h_code.putln(
                    'if (__Pyx_ImportFunction(module, "%s", (void (**)(void))&%s, "%s") < 0) goto bad;'
                    % (entry.name, cname, sig))
            for entry in api_vars:
                cname = env.mangle(Naming.varptr_prefix_api, entry.name)
                sig = entry.type.empty_declaration_code()
                h_code.putln(
                    'if (__Pyx_ImportVoidPtr(module, "%s", (void **)&%s, "%s") < 0) goto bad;'
                    % (entry.name, cname, sig))
            with ModuleImportGenerator(h_code, imported_modules={env.qualified_name: 'module'}) as import_generator:
                for entry in api_extension_types:
                    self.generate_type_import_call(entry.type, h_code, import_generator, error_code="goto bad;")
            h_code.putln("Py_DECREF(module); module = 0;")
            h_code.putln("return 0;")
            h_code.putln("bad:")
            h_code.putln("Py_XDECREF(module);")
            h_code.putln("return -1;")
            h_code.putln("}")
            h_code.putln("")
            h_code.putln("#endif /* !%s */" % api_guard)

            f = open_new_file(result.api_file)
            try:
                h_code.copyto(f)
            finally:
                f.close()

    def generate_cclass_header_code(self, type, h_code):
        h_code.putln("%s %s %s;" % (
            Naming.extern_c_macro,
            PyrexTypes.public_decl("PyTypeObject", "DL_IMPORT"),
            type.typeobj_cname))

    def generate_cclass_include_code(self, type, i_code):
        i_code.putln("cdef extern class %s.%s:" % (
            type.module_name, type.name))
        i_code.indent()
        var_entries = type.scope.var_entries
        if var_entries:
            for entry in var_entries:
                i_code.putln("cdef %s" % (
                    entry.type.declaration_code(entry.cname, pyrex=1)))
        else:
            i_code.putln("pass")
        i_code.dedent()

    def generate_c_code(self, env, options, result):
        modules = self.referenced_modules

        if Options.annotate or options.annotate:
            rootwriter = Annotate.AnnotationCCodeWriter()
        else:
            rootwriter = Code.CCodeWriter()

        c_code_config = generate_c_code_config(env, options)

        globalstate = Code.GlobalState(
            rootwriter, self,
            code_config=c_code_config,
            common_utility_include_dir=options.common_utility_include_dir,
        )
        globalstate.initialize_main_c_code()
        h_code = globalstate['h_code']

        self.generate_module_preamble(env, options, modules, result.embedded_metadata, h_code)

        globalstate.module_pos = self.pos
        globalstate.directives = self.directives

        globalstate.use_utility_code(refnanny_utility_code)

        code = globalstate['before_global_var']
        code.putln('#define __Pyx_MODULE_NAME "%s"' % self.full_module_name)
        module_is_main = "%s%s" % (Naming.module_is_main, self.full_module_name.replace('.', '__'))
        code.putln("extern int %s;" % module_is_main)
        code.putln("int %s = 0;" % module_is_main)
        code.putln("")
        code.putln("/* Implementation of '%s' */" % env.qualified_name)

        code = globalstate['late_includes']
        code.putln("/* Late includes */")
        self.generate_includes(env, modules, code, early=False)

        code = globalstate['all_the_rest']

        self.generate_cached_builtins_decls(env, code)
        self.generate_lambda_definitions(env, code)
        # generate normal variable and function definitions
        self.generate_variable_definitions(env, code)

        self.body.generate_function_definitions(env, code)

        code.mark_pos(None)
        self.generate_typeobj_definitions(env, code)
        self.generate_method_table(env, code)
        if env.has_import_star:
            self.generate_import_star(env, code)
        self.generate_pymoduledef_struct(env, code)

        # initialise the macro to reduce the code size of one-time functionality
        code.putln(UtilityCode.load_as_string("SmallCodeConfig", "ModuleSetupCode.c")[0].strip())

        # init_globals is inserted before this
        self.generate_module_init_func(modules[:-1], env, globalstate['init_module'])
        self.generate_module_cleanup_func(env, globalstate['cleanup_module'])
        if Options.embed:
            self.generate_main_method(env, globalstate['main_method'])
        self.generate_filename_table(globalstate['filename_table'])

        self.generate_declarations_for_modules(env, modules, globalstate)
        h_code.write('\n')

        for utilcode in env.utility_code_list[:]:
            globalstate.use_utility_code(utilcode)
        globalstate.finalize_main_c_code()

        f = open_new_file(result.c_file)
        try:
            rootwriter.copyto(f)
        finally:
            f.close()
        result.c_file_generated = 1
        if options.gdb_debug:
            self._serialize_lineno_map(env, rootwriter)
        if Options.annotate or options.annotate:
            self._generate_annotations(rootwriter, result, options)

    def _generate_annotations(self, rootwriter, result, options):
        self.annotate(rootwriter)

        coverage_xml_filename = Options.annotate_coverage_xml or options.annotate_coverage_xml
        if coverage_xml_filename and os.path.exists(coverage_xml_filename):
            try:
                import xml.etree.cElementTree as ET
            except ImportError:
                import xml.etree.ElementTree as ET
            coverage_xml = ET.parse(coverage_xml_filename).getroot()
            if hasattr(coverage_xml, 'iter'):
                iterator = coverage_xml.iter()  # Python 2.7 & 3.2+
            else:
                iterator = coverage_xml.getiterator()
            for el in iterator:
                el.tail = None  # save some memory
        else:
            coverage_xml = None

        rootwriter.save_annotation(result.main_source_file, result.c_file, coverage_xml=coverage_xml)

        # if we included files, additionally generate one annotation file for each
        if not self.scope.included_files:
            return

        search_include_file = self.scope.context.search_include_directories
        target_dir = os.path.abspath(os.path.dirname(result.c_file))
        for included_file in self.scope.included_files:
            target_file = os.path.abspath(os.path.join(target_dir, included_file))
            target_file_dir = os.path.dirname(target_file)
            if not target_file_dir.startswith(target_dir):
                # any other directories may not be writable => avoid trying
                continue
            source_file = search_include_file(included_file, "", self.pos, include=True)
            if not source_file:
                continue
            if target_file_dir != target_dir and not os.path.exists(target_file_dir):
                try:
                    os.makedirs(target_file_dir)
                except OSError as e:
                    import errno
                    if e.errno != errno.EEXIST:
                        raise
            rootwriter.save_annotation(source_file, target_file, coverage_xml=coverage_xml)

    def _serialize_lineno_map(self, env, ccodewriter):
        tb = env.context.gdb_debug_outputwriter
        markers = ccodewriter.buffer.allmarkers()

        d = defaultdict(list)
        for c_lineno, cython_lineno in enumerate(markers):
            if cython_lineno > 0:
                d[cython_lineno].append(c_lineno + 1)

        tb.start('LineNumberMapping')
        for cython_lineno, c_linenos in sorted(d.items()):
            tb.add_entry(
                'LineNumber',
                c_linenos=' '.join(map(str, c_linenos)),
                cython_lineno=str(cython_lineno),
            )
        tb.end('LineNumberMapping')
        tb.serialize()

    def find_referenced_modules(self, env, module_list, modules_seen):
        if env not in modules_seen:
            modules_seen[env] = 1
            for imported_module in env.cimported_modules:
                self.find_referenced_modules(imported_module, module_list, modules_seen)
            module_list.append(env)

    def sort_types_by_inheritance(self, type_dict, type_order, getkey):
        # copy the types into a list moving each parent type before
        # its first child
        type_list = []
        for i, key in enumerate(type_order):
            new_entry = type_dict[key]

            # collect all base classes to check for children
            hierarchy = set()
            base = new_entry
            while base:
                base_type = base.type.base_type
                if not base_type:
                    break
                base_key = getkey(base_type)
                hierarchy.add(base_key)
                base = type_dict.get(base_key)
            new_entry.base_keys = hierarchy

            # find the first (sub-)subclass and insert before that
            for j in range(i):
                entry = type_list[j]
                if key in entry.base_keys:
                    type_list.insert(j, new_entry)
                    break
            else:
                type_list.append(new_entry)
        return type_list

    def sort_type_hierarchy(self, module_list, env):
        # poor developer's OrderedDict
        vtab_dict, vtab_dict_order = {}, []
        vtabslot_dict, vtabslot_dict_order = {}, []

        for module in module_list:
            for entry in module.c_class_entries:
                if entry.used and not entry.in_cinclude:
                    type = entry.type
                    key = type.vtabstruct_cname
                    if not key:
                        continue
                    if key in vtab_dict:
                        # FIXME: this should *never* happen, but apparently it does
                        # for Cython generated utility code
                        from .UtilityCode import NonManglingModuleScope
                        assert isinstance(entry.scope, NonManglingModuleScope), str(entry.scope)
                        assert isinstance(vtab_dict[key].scope, NonManglingModuleScope), str(vtab_dict[key].scope)
                    else:
                        vtab_dict[key] = entry
                        vtab_dict_order.append(key)
            all_defined_here = module is env
            for entry in module.type_entries:
                if entry.used and (all_defined_here or entry.defined_in_pxd):
                    type = entry.type
                    if type.is_extension_type and not entry.in_cinclude:
                        type = entry.type
                        key = type.objstruct_cname
                        assert key not in vtabslot_dict, key
                        vtabslot_dict[key] = entry
                        vtabslot_dict_order.append(key)

        def vtabstruct_cname(entry_type):
            return entry_type.vtabstruct_cname
        vtab_list = self.sort_types_by_inheritance(
            vtab_dict, vtab_dict_order, vtabstruct_cname)

        def objstruct_cname(entry_type):
            return entry_type.objstruct_cname
        vtabslot_list = self.sort_types_by_inheritance(
            vtabslot_dict, vtabslot_dict_order, objstruct_cname)

        return (vtab_list, vtabslot_list)

    def sort_cdef_classes(self, env):
        key_func = operator.attrgetter('objstruct_cname')
        entry_dict, entry_order = {}, []
        for entry in env.c_class_entries:
            key = key_func(entry.type)
            assert key not in entry_dict, key
            entry_dict[key] = entry
            entry_order.append(key)
        env.c_class_entries[:] = self.sort_types_by_inheritance(
            entry_dict, entry_order, key_func)

    def generate_type_definitions(self, env, modules, vtab_list, vtabslot_list, code):
        # TODO: Why are these separated out?
        for entry in vtabslot_list:
            self.generate_objstruct_predeclaration(entry.type, code)
        vtabslot_entries = set(vtabslot_list)
        ctuple_names = set()
        for module in modules:
            definition = module is env
            type_entries = []
            for entry in module.type_entries:
                if entry.type.is_ctuple and entry.used:
                    if entry.name not in ctuple_names:
                        ctuple_names.add(entry.name)
                        type_entries.append(entry)
                elif definition or entry.defined_in_pxd:
                    type_entries.append(entry)
            type_entries = [t for t in type_entries if t not in vtabslot_entries]
            self.generate_type_header_code(type_entries, code)
        for entry in vtabslot_list:
            self.generate_objstruct_definition(entry.type, code)
            self.generate_typeobj_predeclaration(entry, code)
        for entry in vtab_list:
            self.generate_typeobj_predeclaration(entry, code)
            self.generate_exttype_vtable_struct(entry, code)
            self.generate_exttype_vtabptr_declaration(entry, code)
            self.generate_exttype_final_methods_declaration(entry, code)

    def generate_declarations_for_modules(self, env, modules, globalstate):
        typecode = globalstate['type_declarations']
        typecode.putln("")
        typecode.putln("/*--- Type declarations ---*/")
        # This is to work around the fact that array.h isn't part of the C-API,
        # but we need to declare it earlier than utility code.
        if 'cpython.array' in [m.qualified_name for m in modules]:
            typecode.putln('#ifndef _ARRAYARRAY_H')
            typecode.putln('struct arrayobject;')
            typecode.putln('typedef struct arrayobject arrayobject;')
            typecode.putln('#endif')
        vtab_list, vtabslot_list = self.sort_type_hierarchy(modules, env)
        self.generate_type_definitions(
            env, modules, vtab_list, vtabslot_list, typecode)
        modulecode = globalstate['module_declarations']
        for module in modules:
            defined_here = module is env
            modulecode.putln("")
            modulecode.putln("/* Module declarations from '%s' */" % module.qualified_name)
            self.generate_c_class_declarations(module, modulecode, defined_here)
            self.generate_cvariable_declarations(module, modulecode, defined_here)
            self.generate_cfunction_declarations(module, modulecode, defined_here)

    def _put_setup_code(self, code, name):
        code.put(UtilityCode.load_as_string(name, "ModuleSetupCode.c")[1])

    def generate_module_preamble(self, env, options, cimported_modules, metadata, code):
        code.put_generated_by()
        if metadata:
            code.putln("/* BEGIN: Cython Metadata")
            code.putln(json.dumps(metadata, indent=4, sort_keys=True))
            code.putln("END: Cython Metadata */")
            code.putln("")

        code.putln("#ifndef PY_SSIZE_T_CLEAN")
        code.putln("#define PY_SSIZE_T_CLEAN")
        code.putln("#endif /* PY_SSIZE_T_CLEAN */")

        for inc in sorted(env.c_includes.values(), key=IncludeCode.sortkey):
            if inc.location == inc.INITIAL:
                inc.write(code)
        code.putln("#ifndef Py_PYTHON_H")
        code.putln("    #error Python headers needed to compile C extensions, "
                   "please install development version of Python.")
        code.putln("#elif PY_VERSION_HEX < 0x02060000 || "
                   "(0x03000000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x03030000)")
        code.putln("    #error Cython requires Python 2.6+ or Python 3.3+.")
        code.putln("#else")
        code.globalstate["end"].putln("#endif /* Py_PYTHON_H */")

        from .. import __version__
        code.putln('#define CYTHON_ABI "%s"' % __version__.replace('.', '_'))
        code.putln('#define CYTHON_HEX_VERSION %s' % build_hex_version(__version__))
        code.putln("#define CYTHON_FUTURE_DIVISION %d" % (
            Future.division in env.context.future_directives))

        self._put_setup_code(code, "CModulePreamble")
        if env.context.options.cplus:
            self._put_setup_code(code, "CppInitCode")
        else:
            self._put_setup_code(code, "CInitCode")
        self._put_setup_code(code, "PythonCompatibility")
        self._put_setup_code(code, "MathInitCode")

        # Using "(void)cname" to prevent "unused" warnings.
        if options.c_line_in_traceback:
            cinfo = "%s = %s; (void)%s; " % (Naming.clineno_cname, Naming.line_c_macro, Naming.clineno_cname)
        else:
            cinfo = ""
        code.putln("#define __PYX_MARK_ERR_POS(f_index, lineno) \\")
        code.putln("    { %s = %s[f_index]; (void)%s; %s = lineno; (void)%s; %s}" % (
            Naming.filename_cname, Naming.filetable_cname, Naming.filename_cname,
            Naming.lineno_cname, Naming.lineno_cname,
            cinfo
        ))
        code.putln("#define __PYX_ERR(f_index, lineno, Ln_error) \\")
        code.putln("    { __PYX_MARK_ERR_POS(f_index, lineno) goto Ln_error; }")

        code.putln("")
        self.generate_extern_c_macro_definition(code)
        code.putln("")

        code.putln("#define %s" % Naming.h_guard_prefix + self.api_name(env))
        code.putln("#define %s" % Naming.api_guard_prefix + self.api_name(env))
        code.putln("/* Early includes */")
        self.generate_includes(env, cimported_modules, code, late=False)
        code.putln("")
        code.putln("#if defined(PYREX_WITHOUT_ASSERTIONS) && !defined(CYTHON_WITHOUT_ASSERTIONS)")
        code.putln("#define CYTHON_WITHOUT_ASSERTIONS")
        code.putln("#endif")
        code.putln("")

        if env.directives['ccomplex']:
            code.putln("")
            code.putln("#if !defined(CYTHON_CCOMPLEX)")
            code.putln("#define CYTHON_CCOMPLEX 1")
            code.putln("#endif")
            code.putln("")
        code.put(UtilityCode.load_as_string("UtilityFunctionPredeclarations", "ModuleSetupCode.c")[0])

        c_string_type = env.directives['c_string_type']
        c_string_encoding = env.directives['c_string_encoding']
        if c_string_type not in ('bytes', 'bytearray') and not c_string_encoding:
            error(self.pos, "a default encoding must be provided if c_string_type is not a byte type")
        code.putln('#define __PYX_DEFAULT_STRING_ENCODING_IS_ASCII %s' % int(c_string_encoding == 'ascii'))
        code.putln('#define __PYX_DEFAULT_STRING_ENCODING_IS_UTF8 %s' %
                int(c_string_encoding.replace('-', '').lower() == 'utf8'))
        if c_string_encoding == 'default':
            code.putln('#define __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT 1')
        else:
            code.putln('#define __PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT '
                    '(PY_MAJOR_VERSION >= 3 && __PYX_DEFAULT_STRING_ENCODING_IS_UTF8)')
            code.putln('#define __PYX_DEFAULT_STRING_ENCODING "%s"' % c_string_encoding)
        if c_string_type == 'bytearray':
            c_string_func_name = 'ByteArray'
        else:
            c_string_func_name = c_string_type.title()
        code.putln('#define __Pyx_PyObject_FromString __Pyx_Py%s_FromString' % c_string_func_name)
        code.putln('#define __Pyx_PyObject_FromStringAndSize __Pyx_Py%s_FromStringAndSize' % c_string_func_name)
        code.put(UtilityCode.load_as_string("TypeConversions", "TypeConversion.c")[0])

        # These utility functions are assumed to exist and used elsewhere.
        PyrexTypes.c_long_type.create_to_py_utility_code(env)
        PyrexTypes.c_long_type.create_from_py_utility_code(env)
        PyrexTypes.c_int_type.create_from_py_utility_code(env)

        code.put(Nodes.branch_prediction_macros)
        code.putln('static CYTHON_INLINE void __Pyx_pretend_to_initialize(void* ptr) { (void)ptr; }')
        code.putln('')
        code.putln('static PyObject *%s = NULL;' % env.module_cname)
        code.putln('static PyObject *%s;' % env.module_dict_cname)
        code.putln('static PyObject *%s;' % Naming.builtins_cname)
        code.putln('static PyObject *%s = NULL;' % Naming.cython_runtime_cname)
        code.putln('static PyObject *%s;' % Naming.empty_tuple)
        code.putln('static PyObject *%s;' % Naming.empty_bytes)
        code.putln('static PyObject *%s;' % Naming.empty_unicode)
        if Options.pre_import is not None:
            code.putln('static PyObject *%s;' % Naming.preimport_cname)
        code.putln('static int %s;' % Naming.lineno_cname)
        code.putln('static int %s = 0;' % Naming.clineno_cname)
        code.putln('static const char * %s= %s;' % (Naming.cfilenm_cname, Naming.file_c_macro))
        code.putln('static const char *%s;' % Naming.filename_cname)

        env.use_utility_code(UtilityCode.load_cached("FastTypeChecks", "ModuleSetupCode.c"))
        if has_np_pythran(env):
            env.use_utility_code(UtilityCode.load_cached("PythranConversion", "CppSupport.cpp"))

    def generate_extern_c_macro_definition(self, code):
        name = Naming.extern_c_macro
        code.putln("#ifndef %s" % name)
        code.putln("  #ifdef __cplusplus")
        code.putln('    #define %s extern "C"' % name)
        code.putln("  #else")
        code.putln("    #define %s extern" % name)
        code.putln("  #endif")
        code.putln("#endif")

    def generate_dl_import_macro(self, code):
        code.putln("#ifndef DL_IMPORT")
        code.putln("  #define DL_IMPORT(_T) _T")
        code.putln("#endif")

    def generate_includes(self, env, cimported_modules, code, early=True, late=True):
        includes = []
        for inc in sorted(env.c_includes.values(), key=IncludeCode.sortkey):
            if inc.location == inc.EARLY:
                if early:
                    inc.write(code)
            elif inc.location == inc.LATE:
                if late:
                    inc.write(code)
        if early:
            code.putln_openmp("#include <omp.h>")

    def generate_filename_table(self, code):
        from os.path import isabs, basename
        code.putln("")
        code.putln("static const char *%s[] = {" % Naming.filetable_cname)
        if code.globalstate.filename_list:
            for source_desc in code.globalstate.filename_list:
                file_path = source_desc.get_filenametable_entry()
                if isabs(file_path):
                    file_path = basename(file_path)  # never include absolute paths
                escaped_filename = file_path.replace("\\", "\\\\").replace('"', r'\"')
                code.putln('"%s",' % escaped_filename)
        else:
            # Some C compilers don't like an empty array
            code.putln("0")
        code.putln("};")

    def generate_type_predeclarations(self, env, code):
        pass

    def generate_type_header_code(self, type_entries, code):
        # Generate definitions of structs/unions/enums/typedefs/objstructs.
        #self.generate_gcc33_hack(env, code) # Is this still needed?
        # Forward declarations
        for entry in type_entries:
            if not entry.in_cinclude:
                #print "generate_type_header_code:", entry.name, repr(entry.type) ###
                type = entry.type
                if type.is_typedef: # Must test this first!
                    pass
                elif type.is_struct_or_union or type.is_cpp_class:
                    self.generate_struct_union_predeclaration(entry, code)
                elif type.is_ctuple and entry.used:
                    self.generate_struct_union_predeclaration(entry.type.struct_entry, code)
                elif type.is_extension_type:
                    self.generate_objstruct_predeclaration(type, code)
        # Actual declarations
        for entry in type_entries:
            if not entry.in_cinclude:
                #print "generate_type_header_code:", entry.name, repr(entry.type) ###
                type = entry.type
                if type.is_typedef: # Must test this first!
                    self.generate_typedef(entry, code)
                elif type.is_enum:
                    self.generate_enum_definition(entry, code)
                elif type.is_struct_or_union:
                    self.generate_struct_union_definition(entry, code)
                elif type.is_ctuple and entry.used:
                    self.generate_struct_union_definition(entry.type.struct_entry, code)
                elif type.is_cpp_class:
                    self.generate_cpp_class_definition(entry, code)
                elif type.is_extension_type:
                    self.generate_objstruct_definition(type, code)

    def generate_gcc33_hack(self, env, code):
        # Workaround for spurious warning generation in gcc 3.3
        code.putln("")
        for entry in env.c_class_entries:
            type = entry.type
            if not type.typedef_flag:
                name = type.objstruct_cname
                if name.startswith("__pyx_"):
                    tail = name[6:]
                else:
                    tail = name
                code.putln("typedef struct %s __pyx_gcc33_%s;" % (
                    name, tail))

    def generate_typedef(self, entry, code):
        base_type = entry.type.typedef_base_type
        if base_type.is_numeric:
            try:
                writer = code.globalstate['numeric_typedefs']
            except KeyError:
                writer = code
        else:
            writer = code
        writer.mark_pos(entry.pos)
        writer.putln("typedef %s;" % base_type.declaration_code(entry.cname))

    def sue_predeclaration(self, type, kind, name):
        if type.typedef_flag:
            return "%s %s;\ntypedef %s %s %s;" % (
                kind, name,
                kind, name, name)
        else:
            return "%s %s;" % (kind, name)

    def generate_struct_union_predeclaration(self, entry, code):
        type = entry.type
        if type.is_cpp_class and type.templates:
            code.putln("template <typename %s>" % ", typename ".join(
                [T.empty_declaration_code() for T in type.templates]))
        code.putln(self.sue_predeclaration(type, type.kind, type.cname))

    def sue_header_footer(self, type, kind, name):
        header = "%s %s {" % (kind, name)
        footer = "};"
        return header, footer

    def generate_struct_union_definition(self, entry, code):
        code.mark_pos(entry.pos)
        type = entry.type
        scope = type.scope
        if scope:
            kind = type.kind
            packed = type.is_struct and type.packed
            if packed:
                kind = "%s %s" % (type.kind, "__Pyx_PACKED")
                code.globalstate.use_utility_code(packed_struct_utility_code)
            header, footer = \
                self.sue_header_footer(type, kind, type.cname)
            if packed:
                code.putln("#if defined(__SUNPRO_C)")
                code.putln("  #pragma pack(1)")
                code.putln("#elif !defined(__GNUC__)")
                code.putln("  #pragma pack(push, 1)")
                code.putln("#endif")
            code.putln(header)
            var_entries = scope.var_entries
            if not var_entries:
                error(entry.pos, "Empty struct or union definition not allowed outside a 'cdef extern from' block")
            for attr in var_entries:
                code.putln(
                    "%s;" % attr.type.declaration_code(attr.cname))
            code.putln(footer)
            if packed:
                code.putln("#if defined(__SUNPRO_C)")
                code.putln("  #pragma pack()")
                code.putln("#elif !defined(__GNUC__)")
                code.putln("  #pragma pack(pop)")
                code.putln("#endif")

    def generate_cpp_class_definition(self, entry, code):
        code.mark_pos(entry.pos)
        type = entry.type
        scope = type.scope
        if scope:
            if type.templates:
                code.putln("template <class %s>" % ", class ".join(
                    [T.empty_declaration_code() for T in type.templates]))
            # Just let everything be public.
            code.put("struct %s" % type.cname)
            if type.base_classes:
                base_class_decl = ", public ".join(
                    [base_class.empty_declaration_code() for base_class in type.base_classes])
                code.put(" : public %s" % base_class_decl)
            code.putln(" {")
            py_attrs = [e for e in scope.entries.values()
                        if e.type.is_pyobject and not e.is_inherited]
            has_virtual_methods = False
            constructor = None
            destructor = None
            for attr in scope.var_entries:
                if attr.type.is_cfunction and attr.type.is_static_method:
                    code.put("static ")
                elif attr.name == "<init>":
                    constructor = attr
                elif attr.name == "<del>":
                    destructor = attr
                elif attr.type.is_cfunction:
                    code.put("virtual ")
                    has_virtual_methods = True
                code.putln("%s;" % attr.type.declaration_code(attr.cname))
            is_implementing = 'init_module' in code.globalstate.parts
            if constructor or py_attrs:
                if constructor:
                    arg_decls = []
                    arg_names = []
                    for arg in constructor.type.original_args[
                            :len(constructor.type.args)-constructor.type.optional_arg_count]:
                        arg_decls.append(arg.declaration_code())
                        arg_names.append(arg.cname)
                    if constructor.type.optional_arg_count:
                        arg_decls.append(constructor.type.op_arg_struct.declaration_code(Naming.optional_args_cname))
                        arg_names.append(Naming.optional_args_cname)
                    if not arg_decls:
                        arg_decls = ["void"]
                else:
                    arg_decls = ["void"]
                    arg_names = []
                if is_implementing:
                  code.putln("%s(%s) {" % (type.cname, ", ".join(arg_decls)))
                  if py_attrs:
                      code.put_ensure_gil()
                      for attr in py_attrs:
                          code.put_init_var_to_py_none(attr, nanny=False);
                  if constructor:
                      code.putln("%s(%s);" % (constructor.cname, ", ".join(arg_names)))
                  if py_attrs:
                      code.put_release_ensured_gil()
                  code.putln("}")
                else:
                  code.putln("%s(%s);" % (type.cname, ", ".join(arg_decls)))
            if destructor or py_attrs or has_virtual_methods:
                if has_virtual_methods:
                    code.put("virtual ")
                if is_implementing:
                  code.putln("~%s() {" % type.cname)
                  if py_attrs:
                      code.put_ensure_gil()
                  if destructor:
                      code.putln("%s();" % destructor.cname)
                  if py_attrs:
                      for attr in py_attrs:
                          code.put_var_xdecref(attr, nanny=False);
                      code.put_release_ensured_gil()
                  code.putln("}")
                else:
                  code.putln("~%s();" % type.cname)
            if py_attrs:
                # Also need copy constructor and assignment operators.
                if is_implementing:
                  code.putln("%s(const %s& __Pyx_other) {" % (type.cname, type.cname))
                  code.put_ensure_gil()
                  for attr in scope.var_entries:
                      if not attr.type.is_cfunction:
                          code.putln("%s = __Pyx_other.%s;" % (attr.cname, attr.cname))
                          code.put_var_incref(attr, nanny=False)
                  code.put_release_ensured_gil()
                  code.putln("}")
                  code.putln("%s& operator=(const %s& __Pyx_other) {" % (type.cname, type.cname))
                  code.putln("if (this != &__Pyx_other) {")
                  code.put_ensure_gil()
                  for attr in scope.var_entries:
                      if not attr.type.is_cfunction:
                          code.put_var_xdecref(attr, nanny=False);
                          code.putln("%s = __Pyx_other.%s;" % (attr.cname, attr.cname))
                          code.put_var_incref(attr, nanny=False)
                  code.put_release_ensured_gil()
                  code.putln("}")
                  code.putln("return *this;")
                  code.putln("}")
                else:
                  code.putln("%s(const %s& __Pyx_other);" % (type.cname, type.cname))
                  code.putln("%s& operator=(const %s& __Pyx_other);" % (type.cname, type.cname))
            code.putln("};")

    def generate_enum_definition(self, entry, code):
        code.mark_pos(entry.pos)
        type = entry.type
        name = entry.cname or entry.name or ""
        header, footer = self.sue_header_footer(type, "enum", name)
        code.putln(header)
        enum_values = entry.enum_values
        if not enum_values:
            error(entry.pos, "Empty enum definition not allowed outside a 'cdef extern from' block")
        else:
            last_entry = enum_values[-1]
            # this does not really generate code, just builds the result value
            for value_entry in enum_values:
                if value_entry.value_node is not None:
                    value_entry.value_node.generate_evaluation_code(code)

            for value_entry in enum_values:
                if value_entry.value_node is None:
                    value_code = value_entry.cname
                else:
                    value_code = ("%s = %s" % (
                        value_entry.cname,
                        value_entry.value_node.result()))
                if value_entry is not last_entry:
                    value_code += ","
                code.putln(value_code)
        code.putln(footer)
        if entry.type.typedef_flag:
            # Not pre-declared.
            code.putln("typedef enum %s %s;" % (name, name))

    def generate_typeobj_predeclaration(self, entry, code):
        code.putln("")
        name = entry.type.typeobj_cname
        if name:
            if entry.visibility == 'extern' and not entry.in_cinclude:
                code.putln("%s %s %s;" % (
                    Naming.extern_c_macro,
                    PyrexTypes.public_decl("PyTypeObject", "DL_IMPORT"),
                    name))
            elif entry.visibility == 'public':
                code.putln("%s %s %s;" % (
                    Naming.extern_c_macro,
                    PyrexTypes.public_decl("PyTypeObject", "DL_EXPORT"),
                    name))
            # ??? Do we really need the rest of this? ???
            #else:
            #    code.putln("static PyTypeObject %s;" % name)

    def generate_exttype_vtable_struct(self, entry, code):
        if not entry.used:
            return

        code.mark_pos(entry.pos)
        # Generate struct declaration for an extension type's vtable.
        type = entry.type
        scope = type.scope

        self.specialize_fused_types(scope)

        if type.vtabstruct_cname:
            code.putln("")
            code.putln("struct %s {" % type.vtabstruct_cname)
            if type.base_type and type.base_type.vtabstruct_cname:
                code.putln("struct %s %s;" % (
                    type.base_type.vtabstruct_cname,
                    Naming.obj_base_cname))
            for method_entry in scope.cfunc_entries:
                if not method_entry.is_inherited:
                    code.putln("%s;" % method_entry.type.declaration_code("(*%s)" % method_entry.cname))
            code.putln("};")

    def generate_exttype_vtabptr_declaration(self, entry, code):
        if not entry.used:
            return

        code.mark_pos(entry.pos)
        # Generate declaration of pointer to an extension type's vtable.
        type = entry.type
        if type.vtabptr_cname:
            code.putln("static struct %s *%s;" % (
                type.vtabstruct_cname,
                type.vtabptr_cname))

    def generate_exttype_final_methods_declaration(self, entry, code):
        if not entry.used:
            return

        code.mark_pos(entry.pos)
        # Generate final methods prototypes
        type = entry.type
        for method_entry in entry.type.scope.cfunc_entries:
            if not method_entry.is_inherited and method_entry.final_func_cname:
                declaration = method_entry.type.declaration_code(
                    method_entry.final_func_cname)
                modifiers = code.build_function_modifiers(method_entry.func_modifiers)
                code.putln("static %s%s;" % (modifiers, declaration))

    def generate_objstruct_predeclaration(self, type, code):
        if not type.scope:
            return
        code.putln(self.sue_predeclaration(type, "struct", type.objstruct_cname))

    def generate_objstruct_definition(self, type, code):
        code.mark_pos(type.pos)
        # Generate object struct definition for an
        # extension type.
        if not type.scope:
            return # Forward declared but never defined
        header, footer = \
            self.sue_header_footer(type, "struct", type.objstruct_cname)
        code.putln(header)
        base_type = type.base_type
        if base_type:
            basestruct_cname = base_type.objstruct_cname
            if basestruct_cname == "PyTypeObject":
                # User-defined subclasses of type are heap allocated.
                basestruct_cname = "PyHeapTypeObject"
            code.putln(
                "%s%s %s;" % (
                    ("struct ", "")[base_type.typedef_flag],
                    basestruct_cname,
                    Naming.obj_base_cname))
        else:
            code.putln(
                "PyObject_HEAD")
        if type.vtabslot_cname and not (type.base_type and type.base_type.vtabslot_cname):
            code.putln(
                "struct %s *%s;" % (
                    type.vtabstruct_cname,
                    type.vtabslot_cname))
        for attr in type.scope.var_entries:
            if attr.is_declared_generic:
                attr_type = py_object_type
            else:
                attr_type = attr.type
            code.putln(
                "%s;" % attr_type.declaration_code(attr.cname))
        code.putln(footer)
        if type.objtypedef_cname is not None:
            # Only for exposing public typedef name.
            code.putln("typedef struct %s %s;" % (type.objstruct_cname, type.objtypedef_cname))

    def generate_c_class_declarations(self, env, code, definition):
        for entry in env.c_class_entries:
            if definition or entry.defined_in_pxd:
                code.putln("static PyTypeObject *%s = 0;" % (
                    entry.type.typeptr_cname))

    def generate_cvariable_declarations(self, env, code, definition):
        if env.is_cython_builtin:
            return
        for entry in env.var_entries:
            if (entry.in_cinclude or entry.in_closure or
                    (entry.visibility == 'private' and not (entry.defined_in_pxd or entry.used))):
                continue

            storage_class = None
            dll_linkage = None
            init = None

            if entry.visibility == 'extern':
                storage_class = Naming.extern_c_macro
                dll_linkage = "DL_IMPORT"
            elif entry.visibility == 'public':
                storage_class = Naming.extern_c_macro
                if definition:
                    dll_linkage = "DL_EXPORT"
                else:
                    dll_linkage = "DL_IMPORT"
            elif entry.visibility == 'private':
                storage_class = "static"
                dll_linkage = None
                if entry.init is not None:
                    init = entry.type.literal_code(entry.init)
            type = entry.type
            cname = entry.cname

            if entry.defined_in_pxd and not definition:
                storage_class = "static"
                dll_linkage = None
                type = CPtrType(type)
                cname = env.mangle(Naming.varptr_prefix, entry.name)
                init = 0

            if storage_class:
                code.put("%s " % storage_class)
            code.put(type.declaration_code(
                cname, dll_linkage=dll_linkage))
            if init is not None:
                code.put_safe(" = %s" % init)
            code.putln(";")
            if entry.cname != cname:
                code.putln("#define %s (*%s)" % (entry.cname, cname))

    def generate_cfunction_declarations(self, env, code, definition):
        for entry in env.cfunc_entries:
            if entry.used or (entry.visibility == 'public' or entry.api):
                generate_cfunction_declaration(entry, env, code, definition)

    def generate_variable_definitions(self, env, code):
        for entry in env.var_entries:
            if not entry.in_cinclude and entry.visibility == "public":
                code.put(entry.type.declaration_code(entry.cname))
                if entry.init is not None:
                    init = entry.type.literal_code(entry.init)
                    code.put_safe(" = %s" % init)
                code.putln(";")

    def generate_typeobj_definitions(self, env, code):
        full_module_name = env.qualified_name
        for entry in env.c_class_entries:
            #print "generate_typeobj_definitions:", entry.name
            #print "...visibility =", entry.visibility
            if entry.visibility != 'extern':
                type = entry.type
                scope = type.scope
                if scope: # could be None if there was an error
                    if not scope.directives['c_api_binop_methods']:
                        error(self.pos,
                              "The 'c_api_binop_methods' directive is only supported for forward compatibility"
                              " and must be True.")
                    self.generate_exttype_vtable(scope, code)
                    self.generate_new_function(scope, code, entry)
                    self.generate_dealloc_function(scope, code)
                    if scope.needs_gc():
                        self.generate_traverse_function(scope, code, entry)
                        if scope.needs_tp_clear():
                            self.generate_clear_function(scope, code, entry)
                    if scope.defines_any_special(["__getitem__"]):
                        self.generate_getitem_int_function(scope, code)
                    if scope.defines_any_special(["__setitem__", "__delitem__"]):
                        self.generate_ass_subscript_function(scope, code)
                    if scope.defines_any_special(["__getslice__", "__setslice__", "__delslice__"]):
                        warning(self.pos,
                                "__getslice__, __setslice__, and __delslice__ are not supported by Python 3, "
                                "use __getitem__, __setitem__, and __delitem__ instead", 1)
                        code.putln("#if PY_MAJOR_VERSION >= 3")
                        code.putln("#error __getslice__, __setslice__, and __delslice__ not supported in Python 3.")
                        code.putln("#endif")
                    if scope.defines_any_special(["__setslice__", "__delslice__"]):
                        self.generate_ass_slice_function(scope, code)
                    if scope.defines_any_special(["__getattr__", "__getattribute__"]):
                        self.generate_getattro_function(scope, code)
                    if scope.defines_any_special(["__setattr__", "__delattr__"]):
                        self.generate_setattro_function(scope, code)
                    if scope.defines_any_special(["__get__"]):
                        self.generate_descr_get_function(scope, code)
                    if scope.defines_any_special(["__set__", "__delete__"]):
                        self.generate_descr_set_function(scope, code)
                    if not scope.is_closure_class_scope and scope.defines_any(["__dict__"]):
                        self.generate_dict_getter_function(scope, code)
                    if scope.defines_any_special(TypeSlots.richcmp_special_methods):
                        self.generate_richcmp_function(scope, code)
                    self.generate_property_accessors(scope, code)
                    self.generate_method_table(scope, code)
                    self.generate_getset_table(scope, code)
                    self.generate_typeobj_definition(full_module_name, entry, code)

    def generate_exttype_vtable(self, scope, code):
        # Generate the definition of an extension type's vtable.
        type = scope.parent_type
        if type.vtable_cname:
            code.putln("static struct %s %s;" % (
                type.vtabstruct_cname,
                type.vtable_cname))

    def generate_self_cast(self, scope, code):
        type = scope.parent_type
        code.putln(
            "%s = (%s)o;" % (
                type.declaration_code("p"),
                type.empty_declaration_code()))

    def generate_new_function(self, scope, code, cclass_entry):
        tp_slot = TypeSlots.ConstructorSlot("tp_new", '__new__')
        slot_func = scope.mangle_internal("tp_new")
        type = scope.parent_type
        base_type = type.base_type

        have_entries, (py_attrs, py_buffers, memoryview_slices) = \
                        scope.get_refcounted_entries()
        is_final_type = scope.parent_type.is_final_type
        if scope.is_internal:
            # internal classes (should) never need None inits, normal zeroing will do
            py_attrs = []
        cpp_class_attrs = [entry for entry in scope.var_entries
                           if entry.type.is_cpp_class]

        new_func_entry = scope.lookup_here("__new__")
        if base_type or (new_func_entry and new_func_entry.is_special
                         and not new_func_entry.trivial_signature):
            unused_marker = ''
        else:
            unused_marker = 'CYTHON_UNUSED '

        if base_type:
            freelist_size = 0  # not currently supported
        else:
            freelist_size = scope.directives.get('freelist', 0)
        freelist_name = scope.mangle_internal(Naming.freelist_name)
        freecount_name = scope.mangle_internal(Naming.freecount_name)

        decls = code.globalstate['decls']
        decls.putln("static PyObject *%s(PyTypeObject *t, PyObject *a, PyObject *k); /*proto*/" %
                    slot_func)
        code.putln("")
        if freelist_size:
            code.putln("static %s[%d];" % (
                scope.parent_type.declaration_code(freelist_name),
                freelist_size))
            code.putln("static int %s = 0;" % freecount_name)
            code.putln("")
        code.putln(
            "static PyObject *%s(PyTypeObject *t, %sPyObject *a, %sPyObject *k) {" % (
                slot_func, unused_marker, unused_marker))

        need_self_cast = (type.vtabslot_cname or
                          (py_buffers or memoryview_slices or py_attrs) or
                          cpp_class_attrs)
        if need_self_cast:
            code.putln("%s;" % scope.parent_type.declaration_code("p"))
        if base_type:
            tp_new = TypeSlots.get_base_slot_function(scope, tp_slot)
            if tp_new is None:
                tp_new = "%s->tp_new" % base_type.typeptr_cname
            code.putln("PyObject *o = %s(t, a, k);" % tp_new)
        else:
            code.putln("PyObject *o;")
            if freelist_size:
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached("IncludeStringH", "StringTools.c"))
                if is_final_type:
                    type_safety_check = ''
                else:
                    type_safety_check = ' & ((t->tp_flags & (Py_TPFLAGS_IS_ABSTRACT | Py_TPFLAGS_HEAPTYPE)) == 0)'
                obj_struct = type.declaration_code("", deref=True)
                code.putln(
                    "if (CYTHON_COMPILING_IN_CPYTHON && likely((%s > 0) & (t->tp_basicsize == sizeof(%s))%s)) {" % (
                        freecount_name, obj_struct, type_safety_check))
                code.putln("o = (PyObject*)%s[--%s];" % (
                    freelist_name, freecount_name))
                code.putln("memset(o, 0, sizeof(%s));" % obj_struct)
                code.putln("(void) PyObject_INIT(o, t);")
                if scope.needs_gc():
                    code.putln("PyObject_GC_Track(o);")
                code.putln("} else {")
            if not is_final_type:
                code.putln("if (likely((t->tp_flags & Py_TPFLAGS_IS_ABSTRACT) == 0)) {")
            code.putln("o = (*t->tp_alloc)(t, 0);")
            if not is_final_type:
                code.putln("} else {")
                code.putln("o = (PyObject *) PyBaseObject_Type.tp_new(t, %s, 0);" % Naming.empty_tuple)
                code.putln("}")
        code.putln("if (unlikely(!o)) return 0;")
        if freelist_size and not base_type:
            code.putln('}')
        if need_self_cast:
            code.putln("p = %s;" % type.cast_code("o"))
        #if need_self_cast:
        #    self.generate_self_cast(scope, code)

        # from this point on, ensure DECREF(o) on failure
        needs_error_cleanup = False

        if type.vtabslot_cname:
            vtab_base_type = type
            while vtab_base_type.base_type and vtab_base_type.base_type.vtabstruct_cname:
                vtab_base_type = vtab_base_type.base_type
            if vtab_base_type is not type:
                struct_type_cast = "(struct %s*)" % vtab_base_type.vtabstruct_cname
            else:
                struct_type_cast = ""
            code.putln("p->%s = %s%s;" % (
                type.vtabslot_cname,
                struct_type_cast, type.vtabptr_cname))

        for entry in cpp_class_attrs:
            code.putln("new((void*)&(p->%s)) %s();" % (
                entry.cname, entry.type.empty_declaration_code()))

        for entry in py_attrs:
            if entry.name == "__dict__":
                needs_error_cleanup = True
                code.put("p->%s = PyDict_New(); if (unlikely(!p->%s)) goto bad;" % (
                    entry.cname, entry.cname))
            else:
                code.put_init_var_to_py_none(entry, "p->%s", nanny=False)

        for entry in memoryview_slices:
            code.putln("p->%s.data = NULL;" % entry.cname)
            code.putln("p->%s.memview = NULL;" % entry.cname)

        for entry in py_buffers:
            code.putln("p->%s.obj = NULL;" % entry.cname)

        if cclass_entry.cname == '__pyx_memoryviewslice':
            code.putln("p->from_slice.memview = NULL;")

        if new_func_entry and new_func_entry.is_special:
            if new_func_entry.trivial_signature:
                cinit_args = "o, %s, NULL" % Naming.empty_tuple
            else:
                cinit_args = "o, a, k"
            needs_error_cleanup = True
            code.putln("if (unlikely(%s(%s) < 0)) goto bad;" % (
                new_func_entry.func_cname, cinit_args))

        code.putln(
            "return o;")
        if needs_error_cleanup:
            code.putln("bad:")
            code.put_decref_clear("o", py_object_type, nanny=False)
            code.putln("return NULL;")
        code.putln(
            "}")

    def generate_dealloc_function(self, scope, code):
        tp_slot = TypeSlots.ConstructorSlot("tp_dealloc", '__dealloc__')
        slot_func = scope.mangle_internal("tp_dealloc")
        base_type = scope.parent_type.base_type
        if tp_slot.slot_code(scope) != slot_func:
            return  # never used

        slot_func_cname = scope.mangle_internal("tp_dealloc")
        code.putln("")
        code.putln(
            "static void %s(PyObject *o) {" % slot_func_cname)

        is_final_type = scope.parent_type.is_final_type
        needs_gc = scope.needs_gc()

        weakref_slot = scope.lookup_here("__weakref__") if not scope.is_closure_class_scope else None
        if weakref_slot not in scope.var_entries:
            weakref_slot = None

        dict_slot = scope.lookup_here("__dict__") if not scope.is_closure_class_scope else None
        if dict_slot not in scope.var_entries:
            dict_slot = None

        _, (py_attrs, _, memoryview_slices) = scope.get_refcounted_entries()
        cpp_class_attrs = [entry for entry in scope.var_entries
                           if entry.type.is_cpp_class]

        if py_attrs or cpp_class_attrs or memoryview_slices or weakref_slot or dict_slot:
            self.generate_self_cast(scope, code)

        if not is_final_type:
            # in Py3.4+, call tp_finalize() as early as possible
            code.putln("#if CYTHON_USE_TP_FINALIZE")
            if needs_gc:
                finalised_check = '!_PyGC_FINALIZED(o)'
            else:
                finalised_check = (
                    '(!PyType_IS_GC(Py_TYPE(o)) || !_PyGC_FINALIZED(o))')
            code.putln(
                "if (unlikely(PyType_HasFeature(Py_TYPE(o), Py_TPFLAGS_HAVE_FINALIZE)"
                " && Py_TYPE(o)->tp_finalize) && %s) {" % finalised_check)
            # if instance was resurrected by finaliser, return
            code.putln("if (PyObject_CallFinalizerFromDealloc(o)) return;")
            code.putln("}")
            code.putln("#endif")

        if needs_gc:
            # We must mark this object as (gc) untracked while tearing
            # it down, lest the garbage collection is invoked while
            # running this destructor.
            code.putln("PyObject_GC_UnTrack(o);")

        # call the user's __dealloc__
        self.generate_usr_dealloc_call(scope, code)

        if weakref_slot:
            code.putln("if (p->__weakref__) PyObject_ClearWeakRefs(o);")

        if dict_slot:
            code.putln("if (p->__dict__) PyDict_Clear(p->__dict__);")

        for entry in cpp_class_attrs:
            code.putln("__Pyx_call_destructor(p->%s);" % entry.cname)

        for entry in py_attrs:
            code.put_xdecref_clear("p->%s" % entry.cname, entry.type, nanny=False,
                                   clear_before_decref=True)

        for entry in memoryview_slices:
            code.put_xdecref_memoryviewslice("p->%s" % entry.cname,
                                             have_gil=True)

        if base_type:
            if needs_gc:
                # The base class deallocator probably expects this to be tracked,
                # so undo the untracking above.
                if base_type.scope and base_type.scope.needs_gc():
                    code.putln("PyObject_GC_Track(o);")
                else:
                    code.putln("#if CYTHON_USE_TYPE_SLOTS")
                    code.putln("if (PyType_IS_GC(Py_TYPE(o)->tp_base))")
                    code.putln("#endif")
                    code.putln("PyObject_GC_Track(o);")

            tp_dealloc = TypeSlots.get_base_slot_function(scope, tp_slot)
            if tp_dealloc is not None:
                code.putln("%s(o);" % tp_dealloc)
            elif base_type.is_builtin_type:
                code.putln("%s->tp_dealloc(o);" % base_type.typeptr_cname)
            else:
                # This is an externally defined type.  Calling through the
                # cimported base type pointer directly interacts badly with
                # the module cleanup, which may already have cleared it.
                # In that case, fall back to traversing the type hierarchy.
                base_cname = base_type.typeptr_cname
                code.putln("if (likely(%s)) %s->tp_dealloc(o); "
                           "else __Pyx_call_next_tp_dealloc(o, %s);" % (
                               base_cname, base_cname, slot_func_cname))
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached("CallNextTpDealloc", "ExtensionTypes.c"))
        else:
            freelist_size = scope.directives.get('freelist', 0)
            if freelist_size:
                freelist_name = scope.mangle_internal(Naming.freelist_name)
                freecount_name = scope.mangle_internal(Naming.freecount_name)

                if is_final_type:
                    type_safety_check = ''
                else:
                    type_safety_check = (
                        ' & ((Py_TYPE(o)->tp_flags & (Py_TPFLAGS_IS_ABSTRACT | Py_TPFLAGS_HEAPTYPE)) == 0)')

                type = scope.parent_type
                code.putln(
                    "if (CYTHON_COMPILING_IN_CPYTHON && ((%s < %d) & (Py_TYPE(o)->tp_basicsize == sizeof(%s))%s)) {" % (
                        freecount_name,
                        freelist_size,
                        type.declaration_code("", deref=True),
                        type_safety_check))
                code.putln("%s[%s++] = %s;" % (
                    freelist_name, freecount_name, type.cast_code("o")))
                code.putln("} else {")
            code.putln("(*Py_TYPE(o)->tp_free)(o);")
            if freelist_size:
                code.putln("}")
        code.putln(
            "}")

    def generate_usr_dealloc_call(self, scope, code):
        entry = scope.lookup_here("__dealloc__")
        if not entry:
            return

        code.putln("{")
        code.putln("PyObject *etype, *eval, *etb;")
        code.putln("PyErr_Fetch(&etype, &eval, &etb);")
        # increase the refcount while we are calling into user code
        # to prevent recursive deallocation
        code.putln("__Pyx_SET_REFCNT(o, Py_REFCNT(o) + 1);")
        code.putln("%s(o);" % entry.func_cname)
        code.putln("__Pyx_SET_REFCNT(o, Py_REFCNT(o) - 1);")
        code.putln("PyErr_Restore(etype, eval, etb);")
        code.putln("}")

    def generate_traverse_function(self, scope, code, cclass_entry):
        tp_slot = TypeSlots.GCDependentSlot("tp_traverse")
        slot_func = scope.mangle_internal("tp_traverse")
        base_type = scope.parent_type.base_type
        if tp_slot.slot_code(scope) != slot_func:
            return  # never used
        code.putln("")
        code.putln(
            "static int %s(PyObject *o, visitproc v, void *a) {" % slot_func)

        have_entries, (py_attrs, py_buffers, memoryview_slices) = (
            scope.get_refcounted_entries(include_gc_simple=False))

        if base_type or py_attrs:
            code.putln("int e;")

        if py_attrs or py_buffers:
            self.generate_self_cast(scope, code)

        if base_type:
            # want to call it explicitly if possible so inlining can be performed
            static_call = TypeSlots.get_base_slot_function(scope, tp_slot)
            if static_call:
                code.putln("e = %s(o, v, a); if (e) return e;" % static_call)
            elif base_type.is_builtin_type:
                base_cname = base_type.typeptr_cname
                code.putln("if (!%s->tp_traverse); else { e = %s->tp_traverse(o,v,a); if (e) return e; }" % (
                    base_cname, base_cname))
            else:
                # This is an externally defined type.  Calling through the
                # cimported base type pointer directly interacts badly with
                # the module cleanup, which may already have cleared it.
                # In that case, fall back to traversing the type hierarchy.
                base_cname = base_type.typeptr_cname
                code.putln(
                    "e = ((likely(%s)) ? ((%s->tp_traverse) ? %s->tp_traverse(o, v, a) : 0) : "
                    "__Pyx_call_next_tp_traverse(o, v, a, %s)); if (e) return e;" % (
                        base_cname, base_cname, base_cname, slot_func))
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached("CallNextTpTraverse", "ExtensionTypes.c"))

        for entry in py_attrs:
            var_code = "p->%s" % entry.cname
            var_as_pyobject = PyrexTypes.typecast(py_object_type, entry.type, var_code)
            code.putln("if (%s) {" % var_code)
            code.putln("e = (*v)(%s, a); if (e) return e;" % var_as_pyobject)
            code.putln("}")

        # Traverse buffer exporting objects.
        # Note: not traversing memoryview attributes of memoryview slices!
        # When triggered by the GC, it would cause multiple visits (gc_refs
        # subtractions which is not matched by its reference count!)
        for entry in py_buffers:
            cname = entry.cname + ".obj"
            code.putln("if (p->%s) {" % cname)
            code.putln("e = (*v)(p->%s, a); if (e) return e;" % cname)
            code.putln("}")

        code.putln("return 0;")
        code.putln("}")

    def generate_clear_function(self, scope, code, cclass_entry):
        tp_slot = TypeSlots.get_slot_by_name("tp_clear")
        slot_func = scope.mangle_internal("tp_clear")
        base_type = scope.parent_type.base_type
        if tp_slot.slot_code(scope) != slot_func:
            return # never used

        have_entries, (py_attrs, py_buffers, memoryview_slices) = (
            scope.get_refcounted_entries(include_gc_simple=False))

        if py_attrs or py_buffers or base_type:
            unused = ''
        else:
            unused = 'CYTHON_UNUSED '

        code.putln("")
        code.putln("static int %s(%sPyObject *o) {" % (slot_func, unused))

        if py_attrs and Options.clear_to_none:
            code.putln("PyObject* tmp;")

        if py_attrs or py_buffers:
            self.generate_self_cast(scope, code)

        if base_type:
            # want to call it explicitly if possible so inlining can be performed
            static_call = TypeSlots.get_base_slot_function(scope, tp_slot)
            if static_call:
                code.putln("%s(o);" % static_call)
            elif base_type.is_builtin_type:
                base_cname = base_type.typeptr_cname
                code.putln("if (!%s->tp_clear); else %s->tp_clear(o);" % (
                    base_cname, base_cname))
            else:
                # This is an externally defined type.  Calling through the
                # cimported base type pointer directly interacts badly with
                # the module cleanup, which may already have cleared it.
                # In that case, fall back to traversing the type hierarchy.
                base_cname = base_type.typeptr_cname
                code.putln(
                    "if (likely(%s)) { if (%s->tp_clear) %s->tp_clear(o); } else __Pyx_call_next_tp_clear(o, %s);" % (
                        base_cname, base_cname, base_cname, slot_func))
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached("CallNextTpClear", "ExtensionTypes.c"))

        if Options.clear_to_none:
            for entry in py_attrs:
                name = "p->%s" % entry.cname
                code.putln("tmp = ((PyObject*)%s);" % name)
                if entry.is_declared_generic:
                    code.put_init_to_py_none(name, py_object_type, nanny=False)
                else:
                    code.put_init_to_py_none(name, entry.type, nanny=False)
                code.putln("Py_XDECREF(tmp);")
        else:
            for entry in py_attrs:
                code.putln("Py_CLEAR(p->%s);" % entry.cname)

        for entry in py_buffers:
            # Note: shouldn't this call __Pyx_ReleaseBuffer ??
            code.putln("Py_CLEAR(p->%s.obj);" % entry.cname)

        if cclass_entry.cname == '__pyx_memoryviewslice':
            code.putln("__PYX_XDEC_MEMVIEW(&p->from_slice, 1);")

        code.putln("return 0;")
        code.putln("}")

    def generate_getitem_int_function(self, scope, code):
        # This function is put into the sq_item slot when
        # a __getitem__ method is present. It converts its
        # argument to a Python integer and calls mp_subscript.
        code.putln(
            "static PyObject *%s(PyObject *o, Py_ssize_t i) {" % (
                scope.mangle_internal("sq_item")))
        code.putln(
            "PyObject *r;")
        code.putln(
            "PyObject *x = PyInt_FromSsize_t(i); if(!x) return 0;")
        code.putln(
            "r = Py_TYPE(o)->tp_as_mapping->mp_subscript(o, x);")
        code.putln(
            "Py_DECREF(x);")
        code.putln(
            "return r;")
        code.putln(
            "}")

    def generate_ass_subscript_function(self, scope, code):
        # Setting and deleting an item are both done through
        # the ass_subscript method, so we dispatch to user's __setitem__
        # or __delitem__, or raise an exception.
        base_type = scope.parent_type.base_type
        set_entry = scope.lookup_here("__setitem__")
        del_entry = scope.lookup_here("__delitem__")
        code.putln("")
        code.putln(
            "static int %s(PyObject *o, PyObject *i, PyObject *v) {" % (
                scope.mangle_internal("mp_ass_subscript")))
        code.putln(
            "if (v) {")
        if set_entry:
            code.putln("return %s(o, i, v);" % set_entry.func_cname)
        else:
            self.generate_guarded_basetype_call(
                base_type, "tp_as_mapping", "mp_ass_subscript", "o, i, v", code)
            code.putln(
                "PyErr_Format(PyExc_NotImplementedError,")
            code.putln(
                '  "Subscript assignment not supported by %.200s", Py_TYPE(o)->tp_name);')
            code.putln(
                "return -1;")
        code.putln(
            "}")
        code.putln(
            "else {")
        if del_entry:
            code.putln(
                "return %s(o, i);" % (
                    del_entry.func_cname))
        else:
            self.generate_guarded_basetype_call(
                base_type, "tp_as_mapping", "mp_ass_subscript", "o, i, v", code)
            code.putln(
                "PyErr_Format(PyExc_NotImplementedError,")
            code.putln(
                '  "Subscript deletion not supported by %.200s", Py_TYPE(o)->tp_name);')
            code.putln(
                "return -1;")
        code.putln(
            "}")
        code.putln(
            "}")

    def generate_guarded_basetype_call(
            self, base_type, substructure, slot, args, code):
        if base_type:
            base_tpname = base_type.typeptr_cname
            if substructure:
                code.putln(
                    "if (%s->%s && %s->%s->%s)" % (
                        base_tpname, substructure, base_tpname, substructure, slot))
                code.putln(
                    "  return %s->%s->%s(%s);" % (
                        base_tpname, substructure, slot, args))
            else:
                code.putln(
                    "if (%s->%s)" % (
                        base_tpname, slot))
                code.putln(
                    "  return %s->%s(%s);" % (
                        base_tpname, slot, args))

    def generate_ass_slice_function(self, scope, code):
        # Setting and deleting a slice are both done through
        # the ass_slice method, so we dispatch to user's __setslice__
        # or __delslice__, or raise an exception.
        base_type = scope.parent_type.base_type
        set_entry = scope.lookup_here("__setslice__")
        del_entry = scope.lookup_here("__delslice__")
        code.putln("")
        code.putln(
            "static int %s(PyObject *o, Py_ssize_t i, Py_ssize_t j, PyObject *v) {" % (
                scope.mangle_internal("sq_ass_slice")))
        code.putln(
            "if (v) {")
        if set_entry:
            code.putln(
                "return %s(o, i, j, v);" % (
                    set_entry.func_cname))
        else:
            self.generate_guarded_basetype_call(
                base_type, "tp_as_sequence", "sq_ass_slice", "o, i, j, v", code)
            code.putln(
                "PyErr_Format(PyExc_NotImplementedError,")
            code.putln(
                '  "2-element slice assignment not supported by %.200s", Py_TYPE(o)->tp_name);')
            code.putln(
                "return -1;")
        code.putln(
            "}")
        code.putln(
            "else {")
        if del_entry:
            code.putln(
                "return %s(o, i, j);" % (
                    del_entry.func_cname))
        else:
            self.generate_guarded_basetype_call(
                base_type, "tp_as_sequence", "sq_ass_slice", "o, i, j, v", code)
            code.putln(
                "PyErr_Format(PyExc_NotImplementedError,")
            code.putln(
                '  "2-element slice deletion not supported by %.200s", Py_TYPE(o)->tp_name);')
            code.putln(
                "return -1;")
        code.putln(
            "}")
        code.putln(
            "}")

    def generate_richcmp_function(self, scope, code):
        if scope.lookup_here("__richcmp__"):
            # user implemented, nothing to do
            return
        # otherwise, we have to generate it from the Python special methods
        richcmp_cfunc = scope.mangle_internal("tp_richcompare")
        code.putln("")
        code.putln("static PyObject *%s(PyObject *o1, PyObject *o2, int op) {" % richcmp_cfunc)
        code.putln("switch (op) {")

        class_scopes = []
        cls = scope.parent_type
        while cls is not None and not cls.entry.visibility == 'extern':
            class_scopes.append(cls.scope)
            cls = cls.scope.parent_type.base_type
        assert scope in class_scopes

        extern_parent = None
        if cls and cls.entry.visibility == 'extern':
            # need to call up into base classes as we may not know all implemented comparison methods
            extern_parent = cls if cls.typeptr_cname else scope.parent_type.base_type

        eq_entry = None
        has_ne = False
        for cmp_method in TypeSlots.richcmp_special_methods:
            for class_scope in class_scopes:
                entry = class_scope.lookup_here(cmp_method)
                if entry is not None:
                    break
            else:
                continue

            cmp_type = cmp_method.strip('_').upper()  # e.g. "__eq__" -> EQ
            code.putln("case Py_%s: {" % cmp_type)
            if cmp_method == '__eq__':
                eq_entry = entry
                # Python itself does not do this optimisation, it seems...
                #code.putln("if (o1 == o2) return __Pyx_NewRef(Py_True);")
            elif cmp_method == '__ne__':
                has_ne = True
                # Python itself does not do this optimisation, it seems...
                #code.putln("if (o1 == o2) return __Pyx_NewRef(Py_False);")
            code.putln("return %s(o1, o2);" % entry.func_cname)
            code.putln("}")

        if eq_entry and not has_ne and not extern_parent:
            code.putln("case Py_NE: {")
            code.putln("PyObject *ret;")
            # Python itself does not do this optimisation, it seems...
            #code.putln("if (o1 == o2) return __Pyx_NewRef(Py_False);")
            code.putln("ret = %s(o1, o2);" % eq_entry.func_cname)
            code.putln("if (likely(ret && ret != Py_NotImplemented)) {")
            code.putln("int b = __Pyx_PyObject_IsTrue(ret); Py_DECREF(ret);")
            code.putln("if (unlikely(b < 0)) return NULL;")
            code.putln("ret = (b) ? Py_False : Py_True;")
            code.putln("Py_INCREF(ret);")
            code.putln("}")
            code.putln("return ret;")
            code.putln("}")

        code.putln("default: {")
        if extern_parent and extern_parent.typeptr_cname:
            code.putln("if (likely(%s->tp_richcompare)) return %s->tp_richcompare(o1, o2, op);" % (
                extern_parent.typeptr_cname, extern_parent.typeptr_cname))
        code.putln("return __Pyx_NewRef(Py_NotImplemented);")
        code.putln("}")

        code.putln("}")  # switch
        code.putln("}")

    def generate_getattro_function(self, scope, code):
        # First try to get the attribute using __getattribute__, if defined, or
        # PyObject_GenericGetAttr.
        #
        # If that raises an AttributeError, call the __getattr__ if defined.
        #
        # In both cases, defined can be in this class, or any base class.
        def lookup_here_or_base(n, tp=None, extern_return=None):
            # Recursive lookup
            if tp is None:
                tp = scope.parent_type
            r = tp.scope.lookup_here(n)
            if r is None:
                if tp.is_external and extern_return is not None:
                    return extern_return
                if tp.base_type is not None:
                    return lookup_here_or_base(n, tp.base_type)
            return r

        has_instance_dict = lookup_here_or_base("__dict__", extern_return="extern")
        getattr_entry = lookup_here_or_base("__getattr__")
        getattribute_entry = lookup_here_or_base("__getattribute__")
        code.putln("")
        code.putln(
            "static PyObject *%s(PyObject *o, PyObject *n) {" % (
                scope.mangle_internal("tp_getattro")))
        if getattribute_entry is not None:
            code.putln(
                "PyObject *v = %s(o, n);" % (
                    getattribute_entry.func_cname))
        else:
            if not has_instance_dict and scope.parent_type.is_final_type:
                # Final with no dict => use faster type attribute lookup.
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached("PyObject_GenericGetAttrNoDict", "ObjectHandling.c"))
                generic_getattr_cfunc = "__Pyx_PyObject_GenericGetAttrNoDict"
            elif not has_instance_dict or has_instance_dict == "extern":
                # No dict in the known ancestors, but don't know about extern ancestors or subtypes.
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached("PyObject_GenericGetAttr", "ObjectHandling.c"))
                generic_getattr_cfunc = "__Pyx_PyObject_GenericGetAttr"
            else:
                generic_getattr_cfunc = "PyObject_GenericGetAttr"
            code.putln(
                "PyObject *v = %s(o, n);" % generic_getattr_cfunc)
        if getattr_entry is not None:
            code.putln(
                "if (!v && PyErr_ExceptionMatches(PyExc_AttributeError)) {")
            code.putln(
                "PyErr_Clear();")
            code.putln(
                "v = %s(o, n);" % (
                    getattr_entry.func_cname))
            code.putln(
                "}")
        code.putln(
            "return v;")
        code.putln(
            "}")

    def generate_setattro_function(self, scope, code):
        # Setting and deleting an attribute are both done through
        # the setattro method, so we dispatch to user's __setattr__
        # or __delattr__ or fall back on PyObject_GenericSetAttr.
        base_type = scope.parent_type.base_type
        set_entry = scope.lookup_here("__setattr__")
        del_entry = scope.lookup_here("__delattr__")
        code.putln("")
        code.putln(
            "static int %s(PyObject *o, PyObject *n, PyObject *v) {" % (
                scope.mangle_internal("tp_setattro")))
        code.putln(
            "if (v) {")
        if set_entry:
            code.putln(
                "return %s(o, n, v);" % (
                    set_entry.func_cname))
        else:
            self.generate_guarded_basetype_call(
                base_type, None, "tp_setattro", "o, n, v", code)
            code.putln(
                "return PyObject_GenericSetAttr(o, n, v);")
        code.putln(
            "}")
        code.putln(
            "else {")
        if del_entry:
            code.putln(
                "return %s(o, n);" % (
                    del_entry.func_cname))
        else:
            self.generate_guarded_basetype_call(
                base_type, None, "tp_setattro", "o, n, v", code)
            code.putln(
                "return PyObject_GenericSetAttr(o, n, 0);")
        code.putln(
            "}")
        code.putln(
            "}")

    def generate_descr_get_function(self, scope, code):
        # The __get__ function of a descriptor object can be
        # called with NULL for the second or third arguments
        # under some circumstances, so we replace them with
        # None in that case.
        user_get_entry = scope.lookup_here("__get__")
        code.putln("")
        code.putln(
            "static PyObject *%s(PyObject *o, PyObject *i, PyObject *c) {" % (
                scope.mangle_internal("tp_descr_get")))
        code.putln(
            "PyObject *r = 0;")
        code.putln(
            "if (!i) i = Py_None;")
        code.putln(
            "if (!c) c = Py_None;")
        #code.put_incref("i", py_object_type)
        #code.put_incref("c", py_object_type)
        code.putln(
            "r = %s(o, i, c);" % (
                user_get_entry.func_cname))
        #code.put_decref("i", py_object_type)
        #code.put_decref("c", py_object_type)
        code.putln(
            "return r;")
        code.putln(
            "}")

    def generate_descr_set_function(self, scope, code):
        # Setting and deleting are both done through the __set__
        # method of a descriptor, so we dispatch to user's __set__
        # or __delete__ or raise an exception.
        base_type = scope.parent_type.base_type
        user_set_entry = scope.lookup_here("__set__")
        user_del_entry = scope.lookup_here("__delete__")
        code.putln("")
        code.putln(
            "static int %s(PyObject *o, PyObject *i, PyObject *v) {" % (
                scope.mangle_internal("tp_descr_set")))
        code.putln(
            "if (v) {")
        if user_set_entry:
            code.putln(
                "return %s(o, i, v);" % (
                    user_set_entry.func_cname))
        else:
            self.generate_guarded_basetype_call(
                base_type, None, "tp_descr_set", "o, i, v", code)
            code.putln(
                'PyErr_SetString(PyExc_NotImplementedError, "__set__");')
            code.putln(
                "return -1;")
        code.putln(
            "}")
        code.putln(
            "else {")
        if user_del_entry:
            code.putln(
                "return %s(o, i);" % (
                    user_del_entry.func_cname))
        else:
            self.generate_guarded_basetype_call(
                base_type, None, "tp_descr_set", "o, i, v", code)
            code.putln(
                'PyErr_SetString(PyExc_NotImplementedError, "__delete__");')
            code.putln(
                "return -1;")
        code.putln(
            "}")
        code.putln(
            "}")

    def generate_property_accessors(self, cclass_scope, code):
        for entry in cclass_scope.property_entries:
            property_scope = entry.scope
            if property_scope.defines_any(["__get__"]):
                self.generate_property_get_function(entry, code)
            if property_scope.defines_any(["__set__", "__del__"]):
                self.generate_property_set_function(entry, code)

    def generate_property_get_function(self, property_entry, code):
        property_scope = property_entry.scope
        property_entry.getter_cname = property_scope.parent_scope.mangle(
            Naming.prop_get_prefix, property_entry.name)
        get_entry = property_scope.lookup_here("__get__")
        code.putln("")
        code.putln(
            "static PyObject *%s(PyObject *o, CYTHON_UNUSED void *x) {" % (
                property_entry.getter_cname))
        code.putln(
            "return %s(o);" % (
                get_entry.func_cname))
        code.putln(
            "}")

    def generate_property_set_function(self, property_entry, code):
        property_scope = property_entry.scope
        property_entry.setter_cname = property_scope.parent_scope.mangle(
            Naming.prop_set_prefix, property_entry.name)
        set_entry = property_scope.lookup_here("__set__")
        del_entry = property_scope.lookup_here("__del__")
        code.putln("")
        code.putln(
            "static int %s(PyObject *o, PyObject *v, CYTHON_UNUSED void *x) {" % (
                property_entry.setter_cname))
        code.putln(
            "if (v) {")
        if set_entry:
            code.putln(
                "return %s(o, v);" % (
                    set_entry.func_cname))
        else:
            code.putln(
                'PyErr_SetString(PyExc_NotImplementedError, "__set__");')
            code.putln(
                "return -1;")
        code.putln(
            "}")
        code.putln(
            "else {")
        if del_entry:
            code.putln(
                "return %s(o);" % (
                    del_entry.func_cname))
        else:
            code.putln(
                'PyErr_SetString(PyExc_NotImplementedError, "__del__");')
            code.putln(
                "return -1;")
        code.putln(
            "}")
        code.putln(
            "}")

    def generate_typeobj_definition(self, modname, entry, code):
        type = entry.type
        scope = type.scope
        for suite in TypeSlots.substructures:
            suite.generate_substructure(scope, code)
        code.putln("")
        if entry.visibility == 'public':
            header = "DL_EXPORT(PyTypeObject) %s = {"
        else:
            header = "static PyTypeObject %s = {"
        #code.putln(header % scope.parent_type.typeobj_cname)
        code.putln(header % type.typeobj_cname)
        code.putln(
            "PyVarObject_HEAD_INIT(0, 0)")
        code.putln(
            '"%s.%s", /*tp_name*/' % (
                self.full_module_name, scope.class_name))
        if type.typedef_flag:
            objstruct = type.objstruct_cname
        else:
            objstruct = "struct %s" % type.objstruct_cname
        code.putln(
            "sizeof(%s), /*tp_basicsize*/" % objstruct)
        code.putln(
            "0, /*tp_itemsize*/")
        for slot in TypeSlots.slot_table:
            slot.generate(scope, code)
        code.putln(
            "};")

    def generate_method_table(self, env, code):
        if env.is_c_class_scope and not env.pyfunc_entries:
            return
        binding = env.directives['binding']

        code.putln("")
        wrapper_code_writer = code.insertion_point()

        code.putln(
            "static PyMethodDef %s[] = {" % (
                env.method_table_cname))
        for entry in env.pyfunc_entries:
            if not entry.fused_cfunction and not (binding and entry.is_overridable):
                code.put_pymethoddef(entry, ",", wrapper_code_writer=wrapper_code_writer)
        code.putln(
            "{0, 0, 0, 0}")
        code.putln(
            "};")

        if wrapper_code_writer.getvalue():
            wrapper_code_writer.putln("")

    def generate_dict_getter_function(self, scope, code):
        dict_attr = scope.lookup_here("__dict__")
        if not dict_attr or not dict_attr.is_variable:
            return
        func_name = scope.mangle_internal("__dict__getter")
        dict_name = dict_attr.cname
        code.putln("")
        code.putln("static PyObject *%s(PyObject *o, CYTHON_UNUSED void *x) {" % func_name)
        self.generate_self_cast(scope, code)
        code.putln("if (unlikely(!p->%s)){" % dict_name)
        code.putln("p->%s = PyDict_New();" % dict_name)
        code.putln("}")
        code.putln("Py_XINCREF(p->%s);" % dict_name)
        code.putln("return p->%s;" % dict_name)
        code.putln("}")

    def generate_getset_table(self, env, code):
        if env.property_entries:
            code.putln("")
            code.putln(
                "static struct PyGetSetDef %s[] = {" %
                env.getset_table_cname)
            for entry in env.property_entries:
                doc = entry.doc
                if doc:
                    if doc.is_unicode:
                        doc = doc.as_utf8_string()
                    doc_code = doc.as_c_string_literal()
                else:
                    doc_code = "0"
                code.putln(
                    '{(char *)"%s", %s, %s, (char *)%s, 0},' % (
                        entry.name,
                        entry.getter_cname or "0",
                        entry.setter_cname or "0",
                        doc_code))
            code.putln(
                "{0, 0, 0, 0, 0}")
            code.putln(
                "};")

    def create_import_star_conversion_utility_code(self, env):
        # Create all conversion helpers that are needed for "import *" assignments.
        # Must be done before code generation to support CythonUtilityCode.
        for name, entry in sorted(env.entries.items()):
            if entry.is_cglobal and entry.used:
                if not entry.type.is_pyobject:
                    entry.type.create_from_py_utility_code(env)

    def generate_import_star(self, env, code):
        env.use_utility_code(UtilityCode.load_cached("CStringEquals", "StringTools.c"))
        code.putln()
        code.enter_cfunc_scope()  # as we need labels
        code.putln("static int %s(PyObject *o, PyObject* py_name, char *name) {" % Naming.import_star_set)

        code.putln("static const char* internal_type_names[] = {")
        for name, entry in sorted(env.entries.items()):
            if entry.is_type:
                code.putln('"%s",' % name)
        code.putln("0")
        code.putln("};")

        code.putln("const char** type_name = internal_type_names;")
        code.putln("while (*type_name) {")
        code.putln("if (__Pyx_StrEq(name, *type_name)) {")
        code.putln('PyErr_Format(PyExc_TypeError, "Cannot overwrite C type %s", name);')
        code.putln('goto bad;')
        code.putln("}")
        code.putln("type_name++;")
        code.putln("}")

        old_error_label = code.new_error_label()
        code.putln("if (0);")  # so the first one can be "else if"
        msvc_count = 0
        for name, entry in sorted(env.entries.items()):
            if entry.is_cglobal and entry.used and not entry.type.is_const:
                msvc_count += 1
                if msvc_count % 100 == 0:
                    code.putln("#ifdef _MSC_VER")
                    code.putln("if (0);  /* Workaround for MSVC C1061. */")
                    code.putln("#endif")
                code.putln('else if (__Pyx_StrEq(name, "%s")) {' % name)
                if entry.type.is_pyobject:
                    if entry.type.is_extension_type or entry.type.is_builtin_type:
                        code.putln("if (!(%s)) %s;" % (
                            entry.type.type_test_code("o"),
                            code.error_goto(entry.pos)))
                    code.putln("Py_INCREF(o);")
                    code.put_decref(entry.cname, entry.type, nanny=False)
                    code.putln("%s = %s;" % (
                        entry.cname,
                        PyrexTypes.typecast(entry.type, py_object_type, "o")))
                elif entry.type.create_from_py_utility_code(env):
                    # if available, utility code was already created in self.prepare_utility_code()
                    code.putln(entry.type.from_py_call_code(
                        'o', entry.cname, entry.pos, code))
                else:
                    code.putln('PyErr_Format(PyExc_TypeError, "Cannot convert Python object %s to %s");' % (
                        name, entry.type))
                    code.putln(code.error_goto(entry.pos))
                code.putln("}")
        code.putln("else {")
        code.putln("if (PyObject_SetAttr(%s, py_name, o) < 0) goto bad;" % Naming.module_cname)
        code.putln("}")
        code.putln("return 0;")
        if code.label_used(code.error_label):
            code.put_label(code.error_label)
            # This helps locate the offending name.
            code.put_add_traceback(self.full_module_name)
        code.error_label = old_error_label
        code.putln("bad:")
        code.putln("return -1;")
        code.putln("}")
        code.putln("")
        code.putln(UtilityCode.load_as_string("ImportStar", "ImportExport.c")[1])
        code.exit_cfunc_scope()  # done with labels

    def generate_module_init_func(self, imported_modules, env, code):
        subfunction = self.mod_init_subfunction(self.pos, self.scope, code)

        code.enter_cfunc_scope(self.scope)
        code.putln("")
        code.putln(UtilityCode.load_as_string("PyModInitFuncType", "ModuleSetupCode.c")[0])
        header2 = "__Pyx_PyMODINIT_FUNC init%s(void)" % env.module_name
        header3 = "__Pyx_PyMODINIT_FUNC %s(void)" % self.mod_init_func_cname('PyInit', env)
        code.putln("#if PY_MAJOR_VERSION < 3")
        # Optimise for small code size as the module init function is only executed once.
        code.putln("%s CYTHON_SMALL_CODE; /*proto*/" % header2)
        code.putln(header2)
        code.putln("#else")
        code.putln("%s CYTHON_SMALL_CODE; /*proto*/" % header3)
        code.putln(header3)

        # CPython 3.5+ supports multi-phase module initialisation (gives access to __spec__, __file__, etc.)
        code.putln("#if CYTHON_PEP489_MULTI_PHASE_INIT")
        code.putln("{")
        code.putln("return PyModuleDef_Init(&%s);" % Naming.pymoduledef_cname)
        code.putln("}")

        mod_create_func = UtilityCode.load_as_string("ModuleCreationPEP489", "ModuleSetupCode.c")[1]
        code.put(mod_create_func)

        code.putln("")
        # main module init code lives in Py_mod_exec function, not in PyInit function
        code.putln("static CYTHON_SMALL_CODE int %s(PyObject *%s)" % (
            self.mod_init_func_cname(Naming.pymodule_exec_func_cname, env),
            Naming.pymodinit_module_arg))
        code.putln("#endif")  # PEP489

        code.putln("#endif")  # Py3

        # start of module init/exec function (pre/post PEP 489)
        code.putln("{")

        tempdecl_code = code.insertion_point()

        profile = code.globalstate.directives['profile']
        linetrace = code.globalstate.directives['linetrace']
        if profile or linetrace:
            code.globalstate.use_utility_code(UtilityCode.load_cached("Profile", "Profile.c"))

        code.put_declare_refcount_context()
        code.putln("#if CYTHON_PEP489_MULTI_PHASE_INIT")
        # Most extension modules simply can't deal with it, and Cython isn't ready either.
        # See issues listed here: https://docs.python.org/3/c-api/init.html#sub-interpreter-support
        code.putln("if (%s) {" % Naming.module_cname)
        # Hack: enforce single initialisation.
        code.putln("if (%s == %s) return 0;" % (
            Naming.module_cname,
            Naming.pymodinit_module_arg,
        ))
        code.putln('PyErr_SetString(PyExc_RuntimeError,'
                   ' "Module \'%s\' has already been imported. Re-initialisation is not supported.");' %
                   env.module_name)
        code.putln("return -1;")
        code.putln("}")
        code.putln("#elif PY_MAJOR_VERSION >= 3")
        # Hack: enforce single initialisation also on reimports under different names on Python 3 (with PEP 3121/489).
        code.putln("if (%s) return __Pyx_NewRef(%s);" % (
            Naming.module_cname,
            Naming.module_cname,
        ))
        code.putln("#endif")

        if profile or linetrace:
            tempdecl_code.put_trace_declarations()
            code.put_trace_frame_init()

        refnanny_import_code = UtilityCode.load_as_string("ImportRefnannyAPI", "ModuleSetupCode.c")[1]
        code.putln(refnanny_import_code.rstrip())
        code.put_setup_refcount_context(header3)

        env.use_utility_code(UtilityCode.load("CheckBinaryVersion", "ModuleSetupCode.c"))
        code.put_error_if_neg(self.pos, "__Pyx_check_binary_version()")

        code.putln("#ifdef __Pxy_PyFrame_Initialize_Offsets")
        code.putln("__Pxy_PyFrame_Initialize_Offsets();")
        code.putln("#endif")
        code.putln("%s = PyTuple_New(0); %s" % (
            Naming.empty_tuple, code.error_goto_if_null(Naming.empty_tuple, self.pos)))
        code.putln("%s = PyBytes_FromStringAndSize(\"\", 0); %s" % (
            Naming.empty_bytes, code.error_goto_if_null(Naming.empty_bytes, self.pos)))
        code.putln("%s = PyUnicode_FromStringAndSize(\"\", 0); %s" % (
            Naming.empty_unicode, code.error_goto_if_null(Naming.empty_unicode, self.pos)))

        for ext_type in ('CyFunction', 'FusedFunction', 'Coroutine', 'Generator', 'AsyncGen', 'StopAsyncIteration'):
            code.putln("#ifdef __Pyx_%s_USED" % ext_type)
            code.put_error_if_neg(self.pos, "__pyx_%s_init()" % ext_type)
            code.putln("#endif")

        code.putln("/*--- Library function declarations ---*/")
        if env.directives['np_pythran']:
            code.put_error_if_neg(self.pos, "_import_array()")

        code.putln("/*--- Threads initialization code ---*/")
        code.putln("#if defined(WITH_THREAD) && PY_VERSION_HEX < 0x030700F0 "
                   "&& defined(__PYX_FORCE_INIT_THREADS) && __PYX_FORCE_INIT_THREADS")
        code.putln("PyEval_InitThreads();")
        code.putln("#endif")

        code.putln("/*--- Module creation code ---*/")
        self.generate_module_creation_code(env, code)

        code.putln("/*--- Initialize various global constants etc. ---*/")
        code.put_error_if_neg(self.pos, "__Pyx_InitGlobals()")

        code.putln("#if PY_MAJOR_VERSION < 3 && (__PYX_DEFAULT_STRING_ENCODING_IS_ASCII || "
                   "__PYX_DEFAULT_STRING_ENCODING_IS_DEFAULT)")
        code.put_error_if_neg(self.pos, "__Pyx_init_sys_getdefaultencoding_params()")
        code.putln("#endif")

        code.putln("if (%s%s) {" % (Naming.module_is_main, self.full_module_name.replace('.', '__')))
        code.put_error_if_neg(self.pos, 'PyObject_SetAttr(%s, %s, %s)' % (
            env.module_cname,
            code.intern_identifier(EncodedString("__name__")),
            code.intern_identifier(EncodedString("__main__"))))
        code.putln("}")

        # set up __file__ and __path__, then add the module to sys.modules
        self.generate_module_import_setup(env, code)

        if Options.cache_builtins:
            code.putln("/*--- Builtin init code ---*/")
            code.put_error_if_neg(self.pos, "__Pyx_InitCachedBuiltins()")

        code.putln("/*--- Constants init code ---*/")
        code.put_error_if_neg(self.pos, "__Pyx_InitCachedConstants()")

        code.putln("/*--- Global type/function init code ---*/")

        with subfunction("Global init code") as inner_code:
            self.generate_global_init_code(env, inner_code)

        with subfunction("Variable export code") as inner_code:
            self.generate_c_variable_export_code(env, inner_code)

        with subfunction("Function export code") as inner_code:
            self.generate_c_function_export_code(env, inner_code)

        with subfunction("Type init code") as inner_code:
            self.generate_type_init_code(env, inner_code)

        with subfunction("Type import code") as inner_code:
            for module in imported_modules:
                self.generate_type_import_code_for_module(module, env, inner_code)

        with subfunction("Variable import code") as inner_code:
            for module in imported_modules:
                self.generate_c_variable_import_code_for_module(module, env, inner_code)

        with subfunction("Function import code") as inner_code:
            for module in imported_modules:
                self.specialize_fused_types(module)
                self.generate_c_function_import_code_for_module(module, env, inner_code)

        code.putln("/*--- Execution code ---*/")
        code.mark_pos(None)

        code.putln("#if defined(__Pyx_Generator_USED) || defined(__Pyx_Coroutine_USED)")
        code.put_error_if_neg(self.pos, "__Pyx_patch_abc()")
        code.putln("#endif")

        if profile or linetrace:
            code.put_trace_call(header3, self.pos, nogil=not code.funcstate.gil_owned)
            code.funcstate.can_trace = True

        self.body.generate_execution_code(code)

        if profile or linetrace:
            code.funcstate.can_trace = False
            code.put_trace_return("Py_None", nogil=not code.funcstate.gil_owned)

        code.putln()
        code.putln("/*--- Wrapped vars code ---*/")
        self.generate_wrapped_entries_code(env, code)
        code.putln()

        if Options.generate_cleanup_code:
            code.globalstate.use_utility_code(
                UtilityCode.load_cached("RegisterModuleCleanup", "ModuleSetupCode.c"))
            code.putln("if (__Pyx_RegisterCleanup()) %s;" % code.error_goto(self.pos))

        code.put_goto(code.return_label)
        code.put_label(code.error_label)
        for cname, type in code.funcstate.all_managed_temps():
            code.put_xdecref(cname, type)
        code.putln('if (%s) {' % env.module_cname)
        code.putln('if (%s) {' % env.module_dict_cname)
        code.put_add_traceback("init %s" % env.qualified_name)
        code.globalstate.use_utility_code(Nodes.traceback_utility_code)
        # Module reference and module dict are in global variables which might still be needed
        # for cleanup, atexit code, etc., so leaking is better than crashing.
        # At least clearing the module dict here might be a good idea, but could still break
        # user code in atexit or other global registries.
        ##code.put_decref_clear(env.module_dict_cname, py_object_type, nanny=False)
        code.putln('}')
        code.put_decref_clear(env.module_cname, py_object_type, nanny=False, clear_before_decref=True)
        code.putln('} else if (!PyErr_Occurred()) {')
        code.putln('PyErr_SetString(PyExc_ImportError, "init %s");' % env.qualified_name)
        code.putln('}')
        code.put_label(code.return_label)

        code.put_finish_refcount_context()

        code.putln("#if CYTHON_PEP489_MULTI_PHASE_INIT")
        code.putln("return (%s != NULL) ? 0 : -1;" % env.module_cname)
        code.putln("#elif PY_MAJOR_VERSION >= 3")
        code.putln("return %s;" % env.module_cname)
        code.putln("#else")
        code.putln("return;")
        code.putln("#endif")
        code.putln('}')

        tempdecl_code.put_temp_declarations(code.funcstate)

        code.exit_cfunc_scope()

    def mod_init_subfunction(self, pos, scope, orig_code):
        """
        Return a context manager that allows deviating the module init code generation
        into a separate function and instead inserts a call to it.

        Can be reused sequentially to create multiple functions.
        The functions get inserted at the point where the context manager was created.
        The call gets inserted where the context manager is used (on entry).
        """
        prototypes = orig_code.insertion_point()
        prototypes.putln("")
        function_code = orig_code.insertion_point()
        function_code.putln("")

        class ModInitSubfunction(object):
            def __init__(self, code_type):
                cname = '_'.join(code_type.lower().split())
                assert re.match("^[a-z0-9_]+$", cname)
                self.cfunc_name = "__Pyx_modinit_%s" % cname
                self.description = code_type
                self.tempdecl_code = None
                self.call_code = None

            def __enter__(self):
                self.call_code = orig_code.insertion_point()
                code = function_code
                code.enter_cfunc_scope(scope)
                prototypes.putln("static CYTHON_SMALL_CODE int %s(void); /*proto*/" % self.cfunc_name)
                code.putln("static int %s(void) {" % self.cfunc_name)
                code.put_declare_refcount_context()
                self.tempdecl_code = code.insertion_point()
                code.put_setup_refcount_context(self.cfunc_name)
                # Leave a grepable marker that makes it easy to find the generator source.
                code.putln("/*--- %s ---*/" % self.description)
                return code

            def __exit__(self, *args):
                code = function_code
                code.put_finish_refcount_context()
                code.putln("return 0;")

                self.tempdecl_code.put_temp_declarations(code.funcstate)
                self.tempdecl_code = None

                needs_error_handling = code.label_used(code.error_label)
                if needs_error_handling:
                    code.put_label(code.error_label)
                    for cname, type in code.funcstate.all_managed_temps():
                        code.put_xdecref(cname, type)
                    code.put_finish_refcount_context()
                    code.putln("return -1;")
                code.putln("}")
                code.exit_cfunc_scope()
                code.putln("")

                if needs_error_handling:
                    self.call_code.putln(
                        self.call_code.error_goto_if_neg("%s()" % self.cfunc_name, pos))
                else:
                    self.call_code.putln("(void)%s();" % self.cfunc_name)
                self.call_code = None

        return ModInitSubfunction

    def generate_module_import_setup(self, env, code):
        module_path = env.directives['set_initial_path']
        if module_path == 'SOURCEFILE':
            module_path = self.pos[0].filename

        if module_path:
            code.putln('if (!CYTHON_PEP489_MULTI_PHASE_INIT) {')
            code.putln('if (PyObject_SetAttrString(%s, "__file__", %s) < 0) %s;' % (
                env.module_cname,
                code.globalstate.get_py_string_const(
                    EncodedString(decode_filename(module_path))).cname,
                code.error_goto(self.pos)))
            code.putln("}")

            if env.is_package:
                # set __path__ to mark the module as package
                code.putln('if (!CYTHON_PEP489_MULTI_PHASE_INIT) {')
                temp = code.funcstate.allocate_temp(py_object_type, True)
                code.putln('%s = Py_BuildValue("[O]", %s); %s' % (
                    temp,
                    code.globalstate.get_py_string_const(
                        EncodedString(decode_filename(
                            os.path.dirname(module_path)))).cname,
                    code.error_goto_if_null(temp, self.pos)))
                code.put_gotref(temp)
                code.putln(
                    'if (PyObject_SetAttrString(%s, "__path__", %s) < 0) %s;' % (
                        env.module_cname, temp, code.error_goto(self.pos)))
                code.put_decref_clear(temp, py_object_type)
                code.funcstate.release_temp(temp)
                code.putln("}")

        elif env.is_package:
            # packages require __path__, so all we can do is try to figure
            # out the module path at runtime by rerunning the import lookup
            code.putln("if (!CYTHON_PEP489_MULTI_PHASE_INIT) {")
            package_name, _ = self.full_module_name.rsplit('.', 1)
            if '.' in package_name:
                parent_name = '"%s"' % (package_name.rsplit('.', 1)[0],)
            else:
                parent_name = 'NULL'
            code.globalstate.use_utility_code(UtilityCode.load(
                "SetPackagePathFromImportLib", "ImportExport.c"))
            code.putln(code.error_goto_if_neg(
                '__Pyx_SetPackagePathFromImportLib(%s, %s)' % (
                    parent_name,
                    code.globalstate.get_py_string_const(
                        EncodedString(env.module_name)).cname),
                self.pos))
            code.putln("}")

        # CPython may not have put us into sys.modules yet, but relative imports and reimports require it
        fq_module_name = self.full_module_name
        if fq_module_name.endswith('.__init__'):
            fq_module_name = fq_module_name[:-len('.__init__')]
        code.putln("#if PY_MAJOR_VERSION >= 3")
        code.putln("{")
        code.putln("PyObject *modules = PyImport_GetModuleDict(); %s" %
                   code.error_goto_if_null("modules", self.pos))
        code.putln('if (!PyDict_GetItemString(modules, "%s")) {' % fq_module_name)
        code.putln(code.error_goto_if_neg('PyDict_SetItemString(modules, "%s", %s)' % (
            fq_module_name, env.module_cname), self.pos))
        code.putln("}")
        code.putln("}")
        code.putln("#endif")

    def generate_module_cleanup_func(self, env, code):
        if not Options.generate_cleanup_code:
            return

        code.putln('static void %s(CYTHON_UNUSED PyObject *self) {' %
                   Naming.cleanup_cname)
        code.enter_cfunc_scope(env)

        if Options.generate_cleanup_code >= 2:
            code.putln("/*--- Global cleanup code ---*/")
            rev_entries = list(env.var_entries)
            rev_entries.reverse()
            for entry in rev_entries:
                if entry.visibility != 'extern':
                    if entry.type.is_pyobject and entry.used:
                        code.put_xdecref_clear(
                            entry.cname, entry.type,
                            clear_before_decref=True,
                            nanny=False)
        code.putln("__Pyx_CleanupGlobals();")
        if Options.generate_cleanup_code >= 3:
            code.putln("/*--- Type import cleanup code ---*/")
            for ext_type in sorted(env.types_imported, key=operator.attrgetter('typeptr_cname')):
                code.put_xdecref_clear(
                    ext_type.typeptr_cname, ext_type,
                    clear_before_decref=True,
                    nanny=False)
        if Options.cache_builtins:
            code.putln("/*--- Builtin cleanup code ---*/")
            for entry in env.cached_builtins:
                code.put_xdecref_clear(
                    entry.cname, PyrexTypes.py_object_type,
                    clear_before_decref=True,
                    nanny=False)
        code.putln("/*--- Intern cleanup code ---*/")
        code.put_decref_clear(Naming.empty_tuple,
                              PyrexTypes.py_object_type,
                              clear_before_decref=True,
                              nanny=False)
        for entry in env.c_class_entries:
            cclass_type = entry.type
            if cclass_type.is_external or cclass_type.base_type:
                continue
            if cclass_type.scope.directives.get('freelist', 0):
                scope = cclass_type.scope
                freelist_name = scope.mangle_internal(Naming.freelist_name)
                freecount_name = scope.mangle_internal(Naming.freecount_name)
                code.putln("while (%s > 0) {" % freecount_name)
                code.putln("PyObject* o = (PyObject*)%s[--%s];" % (
                    freelist_name, freecount_name))
                code.putln("(*Py_TYPE(o)->tp_free)(o);")
                code.putln("}")
#        for entry in env.pynum_entries:
#            code.put_decref_clear(entry.cname,
#                                  PyrexTypes.py_object_type,
#                                  nanny=False)
#        for entry in env.all_pystring_entries:
#            if entry.is_interned:
#                code.put_decref_clear(entry.pystring_cname,
#                                      PyrexTypes.py_object_type,
#                                      nanny=False)
#        for entry in env.default_entries:
#            if entry.type.is_pyobject and entry.used:
#                code.putln("Py_DECREF(%s); %s = 0;" % (
#                    code.entry_as_pyobject(entry), entry.cname))
        if Options.pre_import is not None:
            code.put_decref_clear(Naming.preimport_cname, py_object_type,
                                  nanny=False, clear_before_decref=True)
        for cname in [env.module_dict_cname, Naming.cython_runtime_cname, Naming.builtins_cname]:
            code.put_decref_clear(cname, py_object_type, nanny=False, clear_before_decref=True)

    def generate_main_method(self, env, code):
        module_is_main = "%s%s" % (Naming.module_is_main, self.full_module_name.replace('.', '__'))
        if Options.embed == "main":
            wmain = "wmain"
        else:
            wmain = Options.embed
        main_method = UtilityCode.load_cached("MainFunction", "Embed.c")
        code.globalstate.use_utility_code(
            main_method.specialize(
                module_name=env.module_name,
                module_is_main=module_is_main,
                main_method=Options.embed,
                wmain_method=wmain))

    def mod_init_func_cname(self, prefix, env):
        return '%s_%s' % (prefix, env.module_name)

    def generate_pymoduledef_struct(self, env, code):
        if env.doc:
            doc = "%s" % code.get_string_const(env.doc)
        else:
            doc = "0"
        if Options.generate_cleanup_code:
            cleanup_func = "(freefunc)%s" % Naming.cleanup_cname
        else:
            cleanup_func = 'NULL'

        code.putln("")
        code.putln("#if PY_MAJOR_VERSION >= 3")
        code.putln("#if CYTHON_PEP489_MULTI_PHASE_INIT")
        exec_func_cname = self.mod_init_func_cname(Naming.pymodule_exec_func_cname, env)
        code.putln("static PyObject* %s(PyObject *spec, PyModuleDef *def); /*proto*/" %
                   Naming.pymodule_create_func_cname)
        code.putln("static int %s(PyObject* module); /*proto*/" % exec_func_cname)

        code.putln("static PyModuleDef_Slot %s[] = {" % Naming.pymoduledef_slots_cname)
        code.putln("{Py_mod_create, (void*)%s}," % Naming.pymodule_create_func_cname)
        code.putln("{Py_mod_exec, (void*)%s}," % exec_func_cname)
        code.putln("{0, NULL}")
        code.putln("};")
        code.putln("#endif")

        code.putln("")
        code.putln("static struct PyModuleDef %s = {" % Naming.pymoduledef_cname)
        code.putln("  PyModuleDef_HEAD_INIT,")
        code.putln('  "%s",' % env.module_name)
        code.putln("  %s, /* m_doc */" % doc)
        code.putln("#if CYTHON_PEP489_MULTI_PHASE_INIT")
        code.putln("  0, /* m_size */")
        code.putln("#else")
        code.putln("  -1, /* m_size */")
        code.putln("#endif")
        code.putln("  %s /* m_methods */," % env.method_table_cname)
        code.putln("#if CYTHON_PEP489_MULTI_PHASE_INIT")
        code.putln("  %s, /* m_slots */" % Naming.pymoduledef_slots_cname)
        code.putln("#else")
        code.putln("  NULL, /* m_reload */")
        code.putln("#endif")
        code.putln("  NULL, /* m_traverse */")
        code.putln("  NULL, /* m_clear */")
        code.putln("  %s /* m_free */" % cleanup_func)
        code.putln("};")
        code.putln("#endif")

    def generate_module_creation_code(self, env, code):
        # Generate code to create the module object and
        # install the builtins.
        if env.doc:
            doc = "%s" % code.get_string_const(env.doc)
        else:
            doc = "0"

        code.putln("#if CYTHON_PEP489_MULTI_PHASE_INIT")
        code.putln("%s = %s;" % (
            env.module_cname,
            Naming.pymodinit_module_arg))
        code.put_incref(env.module_cname, py_object_type, nanny=False)
        code.putln("#else")
        code.putln("#if PY_MAJOR_VERSION < 3")
        code.putln(
            '%s = Py_InitModule4("%s", %s, %s, 0, PYTHON_API_VERSION); Py_XINCREF(%s);' % (
                env.module_cname,
                env.module_name,
                env.method_table_cname,
                doc,
                env.module_cname))
        code.putln("#else")
        code.putln(
            "%s = PyModule_Create(&%s);" % (
                env.module_cname,
                Naming.pymoduledef_cname))
        code.putln("#endif")
        code.putln(code.error_goto_if_null(env.module_cname, self.pos))
        code.putln("#endif")  # CYTHON_PEP489_MULTI_PHASE_INIT

        code.putln(
            "%s = PyModule_GetDict(%s); %s" % (
                env.module_dict_cname, env.module_cname,
                code.error_goto_if_null(env.module_dict_cname, self.pos)))
        code.put_incref(env.module_dict_cname, py_object_type, nanny=False)

        code.putln(
            '%s = PyImport_AddModule(__Pyx_BUILTIN_MODULE_NAME); %s' % (
                Naming.builtins_cname,
                code.error_goto_if_null(Naming.builtins_cname, self.pos)))
        code.put_incref(Naming.builtins_cname, py_object_type, nanny=False)
        code.putln(
            '%s = PyImport_AddModule((char *) "cython_runtime"); %s' % (
                Naming.cython_runtime_cname,
                code.error_goto_if_null(Naming.cython_runtime_cname, self.pos)))
        code.put_incref(Naming.cython_runtime_cname, py_object_type, nanny=False)
        code.putln(
            'if (PyObject_SetAttrString(%s, "__builtins__", %s) < 0) %s;' % (
                env.module_cname,
                Naming.builtins_cname,
                code.error_goto(self.pos)))
        if Options.pre_import is not None:
            code.putln(
                '%s = PyImport_AddModule("%s"); %s' % (
                    Naming.preimport_cname,
                    Options.pre_import,
                    code.error_goto_if_null(Naming.preimport_cname, self.pos)))
            code.put_incref(Naming.preimport_cname, py_object_type, nanny=False)

    def generate_global_init_code(self, env, code):
        # Generate code to initialise global PyObject *
        # variables to None.
        for entry in env.var_entries:
            if entry.visibility != 'extern':
                if entry.used:
                    entry.type.global_init_code(entry, code)

    def generate_wrapped_entries_code(self, env, code):
        for name, entry in sorted(env.entries.items()):
            if (entry.create_wrapper
                    and not entry.is_type
                    and entry.scope is env):
                if not entry.type.create_to_py_utility_code(env):
                    error(entry.pos, "Cannot convert '%s' to Python object" % entry.type)
                code.putln("{")
                code.putln("PyObject* wrapped = %s(%s);"  % (
                    entry.type.to_py_function,
                    entry.cname))
                code.putln(code.error_goto_if_null("wrapped", entry.pos))
                code.putln(
                    'if (PyObject_SetAttrString(%s, "%s", wrapped) < 0) %s;' % (
                        env.module_cname,
                        name,
                        code.error_goto(entry.pos)))
                code.putln("}")

    def generate_c_variable_export_code(self, env, code):
        # Generate code to create PyCFunction wrappers for exported C functions.
        entries = []
        for entry in env.var_entries:
            if (entry.api
                    or entry.defined_in_pxd
                    or (Options.cimport_from_pyx and not entry.visibility == 'extern')):
                entries.append(entry)
        if entries:
            env.use_utility_code(UtilityCode.load_cached("VoidPtrExport", "ImportExport.c"))
            for entry in entries:
                signature = entry.type.empty_declaration_code()
                name = code.intern_identifier(entry.name)
                code.putln('if (__Pyx_ExportVoidPtr(%s, (void *)&%s, "%s") < 0) %s' % (
                    name, entry.cname, signature,
                    code.error_goto(self.pos)))

    def generate_c_function_export_code(self, env, code):
        # Generate code to create PyCFunction wrappers for exported C functions.
        entries = []
        for entry in env.cfunc_entries:
            if (entry.api
                    or entry.defined_in_pxd
                    or (Options.cimport_from_pyx and not entry.visibility == 'extern')):
                entries.append(entry)
        if entries:
            env.use_utility_code(
                UtilityCode.load_cached("FunctionExport", "ImportExport.c"))
            # Note: while this looks like it could be more cheaply stored and read from a struct array,
            # investigation shows that the resulting binary is smaller with repeated functions calls.
            for entry in entries:
                signature = entry.type.signature_string()
                code.putln('if (__Pyx_ExportFunction("%s", (void (*)(void))%s, "%s") < 0) %s' % (
                    entry.name,
                    entry.cname,
                    signature,
                    code.error_goto(self.pos)))

    def generate_type_import_code_for_module(self, module, env, code):
        # Generate type import code for all exported extension types in
        # an imported module.
        #if module.c_class_entries:
        with ModuleImportGenerator(code) as import_generator:
            for entry in module.c_class_entries:
                if entry.defined_in_pxd:
                    self.generate_type_import_code(env, entry.type, entry.pos, code, import_generator)

    def specialize_fused_types(self, pxd_env):
        """
        If fused c(p)def functions are defined in an imported pxd, but not
        used in this implementation file, we still have fused entries and
        not specialized ones. This method replaces any fused entries with their
        specialized ones.
        """
        for entry in pxd_env.cfunc_entries[:]:
            if entry.type.is_fused:
                # This call modifies the cfunc_entries in-place
                entry.type.get_all_specialized_function_types()

    def generate_c_variable_import_code_for_module(self, module, env, code):
        # Generate import code for all exported C functions in a cimported module.
        entries = []
        for entry in module.var_entries:
            if entry.defined_in_pxd:
                entries.append(entry)
        if entries:
            env.use_utility_code(
                UtilityCode.load_cached("VoidPtrImport", "ImportExport.c"))
            temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
            code.putln(
                '%s = PyImport_ImportModule("%s"); if (!%s) %s' % (
                    temp,
                    module.qualified_name,
                    temp,
                    code.error_goto(self.pos)))
            code.put_gotref(temp)
            for entry in entries:
                if env is module:
                    cname = entry.cname
                else:
                    cname = module.mangle(Naming.varptr_prefix, entry.name)
                signature = entry.type.empty_declaration_code()
                code.putln(
                    'if (__Pyx_ImportVoidPtr(%s, "%s", (void **)&%s, "%s") < 0) %s' % (
                        temp, entry.name, cname, signature,
                        code.error_goto(self.pos)))
            code.put_decref_clear(temp, py_object_type)
            code.funcstate.release_temp(temp)

    def generate_c_function_import_code_for_module(self, module, env, code):
        # Generate import code for all exported C functions in a cimported module.
        entries = []
        for entry in module.cfunc_entries:
            if entry.defined_in_pxd and entry.used:
                entries.append(entry)
        if entries:
            env.use_utility_code(
                UtilityCode.load_cached("FunctionImport", "ImportExport.c"))
            temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
            code.putln(
                '%s = PyImport_ImportModule("%s"); if (!%s) %s' % (
                    temp,
                    module.qualified_name,
                    temp,
                    code.error_goto(self.pos)))
            code.put_gotref(temp)
            for entry in entries:
                code.putln(
                    'if (__Pyx_ImportFunction(%s, "%s", (void (**)(void))&%s, "%s") < 0) %s' % (
                        temp,
                        entry.name,
                        entry.cname,
                        entry.type.signature_string(),
                        code.error_goto(self.pos)))
            code.put_decref_clear(temp, py_object_type)
            code.funcstate.release_temp(temp)

    def generate_type_init_code(self, env, code):
        # Generate type import code for extern extension types
        # and type ready code for non-extern ones.
        with ModuleImportGenerator(code) as import_generator:
            for entry in env.c_class_entries:
                if entry.visibility == 'extern' and not entry.utility_code_definition:
                    self.generate_type_import_code(env, entry.type, entry.pos, code, import_generator)
                else:
                    self.generate_base_type_import_code(env, entry, code, import_generator)
                    self.generate_exttype_vtable_init_code(entry, code)
                    if entry.type.early_init:
                        self.generate_type_ready_code(entry, code)

    def generate_base_type_import_code(self, env, entry, code, import_generator):
        base_type = entry.type.base_type
        if (base_type and base_type.module_name != env.qualified_name and not
                base_type.is_builtin_type and not entry.utility_code_definition):
            self.generate_type_import_code(env, base_type, self.pos, code, import_generator)

    def generate_type_import_code(self, env, type, pos, code, import_generator):
        # If not already done, generate code to import the typeobject of an
        # extension type defined in another module, and extract its C method
        # table pointer if any.
        if type in env.types_imported:
            return
        if type.name not in Code.ctypedef_builtins_map:
            # see corresponding condition in generate_type_import_call() below!
            code.globalstate.use_utility_code(
                UtilityCode.load_cached("TypeImport", "ImportExport.c"))
        self.generate_type_import_call(type, code, import_generator, error_pos=pos)
        if type.vtabptr_cname:
            code.globalstate.use_utility_code(
                UtilityCode.load_cached('GetVTable', 'ImportExport.c'))
            code.putln("%s = (struct %s*)__Pyx_GetVtable(%s->tp_dict); %s" % (
                type.vtabptr_cname,
                type.vtabstruct_cname,
                type.typeptr_cname,
                code.error_goto_if_null(type.vtabptr_cname, pos)))
        env.types_imported.add(type)

    def generate_type_import_call(self, type, code, import_generator, error_code=None, error_pos=None):
        if type.typedef_flag:
            objstruct = type.objstruct_cname
        else:
            objstruct = "struct %s" % type.objstruct_cname
        sizeof_objstruct = objstruct
        module_name = type.module_name
        condition = replacement = None
        if module_name not in ('__builtin__', 'builtins'):
            module_name = '"%s"' % module_name
        elif type.name in Code.ctypedef_builtins_map:
            # Fast path for special builtins, don't actually import
            ctypename = Code.ctypedef_builtins_map[type.name]
            code.putln('%s = %s;' % (type.typeptr_cname, ctypename))
            return
        else:
            module_name = '__Pyx_BUILTIN_MODULE_NAME'
            if type.name in Code.non_portable_builtins_map:
                condition, replacement = Code.non_portable_builtins_map[type.name]
            if objstruct in Code.basicsize_builtins_map:
                # Some builtin types have a tp_basicsize which differs from sizeof(...):
                sizeof_objstruct = Code.basicsize_builtins_map[objstruct]

        if not error_code:
            assert error_pos is not None
            error_code = code.error_goto(error_pos)

        module = import_generator.imported_module(module_name, error_code)
        code.put('%s = __Pyx_ImportType(%s, %s,' % (
            type.typeptr_cname,
            module,
            module_name))

        if condition and replacement:
            code.putln("")  # start in new line
            code.putln("#if %s" % condition)
            code.putln('"%s",' % replacement)
            code.putln("#else")
            code.putln('"%s",' % type.name)
            code.putln("#endif")
        else:
            code.put(' "%s", ' % type.name)

        if sizeof_objstruct != objstruct:
            if not condition:
                code.putln("")  # start in new line
            code.putln("#if defined(PYPY_VERSION_NUM) && PYPY_VERSION_NUM < 0x050B0000")
            code.putln('sizeof(%s),' % objstruct)
            code.putln("#else")
            code.putln('sizeof(%s),' % sizeof_objstruct)
            code.putln("#endif")
        else:
            code.put('sizeof(%s), ' % objstruct)

        # check_size
        if type.check_size and type.check_size in ('error', 'warn', 'ignore'):
            check_size = type.check_size
        elif not type.is_external or type.is_subclassed:
            check_size = 'error'
        else:
            raise RuntimeError("invalid value for check_size '%s' when compiling %s.%s" % (
                type.check_size, module_name, type.name))
        code.putln('__Pyx_ImportType_CheckSize_%s);' % check_size.title())

        code.putln(' if (!%s) %s' % (type.typeptr_cname, error_code))

    def generate_type_ready_code(self, entry, code):
        Nodes.CClassDefNode.generate_type_ready_code(entry, code)

    def generate_exttype_vtable_init_code(self, entry, code):
        # Generate code to initialise the C method table of an
        # extension type.
        type = entry.type
        if type.vtable_cname:
            code.putln(
                "%s = &%s;" % (
                    type.vtabptr_cname,
                    type.vtable_cname))
            if type.base_type and type.base_type.vtabptr_cname:
                code.putln(
                    "%s.%s = *%s;" % (
                        type.vtable_cname,
                        Naming.obj_base_cname,
                        type.base_type.vtabptr_cname))

            c_method_entries = [
                entry for entry in type.scope.cfunc_entries
                if entry.func_cname]
            if c_method_entries:
                for meth_entry in c_method_entries:
                    cast = meth_entry.type.signature_cast_string()
                    code.putln(
                        "%s.%s = %s%s;" % (
                            type.vtable_cname,
                            meth_entry.cname,
                            cast,
                            meth_entry.func_cname))


class ModuleImportGenerator(object):
    """
    Helper to generate module import while importing external types.
    This is used to avoid excessive re-imports of external modules when multiple types are looked up.
    """
    def __init__(self, code, imported_modules=None):
        self.code = code
        self.imported = {}
        if imported_modules:
            for name, cname in imported_modules.items():
                self.imported['"%s"' % name] = cname
        self.temps = []  # remember original import order for freeing

    def imported_module(self, module_name_string, error_code):
        if module_name_string in self.imported:
            return self.imported[module_name_string]

        code = self.code
        temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        self.temps.append(temp)
        code.putln('%s = PyImport_ImportModule(%s); if (unlikely(!%s)) %s' % (
            temp, module_name_string, temp, error_code))
        code.put_gotref(temp)
        self.imported[module_name_string] = temp
        return temp

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        code = self.code
        for temp in self.temps:
            code.put_decref_clear(temp, py_object_type)
            code.funcstate.release_temp(temp)


def generate_cfunction_declaration(entry, env, code, definition):
    from_cy_utility = entry.used and entry.utility_code_definition
    if entry.used and entry.inline_func_in_pxd or (not entry.in_cinclude and (
            definition or entry.defined_in_pxd or entry.visibility == 'extern' or from_cy_utility)):
        if entry.visibility == 'extern':
            storage_class = Naming.extern_c_macro
            dll_linkage = "DL_IMPORT"
        elif entry.visibility == 'public':
            storage_class = Naming.extern_c_macro
            dll_linkage = None
        elif entry.visibility == 'private':
            storage_class = "static"
            dll_linkage = None
        else:
            storage_class = "static"
            dll_linkage = None
        type = entry.type

        if entry.defined_in_pxd and not definition:
            storage_class = "static"
            dll_linkage = None
            type = CPtrType(type)

        header = type.declaration_code(
            entry.cname, dll_linkage=dll_linkage)
        modifiers = code.build_function_modifiers(entry.func_modifiers)
        code.putln("%s %s%s; /*proto*/" % (
            storage_class,
            modifiers,
            header))

#------------------------------------------------------------------------------------
#
#  Runtime support code
#
#------------------------------------------------------------------------------------

refnanny_utility_code = UtilityCode.load("Refnanny", "ModuleSetupCode.c")

packed_struct_utility_code = UtilityCode(proto="""
#if defined(__GNUC__)
#define __Pyx_PACKED __attribute__((__packed__))
#else
#define __Pyx_PACKED
#endif
""", impl="", proto_block='utility_code_proto_before_types')

capsule_utility_code = UtilityCode.load("Capsule")
