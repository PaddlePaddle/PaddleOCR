from __future__ import absolute_import

import itertools
from time import time

from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming

#
# Really small pipeline stages
#
def dumptree(t):
    # For quick debugging in pipelines
    print(t.dump())
    return t

def abort_on_errors(node):
    # Stop the pipeline if there are any errors.
    if Errors.num_errors != 0:
        raise AbortError("pipeline break")
    return node

def parse_stage_factory(context):
    def parse(compsrc):
        source_desc = compsrc.source_desc
        full_module_name = compsrc.full_module_name
        initial_pos = (source_desc, 1, 0)
        saved_cimport_from_pyx, Options.cimport_from_pyx = Options.cimport_from_pyx, False
        scope = context.find_module(full_module_name, pos = initial_pos, need_pxd = 0)
        Options.cimport_from_pyx = saved_cimport_from_pyx
        tree = context.parse(source_desc, scope, pxd = 0, full_module_name = full_module_name)
        tree.compilation_source = compsrc
        tree.scope = scope
        tree.is_pxd = False
        return tree
    return parse

def parse_pxd_stage_factory(context, scope, module_name):
    def parse(source_desc):
        tree = context.parse(source_desc, scope, pxd=True,
                             full_module_name=module_name)
        tree.scope = scope
        tree.is_pxd = True
        return tree
    return parse

def generate_pyx_code_stage_factory(options, result):
    def generate_pyx_code_stage(module_node):
        module_node.process_implementation(options, result)
        result.compilation_source = module_node.compilation_source
        return result
    return generate_pyx_code_stage


def inject_pxd_code_stage_factory(context):
    def inject_pxd_code_stage(module_node):
        for name, (statlistnode, scope) in context.pxds.items():
            module_node.merge_in(statlistnode, scope)
        return module_node
    return inject_pxd_code_stage


def use_utility_code_definitions(scope, target, seen=None):
    if seen is None:
        seen = set()

    for entry in scope.entries.values():
        if entry in seen:
            continue

        seen.add(entry)
        if entry.used and entry.utility_code_definition:
            target.use_utility_code(entry.utility_code_definition)
            for required_utility in entry.utility_code_definition.requires:
                target.use_utility_code(required_utility)
        elif entry.as_module:
            use_utility_code_definitions(entry.as_module, target, seen)


def sort_utility_codes(utilcodes):
    ranks = {}
    def get_rank(utilcode):
        if utilcode not in ranks:
            ranks[utilcode] = 0  # prevent infinite recursion on circular dependencies
            original_order = len(ranks)
            ranks[utilcode] = 1 + min([get_rank(dep) for dep in utilcode.requires or ()] or [-1]) + original_order * 1e-8
        return ranks[utilcode]
    for utilcode in utilcodes:
        get_rank(utilcode)
    return [utilcode for utilcode, _ in sorted(ranks.items(), key=lambda kv: kv[1])]


def normalize_deps(utilcodes):
    deps = {}
    for utilcode in utilcodes:
        deps[utilcode] = utilcode

    def unify_dep(dep):
        if dep in deps:
            return deps[dep]
        else:
            deps[dep] = dep
            return dep

    for utilcode in utilcodes:
        utilcode.requires = [unify_dep(dep) for dep in utilcode.requires or ()]


def inject_utility_code_stage_factory(context):
    def inject_utility_code_stage(module_node):
        module_node.prepare_utility_code()
        use_utility_code_definitions(context.cython_scope, module_node.scope)
        module_node.scope.utility_code_list = sort_utility_codes(module_node.scope.utility_code_list)
        normalize_deps(module_node.scope.utility_code_list)
        added = []
        # Note: the list might be extended inside the loop (if some utility code
        # pulls in other utility code, explicitly or implicitly)
        for utilcode in module_node.scope.utility_code_list:
            if utilcode in added:
                continue
            added.append(utilcode)
            if utilcode.requires:
                for dep in utilcode.requires:
                    if dep not in added and dep not in module_node.scope.utility_code_list:
                        module_node.scope.utility_code_list.append(dep)
            tree = utilcode.get_tree(cython_scope=context.cython_scope)
            if tree:
                module_node.merge_in(tree.body, tree.scope, merge_scope=True)
        return module_node
    return inject_utility_code_stage


#
# Pipeline factories
#

def create_pipeline(context, mode, exclude_classes=()):
    assert mode in ('pyx', 'py', 'pxd')
    from .Visitor import PrintTree
    from .ParseTreeTransforms import WithTransform, NormalizeTree, PostParse, PxdPostParse
    from .ParseTreeTransforms import ForwardDeclareTypes, InjectGilHandling, AnalyseDeclarationsTransform
    from .ParseTreeTransforms import AnalyseExpressionsTransform, FindInvalidUseOfFusedTypes
    from .ParseTreeTransforms import CreateClosureClasses, MarkClosureVisitor, DecoratorTransform
    from .ParseTreeTransforms import TrackNumpyAttributes, InterpretCompilerDirectives, TransformBuiltinMethods
    from .ParseTreeTransforms import ExpandInplaceOperators, ParallelRangeTransform
    from .ParseTreeTransforms import CalculateQualifiedNamesTransform
    from .TypeInference import MarkParallelAssignments, MarkOverflowingArithmetic
    from .ParseTreeTransforms import AdjustDefByDirectives, AlignFunctionDefinitions
    from .ParseTreeTransforms import RemoveUnreachableCode, GilCheck
    from .FlowControl import ControlFlowAnalysis
    from .AnalysedTreeTransforms import AutoTestDictTransform
    from .AutoDocTransforms import EmbedSignature
    from .Optimize import FlattenInListTransform, SwitchTransform, IterationTransform
    from .Optimize import EarlyReplaceBuiltinCalls, OptimizeBuiltinCalls
    from .Optimize import InlineDefNodeCalls
    from .Optimize import ConstantFolding, FinalOptimizePhase
    from .Optimize import DropRefcountingTransform
    from .Optimize import ConsolidateOverflowCheck
    from .Buffer import IntroduceBufferAuxiliaryVars
    from .ModuleNode import check_c_declarations, check_c_declarations_pxd


    if mode == 'pxd':
        _check_c_declarations = check_c_declarations_pxd
        _specific_post_parse = PxdPostParse(context)
    else:
        _check_c_declarations = check_c_declarations
        _specific_post_parse = None

    if mode == 'py':
        _align_function_definitions = AlignFunctionDefinitions(context)
    else:
        _align_function_definitions = None

    # NOTE: This is the "common" parts of the pipeline, which is also
    # code in pxd files. So it will be run multiple times in a
    # compilation stage.
    stages = [
        NormalizeTree(context),
        PostParse(context),
        _specific_post_parse,
        TrackNumpyAttributes(),
        InterpretCompilerDirectives(context, context.compiler_directives),
        ParallelRangeTransform(context),
        AdjustDefByDirectives(context),
        WithTransform(context),
        MarkClosureVisitor(context),
        _align_function_definitions,
        RemoveUnreachableCode(context),
        ConstantFolding(),
        FlattenInListTransform(),
        DecoratorTransform(context),
        ForwardDeclareTypes(context),
        InjectGilHandling(),
        AnalyseDeclarationsTransform(context),
        AutoTestDictTransform(context),
        EmbedSignature(context),
        EarlyReplaceBuiltinCalls(context),  ## Necessary?
        TransformBuiltinMethods(context),
        MarkParallelAssignments(context),
        ControlFlowAnalysis(context),
        RemoveUnreachableCode(context),
        # MarkParallelAssignments(context),
        MarkOverflowingArithmetic(context),
        IntroduceBufferAuxiliaryVars(context),
        _check_c_declarations,
        InlineDefNodeCalls(context),
        AnalyseExpressionsTransform(context),
        FindInvalidUseOfFusedTypes(context),
        ExpandInplaceOperators(context),
        IterationTransform(context),
        SwitchTransform(context),
        OptimizeBuiltinCalls(context),  ## Necessary?
        CreateClosureClasses(context),  ## After all lookups and type inference
        CalculateQualifiedNamesTransform(context),
        ConsolidateOverflowCheck(context),
        DropRefcountingTransform(),
        FinalOptimizePhase(context),
        GilCheck(),
        ]
    filtered_stages = []
    for s in stages:
        if s.__class__ not in exclude_classes:
            filtered_stages.append(s)
    return filtered_stages

def create_pyx_pipeline(context, options, result, py=False, exclude_classes=()):
    if py:
        mode = 'py'
    else:
        mode = 'pyx'
    test_support = []
    if options.evaluate_tree_assertions:
        from ..TestUtils import TreeAssertVisitor
        test_support.append(TreeAssertVisitor())

    if options.gdb_debug:
        from ..Debugger import DebugWriter # requires Py2.5+
        from .ParseTreeTransforms import DebugTransform
        context.gdb_debug_outputwriter = DebugWriter.CythonDebugWriter(
            options.output_dir)
        debug_transform = [DebugTransform(context, options, result)]
    else:
        debug_transform = []

    return list(itertools.chain(
        [parse_stage_factory(context)],
        create_pipeline(context, mode, exclude_classes=exclude_classes),
        test_support,
        [inject_pxd_code_stage_factory(context),
         inject_utility_code_stage_factory(context),
         abort_on_errors],
        debug_transform,
        [generate_pyx_code_stage_factory(options, result)]))

def create_pxd_pipeline(context, scope, module_name):
    from .CodeGeneration import ExtractPxdCode

    # The pxd pipeline ends up with a CCodeWriter containing the
    # code of the pxd, as well as a pxd scope.
    return [
        parse_pxd_stage_factory(context, scope, module_name)
        ] + create_pipeline(context, 'pxd') + [
        ExtractPxdCode()
        ]

def create_py_pipeline(context, options, result):
    return create_pyx_pipeline(context, options, result, py=True)

def create_pyx_as_pxd_pipeline(context, result):
    from .ParseTreeTransforms import AlignFunctionDefinitions, \
        MarkClosureVisitor, WithTransform, AnalyseDeclarationsTransform
    from .Optimize import ConstantFolding, FlattenInListTransform
    from .Nodes import StatListNode
    pipeline = []
    pyx_pipeline = create_pyx_pipeline(context, context.options, result,
                                       exclude_classes=[
                                           AlignFunctionDefinitions,
                                           MarkClosureVisitor,
                                           ConstantFolding,
                                           FlattenInListTransform,
                                           WithTransform
                                           ])
    for stage in pyx_pipeline:
        pipeline.append(stage)
        if isinstance(stage, AnalyseDeclarationsTransform):
            # This is the last stage we need.
            break
    def fake_pxd(root):
        for entry in root.scope.entries.values():
            if not entry.in_cinclude:
                entry.defined_in_pxd = 1
                if entry.name == entry.cname and entry.visibility != 'extern':
                    # Always mangle non-extern cimported entries.
                    entry.cname = entry.scope.mangle(Naming.func_prefix, entry.name)
        return StatListNode(root.pos, stats=[]), root.scope
    pipeline.append(fake_pxd)
    return pipeline

def insert_into_pipeline(pipeline, transform, before=None, after=None):
    """
    Insert a new transform into the pipeline after or before an instance of
    the given class. e.g.

        pipeline = insert_into_pipeline(pipeline, transform,
                                        after=AnalyseDeclarationsTransform)
    """
    assert before or after

    cls = before or after
    for i, t in enumerate(pipeline):
        if isinstance(t, cls):
            break

    if after:
        i += 1

    return pipeline[:i] + [transform] + pipeline[i:]

#
# Running a pipeline
#

_pipeline_entry_points = {}


def run_pipeline(pipeline, source, printtree=True):
    from .Visitor import PrintTree
    exec_ns = globals().copy() if DebugFlags.debug_verbose_pipeline else None

    def run(phase, data):
        return phase(data)

    error = None
    data = source
    try:
        try:
            for phase in pipeline:
                if phase is not None:
                    if not printtree and isinstance(phase, PrintTree):
                        continue
                    if DebugFlags.debug_verbose_pipeline:
                        t = time()
                        print("Entering pipeline phase %r" % phase)
                        # create a new wrapper for each step to show the name in profiles
                        phase_name = getattr(phase, '__name__', type(phase).__name__)
                        try:
                            run = _pipeline_entry_points[phase_name]
                        except KeyError:
                            exec("def %s(phase, data): return phase(data)" % phase_name, exec_ns)
                            run = _pipeline_entry_points[phase_name] = exec_ns[phase_name]
                    data = run(phase, data)
                    if DebugFlags.debug_verbose_pipeline:
                        print("    %.3f seconds" % (time() - t))
        except CompileError as err:
            # err is set
            Errors.report_error(err, use_stack=False)
            error = err
    except InternalError as err:
        # Only raise if there was not an earlier error
        if Errors.num_errors == 0:
            raise
        error = err
    except AbortError as err:
        error = err
    return (error, data)
