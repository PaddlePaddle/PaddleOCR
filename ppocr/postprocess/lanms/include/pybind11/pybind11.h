/*
    pybind11/pybind11.h: Main header file of the C++11 python
    binding generator library

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4100) // warning C4100: Unreferenced formal parameter
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4512) // warning C4512: Assignment operator was implicitly defined as deleted
#  pragma warning(disable: 4800) // warning C4800: 'int': forcing value to bool 'true' or 'false' (performance warning)
#  pragma warning(disable: 4996) // warning C4996: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name
#  pragma warning(disable: 4702) // warning C4702: unreachable code
#  pragma warning(disable: 4522) // warning C4522: multiple assignment operators specified
#elif defined(__INTEL_COMPILER)
#  pragma warning(push)
#  pragma warning(disable: 68)    // integer conversion resulted in a change of sign
#  pragma warning(disable: 186)   // pointless comparison of unsigned integer with zero
#  pragma warning(disable: 878)   // incompatible exception specifications
#  pragma warning(disable: 1334)  // the "template" keyword used for syntactic disambiguation may only be used within a template
#  pragma warning(disable: 1682)  // implicit conversion of a 64-bit integral type to a smaller integral type (potential portability problem)
#  pragma warning(disable: 1875)  // offsetof applied to non-POD (Plain Old Data) types is nonstandard
#  pragma warning(disable: 2196)  // warning #2196: routine is both "inline" and "noinline"
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#  pragma GCC diagnostic ignored "-Wattributes"
#  if __GNUC__ >= 7
#    pragma GCC diagnostic ignored "-Wnoexcept-type"
#  endif
#endif

#include "attr.h"
#include "options.h"
#include "class_support.h"

NAMESPACE_BEGIN(pybind11)

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
public:
    cpp_function() { }

    /// Construct a cpp_function from a vanilla function pointer
    template <typename Return, typename... Args, typename... Extra>
    cpp_function(Return (*f)(Args...), const Extra&... extra) {
        initialize(f, f, extra...);
    }

    /// Construct a cpp_function from a lambda function (possibly with internal state)
    template <typename Func, typename... Extra, typename = detail::enable_if_t<
        detail::satisfies_none_of<
            detail::remove_reference_t<Func>,
            std::is_function, std::is_pointer, std::is_member_pointer
        >::value>
    >
    cpp_function(Func &&f, const Extra&... extra) {
        using FuncType = typename detail::remove_class<decltype(&detail::remove_reference_t<Func>::operator())>::type;
        initialize(std::forward<Func>(f),
                   (FuncType *) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...), const Extra&... extra) {
        initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*) (Class *, Arg...)) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (const)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...) const, const Extra&... extra) {
        initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*)(const Class *, Arg ...)) nullptr, extra...);
    }

    /// Return the function name
    object name() const { return attr("__name__"); }

protected:
    /// Space optimization: don't inline this frequently instantiated fragment
    PYBIND11_NOINLINE detail::function_record *make_function_record() {
        return new detail::function_record();
    }

    /// Special internal constructor for functors, lambda functions, etc.
    template <typename Func, typename Return, typename... Args, typename... Extra>
    void initialize(Func &&f, Return (*)(Args...), const Extra&... extra) {

        struct capture { detail::remove_reference_t<Func> f; };

        /* Store the function including any extra state it might have (e.g. a lambda capture object) */
        auto rec = make_function_record();

        /* Store the capture object directly in the function record if there is enough space */
        if (sizeof(capture) <= sizeof(rec->data)) {
            /* Without these pragmas, GCC warns that there might not be
               enough space to use the placement new operator. However, the
               'if' statement above ensures that this is the case. */
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 6
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wplacement-new"
#endif
            new ((capture *) &rec->data) capture { std::forward<Func>(f) };
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 6
#  pragma GCC diagnostic pop
#endif
            if (!std::is_trivially_destructible<Func>::value)
                rec->free_data = [](detail::function_record *r) { ((capture *) &r->data)->~capture(); };
        } else {
            rec->data[0] = new capture { std::forward<Func>(f) };
            rec->free_data = [](detail::function_record *r) { delete ((capture *) r->data[0]); };
        }

        /* Type casters for the function arguments and return value */
        using cast_in = detail::argument_loader<Args...>;
        using cast_out = detail::make_caster<
            detail::conditional_t<std::is_void<Return>::value, detail::void_type, Return>
        >;

        static_assert(detail::expected_num_args<Extra...>(sizeof...(Args), cast_in::has_args, cast_in::has_kwargs),
                      "The number of argument annotations does not match the number of function arguments");

        /* Dispatch code which converts function arguments and performs the actual function call */
        rec->impl = [](detail::function_call &call) -> handle {
            cast_in args_converter;

            /* Try to cast the function arguments into the C++ domain */
            if (!args_converter.load_args(call))
                return PYBIND11_TRY_NEXT_OVERLOAD;

            /* Invoke call policy pre-call hook */
            detail::process_attributes<Extra...>::precall(call);

            /* Get a pointer to the capture object */
            auto data = (sizeof(capture) <= sizeof(call.func.data)
                         ? &call.func.data : call.func.data[0]);
            capture *cap = const_cast<capture *>(reinterpret_cast<const capture *>(data));

            /* Override policy for rvalues -- usually to enforce rvp::move on an rvalue */
            const auto policy = detail::return_value_policy_override<Return>::policy(call.func.policy);

            /* Function scope guard -- defaults to the compile-to-nothing `void_type` */
            using Guard = detail::extract_guard_t<Extra...>;

            /* Perform the function call */
            handle result = cast_out::cast(
                std::move(args_converter).template call<Return, Guard>(cap->f), policy, call.parent);

            /* Invoke call policy post-call hook */
            detail::process_attributes<Extra...>::postcall(call, result);

            return result;
        };

        /* Process any user-provided function attributes */
        detail::process_attributes<Extra...>::init(extra..., rec);

        /* Generate a readable signature describing the function's arguments and return value types */
        using detail::descr; using detail::_;
        PYBIND11_DESCR signature = _("(") + cast_in::arg_names() + _(") -> ") + cast_out::name();

        /* Register the function with Python from generic (non-templated) code */
        initialize_generic(rec, signature.text(), signature.types(), sizeof...(Args));

        if (cast_in::has_args) rec->has_args = true;
        if (cast_in::has_kwargs) rec->has_kwargs = true;

        /* Stash some additional information used by an important optimization in 'functional.h' */
        using FunctionType = Return (*)(Args...);
        constexpr bool is_function_ptr =
            std::is_convertible<Func, FunctionType>::value &&
            sizeof(capture) == sizeof(void *);
        if (is_function_ptr) {
            rec->is_stateless = true;
            rec->data[1] = const_cast<void *>(reinterpret_cast<const void *>(&typeid(FunctionType)));
        }
    }

    /// Register a function call with Python (generic non-templated code goes here)
    void initialize_generic(detail::function_record *rec, const char *text,
                            const std::type_info *const *types, size_t args) {

        /* Create copies of all referenced C-style strings */
        rec->name = strdup(rec->name ? rec->name : "");
        if (rec->doc) rec->doc = strdup(rec->doc);
        for (auto &a: rec->args) {
            if (a.name)
                a.name = strdup(a.name);
            if (a.descr)
                a.descr = strdup(a.descr);
            else if (a.value)
                a.descr = strdup(a.value.attr("__repr__")().cast<std::string>().c_str());
        }

        /* Generate a proper function signature */
        std::string signature;
        size_t type_depth = 0, char_index = 0, type_index = 0, arg_index = 0;
        while (true) {
            char c = text[char_index++];
            if (c == '\0')
                break;

            if (c == '{') {
                // Write arg name for everything except *args, **kwargs and return type.
                if (type_depth == 0 && text[char_index] != '*' && arg_index < args) {
                    if (!rec->args.empty() && rec->args[arg_index].name) {
                        signature += rec->args[arg_index].name;
                    } else if (arg_index == 0 && rec->is_method) {
                        signature += "self";
                    } else {
                        signature += "arg" + std::to_string(arg_index - (rec->is_method ? 1 : 0));
                    }
                    signature += ": ";
                }
                ++type_depth;
            } else if (c == '}') {
                --type_depth;
                if (type_depth == 0) {
                    if (arg_index < rec->args.size() && rec->args[arg_index].descr) {
                        signature += "=";
                        signature += rec->args[arg_index].descr;
                    }
                    arg_index++;
                }
            } else if (c == '%') {
                const std::type_info *t = types[type_index++];
                if (!t)
                    pybind11_fail("Internal error while parsing type signature (1)");
                if (auto tinfo = detail::get_type_info(*t)) {
#if defined(PYPY_VERSION)
                    signature += handle((PyObject *) tinfo->type)
                                     .attr("__module__")
                                     .cast<std::string>() + ".";
#endif
                    signature += tinfo->type->tp_name;
                } else {
                    std::string tname(t->name());
                    detail::clean_type_id(tname);
                    signature += tname;
                }
            } else {
                signature += c;
            }
        }
        if (type_depth != 0 || types[type_index] != nullptr)
            pybind11_fail("Internal error while parsing type signature (2)");

        #if !defined(PYBIND11_CONSTEXPR_DESCR)
            delete[] types;
            delete[] text;
        #endif

#if PY_MAJOR_VERSION < 3
        if (strcmp(rec->name, "__next__") == 0) {
            std::free(rec->name);
            rec->name = strdup("next");
        } else if (strcmp(rec->name, "__bool__") == 0) {
            std::free(rec->name);
            rec->name = strdup("__nonzero__");
        }
#endif
        rec->signature = strdup(signature.c_str());
        rec->args.shrink_to_fit();
        rec->is_constructor = !strcmp(rec->name, "__init__") || !strcmp(rec->name, "__setstate__");
        rec->nargs = (std::uint16_t) args;

        if (rec->sibling && PYBIND11_INSTANCE_METHOD_CHECK(rec->sibling.ptr()))
            rec->sibling = PYBIND11_INSTANCE_METHOD_GET_FUNCTION(rec->sibling.ptr());

        detail::function_record *chain = nullptr, *chain_start = rec;
        if (rec->sibling) {
            if (PyCFunction_Check(rec->sibling.ptr())) {
                auto rec_capsule = reinterpret_borrow<capsule>(PyCFunction_GET_SELF(rec->sibling.ptr()));
                chain = (detail::function_record *) rec_capsule;
                /* Never append a method to an overload chain of a parent class;
                   instead, hide the parent's overloads in this case */
                if (!chain->scope.is(rec->scope))
                    chain = nullptr;
            }
            // Don't trigger for things like the default __init__, which are wrapper_descriptors that we are intentionally replacing
            else if (!rec->sibling.is_none() && rec->name[0] != '_')
                pybind11_fail("Cannot overload existing non-function object \"" + std::string(rec->name) +
                        "\" with a function of the same name");
        }

        if (!chain) {
            /* No existing overload was found, create a new function object */
            rec->def = new PyMethodDef();
            std::memset(rec->def, 0, sizeof(PyMethodDef));
            rec->def->ml_name = rec->name;
            rec->def->ml_meth = reinterpret_cast<PyCFunction>(*dispatcher);
            rec->def->ml_flags = METH_VARARGS | METH_KEYWORDS;

            capsule rec_capsule(rec, [](void *ptr) {
                destruct((detail::function_record *) ptr);
            });

            object scope_module;
            if (rec->scope) {
                if (hasattr(rec->scope, "__module__")) {
                    scope_module = rec->scope.attr("__module__");
                } else if (hasattr(rec->scope, "__name__")) {
                    scope_module = rec->scope.attr("__name__");
                }
            }

            m_ptr = PyCFunction_NewEx(rec->def, rec_capsule.ptr(), scope_module.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate function object");
        } else {
            /* Append at the end of the overload chain */
            m_ptr = rec->sibling.ptr();
            inc_ref();
            chain_start = chain;
            if (chain->is_method != rec->is_method)
                pybind11_fail("overloading a method with both static and instance methods is not supported; "
                    #if defined(NDEBUG)
                        "compile in debug mode for more details"
                    #else
                        "error while attempting to bind " + std::string(rec->is_method ? "instance" : "static") + " method " +
                        std::string(pybind11::str(rec->scope.attr("__name__"))) + "." + std::string(rec->name) + signature
                    #endif
                );
            while (chain->next)
                chain = chain->next;
            chain->next = rec;
        }

        std::string signatures;
        int index = 0;
        /* Create a nice pydoc rec including all signatures and
           docstrings of the functions in the overload chain */
        if (chain && options::show_function_signatures()) {
            // First a generic signature
            signatures += rec->name;
            signatures += "(*args, **kwargs)\n";
            signatures += "Overloaded function.\n\n";
        }
        // Then specific overload signatures
        bool first_user_def = true;
        for (auto it = chain_start; it != nullptr; it = it->next) {
            if (options::show_function_signatures()) {
                if (index > 0) signatures += "\n";
                if (chain)
                    signatures += std::to_string(++index) + ". ";
                signatures += rec->name;
                signatures += it->signature;
                signatures += "\n";
            }
            if (it->doc && strlen(it->doc) > 0 && options::show_user_defined_docstrings()) {
                // If we're appending another docstring, and aren't printing function signatures, we
                // need to append a newline first:
                if (!options::show_function_signatures()) {
                    if (first_user_def) first_user_def = false;
                    else signatures += "\n";
                }
                if (options::show_function_signatures()) signatures += "\n";
                signatures += it->doc;
                if (options::show_function_signatures()) signatures += "\n";
            }
        }

        /* Install docstring */
        PyCFunctionObject *func = (PyCFunctionObject *) m_ptr;
        if (func->m_ml->ml_doc)
            std::free(const_cast<char *>(func->m_ml->ml_doc));
        func->m_ml->ml_doc = strdup(signatures.c_str());

        if (rec->is_method) {
            m_ptr = PYBIND11_INSTANCE_METHOD_NEW(m_ptr, rec->scope.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate instance method object");
            Py_DECREF(func);
        }
    }

    /// When a cpp_function is GCed, release any memory allocated by pybind11
    static void destruct(detail::function_record *rec) {
        while (rec) {
            detail::function_record *next = rec->next;
            if (rec->free_data)
                rec->free_data(rec);
            std::free((char *) rec->name);
            std::free((char *) rec->doc);
            std::free((char *) rec->signature);
            for (auto &arg: rec->args) {
                std::free(const_cast<char *>(arg.name));
                std::free(const_cast<char *>(arg.descr));
                arg.value.dec_ref();
            }
            if (rec->def) {
                std::free(const_cast<char *>(rec->def->ml_doc));
                delete rec->def;
            }
            delete rec;
            rec = next;
        }
    }

    /// Main dispatch logic for calls to functions bound using pybind11
    static PyObject *dispatcher(PyObject *self, PyObject *args_in, PyObject *kwargs_in) {
        using namespace detail;

        /* Iterator over the list of potentially admissible overloads */
        function_record *overloads = (function_record *) PyCapsule_GetPointer(self, nullptr),
                        *it = overloads;

        /* Need to know how many arguments + keyword arguments there are to pick the right overload */
        const size_t n_args_in = (size_t) PyTuple_GET_SIZE(args_in);

        handle parent = n_args_in > 0 ? PyTuple_GET_ITEM(args_in, 0) : nullptr,
               result = PYBIND11_TRY_NEXT_OVERLOAD;

        try {
            // We do this in two passes: in the first pass, we load arguments with `convert=false`;
            // in the second, we allow conversion (except for arguments with an explicit
            // py::arg().noconvert()).  This lets us prefer calls without conversion, with
            // conversion as a fallback.
            std::vector<function_call> second_pass;

            // However, if there are no overloads, we can just skip the no-convert pass entirely
            const bool overloaded = it != nullptr && it->next != nullptr;

            for (; it != nullptr; it = it->next) {

                /* For each overload:
                   1. Copy all positional arguments we were given, also checking to make sure that
                      named positional arguments weren't *also* specified via kwarg.
                   2. If we weren't given enough, try to make up the omitted ones by checking
                      whether they were provided by a kwarg matching the `py::arg("name")` name.  If
                      so, use it (and remove it from kwargs; if not, see if the function binding
                      provided a default that we can use.
                   3. Ensure that either all keyword arguments were "consumed", or that the function
                      takes a kwargs argument to accept unconsumed kwargs.
                   4. Any positional arguments still left get put into a tuple (for args), and any
                      leftover kwargs get put into a dict.
                   5. Pack everything into a vector; if we have py::args or py::kwargs, they are an
                      extra tuple or dict at the end of the positional arguments.
                   6. Call the function call dispatcher (function_record::impl)

                   If one of these fail, move on to the next overload and keep trying until we get a
                   result other than PYBIND11_TRY_NEXT_OVERLOAD.
                 */

                function_record &func = *it;
                size_t pos_args = func.nargs;    // Number of positional arguments that we need
                if (func.has_args) --pos_args;   // (but don't count py::args
                if (func.has_kwargs) --pos_args; //  or py::kwargs)

                if (!func.has_args && n_args_in > pos_args)
                    continue; // Too many arguments for this overload

                if (n_args_in < pos_args && func.args.size() < pos_args)
                    continue; // Not enough arguments given, and not enough defaults to fill in the blanks

                function_call call(func, parent);

                size_t args_to_copy = std::min(pos_args, n_args_in);
                size_t args_copied = 0;

                // 1. Copy any position arguments given.
                bool bad_arg = false;
                for (; args_copied < args_to_copy; ++args_copied) {
                    argument_record *arg_rec = args_copied < func.args.size() ? &func.args[args_copied] : nullptr;
                    if (kwargs_in && arg_rec && arg_rec->name && PyDict_GetItemString(kwargs_in, arg_rec->name)) {
                        bad_arg = true;
                        break;
                    }

                    handle arg(PyTuple_GET_ITEM(args_in, args_copied));
                    if (arg_rec && !arg_rec->none && arg.is_none()) {
                        bad_arg = true;
                        break;
                    }
                    call.args.push_back(arg);
                    call.args_convert.push_back(arg_rec ? arg_rec->convert : true);
                }
                if (bad_arg)
                    continue; // Maybe it was meant for another overload (issue #688)

                // We'll need to copy this if we steal some kwargs for defaults
                dict kwargs = reinterpret_borrow<dict>(kwargs_in);

                // 2. Check kwargs and, failing that, defaults that may help complete the list
                if (args_copied < pos_args) {
                    bool copied_kwargs = false;

                    for (; args_copied < pos_args; ++args_copied) {
                        const auto &arg = func.args[args_copied];

                        handle value;
                        if (kwargs_in && arg.name)
                            value = PyDict_GetItemString(kwargs.ptr(), arg.name);

                        if (value) {
                            // Consume a kwargs value
                            if (!copied_kwargs) {
                                kwargs = reinterpret_steal<dict>(PyDict_Copy(kwargs.ptr()));
                                copied_kwargs = true;
                            }
                            PyDict_DelItemString(kwargs.ptr(), arg.name);
                        } else if (arg.value) {
                            value = arg.value;
                        }

                        if (value) {
                            call.args.push_back(value);
                            call.args_convert.push_back(arg.convert);
                        }
                        else
                            break;
                    }

                    if (args_copied < pos_args)
                        continue; // Not enough arguments, defaults, or kwargs to fill the positional arguments
                }

                // 3. Check everything was consumed (unless we have a kwargs arg)
                if (kwargs && kwargs.size() > 0 && !func.has_kwargs)
                    continue; // Unconsumed kwargs, but no py::kwargs argument to accept them

                // 4a. If we have a py::args argument, create a new tuple with leftovers
                tuple extra_args;
                if (func.has_args) {
                    if (args_to_copy == 0) {
                        // We didn't copy out any position arguments from the args_in tuple, so we
                        // can reuse it directly without copying:
                        extra_args = reinterpret_borrow<tuple>(args_in);
                    } else if (args_copied >= n_args_in) {
                        extra_args = tuple(0);
                    } else {
                        size_t args_size = n_args_in - args_copied;
                        extra_args = tuple(args_size);
                        for (size_t i = 0; i < args_size; ++i) {
                            handle item = PyTuple_GET_ITEM(args_in, args_copied + i);
                            extra_args[i] = item.inc_ref().ptr();
                        }
                    }
                    call.args.push_back(extra_args);
                    call.args_convert.push_back(false);
                }

                // 4b. If we have a py::kwargs, pass on any remaining kwargs
                if (func.has_kwargs) {
                    if (!kwargs.ptr())
                        kwargs = dict(); // If we didn't get one, send an empty one
                    call.args.push_back(kwargs);
                    call.args_convert.push_back(false);
                }

                // 5. Put everything in a vector.  Not technically step 5, we've been building it
                // in `call.args` all along.
                #if !defined(NDEBUG)
                if (call.args.size() != func.nargs || call.args_convert.size() != func.nargs)
                    pybind11_fail("Internal error: function call dispatcher inserted wrong number of arguments!");
                #endif

                std::vector<bool> second_pass_convert;
                if (overloaded) {
                    // We're in the first no-convert pass, so swap out the conversion flags for a
                    // set of all-false flags.  If the call fails, we'll swap the flags back in for
                    // the conversion-allowed call below.
                    second_pass_convert.resize(func.nargs, false);
                    call.args_convert.swap(second_pass_convert);
                }

                // 6. Call the function.
                try {
                    loader_life_support guard{};
                    result = func.impl(call);
                } catch (reference_cast_error &) {
                    result = PYBIND11_TRY_NEXT_OVERLOAD;
                }

                if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD)
                    break;

                if (overloaded) {
                    // The (overloaded) call failed; if the call has at least one argument that
                    // permits conversion (i.e. it hasn't been explicitly specified `.noconvert()`)
                    // then add this call to the list of second pass overloads to try.
                    for (size_t i = func.is_method ? 1 : 0; i < pos_args; i++) {
                        if (second_pass_convert[i]) {
                            // Found one: swap the converting flags back in and store the call for
                            // the second pass.
                            call.args_convert.swap(second_pass_convert);
                            second_pass.push_back(std::move(call));
                            break;
                        }
                    }
                }
            }

            if (overloaded && !second_pass.empty() && result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD) {
                // The no-conversion pass finished without success, try again with conversion allowed
                for (auto &call : second_pass) {
                    try {
                        loader_life_support guard{};
                        result = call.func.impl(call);
                    } catch (reference_cast_error &) {
                        result = PYBIND11_TRY_NEXT_OVERLOAD;
                    }

                    if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD)
                        break;
                }
            }
        } catch (error_already_set &e) {
            e.restore();
            return nullptr;
        } catch (...) {
            /* When an exception is caught, give each registered exception
               translator a chance to translate it to a Python exception
               in reverse order of registration.

               A translator may choose to do one of the following:

                - catch the exception and call PyErr_SetString or PyErr_SetObject
                  to set a standard (or custom) Python exception, or
                - do nothing and let the exception fall through to the next translator, or
                - delegate translation to the next translator by throwing a new type of exception. */

            auto last_exception = std::current_exception();
            auto &registered_exception_translators = get_internals().registered_exception_translators;
            for (auto& translator : registered_exception_translators) {
                try {
                    translator(last_exception);
                } catch (...) {
                    last_exception = std::current_exception();
                    continue;
                }
                return nullptr;
            }
            PyErr_SetString(PyExc_SystemError, "Exception escaped from default exception translator!");
            return nullptr;
        }

        if (result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD) {
            if (overloads->is_operator)
                return handle(Py_NotImplemented).inc_ref().ptr();

            std::string msg = std::string(overloads->name) + "(): incompatible " +
                std::string(overloads->is_constructor ? "constructor" : "function") +
                " arguments. The following argument types are supported:\n";

            int ctr = 0;
            for (function_record *it2 = overloads; it2 != nullptr; it2 = it2->next) {
                msg += "    "+ std::to_string(++ctr) + ". ";

                bool wrote_sig = false;
                if (overloads->is_constructor) {
                    // For a constructor, rewrite `(self: Object, arg0, ...) -> NoneType` as `Object(arg0, ...)`
                    std::string sig = it2->signature;
                    size_t start = sig.find('(') + 7; // skip "(self: "
                    if (start < sig.size()) {
                        // End at the , for the next argument
                        size_t end = sig.find(", "), next = end + 2;
                        size_t ret = sig.rfind(" -> ");
                        // Or the ), if there is no comma:
                        if (end >= sig.size()) next = end = sig.find(')');
                        if (start < end && next < sig.size()) {
                            msg.append(sig, start, end - start);
                            msg += '(';
                            msg.append(sig, next, ret - next);
                            wrote_sig = true;
                        }
                    }
                }
                if (!wrote_sig) msg += it2->signature;

                msg += "\n";
            }
            msg += "\nInvoked with: ";
            auto args_ = reinterpret_borrow<tuple>(args_in);
            bool some_args = false;
            for (size_t ti = overloads->is_constructor ? 1 : 0; ti < args_.size(); ++ti) {
                if (!some_args) some_args = true;
                else msg += ", ";
                msg += pybind11::repr(args_[ti]);
            }
            if (kwargs_in) {
                auto kwargs = reinterpret_borrow<dict>(kwargs_in);
                if (kwargs.size() > 0) {
                    if (some_args) msg += "; ";
                    msg += "kwargs: ";
                    bool first = true;
                    for (auto kwarg : kwargs) {
                        if (first) first = false;
                        else msg += ", ";
                        msg += pybind11::str("{}={!r}").format(kwarg.first, kwarg.second);
                    }
                }
            }

            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else if (!result) {
            std::string msg = "Unable to convert function return value to a "
                              "Python type! The signature was\n\t";
            msg += it->signature;
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else {
            if (overloads->is_constructor) {
                auto tinfo = get_type_info((PyTypeObject *) overloads->scope.ptr());
                tinfo->init_instance(reinterpret_cast<instance *>(parent.ptr()), nullptr);
            }
            return result.ptr();
        }
    }
};

/// Wrapper for Python extension modules
class module : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module, object, PyModule_Check)

    /// Create a new top-level Python module with the given name and docstring
    explicit module(const char *name, const char *doc = nullptr) {
        if (!options::show_user_defined_docstrings()) doc = nullptr;
#if PY_MAJOR_VERSION >= 3
        PyModuleDef *def = new PyModuleDef();
        std::memset(def, 0, sizeof(PyModuleDef));
        def->m_name = name;
        def->m_doc = doc;
        def->m_size = -1;
        Py_INCREF(def);
        m_ptr = PyModule_Create(def);
#else
        m_ptr = Py_InitModule3(name, nullptr, doc);
#endif
        if (m_ptr == nullptr)
            pybind11_fail("Internal error in module::module()");
        inc_ref();
    }

    /** \rst
        Create Python binding for a new function within the module scope. ``Func``
        can be a plain C++ function, a function pointer, or a lambda function. For
        details on the ``Extra&& ... extra`` argument, see section :ref:`extras`.
    \endrst */
    template <typename Func, typename... Extra>
    module &def(const char *name_, Func &&f, const Extra& ... extra) {
        cpp_function func(std::forward<Func>(f), name(name_), scope(*this),
                          sibling(getattr(*this, name_, none())), extra...);
        // NB: allow overwriting here because cpp_function sets up a chain with the intention of
        // overwriting (and has already checked internally that it isn't overwriting non-functions).
        add_object(name_, func, true /* overwrite */);
        return *this;
    }

    /** \rst
        Create and return a new Python submodule with the given name and docstring.
        This also works recursively, i.e.

        .. code-block:: cpp

            py::module m("example", "pybind11 example plugin");
            py::module m2 = m.def_submodule("sub", "A submodule of 'example'");
            py::module m3 = m2.def_submodule("subsub", "A submodule of 'example.sub'");
    \endrst */
    module def_submodule(const char *name, const char *doc = nullptr) {
        std::string full_name = std::string(PyModule_GetName(m_ptr))
            + std::string(".") + std::string(name);
        auto result = reinterpret_borrow<module>(PyImport_AddModule(full_name.c_str()));
        if (doc && options::show_user_defined_docstrings())
            result.attr("__doc__") = pybind11::str(doc);
        attr(name) = result;
        return result;
    }

    /// Import and return a module or throws `error_already_set`.
    static module import(const char *name) {
        PyObject *obj = PyImport_ImportModule(name);
        if (!obj)
            throw error_already_set();
        return reinterpret_steal<module>(obj);
    }

    // Adds an object to the module using the given name.  Throws if an object with the given name
    // already exists.
    //
    // overwrite should almost always be false: attempting to overwrite objects that pybind11 has
    // established will, in most cases, break things.
    PYBIND11_NOINLINE void add_object(const char *name, handle obj, bool overwrite = false) {
        if (!overwrite && hasattr(*this, name))
            pybind11_fail("Error during initialization: multiple incompatible definitions with name \"" +
                    std::string(name) + "\"");

        PyModule_AddObject(ptr(), name, obj.inc_ref().ptr() /* steals a reference */);
    }
};

/// \ingroup python_builtins
/// Return a dictionary representing the global variables in the current execution frame,
/// or ``__main__.__dict__`` if there is no frame (usually when the interpreter is embedded).
inline dict globals() {
    PyObject *p = PyEval_GetGlobals();
    return reinterpret_borrow<dict>(p ? p : module::import("__main__").attr("__dict__").ptr());
}

NAMESPACE_BEGIN(detail)
/// Generic support for creating new Python heap types
class generic_type : public object {
    template <typename...> friend class class_;
public:
    PYBIND11_OBJECT_DEFAULT(generic_type, object, PyType_Check)
protected:
    void initialize(const type_record &rec) {
        if (rec.scope && hasattr(rec.scope, rec.name))
            pybind11_fail("generic_type: cannot initialize type \"" + std::string(rec.name) +
                          "\": an object with that name is already defined");

        if (get_type_info(*rec.type))
            pybind11_fail("generic_type: type \"" + std::string(rec.name) +
                          "\" is already registered!");

        m_ptr = make_new_python_type(rec);

        /* Register supplemental type information in C++ dict */
        auto *tinfo = new detail::type_info();
        tinfo->type = (PyTypeObject *) m_ptr;
        tinfo->cpptype = rec.type;
        tinfo->type_size = rec.type_size;
        tinfo->operator_new = rec.operator_new;
        tinfo->holder_size_in_ptrs = size_in_ptrs(rec.holder_size);
        tinfo->init_instance = rec.init_instance;
        tinfo->dealloc = rec.dealloc;
        tinfo->simple_type = true;
        tinfo->simple_ancestors = true;

        auto &internals = get_internals();
        auto tindex = std::type_index(*rec.type);
        tinfo->direct_conversions = &internals.direct_conversions[tindex];
        tinfo->default_holder = rec.default_holder;
        internals.registered_types_cpp[tindex] = tinfo;
        internals.registered_types_py[(PyTypeObject *) m_ptr] = { tinfo };

        if (rec.bases.size() > 1 || rec.multiple_inheritance) {
            mark_parents_nonsimple(tinfo->type);
            tinfo->simple_ancestors = false;
        }
        else if (rec.bases.size() == 1) {
            auto parent_tinfo = get_type_info((PyTypeObject *) rec.bases[0].ptr());
            tinfo->simple_ancestors = parent_tinfo->simple_ancestors;
        }
    }

    /// Helper function which tags all parents of a type using mult. inheritance
    void mark_parents_nonsimple(PyTypeObject *value) {
        auto t = reinterpret_borrow<tuple>(value->tp_bases);
        for (handle h : t) {
            auto tinfo2 = get_type_info((PyTypeObject *) h.ptr());
            if (tinfo2)
                tinfo2->simple_type = false;
            mark_parents_nonsimple((PyTypeObject *) h.ptr());
        }
    }

    void install_buffer_funcs(
            buffer_info *(*get_buffer)(PyObject *, void *),
            void *get_buffer_data) {
        PyHeapTypeObject *type = (PyHeapTypeObject*) m_ptr;
        auto tinfo = detail::get_type_info(&type->ht_type);

        if (!type->ht_type.tp_as_buffer)
            pybind11_fail(
                "To be able to register buffer protocol support for the type '" +
                std::string(tinfo->type->tp_name) +
                "' the associated class<>(..) invocation must "
                "include the pybind11::buffer_protocol() annotation!");

        tinfo->get_buffer = get_buffer;
        tinfo->get_buffer_data = get_buffer_data;
    }

    void def_property_static_impl(const char *name,
                                  handle fget, handle fset,
                                  detail::function_record *rec_fget) {
        const auto is_static = !(rec_fget->is_method && rec_fget->scope);
        const auto has_doc = rec_fget->doc && pybind11::options::show_user_defined_docstrings();

        auto property = handle((PyObject *) (is_static ? get_internals().static_property_type
                                                       : &PyProperty_Type));
        attr(name) = property(fget.ptr() ? fget : none(),
                              fset.ptr() ? fset : none(),
                              /*deleter*/none(),
                              pybind11::str(has_doc ? rec_fget->doc : ""));
    }
};

/// Set the pointer to operator new if it exists. The cast is needed because it can be overloaded.
template <typename T, typename = void_t<decltype(static_cast<void *(*)(size_t)>(T::operator new))>>
void set_operator_new(type_record *r) { r->operator_new = &T::operator new; }

template <typename> void set_operator_new(...) { }

template <typename T, typename SFINAE = void> struct has_operator_delete : std::false_type { };
template <typename T> struct has_operator_delete<T, void_t<decltype(static_cast<void (*)(void *)>(T::operator delete))>>
    : std::true_type { };
template <typename T, typename SFINAE = void> struct has_operator_delete_size : std::false_type { };
template <typename T> struct has_operator_delete_size<T, void_t<decltype(static_cast<void (*)(void *, size_t)>(T::operator delete))>>
    : std::true_type { };
/// Call class-specific delete if it exists or global otherwise. Can also be an overload set.
template <typename T, enable_if_t<has_operator_delete<T>::value, int> = 0>
void call_operator_delete(T *p, size_t) { T::operator delete(p); }
template <typename T, enable_if_t<!has_operator_delete<T>::value && has_operator_delete_size<T>::value, int> = 0>
void call_operator_delete(T *p, size_t s) { T::operator delete(p, s); }

inline void call_operator_delete(void *p, size_t) { ::operator delete(p); }

NAMESPACE_END(detail)

/// Given a pointer to a member function, cast it to its `Derived` version.
/// Forward everything else unchanged.
template <typename /*Derived*/, typename F>
auto method_adaptor(F &&f) -> decltype(std::forward<F>(f)) { return std::forward<F>(f); }

template <typename Derived, typename Return, typename Class, typename... Args>
auto method_adaptor(Return (Class::*pmf)(Args...)) -> Return (Derived::*)(Args...) { return pmf; }

template <typename Derived, typename Return, typename Class, typename... Args>
auto method_adaptor(Return (Class::*pmf)(Args...) const) -> Return (Derived::*)(Args...) const { return pmf; }

template <typename type_, typename... options>
class class_ : public detail::generic_type {
    template <typename T> using is_holder = detail::is_holder_type<type_, T>;
    template <typename T> using is_subtype = detail::is_strict_base_of<type_, T>;
    template <typename T> using is_base = detail::is_strict_base_of<T, type_>;
    // struct instead of using here to help MSVC:
    template <typename T> struct is_valid_class_option :
        detail::any_of<is_holder<T>, is_subtype<T>, is_base<T>> {};

public:
    using type = type_;
    using type_alias = detail::exactly_one_t<is_subtype, void, options...>;
    constexpr static bool has_alias = !std::is_void<type_alias>::value;
    using holder_type = detail::exactly_one_t<is_holder, std::unique_ptr<type>, options...>;

    static_assert(detail::all_of<is_valid_class_option<options>...>::value,
            "Unknown/invalid class_ template parameters provided");

    PYBIND11_OBJECT(class_, generic_type, PyType_Check)

    template <typename... Extra>
    class_(handle scope, const char *name, const Extra &... extra) {
        using namespace detail;

        // MI can only be specified via class_ template options, not constructor parameters
        static_assert(
            none_of<is_pyobject<Extra>...>::value || // no base class arguments, or:
            (   constexpr_sum(is_pyobject<Extra>::value...) == 1 && // Exactly one base
                constexpr_sum(is_base<options>::value...)   == 0 && // no template option bases
                none_of<std::is_same<multiple_inheritance, Extra>...>::value), // no multiple_inheritance attr
            "Error: multiple inheritance bases must be specified via class_ template options");

        type_record record;
        record.scope = scope;
        record.name = name;
        record.type = &typeid(type);
        record.type_size = sizeof(conditional_t<has_alias, type_alias, type>);
        record.holder_size = sizeof(holder_type);
        record.init_instance = init_instance;
        record.dealloc = dealloc;
        record.default_holder = std::is_same<holder_type, std::unique_ptr<type>>::value;

        set_operator_new<type>(&record);

        /* Register base classes specified via template arguments to class_, if any */
        PYBIND11_EXPAND_SIDE_EFFECTS(add_base<options>(record));

        /* Process optional arguments, if any */
        process_attributes<Extra...>::init(extra..., &record);

        generic_type::initialize(record);

        if (has_alias) {
            auto &instances = get_internals().registered_types_cpp;
            instances[std::type_index(typeid(type_alias))] = instances[std::type_index(typeid(type))];
        }
    }

    template <typename Base, detail::enable_if_t<is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &rec) {
        rec.add_base(typeid(Base), [](void *src) -> void * {
            return static_cast<Base *>(reinterpret_cast<type *>(src));
        });
    }

    template <typename Base, detail::enable_if_t<!is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &) { }

    template <typename Func, typename... Extra>
    class_ &def(const char *name_, Func&& f, const Extra&... extra) {
        cpp_function cf(method_adaptor<type>(std::forward<Func>(f)), name(name_), is_method(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        attr(cf.name()) = cf;
        return *this;
    }

    template <typename Func, typename... Extra> class_ &
    def_static(const char *name_, Func &&f, const Extra&... extra) {
        static_assert(!std::is_member_function_pointer<Func>::value,
                "def_static(...) called with a non-static member function pointer");
        cpp_function cf(std::forward<Func>(f), name(name_), scope(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        attr(cf.name()) = cf;
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ &def(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute(*this, extra...);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ & def_cast(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute_cast(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::init<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::init_alias<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename Func> class_& def_buffer(Func &&func) {
        struct capture { Func func; };
        capture *ptr = new capture { std::forward<Func>(func) };
        install_buffer_funcs([](PyObject *obj, void *ptr) -> buffer_info* {
            detail::make_caster<type> caster;
            if (!caster.load(obj, false))
                return nullptr;
            return new buffer_info(((capture *) ptr)->func(caster));
        }, ptr);
        return *this;
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...)) {
        return def_buffer([func] (type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) const) {
        return def_buffer([func] (const type &obj) { return (obj.*func)(); });
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readwrite(const char *name, D C::*pm, const Extra&... extra) {
        static_assert(std::is_base_of<C, type>::value, "def_readwrite() requires a class member (or base class member)");
        cpp_function fget([pm](const type &c) -> const D &{ return c.*pm; }, is_method(*this)),
                     fset([pm](type &c, const D &value) { c.*pm = value; }, is_method(*this));
        def_property(name, fget, fset, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readonly(const char *name, const D C::*pm, const Extra& ...extra) {
        static_assert(std::is_base_of<C, type>::value, "def_readonly() requires a class member (or base class member)");
        cpp_function fget([pm](const type &c) -> const D &{ return c.*pm; }, is_method(*this));
        def_property_readonly(name, fget, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readwrite_static(const char *name, D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this)),
                     fset([pm](object, const D &value) { *pm = value; }, scope(*this));
        def_property_static(name, fget, fset, return_value_policy::reference, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readonly_static(const char *name, const D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this));
        def_property_readonly_static(name, fget, return_value_policy::reference, extra...);
        return *this;
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly(name, cpp_function(method_adaptor<type>(fget)),
                                     return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property(name, fget, cpp_function(), extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly_static(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly_static(name, cpp_function(fget), return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly_static(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property_static(name, fget, cpp_function(), extra...);
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename Setter, typename... Extra>
    class_ &def_property(const char *name, const Getter &fget, const Setter &fset, const Extra& ...extra) {
        return def_property(name, fget, cpp_function(method_adaptor<type>(fset)), extra...);
    }
    template <typename Getter, typename... Extra>
    class_ &def_property(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property(name, cpp_function(method_adaptor<type>(fget)), fset,
                            return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, fget, fset, is_method(*this), extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_static(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, cpp_function(fget), fset, return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_static(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        auto rec_fget = get_function_record(fget), rec_fset = get_function_record(fset);
        char *doc_prev = rec_fget->doc; /* 'extra' field may include a property-specific documentation string */
        detail::process_attributes<Extra...>::init(extra..., rec_fget);
        if (rec_fget->doc && rec_fget->doc != doc_prev) {
            free(doc_prev);
            rec_fget->doc = strdup(rec_fget->doc);
        }
        if (rec_fset) {
            doc_prev = rec_fset->doc;
            detail::process_attributes<Extra...>::init(extra..., rec_fset);
            if (rec_fset->doc && rec_fset->doc != doc_prev) {
                free(doc_prev);
                rec_fset->doc = strdup(rec_fset->doc);
            }
        }
        def_property_static_impl(name, fget, fset, rec_fget);
        return *this;
    }

private:
    /// Initialize holder object, variant 1: object derives from enable_shared_from_this
    template <typename T>
    static void init_holder(detail::instance *inst, detail::value_and_holder &v_h,
            const holder_type * /* unused */, const std::enable_shared_from_this<T> * /* dummy */) {
        try {
            auto sh = std::dynamic_pointer_cast<typename holder_type::element_type>(
                    v_h.value_ptr<type>()->shared_from_this());
            if (sh) {
                new (&v_h.holder<holder_type>()) holder_type(std::move(sh));
                v_h.set_holder_constructed();
            }
        } catch (const std::bad_weak_ptr &) {}

        if (!v_h.holder_constructed() && inst->owned) {
            new (&v_h.holder<holder_type>()) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
            const holder_type *holder_ptr, std::true_type /*is_copy_constructible*/) {
        new (&v_h.holder<holder_type>()) holder_type(*reinterpret_cast<const holder_type *>(holder_ptr));
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
            const holder_type *holder_ptr, std::false_type /*is_copy_constructible*/) {
        new (&v_h.holder<holder_type>()) holder_type(std::move(*const_cast<holder_type *>(holder_ptr)));
    }

    /// Initialize holder object, variant 2: try to construct from existing holder object, if possible
    static void init_holder(detail::instance *inst, detail::value_and_holder &v_h,
            const holder_type *holder_ptr, const void * /* dummy -- not enable_shared_from_this<T>) */) {
        if (holder_ptr) {
            init_holder_from_existing(v_h, holder_ptr, std::is_copy_constructible<holder_type>());
            v_h.set_holder_constructed();
        } else if (inst->owned || detail::always_construct_holder<holder_type>::value) {
            new (&v_h.holder<holder_type>()) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    /// Performs instance initialization including constructing a holder and registering the known
    /// instance.  Should be called as soon as the `type` value_ptr is set for an instance.  Takes an
    /// optional pointer to an existing holder to use; if not specified and the instance is
    /// `.owned`, a new holder will be constructed to manage the value pointer.
    static void init_instance(detail::instance *inst, const void *holder_ptr) {
        auto v_h = inst->get_value_and_holder(detail::get_type_info(typeid(type)));
        if (!v_h.instance_registered()) {
            register_instance(inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered();
        }
        init_holder(inst, v_h, (const holder_type *) holder_ptr, v_h.value_ptr<type>());
    }

    /// Deallocates an instance; via holder, if constructed; otherwise via operator delete.
    static void dealloc(const detail::value_and_holder &v_h) {
        if (v_h.holder_constructed())
            v_h.holder<holder_type>().~holder_type();
        else
            detail::call_operator_delete(v_h.value_ptr<type>(), v_h.type->type_size);
    }

    static detail::function_record *get_function_record(handle h) {
        h = detail::get_function(h);
        return h ? (detail::function_record *) reinterpret_borrow<capsule>(PyCFunction_GET_SELF(h.ptr()))
                 : nullptr;
    }
};

/// Binds C++ enumerations and enumeration classes to Python
template <typename Type> class enum_ : public class_<Type> {
public:
    using class_<Type>::def;
    using class_<Type>::def_property_readonly_static;
    using Scalar = typename std::underlying_type<Type>::type;

    template <typename... Extra>
    enum_(const handle &scope, const char *name, const Extra&... extra)
      : class_<Type>(scope, name, extra...), m_entries(), m_parent(scope) {

        constexpr bool is_arithmetic = detail::any_of<std::is_same<arithmetic, Extra>...>::value;

        auto m_entries_ptr = m_entries.inc_ref().ptr();
        def("__repr__", [name, m_entries_ptr](Type value) -> pybind11::str {
            for (const auto &kv : reinterpret_borrow<dict>(m_entries_ptr)) {
                if (pybind11::cast<Type>(kv.second) == value)
                    return pybind11::str("{}.{}").format(name, kv.first);
            }
            return pybind11::str("{}.???").format(name);
        });
        def_property_readonly_static("__members__", [m_entries_ptr](object /* self */) {
            dict m;
            for (const auto &kv : reinterpret_borrow<dict>(m_entries_ptr))
                m[kv.first] = kv.second;
            return m;
        }, return_value_policy::copy);
        def("__init__", [](Type& value, Scalar i) { value = (Type)i; });
        def("__int__", [](Type value) { return (Scalar) value; });
        #if PY_MAJOR_VERSION < 3
            def("__long__", [](Type value) { return (Scalar) value; });
        #endif
        def("__eq__", [](const Type &value, Type *value2) { return value2 && value == *value2; });
        def("__ne__", [](const Type &value, Type *value2) { return !value2 || value != *value2; });
        if (is_arithmetic) {
            def("__lt__", [](const Type &value, Type *value2) { return value2 && value < *value2; });
            def("__gt__", [](const Type &value, Type *value2) { return value2 && value > *value2; });
            def("__le__", [](const Type &value, Type *value2) { return value2 && value <= *value2; });
            def("__ge__", [](const Type &value, Type *value2) { return value2 && value >= *value2; });
        }
        if (std::is_convertible<Type, Scalar>::value) {
            // Don't provide comparison with the underlying type if the enum isn't convertible,
            // i.e. if Type is a scoped enum, mirroring the C++ behaviour.  (NB: we explicitly
            // convert Type to Scalar below anyway because this needs to compile).
            def("__eq__", [](const Type &value, Scalar value2) { return (Scalar) value == value2; });
            def("__ne__", [](const Type &value, Scalar value2) { return (Scalar) value != value2; });
            if (is_arithmetic) {
                def("__lt__", [](const Type &value, Scalar value2) { return (Scalar) value < value2; });
                def("__gt__", [](const Type &value, Scalar value2) { return (Scalar) value > value2; });
                def("__le__", [](const Type &value, Scalar value2) { return (Scalar) value <= value2; });
                def("__ge__", [](const Type &value, Scalar value2) { return (Scalar) value >= value2; });
                def("__invert__", [](const Type &value) { return ~((Scalar) value); });
                def("__and__", [](const Type &value, Scalar value2) { return (Scalar) value & value2; });
                def("__or__", [](const Type &value, Scalar value2) { return (Scalar) value | value2; });
                def("__xor__", [](const Type &value, Scalar value2) { return (Scalar) value ^ value2; });
                def("__rand__", [](const Type &value, Scalar value2) { return (Scalar) value & value2; });
                def("__ror__", [](const Type &value, Scalar value2) { return (Scalar) value | value2; });
                def("__rxor__", [](const Type &value, Scalar value2) { return (Scalar) value ^ value2; });
                def("__and__", [](const Type &value, const Type &value2) { return (Scalar) value & (Scalar) value2; });
                def("__or__", [](const Type &value, const Type &value2) { return (Scalar) value | (Scalar) value2; });
                def("__xor__", [](const Type &value, const Type &value2) { return (Scalar) value ^ (Scalar) value2; });
            }
        }
        def("__hash__", [](const Type &value) { return (Scalar) value; });
        // Pickling and unpickling -- needed for use with the 'multiprocessing' module
        def("__getstate__", [](const Type &value) { return pybind11::make_tuple((Scalar) value); });
        def("__setstate__", [](Type &p, tuple t) { new (&p) Type((Type) t[0].cast<Scalar>()); });
    }

    /// Export enumeration entries into the parent scope
    enum_& export_values() {
        for (const auto &kv : m_entries)
            m_parent.attr(kv.first) = kv.second;
        return *this;
    }

    /// Add an enumeration entry
    enum_& value(char const* name, Type value) {
        auto v = pybind11::cast(value, return_value_policy::copy);
        this->attr(name) = v;
        m_entries[pybind11::str(name)] = v;
        return *this;
    }

private:
    dict m_entries;
    handle m_parent;
};

NAMESPACE_BEGIN(detail)
template <typename... Args> struct init {
    template <typename Class, typename... Extra, enable_if_t<!Class::has_alias, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        using Base = typename Class::type;
        /// Function which calls a specific C++ in-place constructor
        cl.def("__init__", [](Base *self_, Args... args) { new (self_) Base(args...); }, extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          std::is_constructible<typename Class::type, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        using Base = typename Class::type;
        using Alias = typename Class::type_alias;
        handle cl_type = cl;
        cl.def("__init__", [cl_type](handle self_, Args... args) {
                if (self_.get_type().is(cl_type))
                    new (self_.cast<Base *>()) Base(args...);
                else
                    new (self_.cast<Alias *>()) Alias(args...);
            }, extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          !std::is_constructible<typename Class::type, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        init_alias<Args...>::execute(cl, extra...);
    }
};
template <typename... Args> struct init_alias {
    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias && std::is_constructible<typename Class::type_alias, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        using Alias = typename Class::type_alias;
        cl.def("__init__", [](Alias *self_, Args... args) { new (self_) Alias(args...); }, extra...);
    }
};


inline void keep_alive_impl(handle nurse, handle patient) {
    if (!nurse || !patient)
        pybind11_fail("Could not activate keep_alive!");

    if (patient.is_none() || nurse.is_none())
        return; /* Nothing to keep alive or nothing to be kept alive by */

    auto tinfo = all_type_info(Py_TYPE(nurse.ptr()));
    if (!tinfo.empty()) {
        /* It's a pybind-registered type, so we can store the patient in the
         * internal list. */
        add_patient(nurse.ptr(), patient.ptr());
    }
    else {
        /* Fall back to clever approach based on weak references taken from
         * Boost.Python. This is not used for pybind-registered types because
         * the objects can be destroyed out-of-order in a GC pass. */
        cpp_function disable_lifesupport(
            [patient](handle weakref) { patient.dec_ref(); weakref.dec_ref(); });

        weakref wr(nurse, disable_lifesupport);

        patient.inc_ref(); /* reference patient and leak the weak reference */
        (void) wr.release();
    }
}

PYBIND11_NOINLINE inline void keep_alive_impl(size_t Nurse, size_t Patient, function_call &call, handle ret) {
    keep_alive_impl(
        Nurse   == 0 ? ret : Nurse   <= call.args.size() ? call.args[Nurse   - 1] : handle(),
        Patient == 0 ? ret : Patient <= call.args.size() ? call.args[Patient - 1] : handle()
    );
}

inline std::pair<decltype(internals::registered_types_py)::iterator, bool> all_type_info_get_cache(PyTypeObject *type) {
    auto res = get_internals().registered_types_py
#ifdef __cpp_lib_unordered_map_try_emplace
        .try_emplace(type);
#else
        .emplace(type, std::vector<detail::type_info *>());
#endif
    if (res.second) {
        // New cache entry created; set up a weak reference to automatically remove it if the type
        // gets destroyed:
        weakref((PyObject *) type, cpp_function([type](handle wr) {
            get_internals().registered_types_py.erase(type);
            wr.dec_ref();
        })).release();
    }

    return res;
}

template <typename Iterator, typename Sentinel, bool KeyIterator, return_value_policy Policy>
struct iterator_state {
    Iterator it;
    Sentinel end;
    bool first_or_done;
};

NAMESPACE_END(detail)

template <typename... Args> detail::init<Args...> init() { return detail::init<Args...>(); }
template <typename... Args> detail::init_alias<Args...> init_alias() { return detail::init_alias<Args...>(); }

/// Makes a python iterator from a first and past-the-end C++ InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename ValueType = decltype(*std::declval<Iterator>()),
          typename... Extra>
iterator make_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator, Sentinel, false, Policy> state;

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator")
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> ValueType {
                if (!s.first_or_done)
                    ++s.it;
                else
                    s.first_or_done = false;
                if (s.it == s.end) {
                    s.first_or_done = true;
                    throw stop_iteration();
                }
                return *s.it;
            }, std::forward<Extra>(extra)..., Policy);
    }

    return cast(state{first, last, true});
}

/// Makes an python iterator over the keys (`.first`) of a iterator over pairs from a
/// first and past-the-end InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename KeyType = decltype((*std::declval<Iterator>()).first),
          typename... Extra>
iterator make_key_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator, Sentinel, true, Policy> state;

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator")
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> KeyType {
                if (!s.first_or_done)
                    ++s.it;
                else
                    s.first_or_done = false;
                if (s.it == s.end) {
                    s.first_or_done = true;
                    throw stop_iteration();
                }
                return (*s.it).first;
            }, std::forward<Extra>(extra)..., Policy);
    }

    return cast(state{first, last, true});
}

/// Makes an iterator over values of an stl container or other container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type, typename... Extra> iterator make_iterator(Type &value, Extra&&... extra) {
    return make_iterator<Policy>(std::begin(value), std::end(value), extra...);
}

/// Makes an iterator over the keys (`.first`) of a stl map-like container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type, typename... Extra> iterator make_key_iterator(Type &value, Extra&&... extra) {
    return make_key_iterator<Policy>(std::begin(value), std::end(value), extra...);
}

template <typename InputType, typename OutputType> void implicitly_convertible() {
    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject * {
        if (!detail::make_caster<InputType>().load(obj, false))
            return nullptr;
        tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call((PyObject *) type, args.ptr(), nullptr);
        if (result == nullptr)
            PyErr_Clear();
        return result;
    };

    if (auto tinfo = detail::get_type_info(typeid(OutputType)))
        tinfo->implicit_conversions.push_back(implicit_caster);
    else
        pybind11_fail("implicitly_convertible: Unable to find type " + type_id<OutputType>());
}

template <typename ExceptionTranslator>
void register_exception_translator(ExceptionTranslator&& translator) {
    detail::get_internals().registered_exception_translators.push_front(
        std::forward<ExceptionTranslator>(translator));
}

/**
 * Wrapper to generate a new Python exception type.
 *
 * This should only be used with PyErr_SetString for now.
 * It is not (yet) possible to use as a py::base.
 * Template type argument is reserved for future use.
 */
template <typename type>
class exception : public object {
public:
    exception(handle scope, const char *name, PyObject *base = PyExc_Exception) {
        std::string full_name = scope.attr("__name__").cast<std::string>() +
                                std::string(".") + name;
        m_ptr = PyErr_NewException(const_cast<char *>(full_name.c_str()), base, NULL);
        if (hasattr(scope, name))
            pybind11_fail("Error during initialization: multiple incompatible "
                          "definitions with name \"" + std::string(name) + "\"");
        scope.attr(name) = *this;
    }

    // Sets the current python exception to this exception object with the given message
    void operator()(const char *message) {
        PyErr_SetString(m_ptr, message);
    }
};

/**
 * Registers a Python exception in `m` of the given `name` and installs an exception translator to
 * translate the C++ exception to the created Python exception using the exceptions what() method.
 * This is intended for simple exception translations; for more complex translation, register the
 * exception object and translator directly.
 */
template <typename CppException>
exception<CppException> &register_exception(handle scope,
                                            const char *name,
                                            PyObject *base = PyExc_Exception) {
    static exception<CppException> ex(scope, name, base);
    register_exception_translator([](std::exception_ptr p) {
        if (!p) return;
        try {
            std::rethrow_exception(p);
        } catch (const CppException &e) {
            ex(e.what());
        }
    });
    return ex;
}

NAMESPACE_BEGIN(detail)
PYBIND11_NOINLINE inline void print(tuple args, dict kwargs) {
    auto strings = tuple(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        strings[i] = str(args[i]);
    }
    auto sep = kwargs.contains("sep") ? kwargs["sep"] : cast(" ");
    auto line = sep.attr("join")(strings);

    object file;
    if (kwargs.contains("file")) {
        file = kwargs["file"].cast<object>();
    } else {
        try {
            file = module::import("sys").attr("stdout");
        } catch (const error_already_set &) {
            /* If print() is called from code that is executed as
               part of garbage collection during interpreter shutdown,
               importing 'sys' can fail. Give up rather than crashing the
               interpreter in this case. */
            return;
        }
    }

    auto write = file.attr("write");
    write(line);
    write(kwargs.contains("end") ? kwargs["end"] : cast("\n"));

    if (kwargs.contains("flush") && kwargs["flush"].cast<bool>())
        file.attr("flush")();
}
NAMESPACE_END(detail)

template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
void print(Args &&...args) {
    auto c = detail::collect_arguments<policy>(std::forward<Args>(args)...);
    detail::print(c.args(), c.kwargs());
}

#if defined(WITH_THREAD) && !defined(PYPY_VERSION)

/* The functions below essentially reproduce the PyGILState_* API using a RAII
 * pattern, but there are a few important differences:
 *
 * 1. When acquiring the GIL from an non-main thread during the finalization
 *    phase, the GILState API blindly terminates the calling thread, which
 *    is often not what is wanted. This API does not do this.
 *
 * 2. The gil_scoped_release function can optionally cut the relationship
 *    of a PyThreadState and its associated thread, which allows moving it to
 *    another thread (this is a fairly rare/advanced use case).
 *
 * 3. The reference count of an acquired thread state can be controlled. This
 *    can be handy to prevent cases where callbacks issued from an external
 *    thread would otherwise constantly construct and destroy thread state data
 *    structures.
 *
 * See the Python bindings of NanoGUI (http://github.com/wjakob/nanogui) for an
 * example which uses features 2 and 3 to migrate the Python thread of
 * execution to another thread (to run the event loop on the original thread,
 * in this case).
 */

class gil_scoped_acquire {
public:
    PYBIND11_NOINLINE gil_scoped_acquire() {
        auto const &internals = detail::get_internals();
        tstate = (PyThreadState *) PyThread_get_key_value(internals.tstate);

        if (!tstate) {
            tstate = PyThreadState_New(internals.istate);
            #if !defined(NDEBUG)
                if (!tstate)
                    pybind11_fail("scoped_acquire: could not create thread state!");
            #endif
            tstate->gilstate_counter = 0;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(internals.tstate);
            #endif
            PyThread_set_key_value(internals.tstate, tstate);
        } else {
            release = detail::get_thread_state_unchecked() != tstate;
        }

        if (release) {
            /* Work around an annoying assertion in PyThreadState_Swap */
            #if defined(Py_DEBUG)
                PyInterpreterState *interp = tstate->interp;
                tstate->interp = nullptr;
            #endif
            PyEval_AcquireThread(tstate);
            #if defined(Py_DEBUG)
                tstate->interp = interp;
            #endif
        }

        inc_ref();
    }

    void inc_ref() {
        ++tstate->gilstate_counter;
    }

    PYBIND11_NOINLINE void dec_ref() {
        --tstate->gilstate_counter;
        #if !defined(NDEBUG)
            if (detail::get_thread_state_unchecked() != tstate)
                pybind11_fail("scoped_acquire::dec_ref(): thread state must be current!");
            if (tstate->gilstate_counter < 0)
                pybind11_fail("scoped_acquire::dec_ref(): reference count underflow!");
        #endif
        if (tstate->gilstate_counter == 0) {
            #if !defined(NDEBUG)
                if (!release)
                    pybind11_fail("scoped_acquire::dec_ref(): internal error!");
            #endif
            PyThreadState_Clear(tstate);
            PyThreadState_DeleteCurrent();
            PyThread_delete_key_value(detail::get_internals().tstate);
            release = false;
        }
    }

    PYBIND11_NOINLINE ~gil_scoped_acquire() {
        dec_ref();
        if (release)
           PyEval_SaveThread();
    }
private:
    PyThreadState *tstate = nullptr;
    bool release = true;
};

class gil_scoped_release {
public:
    explicit gil_scoped_release(bool disassoc = false) : disassoc(disassoc) {
        // `get_internals()` must be called here unconditionally in order to initialize
        // `internals.tstate` for subsequent `gil_scoped_acquire` calls. Otherwise, an
        // initialization race could occur as multiple threads try `gil_scoped_acquire`.
        const auto &internals = detail::get_internals();
        tstate = PyEval_SaveThread();
        if (disassoc) {
            auto key = internals.tstate;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(key);
            #else
                PyThread_set_key_value(key, nullptr);
            #endif
        }
    }
    ~gil_scoped_release() {
        if (!tstate)
            return;
        PyEval_RestoreThread(tstate);
        if (disassoc) {
            auto key = detail::get_internals().tstate;
            #if PY_MAJOR_VERSION < 3
                PyThread_delete_key_value(key);
            #endif
            PyThread_set_key_value(key, tstate);
        }
    }
private:
    PyThreadState *tstate;
    bool disassoc;
};
#elif defined(PYPY_VERSION)
class gil_scoped_acquire {
    PyGILState_STATE state;
public:
    gil_scoped_acquire() { state = PyGILState_Ensure(); }
    ~gil_scoped_acquire() { PyGILState_Release(state); }
};

class gil_scoped_release {
    PyThreadState *state;
public:
    gil_scoped_release() { state = PyEval_SaveThread(); }
    ~gil_scoped_release() { PyEval_RestoreThread(state); }
};
#else
class gil_scoped_acquire { };
class gil_scoped_release { };
#endif

error_already_set::~error_already_set() {
    if (type) {
        gil_scoped_acquire gil;
        type.release().dec_ref();
        value.release().dec_ref();
        trace.release().dec_ref();
    }
}

inline function get_type_overload(const void *this_ptr, const detail::type_info *this_type, const char *name)  {
    handle self = detail::get_object_handle(this_ptr, this_type);
    if (!self)
        return function();
    handle type = self.get_type();
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overloaded in Python to avoid
       many costly Python dictionary lookups below */
    auto &cache = detail::get_internals().inactive_overload_cache;
    if (cache.find(key) != cache.end())
        return function();

    function overload = getattr(self, name, function());
    if (overload.is_cpp_function()) {
        cache.insert(key);
        return function();
    }

    /* Don't call dispatch code if invoked from overridden function.
       Unfortunately this doesn't work on PyPy. */
#if !defined(PYPY_VERSION)
    PyFrameObject *frame = PyThreadState_Get()->frame;
    if (frame && (std::string) str(frame->f_code->co_name) == name &&
        frame->f_code->co_argcount > 0) {
        PyFrame_FastToLocals(frame);
        PyObject *self_caller = PyDict_GetItem(
            frame->f_locals, PyTuple_GET_ITEM(frame->f_code->co_varnames, 0));
        if (self_caller == self.ptr())
            return function();
    }
#else
    /* PyPy currently doesn't provide a detailed cpyext emulation of
       frame objects, so we have to emulate this using Python. This
       is going to be slow..*/
    dict d; d["self"] = self; d["name"] = pybind11::str(name);
    PyObject *result = PyRun_String(
        "import inspect\n"
        "frame = inspect.currentframe()\n"
        "if frame is not None:\n"
        "    frame = frame.f_back\n"
        "    if frame is not None and str(frame.f_code.co_name) == name and "
        "frame.f_code.co_argcount > 0:\n"
        "        self_caller = frame.f_locals[frame.f_code.co_varnames[0]]\n"
        "        if self_caller == self:\n"
        "            self = None\n",
        Py_file_input, d.ptr(), d.ptr());
    if (result == nullptr)
        throw error_already_set();
    if (d["self"].is_none())
        return function();
    Py_DECREF(result);
#endif

    return overload;
}

template <class T> function get_overload(const T *this_ptr, const char *name) {
    auto tinfo = detail::get_type_info(typeid(T));
    return tinfo ? get_type_overload(this_ptr, tinfo, name) : function();
}

#define PYBIND11_OVERLOAD_INT(ret_type, cname, name, ...) { \
        pybind11::gil_scoped_acquire gil; \
        pybind11::function overload = pybind11::get_overload(static_cast<const cname *>(this), name); \
        if (overload) { \
            auto o = overload(__VA_ARGS__); \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value) { \
                static pybind11::detail::overload_caster_t<ret_type> caster; \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster); \
            } \
            else return pybind11::detail::cast_safe<ret_type>(std::move(o)); \
        } \
    }

#define PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(ret_type, cname, name, __VA_ARGS__) \
    return cname::fn(__VA_ARGS__)

#define PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(ret_type, cname, name, __VA_ARGS__) \
    pybind11::pybind11_fail("Tried to call pure virtual function \"" #cname "::" name "\"");

#define PYBIND11_OVERLOAD(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_NAME(ret_type, cname, #fn, fn, __VA_ARGS__)

#define PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, #fn, fn, __VA_ARGS__)

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#  pragma warning(pop)
#elif defined(__INTEL_COMPILER)
/* Leave ignored warnings on */
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
