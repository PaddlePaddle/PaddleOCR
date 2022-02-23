/////////////// Header.proto ///////////////
//@proto_block: h_code

#if !defined(CYTHON_CCOMPLEX)
  #if defined(__cplusplus)
    #define CYTHON_CCOMPLEX 1
  #elif defined(_Complex_I)
    #define CYTHON_CCOMPLEX 1
  #else
    #define CYTHON_CCOMPLEX 0
  #endif
#endif

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #include <complex>
  #else
    #include <complex.h>
  #endif
#endif

#if CYTHON_CCOMPLEX && !defined(__cplusplus) && defined(__sun__) && defined(__GNUC__)
  #undef _Complex_I
  #define _Complex_I 1.0fj
#endif


/////////////// RealImag.proto ///////////////

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #define __Pyx_CREAL(z) ((z).real())
    #define __Pyx_CIMAG(z) ((z).imag())
  #else
    #define __Pyx_CREAL(z) (__real__(z))
    #define __Pyx_CIMAG(z) (__imag__(z))
  #endif
#else
    #define __Pyx_CREAL(z) ((z).real)
    #define __Pyx_CIMAG(z) ((z).imag)
#endif

#if defined(__cplusplus) && CYTHON_CCOMPLEX \
        && (defined(_WIN32) || defined(__clang__) || (defined(__GNUC__) && (__GNUC__ >= 5 || __GNUC__ == 4 && __GNUC_MINOR__ >= 4 )) || __cplusplus >= 201103)
    #define __Pyx_SET_CREAL(z,x) ((z).real(x))
    #define __Pyx_SET_CIMAG(z,y) ((z).imag(y))
#else
    #define __Pyx_SET_CREAL(z,x) __Pyx_CREAL(z) = (x)
    #define __Pyx_SET_CIMAG(z,y) __Pyx_CIMAG(z) = (y)
#endif


/////////////// Declarations.proto ///////////////
//@proto_block: complex_type_declarations

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< {{real_type}} > {{type_name}};
  #else
    typedef {{real_type}} _Complex {{type_name}};
  #endif
#else
    typedef struct { {{real_type}} real, imag; } {{type_name}};
#endif

static CYTHON_INLINE {{type}} {{type_name}}_from_parts({{real_type}}, {{real_type}});

/////////////// Declarations ///////////////

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE {{type}} {{type_name}}_from_parts({{real_type}} x, {{real_type}} y) {
      return ::std::complex< {{real_type}} >(x, y);
    }
  #else
    static CYTHON_INLINE {{type}} {{type_name}}_from_parts({{real_type}} x, {{real_type}} y) {
      return x + y*({{type}})_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE {{type}} {{type_name}}_from_parts({{real_type}} x, {{real_type}} y) {
      {{type}} z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif


/////////////// ToPy.proto ///////////////

#define __pyx_PyComplex_FromComplex(z) \
        PyComplex_FromDoubles((double)__Pyx_CREAL(z), \
                              (double)__Pyx_CIMAG(z))


/////////////// FromPy.proto ///////////////

static {{type}} __Pyx_PyComplex_As_{{type_name}}(PyObject*);

/////////////// FromPy ///////////////

static {{type}} __Pyx_PyComplex_As_{{type_name}}(PyObject* o) {
    Py_complex cval;
#if !CYTHON_COMPILING_IN_PYPY
    if (PyComplex_CheckExact(o))
        cval = ((PyComplexObject *)o)->cval;
    else
#endif
        cval = PyComplex_AsCComplex(o);
    return {{type_name}}_from_parts(
               ({{real_type}})cval.real,
               ({{real_type}})cval.imag);
}


/////////////// Arithmetic.proto ///////////////

#if CYTHON_CCOMPLEX
    #define __Pyx_c_eq{{func_suffix}}(a, b)   ((a)==(b))
    #define __Pyx_c_sum{{func_suffix}}(a, b)  ((a)+(b))
    #define __Pyx_c_diff{{func_suffix}}(a, b) ((a)-(b))
    #define __Pyx_c_prod{{func_suffix}}(a, b) ((a)*(b))
    #define __Pyx_c_quot{{func_suffix}}(a, b) ((a)/(b))
    #define __Pyx_c_neg{{func_suffix}}(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zero{{func_suffix}}(z) ((z)==({{real_type}})0)
    #define __Pyx_c_conj{{func_suffix}}(z)    (::std::conj(z))
    #if {{is_float}}
        #define __Pyx_c_abs{{func_suffix}}(z)     (::std::abs(z))
        #define __Pyx_c_pow{{func_suffix}}(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zero{{func_suffix}}(z) ((z)==0)
    #define __Pyx_c_conj{{func_suffix}}(z)    (conj{{m}}(z))
    #if {{is_float}}
        #define __Pyx_c_abs{{func_suffix}}(z)     (cabs{{m}}(z))
        #define __Pyx_c_pow{{func_suffix}}(a, b)  (cpow{{m}}(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eq{{func_suffix}}({{type}}, {{type}});
    static CYTHON_INLINE {{type}} __Pyx_c_sum{{func_suffix}}({{type}}, {{type}});
    static CYTHON_INLINE {{type}} __Pyx_c_diff{{func_suffix}}({{type}}, {{type}});
    static CYTHON_INLINE {{type}} __Pyx_c_prod{{func_suffix}}({{type}}, {{type}});
    static CYTHON_INLINE {{type}} __Pyx_c_quot{{func_suffix}}({{type}}, {{type}});
    static CYTHON_INLINE {{type}} __Pyx_c_neg{{func_suffix}}({{type}});
    static CYTHON_INLINE int __Pyx_c_is_zero{{func_suffix}}({{type}});
    static CYTHON_INLINE {{type}} __Pyx_c_conj{{func_suffix}}({{type}});
    #if {{is_float}}
        static CYTHON_INLINE {{real_type}} __Pyx_c_abs{{func_suffix}}({{type}});
        static CYTHON_INLINE {{type}} __Pyx_c_pow{{func_suffix}}({{type}}, {{type}});
    #endif
#endif

/////////////// Arithmetic ///////////////

#if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eq{{func_suffix}}({{type}} a, {{type}} b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE {{type}} __Pyx_c_sum{{func_suffix}}({{type}} a, {{type}} b) {
        {{type}} z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE {{type}} __Pyx_c_diff{{func_suffix}}({{type}} a, {{type}} b) {
        {{type}} z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE {{type}} __Pyx_c_prod{{func_suffix}}({{type}} a, {{type}} b) {
        {{type}} z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }

    #if {{is_float}}
    static CYTHON_INLINE {{type}} __Pyx_c_quot{{func_suffix}}({{type}} a, {{type}} b) {
        if (b.imag == 0) {
            return {{type_name}}_from_parts(a.real / b.real, a.imag / b.real);
        } else if (fabs{{m}}(b.real) >= fabs{{m}}(b.imag)) {
            if (b.real == 0 && b.imag == 0) {
                return {{type_name}}_from_parts(a.real / b.real, a.imag / b.imag);
            } else {
                {{real_type}} r = b.imag / b.real;
                {{real_type}} s = ({{real_type}})(1.0) / (b.real + b.imag * r);
                return {{type_name}}_from_parts(
                    (a.real + a.imag * r) * s, (a.imag - a.real * r) * s);
            }
        } else {
            {{real_type}} r = b.real / b.imag;
            {{real_type}} s = ({{real_type}})(1.0) / (b.imag + b.real * r);
            return {{type_name}}_from_parts(
                (a.real * r + a.imag) * s, (a.imag * r - a.real) * s);
        }
    }
    #else
    static CYTHON_INLINE {{type}} __Pyx_c_quot{{func_suffix}}({{type}} a, {{type}} b) {
        if (b.imag == 0) {
            return {{type_name}}_from_parts(a.real / b.real, a.imag / b.real);
        } else {
            {{real_type}} denom = b.real * b.real + b.imag * b.imag;
            return {{type_name}}_from_parts(
                (a.real * b.real + a.imag * b.imag) / denom,
                (a.imag * b.real - a.real * b.imag) / denom);
        }
    }
    #endif

    static CYTHON_INLINE {{type}} __Pyx_c_neg{{func_suffix}}({{type}} a) {
        {{type}} z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zero{{func_suffix}}({{type}} a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE {{type}} __Pyx_c_conj{{func_suffix}}({{type}} a) {
        {{type}} z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if {{is_float}}
        static CYTHON_INLINE {{real_type}} __Pyx_c_abs{{func_suffix}}({{type}} z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrt{{m}}(z.real*z.real + z.imag*z.imag);
          #else
            return hypot{{m}}(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE {{type}} __Pyx_c_pow{{func_suffix}}({{type}} a, {{type}} b) {
            {{type}} z;
            {{real_type}} r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    {{real_type}} denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        return __Pyx_c_prod{{func_suffix}}(a, a);
                    case 3:
                        z = __Pyx_c_prod{{func_suffix}}(a, a);
                        return __Pyx_c_prod{{func_suffix}}(z, a);
                    case 4:
                        z = __Pyx_c_prod{{func_suffix}}(a, a);
                        return __Pyx_c_prod{{func_suffix}}(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                } else if (b.imag == 0) {
                    z.real = pow{{m}}(a.real, b.real);
                    z.imag = 0;
                    return z;
                } else if (a.real > 0) {
                    r = a.real;
                    theta = 0;
                } else {
                    r = -a.real;
                    theta = atan2{{m}}(0.0, -1.0);
                }
            } else {
                r = __Pyx_c_abs{{func_suffix}}(a);
                theta = atan2{{m}}(a.imag, a.real);
            }
            lnr = log{{m}}(r);
            z_r = exp{{m}}(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cos{{m}}(z_theta);
            z.imag = z_r * sin{{m}}(z_theta);
            return z;
        }
    #endif
#endif
