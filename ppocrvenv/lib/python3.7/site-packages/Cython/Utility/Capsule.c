//////////////// Capsule.proto ////////////////

/* Todo: wrap the rest of the functionality in similar functions */
static CYTHON_INLINE PyObject *__pyx_capsule_create(void *p, const char *sig);

//////////////// Capsule ////////////////

static CYTHON_INLINE PyObject *
__pyx_capsule_create(void *p, CYTHON_UNUSED const char *sig)
{
    PyObject *cobj;

#if PY_VERSION_HEX >= 0x02070000
    cobj = PyCapsule_New(p, sig, NULL);
#else
    cobj = PyCObject_FromVoidPtr(p, NULL);
#endif

    return cobj;
}
