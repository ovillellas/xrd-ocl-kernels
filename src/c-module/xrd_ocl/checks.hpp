#ifndef XRD_OCL_CHECKS_HPP
#define XRD_OCL_CHECKS_HPP

#include <cstdarg>
// A series of validation functions to use with python parse args.
//
// Used mainly to check dimensions and handle error gracefully while parsing
// args.


// a named array struct. It will be used as argument to the different checking
// functions. The name should be initialized at initialization time.
enum named_array_type {
    na_vector3 = 1,
    na_matrix33 = 2,
};

static const char*
named_array_type_to_str(named_array_type t)
{
    switch (t) {
    case na_vector3: return "3-vector";
    case na_matrix33: return "3x3-matrix";
    };
    return "unknown";
}

enum named_array_dim {
    na_0d_or_none = 1,
    na_0d_only = 2,
    na_1d_only = 3,
    na_0d_or_1d = 4
};

static const char *
named_array_dim_to_str(named_array_dim d)
{
    switch(d) {
    case na_0d_or_none: return "an optional";
    case na_0d_only: return "a single";
    case na_1d_only: return "a 1 dimensional array of";
    case na_0d_or_1d: return "a single or a 1 dimensional array of";
    };
    return "unknown";
}

struct named_array {
    const char *name;
    named_array_type kind;
    named_array_dim dim_kind;
    PyArrayObject *pyarray;
};


// returns true if op is an arrayobject, with a floating point element type
// (either single or double precision) and that is is behaved (byte ordering and
// alignment as required by the platform).
static inline bool
is_valid_array(PyObject *op)
{
    PyArrayObject *aop = reinterpret_cast<PyArrayObject *>(op);

    return PyArray_Check(op) &&
        (NPY_FLOAT32 == PyArray_TYPE(aop) || NPY_FLOAT64 == PyArray_TYPE(aop)) &&
        PyArray_ISBEHAVED_RO(aop);
}


static bool check_inner_dims(const PyArrayObject *aop, int ndim, ...)
{
    auto aop_ndim = PyArray_NDIM(aop);
    if (aop_ndim < ndim)
        return false; // not enough dimensions

    
    va_list argp;
    va_start(argp, ndim);

    for (auto dim = aop_ndim-ndim; dim < aop_ndim; dim++)
    {
        int curr_dim = va_arg(argp, int);
        if (PyArray_DIM(aop, dim) != curr_dim)
            goto fail;
    }

    va_end(argp);
    return true;
    
 fail:
    va_end(argp);
    return false;
}

// This is support for checking if a given array is a streaming kind of vector3/
// array33. Only one dimensional streaming arrays are supported.
// Source must be a known array object...

static inline bool
is_vector3(const PyArrayObject *aop)
{
    return (1 == PyArray_NDIM(aop) &&
            3 == PyArray_DIM(aop, 0));
}

static inline bool
is_matrix33(const PyArrayObject *aop)
{
    return (2 == PyArray_NDIM(aop) &&
            3 == PyArray_DIM(aop, 0) &&
            3 == PyArray_DIM(aop, 1));
}

static inline bool
is_streaming_vector2(const PyArrayObject *aop)
{
    return (2 == PyArray_NDIM(aop) &&
            0 < PyArray_DIM(aop, 0) &&
            2 == PyArray_DIM(aop, 1));
}

static inline bool
is_streaming_vector3(const PyArrayObject *aop)
{
    return (2 == PyArray_NDIM(aop) &&
            0 < PyArray_DIM(aop, 0) &&
            3 == PyArray_DIM(aop, 1));
}

static inline bool
is_streaming_matrix33(const PyArrayObject *aop)
{
    return (3 == PyArray_NDIM(aop) &&
            0 < PyArray_DIM(aop, 0) &&
            3 == PyArray_DIM(aop, 1) &&
            3 == PyArray_DIM(aop, 2));
}


static inline int
named_array_converter(PyObject *op, void *result)
{
    int inner_dims, outer_dims, total_dims;
    auto na = static_cast<named_array *>(result);

    if (op == Py_None && na_0d_or_none == na->dim_kind)
    {
        /* this one is ok. Just place a null in the pyarray.
           all other cases will go through the generic array path.
        */
        na->pyarray = nullptr;
        return 1;
    }
    
    if (!is_valid_array(op))
        goto fail;

    {
        auto array = reinterpret_cast<PyArrayObject*>(op);
        auto dims = PyArray_DIMS(array);
        total_dims = PyArray_NDIM(array);
        /* check base dimensions */
        switch (na->kind) {
        case na_vector3:
            inner_dims = 1;
            break;
        case na_matrix33:
            inner_dims = 2;
            break;
        default:
            goto internal_error;
        }

        outer_dims = total_dims - inner_dims;
        switch (na->dim_kind) {
        case na_0d_or_none:
        case na_0d_only:
            if (outer_dims != 0)
                goto fail;
            break;
        case na_1d_only:
            if (outer_dims != 1)
                goto fail;
            break;
        case na_0d_or_1d:
            if (outer_dims < 0 || outer_dims > 1)
                goto fail;
            break;
        default:
            goto internal_error;
        }
    
        // check that the inner dims are all '3' (works for both, vector3 and matrix33)
        for (int i = outer_dims; i < total_dims; i++) {
            if (dims[i] != 3)
                goto fail;
        }
        na->pyarray = array;
    }

    return 1;
 fail:
    PyErr_Format(PyExc_ValueError, "'%s' MUST be %s %s and behaved.",
                 na->name, named_array_dim_to_str(na->dim_kind),
                 named_array_type_to_str(na->kind));
    return 0;

 internal_error:
    PyErr_Format(PyExc_RuntimeError, "Internal error converting arg '%s' (%s::%d)",
                 na->name, __FILE__, __LINE__);
    return 0;
}



#endif // XRD_OCL_CHECKS_HPP
