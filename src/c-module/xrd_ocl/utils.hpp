#ifndef XRD_OCL_UTILS_HPP
#define XRD_OCL_UTILS_HPP

#include "checks.hpp"
#include <cstdint>
#include <type_traits>

template <typename T>
static inline void
zap_to_zero(T& to_zap)
{
    memset(&to_zap, 0, sizeof(to_zap));
}

template <typename T, typename T2>
static inline T
round_up_divide(T base, T2 divisor)
{
    static_assert(std::is_integral<T>::value, "only for integral types");
    static_assert(std::is_integral<T2>::value, "only for integral types");
    return 1+(base-1)/divisor;
}

template <typename T, typename T2>
static inline T
next_multiple(T base, T2 multiple)
{
    return multiple*round_up_divide(base, multiple);
}

// byte_index will offset the base pointer by 'offset' in bytes. This maps
// pretty well to what NumPy uses. Just dividing the offset by sizeof wouldn't
// always work in platforms were unaligned floating point access is legal on
// arrays with strides that are not a multiple of the floating point type
// natural size. Note that this is a fringe case.
template <typename T>
static inline const T *byte_index(const T *base, ptrdiff_t offset)
{
    return reinterpret_cast<const T*>(reinterpret_cast<const char *>(base)+offset);
}

template <typename T>
static inline T* byte_index(T* base, ptrdiff_t offset)
{
    return reinterpret_cast<T*>(reinterpret_cast<char *>(base)+offset);
}

template <typename T>
inline const T *
ndim_index(const T *base, const size_t *idx, const ptrdiff_t *strides, size_t ndim)
{
    return static_cast<const T*>(ndim_index(static_cast<const void*>(base), idx, strides, ndim));
}

template <typename T>
inline T *
ndim_index(T *base, const size_t *idx, const ptrdiff_t *strides, size_t ndim)
{
    return static_cast<T*>(ndim_index(static_cast<void *>(base), idx, strides, ndim));
}

template <>
inline void *
ndim_index<void>(void *base, const size_t *idx, const ptrdiff_t *strides, size_t ndim)
{
    char *addr = static_cast<char *>(base);
    for (size_t i = 0; i<ndim; i++)
        addr += idx[i]*strides[i];

    return static_cast<void *>(addr);
}

template <>
inline const void *
ndim_index<void>(const void *base, const size_t *idx, const ptrdiff_t *strides, size_t ndim)
{
    const char *addr = static_cast<const char *>(base);
    for (size_t i = 0; i<ndim; i++)
        addr += idx[i]*strides[i];

    return static_cast<const void *>(addr);
}


template <typename T>
inline const char *
type_to_cstr()
{
    return "unknown type";
}

template <>
inline const char *
type_to_cstr<float>()
{
    return "float";
}

template <>
inline const char *
type_to_cstr<double>()
{
    return "double";
}

//
// Copy convert operations.
//
// This will perform a copy from a strided source to a contiguous buffer,
// performing type conversion if needed.
//

template<typename SrcRealType, typename DstRealType>
static inline void
vector3_copy_convert(DstRealType * restrict dst_buff,
                     const SrcRealType * src_buff,
                     ptrdiff_t src_byte_stride)
{
    static_assert(std::is_floating_point<SrcRealType>::value,
                  "Types to copy convert must be floating point");
    static_assert(std::is_floating_point<DstRealType>::value,
                  "Types to copy convert must be floating point");

    for (size_t i = 0; i<3; i++)
        dst_buff[i] = *byte_index(src_buff, src_byte_stride*i);
}

template<typename SrcRealType, typename DstRealType>
static inline void
matrix33_copy_convert(DstRealType * restrict dst_buff,
                      const SrcRealType * src_buff,
                      ptrdiff_t src_byte_stride_row, // stride between different rows
                      ptrdiff_t src_byte_stride_column) // stride between different columns
{
    static_assert(std::is_floating_point<SrcRealType>::value,
                  "Types to copy convert must be floating point");
    static_assert(std::is_floating_point<DstRealType>::value,
                  "Types to copy convert must be floating point");
    for (size_t j = 0; j < 3; j++)
        for (size_t i = 0; i < 3; i++)
            dst_buff[3*j+i] = *byte_index(src_buff,
                                          src_byte_stride_row*j +
                                          src_byte_stride_column*i);
}
                                   
//
// copy convert of a single array. Optionally, add dynamic checks for type and
// dimensions.
//

template <typename T>
constexpr int numpy_type()
{
    return NPY_NOTYPE;
}

template <>
constexpr int
numpy_type<float>()
{
    return NPY_FLOAT32;
}

template <>
constexpr int
numpy_type<double>()
{
    return NPY_FLOAT64;
}


enum array_copy_convert_error {
    NO_ERROR = 0,
    BASETYPE_ERROR = 1, // base type error (float, double, etc)
    TYPE_ERROR = 2,     // other type error (element dimension errors)
    DIM_ERROR = 3,      // general dimension related errors (dimension counts)
    UNEXPECTED_ERROR = 4
};


//
// Copy Convert the vector3 in `pyarray` into the `result` buffer. `result` must
// point to a 3 element array of the destination type.
//
// returns 0 on success, which will be always unless RuntimeChecks is enabled.
// If RuntimeChecks is enabled and a check fails, and number will be returned
// that is unique to the failed check.
//
template<typename DstRealType, typename SrcRealType, bool RuntimeChecks=false>
static inline array_copy_convert_error
array_vector3_copy_convert(DstRealType * restrict result,
                           PyArrayObject *pyarray)
{
    static_assert(std::is_floating_point<DstRealType>::value,
                  "Types to copy convert must be floating point");
    static_assert(std::is_floating_point<SrcRealType>::value,
                  "Types to copy convert must be floating point");
    // perform some runtime checks if requested. Compiler should be able to
    // remove this if disabled.
    if (RuntimeChecks)
    {
        if (numpy_type<SrcRealType>() != PyArray_TYPE(pyarray))
            return BASETYPE_ERROR; // incorrect array type.

        // pyarray is one dimensional and its size is (3,)
        if (!is_vector3(pyarray))
            return TYPE_ERROR; // incorrect dimensions of the array
    }
    
    const SrcRealType* data = reinterpret_cast<SrcRealType*>(PyArray_DATA(pyarray));
    ptrdiff_t stride = PyArray_STRIDE(pyarray, 0);

    vector3_copy_convert(result, data, stride);
    
    return NO_ERROR;
}

template<typename DstRealType, typename SrcRealType, bool RuntimeChecks=false>
static inline array_copy_convert_error
array_matrix33_copy_convert(DstRealType * restrict result,
                           PyArrayObject *pyarray)
{
    static_assert(std::is_floating_point<DstRealType>::value,
                  "Types to copy convert must be floating point");
    static_assert(std::is_floating_point<SrcRealType>::value,
                  "Types to copy convert must be floating point");

    // perform some runtime checks if requested. Compiler should be able to
    // remove this if disabled.
    if (RuntimeChecks)
    {
        if (numpy_type<SrcRealType>() != PyArray_TYPE(pyarray))
            return BASETYPE_ERROR; // incorrect array type.

        // pyarray is one dimensional and its size is (3,)
        if (!is_matrix33(pyarray))
            return TYPE_ERROR; // incorrect dimensions of the array
    }
    
    const SrcRealType* data = reinterpret_cast<SrcRealType*>(PyArray_DATA(pyarray));
    ptrdiff_t stride_row = PyArray_STRIDE(pyarray, 0);
    ptrdiff_t stride_column = PyArray_STRIDE(pyarray, 1);

    matrix33_copy_convert(result, data, stride_row, stride_column);
    
    return NO_ERROR;
}


template<typename DstRealType, bool RuntimeChecks=false>
static inline array_copy_convert_error
array_vector3_autoconvert(DstRealType * restrict dst,
                          PyArrayObject *pyarray)
{
    static_assert(std::is_floating_point<DstRealType>::value,
                  "Types to copy convert must be floating point");

    if (RuntimeChecks)
    {
        if (!is_vector3(pyarray))
            return TYPE_ERROR;
    }

    // dispatch to the appropriate copy_convert
    switch (PyArray_TYPE(pyarray))
    {
    case numpy_type<float>():
        return array_vector3_copy_convert<DstRealType, float, RuntimeChecks>(dst, pyarray);
    case numpy_type<double>():
        return array_vector3_copy_convert<DstRealType, double, RuntimeChecks>(dst, pyarray);
    default:
        return BASETYPE_ERROR;
    }
}

template<typename DstRealType, bool RuntimeChecks=false>
static inline array_copy_convert_error
array_matrix33_autoconvert(DstRealType * restrict dst,
                          PyArrayObject *pyarray)
{
    static_assert(std::is_floating_point<DstRealType>::value,
                  "Types to copy convert must be floating point");

    if (RuntimeChecks)
    {
        if (!is_matrix33(pyarray))
            return TYPE_ERROR;
    }

    // dispatch to the appropriate copy_convert
    switch (PyArray_TYPE(pyarray))
    {
    case numpy_type<float>():
        return array_matrix33_copy_convert<DstRealType, float, RuntimeChecks>(dst, pyarray);
    case numpy_type<double>():
        return array_matrix33_copy_convert<DstRealType, double, RuntimeChecks>(dst, pyarray);
    default:
        return BASETYPE_ERROR;
    }
}


//
// utility printing functions, useful for debugging...
//
template <typename REAL>
constexpr const char *
floating_kind_name()
{
    return "not a supported float type";
}

template <>
constexpr const char *
floating_kind_name<float>()
{
    return "float32";
}

template <>
constexpr const char *
floating_kind_name<double>()
{
    return "float64";
}

static inline const char *
numpy_type_to_str(int type)
{
    switch(type) {
    case NPY_FLOAT32:
        return floating_kind_name<float>();
    case NPY_FLOAT64:
        return floating_kind_name<float>();
    default:
        return "unsupported numpy type";
    };
}


template <typename REAL>
static inline void
print_vector3(const char *name, const REAL *val)
{
    printf("%s <3x %s>:\n", name, floating_kind_name<REAL>());
    printf("%6.4f %6.4f %6.4f\n", val[0], val[1], val[2]);
}

template <typename REAL>
static inline void
print_matrix33(const char *name, const REAL *val)
{
    printf("%s <9x %s>:\n", name, floating_kind_name<REAL>());
    printf("%6.4f %6.4f %6.4f\n", val[0], val[1], val[2]);
    printf("%6.4f %6.4f %6.4f\n", val[3], val[4], val[5]);
    printf("%6.4f %6.4f %6.4f\n", val[6], val[7], val[8]);
}

//
// different kind of streams. Note that these are supposed to be POD classes.
//
struct stream_desc {
    static const int STREAM_MAX_NDIMS = 7;
    
    void *base;
    uint8_t base_type; // either NPY_FLOAT32 or NPY_FLOAT64
    uint8_t ndim_outer;
    uint8_t ndim_inner;
    
    ptrdiff_t strides[STREAM_MAX_NDIMS];
    size_t dims[STREAM_MAX_NDIMS];
                           

    bool is_active() const { return base != NULL; }
    const int element_base_type() const { return base_type; }
    size_t ndim() const { return ndim_outer + ndim_inner; }
    size_t stream_ndim() const { return ndim_outer; }
    size_t element_ndim() const { return ndim_inner; }
    const ptrdiff_t *stream_strides() const { return &strides[0]; }
    const ptrdiff_t *element_strides() const { return &strides[ndim_outer]; }
    const size_t *stream_dims() const { return &dims[0]; }
    const size_t *element_dims() const { return &dims[ndim_outer]; }

};

/*
typedef stream_desc<2,1> vector2_2d_stream;
typedef stream_desc<1,1> vector3_1d_stream;
typedef stream_desc<1,2> matrix33_1d_stream;
*/

// converts a PyArray into an equivalent stream structure to help
// stream multiple elements of inner_dims dimensions over its outer_dims.
//
// inner_dims must be the number of dimensions for the elements. The
// pyarray must have at least inner_dims dimensions.
//
// outer_dims is the number of outer dimensions expected. A -1 will mean
// any number of outer dimensions (as long as inner_dims + outer_dims does
// not exceed STREAM_MAX_DIMS.
static inline array_copy_convert_error
array_stream_convert(stream_desc *restrict stream,
                     PyArrayObject *pyarray, int inner_dims, int outer_dims)
{
    int pa_ndim = PyArray_NDIM(pyarray);

    // first, perform all checks that could result in a type error.
    if (inner_dims > pa_ndim)
        return TYPE_ERROR; // not enough dimensions for a single element!

    if (pa_ndim > stream_desc::STREAM_MAX_NDIMS ||
        (outer_dims >= 0 && (outer_dims + inner_dims) != pa_ndim))
        return DIM_ERROR; // 

    stream->base = PyArray_DATA(pyarray);
    stream->base_type = PyArray_TYPE(pyarray);
    stream->ndim_outer = pa_ndim - inner_dims;
    stream->ndim_inner = inner_dims;
    int stream_idx = 0;
    for (stream_idx = 0; stream_idx < pa_ndim; stream_idx++)
    {
        stream->strides[stream_idx] = PyArray_STRIDE(pyarray, stream_idx);
        stream->dims[stream_idx] = PyArray_DIM(pyarray, stream_idx);
    }

    return NO_ERROR;
}

template<typename TYPE>
static inline void
print_vector(const char *el_fmt, const TYPE *elements, size_t count)
{
    if (count < 1)
        return;
    
    printf("("); printf(el_fmt, elements[0]);
    for (size_t i = 1; i<count; i++)
    {
        printf(", "); printf(el_fmt, elements[i]);
    }
    printf(")");
    return;
}


             
template <typename STREAM_DESC>
inline void
print_stream(const char *name, const STREAM_DESC& stream)
{
    printf("%s stream:\n", name);
    if (stream.is_active())
    {
        printf("\tbase: %p dims: ", stream.base);
        print_vector("%d", stream.stream_dims(), stream.stream_ndim());
        printf(" strides: ");
        print_vector("%td", stream.stream_strides(), stream.stream_ndim());
        printf("\n\telement: ");
        print_vector("%d", stream.element_dims(), stream.element_ndim());
        printf(" of %s strides: ",
               numpy_type_to_str(stream.element_base_type()));
        print_vector("%td", stream.element_strides(), stream.element_ndim());
        printf("\n");
        
    }
    else
    {
        printf("\tINACTIVE\n");
    }
}

template <typename TYPE>
static inline void
debug_print_array(const char *name, const TYPE *vect, size_t ndim,
             const char *elfmt, const char *fmt=0)
{
    if (!fmt || strlen(fmt) < 4)
        fmt = "[,]\n";

    if (name)
        printf("%s ", name);
    
    if (0 == ndim)
    {
        printf("%c %c%c", fmt[0], fmt[2], fmt[3]);
    }
    else
    {
        printf("%c ", fmt[0]);
        printf(elfmt, vect[0]);
        for (size_t i=1; i<ndim; i++)
        {
            printf("%c ", fmt[1]);
            printf(elfmt, vect[i]);
        }
        printf(" %c%c", fmt[2], fmt[3]);
    }
}

static inline void
debug_print_dims(const char *name, const size_t *dims, size_t ndim)
{
    debug_print_array(name, dims, ndim, "%zu");
}


static inline void
debug_print_strides(const char *name, const ptrdiff_t *strides, size_t ndim)
{
    debug_print_array(name, strides, ndim, "%zd");
}

#endif
