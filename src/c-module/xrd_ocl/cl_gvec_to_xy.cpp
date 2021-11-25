#include "checks.hpp"
#include "utils.hpp"
#include "dev_help.hpp"

#include <algorithm> // std::max

#define CLXF_PREFERRED_SLICE_SIZE (4*1024*1024)

// parameters for use in gvec_to_xy. Templatized to support both, float and
// double versions.
template <typename REAL>
struct gvec_to_xy_params {
    cl_uint chunk_size[2];
    cl_uint total_size[2];
    REAL gVec_c[3];
    REAL rMat_d[9];
    REAL rMat_s[9];
    REAL rMat_c[9];
    REAL tVec_d[3];
    REAL tVec_s[3];
    REAL tVec_c[3];
    REAL beam[3];
    cl_char has_beam;
};


// streams used in gvec_to_xy. Some arguments may be streamed, so they will
// have a stream represented. It is not templatized, as the streams will be
// converted in a chunk basis for execution. This is to avoid
// creating/maintaining a full size copy of possibly huge arrays.
//
// A disabled stream is marked by a NULL in its base address, but the canonical
// way to set the stream as non-used is to zap it with zeroes.
struct gvec_to_xy_streams {
    stream_desc gVec_c_stream;
    stream_desc rMat_s_stream;
    stream_desc tVec_c_stream;
    stream_desc xy_out_stream;
};

template <typename REAL>
static inline int
init_gvec_to_xy(gvec_to_xy_params<REAL> * restrict params,
                gvec_to_xy_streams * restrict streams,
                PyArrayObject *pa_gVec_c,
                PyArrayObject *pa_rMat_d,
                PyArrayObject *pa_rMat_s,
                PyArrayObject *pa_rMat_c,
                PyArrayObject *pa_tVec_d,
                PyArrayObject *pa_tVec_s,
                PyArrayObject *pa_tVec_c,
                PyArrayObject *pa_beam_vec,
                PyArrayObject *pa_result_xy)
{
    if (!is_streaming_vector3(pa_gVec_c))
    {
        array_vector3_autoconvert<REAL>(params->gVec_c, pa_gVec_c);
        zap_to_zero(streams->gVec_c_stream);
    }
    else
    {
        zap_to_zero(params->gVec_c);
        array_stream_convert(&streams->gVec_c_stream, pa_gVec_c, 1, 1);
    }

    array_matrix33_autoconvert<REAL>(params->rMat_d, pa_rMat_d);

    if (!is_streaming_matrix33(pa_rMat_s))
    {
        array_matrix33_autoconvert<REAL>(params->rMat_s, pa_rMat_s);
        zap_to_zero(streams->rMat_s_stream);
    }
    else
    {
        zap_to_zero(params->rMat_s);
        array_stream_convert(&streams->rMat_s_stream, pa_rMat_s, 2, 1);
    }

    array_matrix33_autoconvert<REAL>(params->rMat_c, pa_rMat_c);
    array_vector3_autoconvert<REAL>(params->tVec_d, pa_tVec_d);
    array_vector3_autoconvert<REAL>(params->tVec_s, pa_tVec_s);

    if (!is_streaming_vector3(pa_tVec_c))
    {
        array_vector3_autoconvert<REAL>(params->tVec_c, pa_tVec_c);
        zap_to_zero(streams->tVec_c_stream);
    }
    else
    {
        zap_to_zero(params->tVec_c);
        array_stream_convert(&streams->tVec_c_stream, pa_tVec_c, 1, 1);
    }

    if (pa_beam_vec)
    {
        array_vector3_autoconvert<REAL>(params->beam, pa_beam_vec);
        params->has_beam = 1;
    }
    else
    {
        zap_to_zero(params->beam);
        params->has_beam = 0;
    }

    array_stream_convert(&streams->xy_out_stream, pa_result_xy, 1, 2);

    size_t ngvec = streams->xy_out_stream.stream_dims()[0];
    size_t npos = streams->xy_out_stream.stream_dims()[1];

    params->chunk_size[0] = static_cast<cl_uint>(ngvec);
    params->chunk_size[1] = static_cast<cl_uint>(npos);
    params->total_size[0] = static_cast<cl_uint>(ngvec);
    params->total_size[1] = static_cast<cl_uint>(npos);

    return 0;
}

template <typename REAL>
static void
print_gvec_to_xy(const gvec_to_xy_params<REAL> *params,
                 const gvec_to_xy_streams *streams)
{
    printf("gvec_to_xy_params\nkind: %s\n", floating_kind_name<REAL>());
    printf("chunk_size: (%u, %u)\n", params->chunk_size[0], params->chunk_size[1]);
    printf("total_size: (%u, %u)\n", params->total_size[0], params->total_size[1]);
    print_vector3("gVec_c", params->gVec_c);
    print_matrix33("rMat_d", params->rMat_d);
    print_matrix33("rMat_s", params->rMat_s);
    print_matrix33("rMat_c", params->rMat_c);
    print_vector3("tVec_d", params->tVec_d);
    print_vector3("tVec_s", params->tVec_s);
    print_vector3("tVec_c", params->tVec_c);
    print_vector3("beam", params->beam);
    printf("has_beam: %s\n", params->has_beam?"True":"False");

    print_stream("gVec_c", streams->gVec_c_stream);
    print_stream("rMat_s", streams->rMat_s_stream);
    print_stream("tVec_c", streams->tVec_c_stream);
    print_stream("xy_out", streams->xy_out_stream);
}

struct g2xy_buffs
{
    cl_mem params;  // the problem "params"
    cl_mem gvec_c;  // gpu buffer for gvectors
    cl_mem rmat_s;  // gpu buffer for rmat_s
    cl_mem tvec_c;  // gpu buffer for tvec_c
    cl_mem xy_result;  // gpu buffer for xy_result
};

static void
release_g2xy_buffs(g2xy_buffs *buffs)
{
    CL_LOG_CHECK(clReleaseMemObject(buffs->xy_result));
    CL_LOG_CHECK(clReleaseMemObject(buffs->tvec_c));
    CL_LOG_CHECK(clReleaseMemObject(buffs->rmat_s));
    CL_LOG_CHECK(clReleaseMemObject(buffs->gvec_c));
    CL_LOG_CHECK(clReleaseMemObject(buffs->params));

    zap_to_zero(*buffs);
}

template<typename REAL>
cl_instance::kernel_slot gvec_to_xy_kernel()
{
    static_assert(sizeof(REAL) == 0, "Unsupported type for gvec_to_xy");
    return cl_instance::kernel_slot::invalid;
}

template<>
cl_instance::kernel_slot gvec_to_xy_kernel<float>()
{
    return cl_instance::kernel_slot::gvec_to_xy_f32;
}

template<>
cl_instance::kernel_slot gvec_to_xy_kernel<double>()
{
    return cl_instance::kernel_slot::gvec_to_xy_f64;
}

static const char *g2xy_source = R"CLC(
/* example kernel to check things are kind of working... */
#if defined(USE_SINGLE_PRECISION) && USE_SINGLE_PRECISION
#  define REAL float
#  define REAL2 float2
#  define REAL3 float3
#else
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#  define REAL double
#  define REAL2 double2
#  define REAL3 double3
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

struct __attribute__((packed)) gvec_to_xy_params {
    uint chunk_size[2];
    uint total_size[2];
    REAL gVec_c[3];
    REAL rMat_d[9];
    REAL rMat_s[9];
    REAL rMat_c[9];
    REAL tVec_d[3];
    REAL tVec_s[3];
    REAL tVec_c[3];
    REAL beam[3];
    char has_beam;
};

REAL3 diffract_z(REAL3 gvec);
REAL3 diffract(REAL3 beam, REAL3 gvec);
REAL3 transform_vector(REAL3 tr, REAL3 rot0, REAL3 rot1, REAL3 rot2, REAL3 v);
REAL3 rotate_vector(REAL3 rot0, REAL3 rot1, REAL3 rot2, REAL3 v);
REAL2 to_detector(REAL3 rD0, REAL3 rD1, REAL3 rD2, REAL3 tD,
                  REAL3 ray_origin, REAL3 ray_vector);

REAL3
diffract_z(REAL3 gvec)
{
    return (REAL3)(((REAL)2.0)*gvec.z*gvec)-(REAL3)((REAL)0.0, (REAL)0.0, (REAL)1.0);
}

REAL3
diffract(REAL3 beam, REAL3 gvec)
{
    REAL3 bm_diag  = ((REAL)2.0)*gvec*gvec - ((REAL)1.0);
    REAL3 bm_other = ((REAL)2.0)*gvec.xxy*gvec.yzz;
    REAL3 row0 = (REAL3)(bm_diag.x, bm_other.x, bm_other.y);
    REAL3 row1 = (REAL3)(bm_other.x, bm_diag.y, bm_other.z);
    REAL3 row2 = (REAL3)(bm_other.y, bm_other.z, bm_diag.z);
    return (REAL3)(dot(row0, beam), dot(row1, beam), dot(row2,beam));
}

REAL3
transform_vector(REAL3 tr, REAL3 rot0, REAL3 rot1, REAL3 rot2, REAL3 v)
{
    return (REAL3)(tr.x + dot(rot0, v), tr.y + dot(rot1, v), tr.z + dot(rot2, v));
}

REAL3
rotate_vector(REAL3 rot0, REAL3 rot1, REAL3 rot2, REAL3 v)
{
    return (REAL3)(dot(rot0, v), dot(rot1, v), dot(rot2, v));
}

REAL2
to_detector(REAL3 rD0, REAL3 rD1, REAL3 rD2, REAL3 tD,
            REAL3 ray_origin, REAL3 ray_vector)
{
    REAL3 ray_origin_det = ray_origin - tD;
    REAL3 rDt0 = (REAL3)(rD0.x, rD1.x, rD2.x);
    REAL3 rDt1 = (REAL3)(rD0.y, rD1.y, rD2.y);
    REAL3 rDt2 = (REAL3)(rD0.z, rD1.z, rD2.z);

    REAL num = dot(rDt2, ray_origin_det);
    REAL denom = dot(rDt2, ray_vector);
    REAL t = num/denom;
    REAL factor = t<0.0?t:NAN;
    REAL3 point = ray_origin_det - factor*ray_vector;
    return (REAL2)(dot(point, rDt0), dot(point, rDt1));
}

__kernel void gvec_to_xy(
    __constant const struct gvec_to_xy_params *params,
    __global const REAL *g_gVec_c,
    __global const REAL *g_rMat_s,
    __global const REAL *g_tVec_c,
    __global REAL * restrict g_xy_result)
{
//    size_t element = get_global_id(0);
    size_t gvec_idx = get_global_id(0);
    size_t pos_idx = get_global_id(1);
    uint ngvec = params->chunk_size[0];
    uint npos = params->chunk_size[1];
//    uint gvec_idx = element / npos;
//    uint npos_idx = element % npos;
    /* handle border case */
    if (gvec_idx >= ngvec || pos_idx >= npos)
        return;

    REAL3 tVec_c = vload3(pos_idx, g_tVec_c);
    REAL3 gVec_c = vload3(gvec_idx, g_gVec_c);
    REAL3 rMat_s0, rMat_s1, rMat_s2;
    REAL3 rMat_c0 = vload3(0, params->rMat_c),
          rMat_c1 = vload3(1, params->rMat_c),
          rMat_c2 = vload3(2, params->rMat_c);
    REAL3 rMat_d0 = vload3(0, params->rMat_d),
          rMat_d1 = vload3(1, params->rMat_d),
          rMat_d2 = vload3(2, params->rMat_d);
    REAL3 tVec_s = vload3(0, params->tVec_s);
    REAL3 tVec_d = vload3(0, params->tVec_d);
    REAL beam[3];

    if (g_rMat_s)
    {
        rMat_s0 = vload3((3*gvec_idx+0), g_rMat_s);
        rMat_s1 = vload3((3*gvec_idx+1), g_rMat_s);
        rMat_s2 = vload3((3*gvec_idx+2), g_rMat_s);
    }
    else
    {
        rMat_s0 = vload3(0, params->rMat_s);
        rMat_s1 = vload3(1, params->rMat_s);
        rMat_s2 = vload3(2, params->rMat_s);
    }

/*    if (element==0)
    {
       printf("\n\nwhat: %9.4v3hlf\n", rMat_c0);
       printf("test: %9.4f %9.4f %9.4f\n", rMat_c0.x, rMat_c0.y, rMat_c0.z);
       printf("gVec_c:\n");
       printf("%9.4v3hlf\n", gVec_c);
       printf("rMat_d:\n");
       printf("%9.4v3hlf\n", rMat_d0);
       printf("%9.4v3hlf\n", rMat_d1);
       printf("%9.4v3hlf\n", rMat_d2);
       printf("rMat_s:\n");
       printf("%9.4v3hlf\n", rMat_s0);
       printf("%9.4v3hlf\n", rMat_s1);
       printf("%9.4v3hlf\n", rMat_s2);
       printf("rMat_c:\n");
       printf("%9.4v3hlf\n", rMat_c0);
       printf("%9.4v3hlf\n", rMat_c1);
       printf("%9.4v3hlf\n", rMat_c2);
       printf("tVec_d:\n");
       printf("%9.4v3hlf\n", tVec_d);
       printf("tVec_s:\n");
       printf("%9.4v3hlf\n", tVec_s);
       printf("tVec_c:\n");
       printf("%9.4v3hlf\n", tVec_c);
       printf("beam:\n");
       printf("%9.4v3hlf\n", beam);
    }
*/
    REAL3 ray_origin = transform_vector(tVec_s, rMat_s0, rMat_s1, rMat_s2, tVec_c);
    REAL3 gVec_sam = rotate_vector(rMat_c0, rMat_c1, rMat_c2, gVec_c);
    REAL3 gVec_lab = rotate_vector(rMat_s0, rMat_s1, rMat_s2, gVec_sam);
    REAL3 ray_vector;
    if (params->has_beam)
        ray_vector = diffract(vload3(0, params->beam), gVec_lab);
    else
        ray_vector = diffract_z(gVec_lab);

    REAL2 projected = to_detector(rMat_d0, rMat_d1, rMat_d2, tVec_d, ray_origin, ray_vector);

    vstore2(projected, gvec_idx*npos + pos_idx , g_xy_result);
}
)CLC";

template<typename REAL>
static const char *
kernel_compile_options()
{
    return "-cl-std=CL1.2";
}

template<>
const char *
kernel_compile_options<float>()
{
    return "-D USE_SINGLE_PRECISION -cl-single-precision-constant -cl-std=CL1.2";
}


template<typename REAL>
static bool
init_g2xy_buffs(g2xy_buffs *return_state,
                cl_context context,
                const gvec_to_xy_params<REAL> *params,
                const gvec_to_xy_streams *streams)
{
    TIME_SCOPE("init_g2xy_buffs");
    g2xy_buffs buffs;
    zap_to_zero(buffs);
    zap_to_zero(*return_state);

    // params buffer, remains constant for the whole execution.
    {
        const cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                        CL_MEM_COPY_HOST_PTR |
                                        CL_MEM_HOST_NO_ACCESS;

        buffs.params = clCreateBuffer(context, buff_flags,
                                      sizeof(*params), (void*)params, NULL);
    }
    // note: chunk_sz contains the chunksize to init for. The chunksize has two
    // dimensions, the number of gvecs and the number of candidate positions.
    // The chunksize should be made in such a way that the buffers fit in memory.
    // Note the chunksize needs to hit a balance, as a bigger chunksize will mean
    // more memory usage as well as more upfront copy-convert time for
    // the parameters, but also will mean bigger batches for each command.
    //
    // It should be possible to double buffer copy-convert and execution so that
    // copy-convert time is overlapped with actual computation in the GPU.

    if (streams->gVec_c_stream.is_active())
    {
        size_t buff_size = 3*sizeof(REAL)*params->chunk_size[0];
        const cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                        CL_MEM_HOST_WRITE_ONLY;
        buffs.gvec_c = clCreateBuffer(context, buff_flags, buff_size,
                                      NULL, NULL);
    }
    else
        goto fail; // not supported yet.

    if (streams->rMat_s_stream.is_active())
    {
        size_t buff_size = 9*sizeof(REAL)*params->chunk_size[0];
        cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                  CL_MEM_HOST_WRITE_ONLY;
        buffs.rmat_s = clCreateBuffer(context, buff_flags, buff_size,
                                          NULL, NULL);
    }
    else
        goto fail; // not supported yet.

    if (streams->tVec_c_stream.is_active())
    {
        size_t buff_size = 3*sizeof(REAL)*params->chunk_size[1];
        cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                  CL_MEM_HOST_WRITE_ONLY;
        buffs.tvec_c = clCreateBuffer(context, buff_flags, buff_size,
                                      NULL, NULL);
    }
    else
        goto fail;

    if (streams->xy_out_stream.is_active())
    {
        size_t buff_size = 2*sizeof(REAL)*params->chunk_size[0]*params->chunk_size[1];
        cl_mem_flags buff_flags = CL_MEM_WRITE_ONLY |
                                  CL_MEM_HOST_READ_ONLY;
        buffs.xy_result = clCreateBuffer(context, buff_flags, buff_size,
                                         NULL, NULL);
    }
    else
        goto fail;

    *return_state = buffs;
    return true;

 fail:
    CL_LOG(1, "Failure to allocate buffers for gvec_to_xy");
    release_g2xy_buffs(&buffs);
    return false;
}


static void
raw_copy_to_buffer(cl_command_queue queue, cl_mem buffer, const void *srcdata, size_t sz)
{
    void *mapped = clEnqueueMapBuffer(queue, buffer, CL_TRUE,
                                      CL_MAP_WRITE_INVALIDATE_REGION,
                                      0, sz, 0, NULL, NULL, NULL);
    memcpy(mapped, srcdata, sz);
    clEnqueueUnmapMemObject(queue, buffer, mapped, 0, NULL, NULL);
}

static void
raw_copy_from_buffer(cl_command_queue queue, cl_mem buffer, void *dstdata, size_t sz)
{
    void *mapped = clEnqueueMapBuffer(queue, buffer, CL_TRUE,
                                      CL_MAP_READ,
                                      0, sz, 0, NULL, NULL, NULL);

    memcpy(dstdata, mapped, sz);
    clEnqueueUnmapMemObject(queue, buffer, mapped, 0, NULL, NULL);
}

//
// copy convert into a given buffer contents from the stream_desc pointed by
// stream. The type on the destination buffer is based on the template, while
// the one in the source stream is based on the info in the stream_desc.
//
// `pos` contains the position in the stream to be copy converted.
// `sz` contains the size of the data to be copy converted.
// `ndims` contains the number of dimensions of `pos` and `sz`.
//
// ndims MUST match the stream_ndim in the stream_desc.
// `pos` and `sz` should define a valid range in stream defined by stream_desc.
//
// all data will be linearized in the target buffer, in 'C order'
//
// returns false on end of iteration.
static inline bool
next_element(size_t * restrict curr, const size_t *dims, size_t ndims)
{
    size_t it_pos = ndims;
    while (it_pos--) {
        if (++curr[it_pos] < dims[it_pos])
        {
            return true;
        }
        curr[it_pos] = 0;
    }

    return false;
}

template<typename DST_REAL, typename SRC_REAL>
static void
copy_convert_chunk(void * restrict dst, const void *src,
                   const size_t *dims, const ptrdiff_t *strides,
                   size_t ndim)
{
    //printf("Chunk in: %s <- %s.\n", type_to_cstr<DST_REAL>(), type_to_cstr<SRC_REAL>());
    DST_REAL * restrict out = static_cast<DST_REAL *>(dst);
    const SRC_REAL *in = static_cast<const SRC_REAL *>(src);
    size_t curr_pos[ndim];
    for (size_t i = 0; i<ndim; i++)
        curr_pos[i] = 0;

    size_t count = 0;
    do {
        *out++ = *ndim_index(in, curr_pos, strides, ndim);
        count ++;
    } while(next_element(curr_pos, dims, ndim));
#if defined(CLXF_LOG_COPY_CONVERT) && CLXF_LOG_COPY_CONVERT
    print_dims("chunk dimensions", dims, ndim);
    printf("Chunk in: %zd %s written (%s original)\n", count,
           type_to_cstr<DST_REAL>(), type_to_cstr<SRC_REAL>());
#endif
}


template<typename DST_REAL, typename SRC_REAL>
static void
copy_convert_chunk_out(void * restrict dst, const void *src,
                       const size_t *dims, const ptrdiff_t *strides,
                       size_t ndim)
{
     // This is likely to write scattered, which is not ideal. Open for optimization
    const SRC_REAL *in = static_cast<const SRC_REAL *>(src);
    DST_REAL * restrict out = static_cast<DST_REAL *>(dst);
    size_t curr_pos[ndim];
    for (size_t i = 0; i<ndim; i++)
        curr_pos[i] = 0;

    size_t count = 0;
    do {
        *(ndim_index(out, curr_pos, strides, ndim)) = *in++;
        count ++;
    } while(next_element(curr_pos, dims, ndim));

#if defined(CLXF_LOG_COPY_CONVERT) && CLXF_LOG_COPY_CONVERT
    print_dims("chunk dimensions", dims, ndim);
    printf("Chunk out: %zd %s written (%s original)\n", count,
           type_to_cstr<DST_REAL>(), type_to_cstr<SRC_REAL>());
#endif
}

template <typename REAL>
inline array_copy_convert_error
copy_convert_to_buffer(cl_command_queue queue, cl_mem buffer,
                       const stream_desc *stream, const size_t *pos,
                       const size_t *sz, size_t ndim)
{
    array_copy_convert_error err = NO_ERROR;
    if (ndim != stream->stream_ndim())
        return DIM_ERROR;

    if (stream->base_type != NPY_FLOAT32 && stream->base_type != NPY_FLOAT64)
        return BASETYPE_ERROR;

    size_t total_ndim = stream->ndim();

    // build a dimension array for the chunk to copy-convert.
    // compute size required in the target buffer
    size_t dims[total_ndim];
    size_t total_size = sizeof(REAL);
    for (size_t i = 0; i < ndim; i ++) {
        // outer dimensions are those of the chunk
        dims[i] = sz[i];
        total_size *= sz[i];
    }
    for (size_t i = 0; i < stream->element_ndim(); i++) {
        // inner dimensions are the stream element dimensions
        dims[ndim+i] = stream->element_dims()[i];
        total_size *= stream->element_dims()[i];
    }
    const ptrdiff_t *strides = stream->stream_strides();
    const void *src = ndim_index(stream->base, pos, strides,
                                 stream->stream_ndim());

    //printf("q: %p b: %p - WRITE_INVALIDATE - sz: %zd.\n", queue, buffer, total_size);
    void *dst = clEnqueueMapBuffer(queue, buffer, CL_TRUE,
                                   CL_MAP_WRITE_INVALIDATE_REGION,
                                   0, total_size, 0, NULL, NULL, NULL);
    switch(stream->base_type)
    {
    case NPY_FLOAT32:
        copy_convert_chunk<REAL, float>(dst, src, dims, strides, total_ndim);
        break;
    case NPY_FLOAT64:
        copy_convert_chunk<REAL, double>(dst, src, dims, strides, total_ndim);
        break;
    default:
        // this should not happen, should be caught at pre-condition
        err = UNEXPECTED_ERROR;
    }
    clEnqueueUnmapMemObject(queue, buffer, dst, 0, NULL, NULL);

    return err;
}

// perform a cl_buffer to main memory slicing to a max size so that DMA and
// memcpy may be overlapped.
static void
sliced_buff_to_mem(cl_command_queue queue, cl_mem buffer, void *dst, size_t total_size, size_t slice_size)
{
    const size_t SLICE_SIZE = slice_size;

    if (total_size < 2*SLICE_SIZE)
    {   TIME_SCOPE("sliced_buff_to_mem: under threshold");
        void *src;
        {
            src = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0,
                                     total_size, 0, NULL, NULL, NULL);
        }
        memcpy(dst, src, total_size);
    }
    else
    {   TIME_SCOPE("sliced_buff_to_mem: the real deal");
        cl_context the_context;
        cl_mem pinned_buffer;
        void *staging_mem;
        cl_event pending[2];
        int curr_buff = 0, next_buff = 1; // buffer that needs memcpy...
        size_t curr_offset = 0;
        size_t next_offset = SLICE_SIZE;
        CL_LOG_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(the_context), &the_context, NULL));
        pinned_buffer = clCreateBuffer(the_context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                       2*SLICE_SIZE, NULL, NULL);

        staging_mem = clEnqueueMapBuffer(queue, pinned_buffer, CL_TRUE, CL_MAP_READ,
                                        0, 2*SLICE_SIZE, 0, NULL, NULL, NULL);

        CL_LOG_CHECK(clEnqueueCopyBuffer(queue, buffer, pinned_buffer, curr_offset, curr_buff*SLICE_SIZE,
                                         SLICE_SIZE, 0, NULL, pending + curr_buff));
        /*
        staging[curr_buff] = clEnqueueMapBuffer(queue, buffer, CL_FALSE, CL_MAP_READ,
                                                curr_offset, SLICE_SIZE, 0, NULL,
                                                pending + curr_buff, NULL);
        */
        do {
            // copy next
            CL_LOG_CHECK(clEnqueueCopyBuffer(queue, buffer, pinned_buffer, next_offset, next_buff*SLICE_SIZE,
                                             std::min(SLICE_SIZE, total_size-next_offset), 0, NULL, pending+next_buff));
            /*
            staging[1-curr_buff] = clEnqueueMapBuffer(queue, buffer, CL_FALSE, CL_MAP_READ,
                                                      next_offset, std::min(SLICE_SIZE, total_size-next_offset),
                                                      0, NULL, pending + (1-curr_buff), NULL);
            */
            // wait for dma for current to finish
            CL_LOG_CHECK(clWaitForEvents(1, pending+curr_buff));

            // memcpy current buffer
            memcpy(byte_index(dst, curr_offset),
                   byte_index(staging_mem, curr_buff*SLICE_SIZE), SLICE_SIZE);

            /*
            // unmap previous buffer
            clEnqueueUnmapMemObject(queue, buffer, staging[curr_buff], 0, NULL, NULL);
            */
            curr_buff = 1-curr_buff;
            next_buff = 1-next_buff;
            curr_offset += SLICE_SIZE;
            next_offset += SLICE_SIZE;
        } while(next_offset < total_size);

        // last buffer needs memcpying
        CL_LOG_CHECK(clWaitForEvents(1, pending+curr_buff));
        memcpy(byte_index(dst, curr_offset),
               byte_index(staging_mem, curr_buff*SLICE_SIZE), total_size-curr_offset);
        //clEnqueueUnmapMemObject(queue, buffer, staging[curr_buff], 0, NULL, NULL);

        CL_LOG_CHECK(clEnqueueUnmapMemObject(queue, pinned_buffer, staging_mem, 0, NULL, NULL));
        CL_LOG_CHECK(clReleaseMemObject(pinned_buffer));
    }
}

template <typename REAL>
inline array_copy_convert_error
copy_convert_from_buffer(cl_command_queue queue, cl_mem buffer,
                         const stream_desc *stream, const size_t *pos,
                         const size_t *sz, size_t ndim)
{
    TIME_SCOPE("copy_convert_from_buffer");
    array_copy_convert_error err = NO_ERROR;
    if (ndim != stream->stream_ndim())
        return DIM_ERROR;

    if (stream->base_type != NPY_FLOAT32 && stream->base_type != NPY_FLOAT64)
        return BASETYPE_ERROR;


    size_t total_ndim = stream->ndim();

    size_t dims[total_ndim];
    const ptrdiff_t *strides = stream->stream_strides();
    void * restrict dst = ndim_index(stream->base, pos, strides,
                                     stream->stream_ndim());

    size_t total_size = 0;
    switch(stream->base_type) {
    case NPY_FLOAT:
        total_size = sizeof(float);
        break;
    case NPY_DOUBLE:
        total_size = sizeof(double);
        break;
    }

    for (size_t i = 0; i < ndim; i++) {
        dims[i] = sz[i];
        total_size *= sz[i];
    }

    for (size_t i = 0; i < stream->element_ndim(); i++) {
        dims[ndim+i] = stream->element_dims()[i];
        total_size *= stream->element_dims()[i];
    }

    // check for trivial layout of the stream...
    bool trivial_layout = true;
    {
        if (numpy_type<REAL>() == stream->base_type)
        {
            // check if strides are just the size of the underlying dimensions
            size_t sz = sizeof(REAL);
             int i;
            for (i = total_ndim-1; i>=0; i--)
            {
                if (strides[i] != sz)
                    break;
                sz *= dims[i];
            }
            // at this point, i should have the dimension upto which the inner
            // dimensions are trivial, and sz the size of those inner dimensions
            // ... this could be used for some optimization.
            trivial_layout = i < 0;
        }
        else
        {
            // bad luck, needs conversion...
            trivial_layout = false;
        }
    }

    if (trivial_layout)
    { /* do this in slices to see how fast we can go */
        void *src;
        sliced_buff_to_mem(queue, buffer, dst, total_size, CLXF_PREFERRED_SLICE_SIZE);
    }
    else
    {
        void *src;
        { TIME_SCOPE("copy_convert_from_buffer: clEnqueueMapBuffer");
            src = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0,
                                     total_size, 0, NULL, NULL, NULL);
        }

        TIME_SCOPE("copy_convert_from_buffer: non-trivial copy-conversion");
        switch(stream->base_type)
        {
        case NPY_FLOAT32:
            copy_convert_chunk_out<float, REAL>(dst, src, dims, strides, total_ndim);
            break;
        case NPY_FLOAT64:
            copy_convert_chunk_out<double, REAL>(dst, src, dims, strides, total_ndim);
            break;
        default:
            err = UNEXPECTED_ERROR;
        }
        clEnqueueUnmapMemObject(queue, buffer, src, 0, NULL, NULL);
    }

    return err;
}

template <typename REAL>
static int
cl_gvec_to_xy(PyArrayObject *gVec_c,
              PyArrayObject *rMat_d, PyArrayObject *rMat_s, PyArrayObject *rMat_c,
              PyArrayObject *tVec_d, PyArrayObject *tVec_s, PyArrayObject *tVec_c,
              PyArrayObject *beam_vec, // may be nullptr
              PyArrayObject *result_xy_array)
{
    TIME_SCOPE("cl_gvec_to_xy");
    auto cl = cl_instance::instance();
    if (!cl)
        return -1;

    cl_context ctxt = cl->context;
    cl_command_queue queue = cl->queue;

    cl_kernel kernel =  cl->get_kernel(gvec_to_xy_kernel<REAL>());
    if (!kernel)
    {
        /* kernel not cached... just build it */
        kernel = cl->build_kernel("gvec_to_xy", g2xy_source,
                                  kernel_compile_options<REAL>());
        if (kernel)
            cl->set_kernel(gvec_to_xy_kernel<REAL>(), kernel);
        else
            return -2;
    }

    gvec_to_xy_params<REAL> params;
    gvec_to_xy_streams streams;
    init_gvec_to_xy(&params, &streams,
                    gVec_c,
                    rMat_d, rMat_s, rMat_c,
                    tVec_d, tVec_s, tVec_c,
                    beam_vec,
                    result_xy_array);

    g2xy_buffs buffs;
    if (init_g2xy_buffs(&buffs, ctxt, &params, &streams))
    {
        size_t chunk_ngvec = static_cast<size_t>(params.chunk_size[0]);
        size_t chunk_npos = static_cast<size_t>(params.chunk_size[1]);
        size_t chunk_size[] = { chunk_ngvec, chunk_npos };
        size_t chunk_offset[] = { 0, 0 };
        /* initial implementation that does all in one go */
        /* 1. fill input buffers */
        /* TODO: copy convert should go here... now just raw copy */
        { TIME_SCOPE("cl_gvec_to_xy - copy args");
            copy_convert_to_buffer<REAL>(queue, buffs.gvec_c,
                                         &streams.gVec_c_stream,
                                         chunk_offset, &chunk_ngvec, 1);
            copy_convert_to_buffer<REAL>(queue, buffs.rmat_s,
                                         &streams.rMat_s_stream,
                                         chunk_offset, &chunk_ngvec, 1);
            copy_convert_to_buffer<REAL>(queue, buffs.tvec_c,
                                         &streams.tVec_c_stream,
                                         chunk_offset+1, &chunk_npos, 1);

            clFinish(queue);
        }
        /* 2. prepare and enqueue the kernel */
        { TIME_SCOPE("cl_gvec_to_xy - execute kernel");
            CL_LOG_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffs.params));
            CL_LOG_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffs.gvec_c));
            CL_LOG_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffs.rmat_s));
            CL_LOG_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffs.tvec_c));
            CL_LOG_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffs.xy_result));
            const size_t WS_NGVEC = 16;
            const size_t WS_NPOS = 16;

            size_t local_work_size[] = { WS_NGVEC, WS_NPOS };
            size_t total_work_size[] = {
                next_multiple(chunk_ngvec, WS_NGVEC),
                next_multiple(chunk_npos, WS_NPOS)
            };

            clEnqueueNDRangeKernel(queue, kernel, 2, NULL, total_work_size,
                                   local_work_size, 0, NULL, NULL);
            clFinish(queue);
        }
        /* 3. wait and copy results */
        { TIME_SCOPE("cl_gvec_to_xy - copy results");
            copy_convert_from_buffer<REAL>(queue, buffs.xy_result,
                                           &streams.xy_out_stream,
                                           chunk_offset, chunk_size, 2);
            clFinish(queue);
        }

        release_g2xy_buffs(&buffs);
    }

    //print_gvec_to_xy(&params, &streams);
    return 0;
}

XRD_PYTHON_WRAPPER PyObject *
python_cl_gvec_to_xy(PyObject *self, PyObject *args, PyObject *kwargs)
{
    TIME_SCOPE("python wrapper");
    static const char *kwlist[] = {"gvec_c", "rmat_d", "rmat_s", "rmat_c", "tvec_d",
        "tvec_s", "tvec_c", "beam_vec", "single_precision", nullptr};
    const char* parse_tuple_fmt = "O&O&O&O&O&O&O&|O&p";

    named_array na_gVec_c = {kwlist[0], na_vector3, na_0d_or_1d, nullptr};
    named_array na_rMat_d = {kwlist[1], na_matrix33, na_0d_only, nullptr};
    named_array na_rMat_s = {kwlist[2], na_matrix33, na_0d_or_1d, nullptr};
    named_array na_rMat_c = {kwlist[3], na_matrix33, na_0d_only, nullptr};
    named_array na_tVec_d = {kwlist[4], na_vector3, na_0d_only, nullptr};
    named_array na_tVec_s = {kwlist[5], na_vector3, na_0d_only, nullptr};
    named_array na_tVec_c = {kwlist[6], na_vector3, na_0d_or_1d, nullptr};
    named_array na_beam_vec = {kwlist[7], na_vector3, na_0d_or_none, nullptr};
    int use_single_precision = 0;
    PyArrayObject *result_array = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, parse_tuple_fmt, (char**)kwlist,
                                     named_array_converter, &na_gVec_c,
                                     named_array_converter, &na_rMat_d,
                                     named_array_converter, &na_rMat_s,
                                     named_array_converter, &na_rMat_c,
                                     named_array_converter, &na_tVec_d,
                                     named_array_converter, &na_tVec_s,
                                     named_array_converter, &na_tVec_c,
                                     named_array_converter, &na_beam_vec,
                                     &use_single_precision))
        goto fail;

    {
        // At this point, the arguments have been extracted and some minimal error
        // checking has been done (the array dimensions are sound). As argument
        // checks are done in an array by array basis, constraints involving
        // multiple arrays need still be done (like gVec_c and rMat_s having the
        // same number of items if both are streams...)
        auto len_gVec_c = is_streaming_vector3(na_gVec_c.pyarray)?
        PyArray_DIM(na_gVec_c.pyarray, 0) : 0;
        auto len_rMat_s = is_streaming_matrix33(na_rMat_s.pyarray)?
            PyArray_DIM(na_rMat_s.pyarray, 0) : 0;
        auto len_tVec_c = is_streaming_vector3(na_tVec_c.pyarray)?
            PyArray_DIM(na_tVec_c.pyarray, 0) : 0;
        auto gVec_count = std::max(len_gVec_c, len_rMat_s);
        auto tVec_count = len_tVec_c;

        // Note subtle behavior: len_* variables will only be 0 when there is no
        // streaming dimension at all. If there is a explicit extra dimension with
        // 1 element, the len_* variable will be 1.
        //
        // If there is the extra dimension, the constraint of having the same
        // dimensions for gVec_c and tVec_c will be enforced.
        if (len_gVec_c != 0 && len_rMat_s != 0 && len_gVec_c != len_rMat_s)
        {
            // incompatible lengths for gVec_c and rMat_s
            PyErr_Format(PyExc_ValueError, "'%s' and '%s' outer dimension mismatch.",
                         na_gVec_c.name, na_rMat_s.name);
            goto fail;
        }

        // At this point, allocate the result buffer. The problem will be of size
        // gVec_count x tVec_count. The result buffer will be of shape (gVec_count,
        // tVec_count, 2). The actual shape will depend on the streaming of the
        // other dimensions, as if one of the dimensions is not 'streamed', it won't
        // appear in the actual shape of the result.
        npy_intp dims[3];
        int dim_count = 0;
        int type = use_single_precision?NPY_FLOAT32:NPY_FLOAT64;
        int layout = 0; // standard (C) layout.

        // outer dim will be gVec_count... if zero, omit
        if (gVec_count)
            dims[dim_count++] = gVec_count;

        // second dim will be tVec_count... if zero, omit
        if (tVec_count)
            dims[dim_count++] = tVec_count;

        // inner dim is 2 ((x,y) coordinates).
        dims[dim_count++] = 2;
        result_array = (PyArrayObject*)PyArray_EMPTY(dim_count, dims, type, layout);

        if (!result_array)
        {
            PyErr_Format(PyExc_RuntimeError, "Internal error allocating result array (%s::%d)",
                         __FILE__, __LINE__);
            goto fail;
        }
    }

    {
        auto func = use_single_precision?cl_gvec_to_xy<float>:cl_gvec_to_xy<double>;
        int err = func(na_gVec_c.pyarray,
                       na_rMat_d.pyarray, na_rMat_s.pyarray, na_rMat_c.pyarray,
                       na_tVec_d.pyarray, na_tVec_s.pyarray, na_tVec_c.pyarray,
                       na_beam_vec.pyarray,
                       result_array);

        if (err)
        {
            PyErr_Format(PyExc_RuntimeError, "Failed to run the kernel. OCL error?");
            goto fail;
        }
    }

    return reinterpret_cast<PyObject*>(result_array);
 fail:
    Py_XDECREF((PyObject*) result_array);

    return NULL;
}
