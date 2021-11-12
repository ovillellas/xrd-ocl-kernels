#include "checks.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>

// use CL_LOG_LEVEL to set the log level of CL code with CL_LOG_CHECK
// CL_LOG_LEVEL 0 - no logging at all.
// CL_LOG_LEVEL 1 - log when is not a CL_SUCCESS
// CL_LOG_LEVEL 2 - always log
#define CL_LOG_LEVEL 0

#define TRACE() \
    do { printf("%s::%d - %s\n", __FILE__, __LINE__, __FUNCTION__); } while(0)

#if 0 == CL_LOG_LEVEL
#  define CL_LOG_CHECK(code) \
    do { \
        code; \
    } while(0)
#elif 1 == CL_LOG_LEVEL
#  define CL_LOG_CHECK(code) \
    do { \
        cl_int cl_log_check_err = code; \
        if (cl_log_check_err != CL_SUCCESS) \ 
            printf("%s - %s\n", cl_error_to_str(cl_log_check_err), #code); \
    } while(0)
#else
#  define CL_LOG_CHECK(code) \
    do { \
        cl_int cl_log_check_err = code; \
        printf("%s - %s\n", cl_error_to_str(cl_log_check_err), #code); \
    } while(0)
#endif

#define CLERR_TO_STR(x) case x: return #x
static
const char *cl_error_to_str(int cl_error)
{
    switch(cl_error)
    {
    CLERR_TO_STR(CL_SUCCESS);

    CLERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
    CLERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
    CLERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
    CLERR_TO_STR(CL_DEVICE_NOT_FOUND);
    CLERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
    CLERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
    CLERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    CLERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
    CLERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    CLERR_TO_STR(CL_INVALID_ARG_INDEX);
    CLERR_TO_STR(CL_INVALID_ARG_SIZE);
    CLERR_TO_STR(CL_INVALID_ARG_VALUE);
    CLERR_TO_STR(CL_INVALID_BINARY);
    CLERR_TO_STR(CL_INVALID_BUFFER_SIZE);
    CLERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
    CLERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
    CLERR_TO_STR(CL_INVALID_CONTEXT);
    CLERR_TO_STR(CL_INVALID_DEVICE);
    CLERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);
    CLERR_TO_STR(CL_INVALID_DEVICE_TYPE);
    CLERR_TO_STR(CL_INVALID_EVENT);
    CLERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
    CLERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
    CLERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
    CLERR_TO_STR(CL_INVALID_HOST_PTR);
    CLERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
    CLERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    CLERR_TO_STR(CL_INVALID_IMAGE_SIZE);
    CLERR_TO_STR(CL_INVALID_KERNEL);
    CLERR_TO_STR(CL_INVALID_KERNEL_ARGS);
    CLERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
    CLERR_TO_STR(CL_INVALID_KERNEL_NAME);
    CLERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
    CLERR_TO_STR(CL_INVALID_MEM_OBJECT);
    CLERR_TO_STR(CL_INVALID_OPERATION);
    CLERR_TO_STR(CL_INVALID_PLATFORM);
    CLERR_TO_STR(CL_INVALID_PROGRAM);
    CLERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
    CLERR_TO_STR(CL_INVALID_PROPERTY);
    CLERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
    CLERR_TO_STR(CL_INVALID_SAMPLER);
    CLERR_TO_STR(CL_INVALID_VALUE);
    CLERR_TO_STR(CL_INVALID_WORK_DIMENSION);
    CLERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
    CLERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
    CLERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
    CLERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
    CLERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
    CLERR_TO_STR(CL_MAP_FAILURE);
    CLERR_TO_STR(CL_MEM_COPY_OVERLAP);
    CLERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    CLERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    CLERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
    CLERR_TO_STR(CL_OUT_OF_RESOURCES);
    CLERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
    default:
        return "Unknown error code";
    }
}
#undef CLERR_TO_STR

// this describes the problem size for this problem. Note that the actual total
// problem size is inferred from the input arrays. This also describes the chunk
// size that will be used in multiprocessing, as compute unit will get
// subproblems to work with.
struct problem_size_t
{
    cl_ulong ngvec;
    cl_ulong npos;
};

// parameters for use in gvec_to_xy. Templatized to support both, float and
// double versions.
template <typename REAL>
struct gvec_to_xy_params {
    REAL gVec_c[3];
    REAL rMat_d[9];
    REAL rMat_s[9];
    REAL rMat_c[9];
    REAL tVec_d[3];
    REAL tVec_s[3];
    REAL tVec_c[3];
    problem_size_t chunk_size;
    problem_size_t total_size;
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

    array_stream_convert(&streams->xy_out_stream, pa_result_xy, 1, 2);

    size_t ngvec = streams->xy_out_stream.stream_dims()[0];
    size_t npos = streams->xy_out_stream.stream_dims()[1];
    
    params->chunk_size = {static_cast<cl_ulong>(ngvec), static_cast<cl_ulong>(npos)};
    params->total_size = {static_cast<cl_ulong>(ngvec), static_cast<cl_ulong>(npos)};

    return 0;
}

template <typename REAL>
static void
print_gvec_to_xy(const gvec_to_xy_params<REAL> *params,
                 const gvec_to_xy_streams *streams)
{
    printf("gvec_to_xy_params\nkind: %s\n", floating_kind_name<REAL>());
    print_vector3("gVec_c", params->gVec_c);
    print_matrix33("rMat_d", params->rMat_d);
    print_matrix33("rMat_s", params->rMat_s);
    print_matrix33("rMat_c", params->rMat_c);
    print_vector3("tVec_d", params->tVec_d);
    print_vector3("tVec_s", params->tVec_s);
    print_vector3("tVec_c", params->tVec_c);

    print_stream("gVec_c", streams->gVec_c_stream);
    print_stream("rMat_s", streams->rMat_s_stream);
    print_stream("tVec_c", streams->tVec_c_stream);
    print_stream("xy_out", streams->xy_out_stream);
}

struct cl_state
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_gvec_to_xy;

    cl_mem params_buf;  // the problem "params"
    cl_mem gvec_c_buf;  // gpu buffer for gvectors
    cl_mem rmat_s_buf;  // gpu buffer for rmat_s
    cl_mem tvec_c_buf;  // gpu buffer for tvec_c
    cl_mem xy_result_buf;  // gpu buffer for xy_result
};

static void
release_cl(cl_state *state)
{
    clReleaseMemObject(state->xy_result_buf);
    clReleaseMemObject(state->tvec_c_buf);
    clReleaseMemObject(state->rmat_s_buf);
    clReleaseMemObject(state->gvec_c_buf);
    clReleaseMemObject(state->params_buf);
    clReleaseKernel(state->kernel_gvec_to_xy);
    clReleaseProgram(state->program);
    clReleaseCommandQueue(state->queue);
    clReleaseContext(state->context);

    zap_to_zero(*state);
}

template<typename REAL>
cl_kernel gvec_to_xy_kernel(const cl_state *state)
{
    return (cl_kernel)NULL;
}

template<>
cl_kernel gvec_to_xy_kernel<float>(const cl_state *state)
{
    return state->kernel_gvec_to_xy;
}

template<>
cl_kernel gvec_to_xy_kernel<double>(const cl_state *state)
{
    return state->kernel_gvec_to_xy;
}

// C++11 raw string literals to the rescue... will make cl programs legible.
static const char *cl_source = R"CLC(
#if defined(USE_SINGLE_PRECISION) && USE_SINGLE_PRECISION
#  define REAL float
#else
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#  define REAL double
#endif

struct problem_size_t
{
    unsigned long ngvec;
    unsigned long npos;
};

struct gvec_to_xy_params {
    REAL gVec_c[3];
    REAL rMat_d[9];
    REAL rMat_s[9];
    REAL rMat_c[9];
    REAL tVec_d[3];
    REAL tVec_s[3];
    REAL tVec_c[3];
    struct problem_size_t chunk_size;
    struct problem_size_t total_size;
};


void v3s_copy(const REAL *src, ptrdiff_t stride, REAL * restrict dst)
{
    dst[0] = src[0];
    dst[1] = src[stride];
    dst[2] = src[2*stride];
}

void v3_v3s_add(const REAL *src1, const REAL *src2, ptrdiff_t stride,
                REAL * restrict dst)
{
    dst[0] = src1[0] + src2[0];
    dst[1] = src1[1] + src2[1*stride];
    dst[2] = src1[2] + src2[2*stride];
}

void v3_v3s_sub(const REAL *src1, const REAL *src2, ptrdiff_t stride,
                REAL * restrict dst)
{
    dst[0] = src1[0] - src2[0];
    dst[1] = src1[1] - src2[1*stride];
    dst[2] = src1[2] - src2[2*stride];
}

REAL v3_v3s_dot(const REAL *v1,
                const REAL *v2, ptrdiff_t stride)
{
    return v1[0]*v2[0] + v1[1]*v2[stride] + v1[2]*v2[2*stride];
}


void m33_v3s_multiply(const REAL *m, const REAL *v, ptrdiff_t stride,
                      REAL * restrict dst)
{
    dst[0] = m[0]*v[0] + m[3]*v[stride] + m[6]*v[2*stride];
    dst[1] = m[1]*v[0] + m[4]*v[stride] + m[7]*v[2*stride];
    dst[2] = m[2]*v[0] + m[5]*v[stride] + m[8]*v[2*stride];
}


void v3s_s_v3_muladd(const REAL *v1, ptrdiff_t stride, REAL factor,
                     const REAL *v2, REAL * restrict result)
{
    /* result = v1*factor + v2 */
    result[0] = factor*v1[0] + v2[0];
    result[1] = factor*v1[1*stride] + v2[1];
    result[2] = factor*v1[2*stride] + v2[2];
}


void diffract_z(const REAL *gvec, REAL * restrict diffracted)
{
    diffracted[0] = 2*gvec[0]*gvec[2];
    diffracted[1] = 2*gvec[1]*gvec[2];
    diffracted[2] = 2*gvec[2]*gvec[2] - 1.0;
}


int ray_plane_intersect(const REAL *origin, const REAL *vect,
                        const REAL *plane, REAL * restrict collision_point)
{
    double t;
    t = (plane[3] - v3_v3s_dot(plane, origin, 1)) / v3_v3s_dot(plane, vect, 1);
    if (t < 0.0)
        return 0;
    v3s_s_v3_muladd(vect, 1, t, origin, collision_point); 
    return 1;
}


void gvec_to_xy_single(const REAL *gVec_c,
    const REAL *rMat_d, const REAL *rMat_s, const REAL *rMat_c,
    const REAL *tVec_d, const REAL *tVec_s, const REAL *tVec_c,
    REAL * restrict xy_result)
{
    REAL plane[4], tVec_sc[3], tVec_ds[3], gVec_s[3], gVec_l[3];
    REAL ray_origin[3], ray_vector[3], point[3];
    REAL rMat_sc[9];

    v3s_copy(rMat_d + 2, 3, plane);
    plane[3] = 0.0;

    v3_v3s_sub(tVec_s, tVec_d, 1, tVec_ds);

    m33_v3s_multiply(rMat_s, tVec_c, 1, tVec_sc);
    v3_v3s_add(tVec_ds, tVec_sc, 1, ray_origin);

    m33_v3s_multiply(rMat_c, gVec_c, 1, gVec_s);
    m33_v3s_multiply(rMat_s, gVec_s, 1, gVec_l);

    diffract_z(gVec_l, ray_vector);

    if (0 != ray_plane_intersect(ray_origin, ray_vector, plane, point))
    {
        xy_result[0] = v3_v3s_dot(rMat_d, point, 1);
        xy_result[1] = v3_v3s_dot(rMat_d + 3, point, 1); 
    }
    else
    {
        xy_result[0] = NAN;
        xy_result[1] = NAN;
    }
}

void g2l(REAL *dst, __global const REAL *src, size_t sz)
{
    for (size_t i=0; i<sz; i++) dst[i] = src[i];
}

void l2g(__global REAL *dst, const REAL *src, size_t sz)
{
    for (size_t i=0; i<sz; i++) dst[i] = src[i];
}

__kernel void gvec_to_xy(
    __global const struct gvec_to_xy_params *g_params,
    __global const REAL *g_gVec_c,
    __global const REAL *g_rMat_s,
    __global const REAL *g_tVec_c,
    __global REAL * restrict g_xy_result)
{
   REAL gVec_c[3], rMat_d[9], rMat_s[9], rMat_c[9], tVec_d[3], tVec_s[3], tVec_c[3];
   REAL xy_result[2];
   size_t gvec_idx, npos_idx, xy_result_offset;
   gvec_idx = get_global_id(0);
   npos_idx = get_global_id(1);

   g2l(gVec_c, g_gVec_c + 3*gvec_idx, 3);
   g2l(rMat_d, g_params->rMat_d, 9);
   g2l(rMat_s, g_params->rMat_s + 9*gvec_idx, 9);
   g2l(rMat_c, g_params->rMat_c, 9);
   g2l(tVec_d, g_params->tVec_d, 3);
   g2l(tVec_s, g_params->tVec_s, 3);
   g2l(tVec_c, g_tVec_c + 3*npos_idx, 3);

   /* first, naive implementation that just does one evaluation per item */
   gvec_to_xy_single(gVec_c, rMat_d, rMat_s, rMat_c, tVec_d, tVec_s, tVec_c,
                     xy_result);   

   xy_result_offset = 2*(g_params->total_size.npos*gvec_idx + npos_idx);
   l2g(g_xy_result + xy_result_offset, xy_result, 2);
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

static void
cl_error_notify(const char *errinfo, const void* private_info, size_t cb, void *user_data)
{
    if (0 == strncmp(errinfo, "OpenCL Build Warning", 20))
        return; // ignore compile warnings
    printf("CLERROR: %s\n", errinfo);
}

template<typename REAL>
static bool
init_cl(cl_state *return_state,
        const gvec_to_xy_params<REAL> *params,
        const gvec_to_xy_streams *streams)
{
    cl_state state;
    cl_platform_id platform;
    cl_device_id device;
    zap_to_zero(state);
    zap_to_zero(*return_state);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    // this is for testing... there should be a config place to set this up
    // (and associated context).
    clGetPlatformIDs(1, &platform, NULL); // the first platform
    //clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL); // first any device
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // first GPU
    //clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL); // first CPU

    //printf("platform: %p device: %p\n", platform, device);
    state.platform = platform;
    state.device = device;
    state.context = clCreateContext(0, 1, &device, cl_error_notify, NULL, NULL);
    state.queue = clCreateCommandQueue(state.context, device, 0, NULL);
    state.program = clCreateProgramWithSource(state.context, 1, &cl_source,
                                              NULL, NULL);

    auto err = clBuildProgram(state.program, 0, NULL,
                              kernel_compile_options<REAL>(), NULL, NULL);
    if (CL_SUCCESS != err) {
        if (CL_BUILD_PROGRAM_FAILURE == err)
        {
            size_t log_size;
            clGetProgramBuildInfo(state.program, state.device,
                                  CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char*)malloc(log_size);
            if (NULL != log)
            {
                clGetProgramBuildInfo(state.program, state.device,
                                      CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                printf("Build log:\n%s\n", log);
                free(log);
            }
            else
                printf("no memory to show log.\n");
        }
    }
    
    state.kernel_gvec_to_xy = clCreateKernel(state.program, "gvec_to_xy", NULL);
    
    // params buffer, remains constant for the whole execution.
    {
        const cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                        CL_MEM_COPY_HOST_PTR |
                                        CL_MEM_HOST_NO_ACCESS;

        state.params_buf = clCreateBuffer(state.context, buff_flags,
                                          sizeof(*params), &params, NULL);
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
        size_t buff_size = params->chunk_size.ngvec * 3 * sizeof(REAL);
        const cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                        CL_MEM_HOST_WRITE_ONLY;
        state.gvec_c_buf = clCreateBuffer(state.context, buff_flags, buff_size,
                                          NULL, NULL);
    }
    else
        goto fail; // not supported yet.

    if (streams->rMat_s_stream.is_active())
    {
        size_t buff_size = params->chunk_size.ngvec * 9 * sizeof(REAL);
        cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                  CL_MEM_HOST_WRITE_ONLY;
        state.rmat_s_buf = clCreateBuffer(state.context, buff_flags, buff_size,
                                          NULL, NULL);
    }
    else
        goto fail; // not supported yet.

    if (streams->tVec_c_stream.is_active())
    {
        size_t buff_size = params->chunk_size.npos * 3 * sizeof(REAL);
        cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                  CL_MEM_HOST_WRITE_ONLY;
        state.tvec_c_buf = clCreateBuffer(state.context, buff_flags, buff_size,
                                          NULL, NULL);
    }
    else
        goto fail;

    if (streams->xy_out_stream.is_active())
    {
        size_t buff_size = params->chunk_size.npos *
                           params->chunk_size.ngvec * 2 * sizeof(REAL);
        cl_mem_flags buff_flags = CL_MEM_WRITE_ONLY |
                                  CL_MEM_HOST_READ_ONLY;
        state.xy_result_buf = clCreateBuffer(state.context, buff_flags, buff_size,
                                             NULL, NULL);
    }
    else
        goto fail;

    *return_state = state;
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        printf("init_cl took %d ms.\n", duration_cast<std::chrono::milliseconds>(t1-t0).count());
    }
    return true;

 fail:
    release_cl(&state);
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
    print_dims("chunk dimensions", dims, ndim);
    printf("Chunk in: %zd %s written (%s original)\n", count,
           type_to_cstr<DST_REAL>(), type_to_cstr<SRC_REAL>());
    
}


template<typename DST_REAL, typename SRC_REAL>
static void
copy_convert_chunk_out(void * restrict dst, const void *src,
                       const size_t *dims, const ptrdiff_t *strides,
                       size_t ndim)
{
    //printf("Chunk out: %s -> %s.\n", type_to_cstr<SRC_REAL>(), type_to_cstr<DST_REAL>());
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
    print_dims("chunk dimensions", dims, ndim);
    printf("Chunk out: %zd %s written (%s original)", count,
           type_to_cstr<DST_REAL>(), type_to_cstr<SRC_REAL>());
    
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

template <typename REAL>
inline array_copy_convert_error
copy_convert_from_buffer(cl_command_queue queue, cl_mem buffer,
                         const stream_desc *stream, const size_t *pos,
                         const size_t *sz, size_t ndim)
{
    //print_stream("to", *stream);
    array_copy_convert_error err = NO_ERROR;
    if (ndim != stream->stream_ndim())
        return DIM_ERROR;

    if (stream->base_type != NPY_FLOAT32 && stream->base_type != NPY_FLOAT64)
        return BASETYPE_ERROR;

    size_t total_ndim = stream->ndim();

    size_t dims[total_ndim];
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

    const ptrdiff_t *strides = stream->stream_strides();
    void * restrict dst = ndim_index(stream->base, pos, strides,
                                     stream->stream_ndim());
    //printf("q: %p b: %p - READ - sz: %zd.\n", queue, buffer, total_size);
    const void *src = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0,
                                         total_size, 0, NULL, NULL, NULL);
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
    clEnqueueUnmapMemObject(queue, buffer, dst, 0, NULL, NULL);

    return err;
}
template <typename REAL>
static int
cl_gvec_to_xy(PyArrayObject *gVec_c,
              PyArrayObject *rMat_d, PyArrayObject *rMat_s, PyArrayObject *rMat_c,
              PyArrayObject *tVec_d, PyArrayObject *tVec_s, PyArrayObject *tVec_c,
              PyArrayObject *result_xy_array)
{
    gvec_to_xy_params<REAL> params;
    gvec_to_xy_streams streams;
    init_gvec_to_xy(&params, &streams,
                    gVec_c,
                    rMat_d, rMat_s, rMat_c,
                    tVec_d, tVec_s, tVec_c,
                    result_xy_array);

    cl_state cl;
    if (init_cl(&cl, &params, &streams))
    {
        size_t chunk_ngvec = static_cast<size_t>(params.chunk_size.ngvec);
        size_t chunk_npos = static_cast<size_t>(params.chunk_size.npos);
        size_t chunk_size[] = { chunk_ngvec, chunk_npos };
        size_t chunk_offset[] = {0, 0};
                                      
        cl_kernel kern = gvec_to_xy_kernel<REAL>(&cl);
        /* initial implementation that does all in one go */
        /* 1. fill input buffers */
        /* TODO: copy convert should go here... now just raw copy */
        auto t0 = std::chrono::high_resolution_clock::now();
        copy_convert_to_buffer<REAL>(cl.queue, cl.gvec_c_buf,
                                     &streams.gVec_c_stream,
                                     chunk_offset, chunk_size, 1);
        copy_convert_to_buffer<REAL>(cl.queue, cl.rmat_s_buf,
                                     &streams.rMat_s_stream,
                                     chunk_offset, chunk_size, 1);
        copy_convert_to_buffer<REAL>(cl.queue, cl.tvec_c_buf,
                                     &streams.tVec_c_stream,
                                     chunk_offset+1, chunk_size+1, 1);

        clFinish(cl.queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        /*                       
        raw_copy_to_buffer(cl.queue, cl.gvec_c_buf, streams.gVec_c_stream.base,
                           3*sizeof(REAL)*chunk_ngvec);
        raw_copy_to_buffer(cl.queue, cl.rmat_s_buf, streams.rMat_s_stream.base,
                           9*sizeof(REAL)*chunk_ngvec);
        raw_copy_to_buffer(cl.queue, cl.tvec_c_buf, streams.tVec_c_stream.base,
                           3*sizeof(REAL)*chunk_npos);
        */
        
        /* 2. prepare and enqueue the kernel */
        CL_LOG_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &cl.params_buf));
        CL_LOG_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &cl.gvec_c_buf));
        CL_LOG_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &cl.rmat_s_buf));
        CL_LOG_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &cl.tvec_c_buf));
        CL_LOG_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &cl.xy_result_buf));

        size_t local_work_size[] = { 1, 8 };

        auto t2 = std::chrono::high_resolution_clock::now();
        clEnqueueNDRangeKernel(cl.queue, kern, 2, NULL, chunk_size,
                               local_work_size, 0, NULL, NULL);
        clFinish(cl.queue);
        auto t3 = std::chrono::high_resolution_clock::now();

        printf("transfer: %d ms.\ncompute: %d ms.\n",
               duration_cast<std::chrono::milliseconds>(t1-t0).count(),
               duration_cast<std::chrono::milliseconds>(t3-t2).count());

        /* 3. wait and copy results */
        /*
        copy_convert_from_buffer<REAL>(cl.queue, cl.xy_result_buf,
                                       &streams.xy_out_stream,
                                       chunk_offset, chunk_size, 2);
        */
        /*                         
        raw_copy_from_buffer(cl.queue, cl.xy_result_buf, streams.xy_out_stream.base,
                             chunk_ngvec*chunk_npos*2*sizeof(REAL));
        */
        release_cl(&cl);
    }

    //print_gvec_to_xy(&params, &streams);
    return 0;
}

XRD_PYTHON_WRAPPER PyObject *
python_cl_gvec_to_xy(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {"gVec_c", "rMat_d", "rMat_s", "rMat_c", "tVec_d",
        "tVec_s", "tVec_c", "single_precision", NULL};
    const char* parse_tuple_fmt = "O&O&O&O&O&O&O&|p";
        
    named_array na_gVec_c = {"gVec_c", na_vector3, na_0d_or_1d, NULL};
    named_array na_rMat_d = {"rMat_d", na_matrix33, na_0d_only, NULL};
    named_array na_rMat_s = {"rMat_s", na_matrix33, na_0d_or_1d, NULL};
    named_array na_rMat_c = {"rMat_c", na_matrix33, na_0d_only, NULL};
    named_array na_tVec_d = {"tVec_d", na_vector3, na_0d_only, NULL};
    named_array na_tVec_s = {"tVec_s", na_vector3, na_0d_only, NULL};
    named_array na_tVec_c = {"tVec_c", na_vector3, na_0d_or_1d, NULL};
    int use_single_precision = 0;
    PyArrayObject *result_array = NULL;
    PyObject *ret_val = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, parse_tuple_fmt, (char**)kwlist,
                                     named_array_converter, &na_gVec_c,
                                     named_array_converter, &na_rMat_d,
                                     named_array_converter, &na_rMat_s,
                                     named_array_converter, &na_rMat_c,
                                     named_array_converter, &na_tVec_d,
                                     named_array_converter, &na_tVec_s,
                                     named_array_converter, &na_tVec_c,
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
    
    if (use_single_precision)
    {
        cl_gvec_to_xy<float>(na_gVec_c.pyarray,
                             na_rMat_d.pyarray, na_rMat_s.pyarray, na_rMat_c.pyarray,
                             na_tVec_d.pyarray, na_tVec_s.pyarray, na_tVec_c.pyarray,
                             result_array);
    }
    else
    {
        cl_gvec_to_xy<double>(na_gVec_c.pyarray,
                              na_rMat_d.pyarray, na_rMat_s.pyarray, na_rMat_c.pyarray,
                              na_tVec_d.pyarray, na_tVec_s.pyarray, na_tVec_c.pyarray,
                              result_array);        
    }
    
    /* for now, just return the arguments to check that everything is working... */
    /*ret_val = PyTuple_New(8);
    PyTuple_SET_ITEM(ret_val, 0, (PyObject*)na_gVec_c.pyarray); Py_XINCREF((PyObject*)na_gVec_c.pyarray);
    PyTuple_SET_ITEM(ret_val, 1, (PyObject*)na_rMat_d.pyarray); Py_XINCREF((PyObject*)na_rMat_d.pyarray);
    PyTuple_SET_ITEM(ret_val, 2, (PyObject*)na_rMat_s.pyarray); Py_XINCREF((PyObject*)na_rMat_s.pyarray);
    PyTuple_SET_ITEM(ret_val, 3, (PyObject*)na_rMat_c.pyarray); Py_XINCREF((PyObject*)na_rMat_c.pyarray);
    PyTuple_SET_ITEM(ret_val, 4, (PyObject*)na_tVec_d.pyarray); Py_XINCREF((PyObject*)na_tVec_d.pyarray);
    PyTuple_SET_ITEM(ret_val, 5, (PyObject*)na_tVec_s.pyarray); Py_XINCREF((PyObject*)na_tVec_s.pyarray);
    PyTuple_SET_ITEM(ret_val, 6, (PyObject*)na_tVec_c.pyarray); Py_XINCREF((PyObject*)na_tVec_c.pyarray);
    PyTuple_SET_ITEM(ret_val, 7, PyBool_FromLong(use_single_precision)); */
    return reinterpret_cast<PyObject*>(result_array);
 fail:
    Py_XDECREF((PyObject*) result_array);
    
    return NULL;
}
