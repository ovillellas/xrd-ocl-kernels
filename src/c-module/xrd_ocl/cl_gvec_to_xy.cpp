#include "checks.hpp"
#include "utils.hpp"
#include "dev_help.hpp"

#include <algorithm> // std::max, std::min

#define CLXF_PREFERRED_SLICE_SIZE ((size_t)4*1024*1024)

// Maximum result size. But will constraint also to 1/4 of device memory in
// any case.
#define CLXF_MAX_RESULT_SIZE ((size_t)512*1024*1024)

// maximum number of npos per thread. In a workgroup X will be handled.
#define CLXF_MAX_NPOS_PER_THREAD ((size_t)128)
#define CLXF_MAX_NGVEC_PER_CHUNK ((size_t)512)

/*
 * When executing gvec_to_xy as an OpenCL kernel, there are several sizes to
 * deal with:
 *
 * 1. total_size: this is the size of the dataset. That is, the parameters
 *    received from the entry function. That is, a number for G vectors and
 *    candidate positions to be projected into the detector.
 *
 * 2. chunk_size: As evaluation takes place in a device with potentially limited
 *    memory size, total_size will be split in 'chunks'. All chunks are to be
 *    processed one after another, each one resulting into an EnqueueNDRangeKernel
 *    call. Results are moved into the result array in host memory.
 *    If done properly, execution of one chunk may be overlappef with memory
 *    copying from another chunk.
 *
 * 3. workgroup_chunk_size: In OpenCL, kernels are executed in workgroups.
 *    A workgroup is a set of threads executing in lockstep. Instead of limiting
 *    each thread to a single (gvec, npos) pair, work is split in a workgroup
 *    basis. Each thread knows what part of its workgroup dataset evaluate.
 *
 * 4. kernel_local_size: This is how the kernel splits the chunk_size between
 *    the different compute units. It is basically how it assigns blocks to
 *    different workgroups.
 */

struct ocl_g2xy_sizes
{
    size_t total_size[2];
    size_t chunk_size[2];
    size_t workgroup_chunk_size[2];
    size_t kernel_local_size[2];
};

inline void
compute_ocl_g2xy_sizes(ocl_g2xy_sizes& sizes, size_t ngvec, size_t npos,
                       size_t device_memory,
                       size_t workgroup_preferred_size,
                       size_t compute_units,
                       size_t base_value_size)
{
    /* criteria for the chunk_size:
       The idea is having each compute unit working on the different "npos" of
       an ngvec (this is due how the kernel executes).
       So in an ideal world, the chunk size would have into account:
       1. Total size of the results should be less than max result size.
       2. its ngvecs is some multiple of the compute units.
       3. its npos is as big as possible, but taking into account the previous
          point.
     */

    // target_result_count is the target count of elements for the chunk. This
    // is made so that it does not exceed the configured memory size nor 1/8th
    // of the device memory.
    const size_t target_result_count =
        std::min(CLXF_MAX_RESULT_SIZE, device_memory/8)/(2*base_value_size);


    // try the whole dataset, but respecting the configured max sizes per chunk
    size_t max_npos_per_chunk = CLXF_MAX_NPOS_PER_THREAD*workgroup_preferred_size;
    size_t chunk_ngvec = std::min(ngvec, (size_t)CLXF_MAX_NGVEC_PER_CHUNK);
    size_t chunk_npos = std::min(npos, max_npos_per_chunk);
    chunk_npos = next_multiple(chunk_npos, workgroup_preferred_size);

    if (chunk_ngvec * chunk_npos > target_result_count)
    {
        // does not fit in the target result count...

        // split by ngvec (kernel happens to deal well with many npos. As each
        // workgroup works on a single gvec, Try to balance the number of gvecs
        // per chunk with the number of compute units, so all compute unit has
        // data to work with.
        chunk_ngvec = target_result_count/chunk_npos;
        chunk_ngvec = (chunk_ngvec/compute_units)*compute_units;
        chunk_ngvec = std::max(chunk_ngvec, std::min(ngvec, compute_units));

        if (chunk_ngvec*chunk_npos > target_result_count)
        {
            // still does not fit in the target result count...

            // split by npos as well. Target a number of npos by chunk so that
            // all threads in a workgroup will have the same amount of work.
            chunk_npos = target_result_count/chunk_ngvec;
            chunk_npos = (chunk_ngvec/workgroup_preferred_size)*workgroup_preferred_size;
            chunk_npos = std::max(chunk_npos, std::min(npos, workgroup_preferred_size));

            if (chunk_ngvec * chunk_npos > target_result_count)
            {
                // still does not fit in the budget...

                // This is a last resort. Being here means that the memory budget
                // is very low. Go for the safest, although it would be very
                // slow. Execute element by element.
                chunk_npos = 1; chunk_ngvec = 1;
            }
        }
    }

    /*
       in general, workgroup_chunk should be a multiple of kernel_local to
       take maximum advantage of all computing threads. The kernel_local_npos
       is configured to split the npos along the workgroup, using the preferred
       size.

       workgroup_chunk will evenly divide the npos in the chunk among the
       different threads.
    */
    size_t kernel_local_ngvec = 1;
    size_t kernel_local_npos = workgroup_preferred_size;
    size_t workgroup_chunk_ngvec = 1;
    size_t workgroup_chunk_npos = chunk_npos;
    sizes.total_size[0] = ngvec;
    sizes.total_size[1] = npos;
    sizes.chunk_size[0] = chunk_ngvec;
    sizes.chunk_size[1] = chunk_npos;
    sizes.workgroup_chunk_size[0] = workgroup_chunk_ngvec;
    sizes.workgroup_chunk_size[1] = workgroup_chunk_npos;
    sizes.kernel_local_size[0] = kernel_local_ngvec;
    sizes.kernel_local_size[1] = kernel_local_npos;
}

// parameters for use in gvec_to_xy. Templatized to support both, float and
// double versions.
template <typename REAL>
struct g2xy_params
{
    cl_uint workgroup_chunk_size[2];
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


// host information for gvec_to_xy. This includes the streams to be able to
// stream in/out the data to/from the device. But also information about
// orchestrating the execution to maximize efficiency.
struct g2xy_host_info {
    size_t kernel_local_size[2];
    stream_desc gVec_c_stream;
    stream_desc rMat_s_stream;
    stream_desc tVec_c_stream;
    stream_desc xy_out_stream;
};

template <typename REAL>
static inline int
init_g2xy(g2xy_params<REAL> * restrict params,
          g2xy_host_info * restrict host_info,
          const cl_instance *cl,
          cl_kernel kernel,
          PyArrayObject *pa_gVec_c,
          PyArrayObject *pa_rMat_d,
          PyArrayObject *pa_rMat_s,
          PyArrayObject *pa_rMat_c,
          PyArrayObject *pa_tVec_d,
          PyArrayObject *pa_tVec_s,
          PyArrayObject *pa_tVec_c,
          PyArrayObject *pa_beam_vec,
          PyArrayObject *pa_result_xy)
{ TIME_SCOPE("init_g2xy");
    if (!is_streaming_vector3(pa_gVec_c))
    {
        array_vector3_autoconvert<REAL>(params->gVec_c, pa_gVec_c);
        zap_to_zero(host_info->gVec_c_stream);
    }
    else
    {
        zap_to_zero(params->gVec_c);
        array_stream_convert(&host_info->gVec_c_stream, pa_gVec_c, 1, 1);
    }

    array_matrix33_autoconvert<REAL>(params->rMat_d, pa_rMat_d);

    if (!is_streaming_matrix33(pa_rMat_s))
    {
        array_matrix33_autoconvert<REAL>(params->rMat_s, pa_rMat_s);
        zap_to_zero(host_info->rMat_s_stream);
    }
    else
    {
        zap_to_zero(params->rMat_s);
        array_stream_convert(&host_info->rMat_s_stream, pa_rMat_s, 2, 1);
    }

    array_matrix33_autoconvert<REAL>(params->rMat_c, pa_rMat_c);
    array_vector3_autoconvert<REAL>(params->tVec_d, pa_tVec_d);
    array_vector3_autoconvert<REAL>(params->tVec_s, pa_tVec_s);

    if (!is_streaming_vector3(pa_tVec_c))
    {
        array_vector3_autoconvert<REAL>(params->tVec_c, pa_tVec_c);
        zap_to_zero(host_info->tVec_c_stream);
    }
    else
    {
        zap_to_zero(params->tVec_c);
        array_stream_convert(&host_info->tVec_c_stream, pa_tVec_c, 1, 1);
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

    array_stream_convert(&host_info->xy_out_stream, pa_result_xy, 1, 2);

    size_t ngvec = host_info->xy_out_stream.stream_dims()[0];
    size_t npos = host_info->xy_out_stream.stream_dims()[1];

    // let compute_ocl_g2xy_sizes apply heuristics for best chunk size meeting
    // constraints.
    ocl_g2xy_sizes sizes;
    compute_ocl_g2xy_sizes(sizes, ngvec, npos,
                           cl->device_global_mem_size(),
                           cl->kernel_preferred_workgroup_size_multiple(kernel),
                           cl->device_max_compute_units(),
                           sizeof(REAL));


    params->workgroup_chunk_size[0] = static_cast<cl_uint>(sizes.workgroup_chunk_size[0]);
    params->workgroup_chunk_size[1] = static_cast<cl_uint>(sizes.workgroup_chunk_size[1]);
    params->chunk_size[0] = static_cast<cl_uint>(sizes.chunk_size[0]);
    params->chunk_size[1] = static_cast<cl_uint>(sizes.chunk_size[1]);
    params->total_size[0] = static_cast<cl_uint>(sizes.total_size[0]);
    params->total_size[1] = static_cast<cl_uint>(sizes.total_size[1]);
    host_info->kernel_local_size[0] = sizes.kernel_local_size[0];
    host_info->kernel_local_size[1] = sizes.kernel_local_size[1];

    return 0;
}

template <typename REAL>
static void
print_g2xy(const g2xy_params<REAL> *params,
                 const g2xy_host_info *host_info)
{
    printf("g2xy_params\nkind: %s\n", floating_kind_name<REAL>());
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

    print_stream("gVec_c", host_info->gVec_c_stream);
    print_stream("rMat_s", host_info->rMat_s_stream);
    print_stream("tVec_c", host_info->tVec_c_stream);
    print_stream("xy_out", host_info->xy_out_stream);
}

struct g2xy_buffs
{
    cl_mem params;        // the problem "params"
    cl_mem gvec_c;        // gpu buffer for gvectors
    cl_mem rmat_s;        // gpu buffer for rmat_s
    cl_mem tvec_c;        // gpu buffer for tvec_c
    cl_mem xy_result[2];  // gpu buffers for xy_result (double buffered)
};

static void
release_g2xy_buffs(g2xy_buffs *buffs)
{
    CL_LOG_CHECK(clReleaseMemObject(buffs->xy_result[1]));
    CL_LOG_CHECK(clReleaseMemObject(buffs->xy_result[0]));
    CL_LOG_CHECK(clReleaseMemObject(buffs->tvec_c));
    CL_LOG_CHECK(clReleaseMemObject(buffs->rmat_s));
    CL_LOG_CHECK(clReleaseMemObject(buffs->gvec_c));
    CL_LOG_CHECK(clReleaseMemObject(buffs->params));

    zap_to_zero(*buffs);
}

template<typename REAL>
cl_instance::kernel_slot g2xy_kernel()
{
    static_assert(sizeof(REAL) == 0, "Unsupported type for gvec_to_xy");
    return cl_instance::kernel_slot::invalid;
}

template<>
cl_instance::kernel_slot g2xy_kernel<float>()
{
    return cl_instance::kernel_slot::gvec_to_xy_f32;
}

template<>
cl_instance::kernel_slot g2xy_kernel<double>()
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

struct __attribute__((packed)) g2xy_params {
    uint workgroup_chunk_size[2];
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
    __constant const struct g2xy_params *params,
    __global const REAL *g_gVec_c,
    __global const REAL *g_rMat_s,
    __global const REAL *g_tVec_c,
    __global REAL * restrict g_xy_result)
{
    size_t gvec_idx = get_global_id(0);
    size_t pos_group_idx = get_group_id(1);
    size_t local_size = get_local_size(1); // limited to npos dimension
    size_t local_idx  = get_local_id(1); // limited to npos dimension
    size_t pos_offset = get_global_offset(1);
    size_t gvec_offset = get_global_offset(0);
    size_t ngvec = params->total_size[0] - gvec_offset;
    size_t npos = params->total_size[1] - pos_offset;
    size_t workgroup_npos = params->workgroup_chunk_size[1];
    size_t result_npos = min(workgroup_npos, npos);
    // this should only kick-in in partial chunks
    if (gvec_idx >= ngvec)
        return;

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
    REAL3 gVec_sam = rotate_vector(rMat_c0, rMat_c1, rMat_c2, gVec_c);
    REAL3 gVec_lab = rotate_vector(rMat_s0, rMat_s1, rMat_s2, gVec_sam);
    REAL3 ray_vector;
    if (params->has_beam)
        ray_vector = diffract(vload3(0, params->beam), gVec_lab);
    else
        ray_vector = diffract_z(gVec_lab);

    size_t pos_group_start = pos_offset; // + pos_group_idx*workgroup_npos;
    size_t pos_end = pos_group_start + result_npos;
    size_t pos_start = pos_group_start + local_idx;

    for (size_t pos_idx = pos_start; pos_idx<pos_end; pos_idx+=local_size)
    {
        REAL3 tVec_c = vload3(pos_idx, g_tVec_c);

        REAL3 ray_origin = transform_vector(tVec_s, rMat_s0, rMat_s1, rMat_s2, tVec_c);
        REAL2 projected = to_detector(rMat_d0, rMat_d1, rMat_d2, tVec_d, ray_origin, ray_vector);
        vstore2(projected, gvec_idx*result_npos + pos_idx, g_xy_result);
    }
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
                const g2xy_params<REAL> *params,
                const g2xy_host_info *host_info)
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

    if (host_info->gVec_c_stream.is_active())
    {
        size_t buff_size = 3*sizeof(REAL)*params->total_size[0];
        const cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                        CL_MEM_HOST_WRITE_ONLY;
        buffs.gvec_c = clCreateBuffer(context, buff_flags, buff_size,
                                      NULL, NULL);
    }
    else
        goto fail; // not supported yet.

    if (host_info->rMat_s_stream.is_active())
    {
        size_t buff_size = 9*sizeof(REAL)*params->total_size[0];
        cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                  CL_MEM_HOST_WRITE_ONLY;
        buffs.rmat_s = clCreateBuffer(context, buff_flags, buff_size,
                                          NULL, NULL);
    }
    else
        goto fail; // not supported yet.

    if (host_info->tVec_c_stream.is_active())
    {
        size_t buff_size = 3*sizeof(REAL)*params->total_size[1];
        cl_mem_flags buff_flags = CL_MEM_READ_ONLY |
                                  CL_MEM_HOST_WRITE_ONLY;
        buffs.tvec_c = clCreateBuffer(context, buff_flags, buff_size,
                                      NULL, NULL);
    }
    else
        goto fail;

    if (host_info->xy_out_stream.is_active())
    {
        size_t buff_size = 2*sizeof(REAL)*params->chunk_size[0]*params->chunk_size[1];
        cl_mem_flags buff_flags = CL_MEM_WRITE_ONLY |
                                  CL_MEM_HOST_READ_ONLY;
        buffs.xy_result[0] = clCreateBuffer(context, buff_flags, buff_size,
                                            NULL, NULL);
        buffs.xy_result[1] = clCreateBuffer(context, buff_flags, buff_size,
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


static inline void
raw_copy_to_buffer(cl_command_queue queue, cl_mem buffer, const void *srcdata, size_t sz)
{
    void *mapped = clEnqueueMapBuffer(queue, buffer, CL_TRUE,
                                      CL_MAP_WRITE_INVALIDATE_REGION,
                                      0, sz, 0, NULL, NULL, NULL);
    memcpy(mapped, srcdata, sz);
    clEnqueueUnmapMemObject(queue, buffer, mapped, 0, NULL, NULL);
}

static inline void
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
copy_convert_chunk(void * restrict dst, size_t dst_size_in_bytes,
                   const void *src,
                   const size_t *dims, const ptrdiff_t *strides,
                   size_t ndim)
{
    void *limit = byte_index(dst, dst_size_in_bytes);
#if defined(CLXF_LOG_COPY_CONVERT) && CLXF_LOG_COPY_CONVERT
    printf("copy_convert (begin): %p - %p(%s) <- %p(%s)\n",
           dst, limit,
           type_to_cstr<DST_REAL>(),
           src, type_to_cstr<SRC_REAL>());
    print_dims("\tdims", dims, ndim);
    print_strides("\tstrides", strides, ndim);
#endif
    DST_REAL * restrict out = static_cast<DST_REAL *>(dst);
    const SRC_REAL *in = static_cast<const SRC_REAL *>(src);
    size_t curr_pos[ndim];
    for (size_t i = 0; i<ndim; i++)
        curr_pos[i] = 0;

    do {
        if (out > limit)
        {
            printf("BAD-BAD-BAD: will write past limit (%p)\n", limit);
            print_dims("\t\tdims", curr_pos, ndim);
            printf("\t\t\tinto %p (%p) offset = %zd\n", out, dst, out - static_cast<DST_REAL*>(dst));
        }
        *out++ = *ndim_index(in, curr_pos, strides, ndim);
    } while(next_element(curr_pos, dims, ndim));
#if defined(CLXF_LOG_COPY_CONVERT) && CLXF_LOG_COPY_CONVERT
    printf("\t%zd values copy-converted.\n", out - static_cast<DST_REAL*>(dst));
#endif
}


template<typename DST_REAL, typename SRC_REAL>
static void
copy_convert_chunk_out(void * restrict dst,
                       const void *src, const size_t *dims,
                       const ptrdiff_t *strides,
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
{ TIME_SCOPE("copy_convert_to_buffer");
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

    // printf("q: %p b: %p - WRITE_INVALIDATE - sz: %zd.\n", queue, buffer, total_size);
    void *dst = clEnqueueMapBuffer(queue, buffer, CL_TRUE,
                                   CL_MAP_WRITE_INVALIDATE_REGION,
                                   0, total_size, 0, NULL, NULL, NULL);
    switch(stream->base_type)
    {
    case NPY_FLOAT32:
        copy_convert_chunk<REAL, float>(dst, total_size, src, dims, strides, total_ndim);
        break;
    case NPY_FLOAT64:
        copy_convert_chunk<REAL, double>(dst, total_size, src, dims, strides, total_ndim);
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
sliced_buff_to_mem(cl_command_queue queue, cl_mem buffer, void *dst,
                   size_t total_size, size_t slice_size)
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
            ptrdiff_t contiguous_stride = sizeof(REAL);
            int i;
            for (i = total_ndim-1; i>=0; i--)
            {
                if (strides[i] != contiguous_stride)
                    break;
                contiguous_stride *= dims[i];
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
static inline void
execute_g2xy_chunked(cl_command_queue queue, cl_kernel kernel,
                     g2xy_params<REAL>& params,
                     g2xy_host_info& host_info,
                     g2xy_buffs& buffs)
{
    /* when executing chunked... do it with different offsets and so... */
    size_t curr_chunk[2] = {0};
    size_t chunk_dims[2] = {
        round_up_divide(params.total_size[0], params.chunk_size[0]),
        round_up_divide(params.total_size[1], params.chunk_size[1])
    };
    int curr_buff = 0; //, next_buff = 1;

    size_t chunk_size[2] = {params.chunk_size[0], params.chunk_size[1]};
    size_t chunk_offset[2];
    do
    {
        { TIME_SCOPE("cl_gvec_to_xy - launch kernel (chunked)");

            /* enqueue next_kernel */
            CL_LOG_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffs.params));
            CL_LOG_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffs.gvec_c));
            CL_LOG_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffs.rmat_s));
            CL_LOG_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffs.tvec_c));
            CL_LOG_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),
                                        &buffs.xy_result[curr_buff]));

            chunk_offset[0] = curr_chunk[0]*params.chunk_size[0];
            chunk_offset[1] = curr_chunk[1]*params.chunk_size[1];
            CL_LOG_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2,
                                                chunk_offset, chunk_size,
                                                host_info.kernel_local_size,
                                                0, NULL,
                                                NULL));
            clFinish(queue);
        }

        { TIME_SCOPE("cl_gvec_to_xy - copy results (chunked)");
            size_t this_chunk_size[2] = {
                std::min(chunk_size[0], params.total_size[0] - chunk_offset[0]),
                std::min(chunk_size[1], params.total_size[1] - chunk_offset[1])
            };
            copy_convert_from_buffer<REAL>(queue, buffs.xy_result[0],
                                           &host_info.xy_out_stream,
                                           chunk_offset, this_chunk_size, 2);
            clFinish(queue);
        }

    } while (next_element(curr_chunk, chunk_dims, 2));

}

template <typename REAL>
static inline void
execute_g2xy_oneshot(cl_command_queue queue, cl_kernel kernel,
                     g2xy_params<REAL>& params,
                     g2xy_host_info& host_info,
                     g2xy_buffs &buffs)
{
    size_t chunk_offset[2] = {0, 0};
    size_t chunk_size[2] = { params.total_size[0], params.total_size[1] };
    /* prepare and enqueue the kernel */
    { TIME_SCOPE("cl_gvec_to_xy - execute kernel (oneshot)");
        CL_LOG_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffs.params));
        CL_LOG_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffs.gvec_c));
        CL_LOG_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffs.rmat_s));
        CL_LOG_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffs.tvec_c));
        CL_LOG_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffs.xy_result[0]));

        // use chunk_size in this case, as it is properly rounded up.
        size_t total_size[] = { params.chunk_size[0], params.chunk_size[1] };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, total_size,
                               host_info.kernel_local_size, 0, NULL, NULL);
        clFinish(queue);
    }
    /* wait and copy results */
    { TIME_SCOPE("cl_gvec_to_xy - copy results (oneshot)");
        size_t this_chunk_size[2] = {
            std::min(chunk_size[0], params.total_size[0] - chunk_offset[0]),
            std::min(chunk_size[1], params.total_size[1] - chunk_offset[1])
        };
        copy_convert_from_buffer<REAL>(queue, buffs.xy_result[0],
                                       &host_info.xy_out_stream,
                                       chunk_offset, this_chunk_size, 2);
        clFinish(queue);
    }
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

    cl_kernel kernel =  cl->get_kernel(g2xy_kernel<REAL>());
    if (!kernel)
    { TIME_SCOPE("compile kernel");
        /* kernel not cached... just build it */
        kernel = cl->build_kernel("gvec_to_xy", g2xy_source,
                                  kernel_compile_options<REAL>());
        if (kernel)
        {
            cl->set_kernel(g2xy_kernel<REAL>(), kernel);
        }
        else
            return -2;
    }

    g2xy_params<REAL> params;
    g2xy_host_info host_info;
    init_g2xy(&params, &host_info, cl, kernel,
              gVec_c,
              rMat_d, rMat_s, rMat_c,
              tVec_d, tVec_s, tVec_c,
              beam_vec,
              result_xy_array);

    g2xy_buffs buffs;
    if (init_g2xy_buffs(&buffs, ctxt, &params, &host_info))
    {
        size_t ngvec = static_cast<size_t>(params.total_size[0]);
        size_t npos = static_cast<size_t>(params.total_size[1]);
        size_t zero = 0;
        /* Right now, all inputs are assumed to fit into device memory */
        { TIME_SCOPE("cl_gvec_to_xy - copy args");
            copy_convert_to_buffer<REAL>(queue, buffs.gvec_c,
                                         &host_info.gVec_c_stream,
                                         &zero, &ngvec, 1);
            copy_convert_to_buffer<REAL>(queue, buffs.rmat_s,
                                         &host_info.rMat_s_stream,
                                         &zero, &ngvec, 1);
            copy_convert_to_buffer<REAL>(queue, buffs.tvec_c,
                                         &host_info.tVec_c_stream,
                                         &zero, &npos, 1);

            clFinish(queue);
        }

        if (params.chunk_size[0] <= params.total_size[0] ||
            params.chunk_size[1] <= params.total_size[1])
        {
            execute_g2xy_chunked(queue, kernel, params, host_info, buffs);
        }
        else
        {
            execute_g2xy_oneshot(queue, kernel, params, host_info, buffs);
        }

        release_g2xy_buffs(&buffs);
    }

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
