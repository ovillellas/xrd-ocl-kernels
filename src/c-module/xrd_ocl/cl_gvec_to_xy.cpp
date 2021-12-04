#include "checks.hpp"
#include "utils.hpp"
#include "dev_help.hpp"

#include <algorithm> // std::max, std::min
#include <cstdint>

// Maximum result size. But will constraint also to 1/4 of device memory in
// any case.
#define CLXF_MAX_RESULT_SIZE ((size_t)512*1024*1024)

// maximum number of npos per thread. In a workgroup X will be handled.
#define CLXF_MAX_WORKGROUP_WIDTH ((size_t)SIZE_MAX)
#define CLXF_MAX_NPOS_PER_THREAD ((size_t)64)
#define CLXF_MAX_NPOS_PER_CHUNK  ((size_t)SIZE_MAX)
#define CLXF_MAX_NGVEC_PER_CHUNK ((size_t)65536)

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
 * 3. tile_size: In OpenCL, kernels are executed in workgroups.
 *    A workgroup is a set of threads executing in lockstep. Instead of limiting
 *    each thread to a single (gvec, npos) pair, work is split in a workgroup
 *    basis. Each thread knows what part of its workgroup dataset evaluate.
 *    TILE is the name I give to the block of data assigned to a workgroup.
 *
 * 4. kernel_local_size: This is how the kernel splits the chunk_size between
 *    the different compute units. It is basically how it assigns blocks to
 *    different workgroups.
 */

struct ocl_g2xy_sizes
{
    size_t total_size[2];
    size_t chunk_size[2];
    size_t tile_size[2];
    size_t kernel_local_size[2];
};

inline void
hardcode_ocl_g2xy_sizes(ocl_g2xy_sizes& sizes, size_t ngvec, size_t npos,
                        size_t device_memory,
                        size_t workgroup_preferred_size,
                        size_t compute_units,
                        size_t base_value_size,
                        size_t chunk_ngvec, size_t chunk_npos)
{
    // just compute everything so that it can be evaluated with the hardcoded
    // chunk size...
    size_t kernel_local_ngvec = 1;
    size_t kernel_local_npos = std::min(chunk_npos, workgroup_preferred_size);
    size_t tile_ngvec = 1;
    size_t tile_npos = chunk_npos;
    
    sizes.total_size[0] = ngvec;
    sizes.total_size[1] = npos;
    sizes.chunk_size[0] = chunk_ngvec;
    sizes.chunk_size[1] = chunk_npos;
    sizes.tile_size[0] = tile_ngvec;
    sizes.tile_size[1] = tile_npos;
    sizes.kernel_local_size[0] = kernel_local_ngvec;
    sizes.kernel_local_size[1] = kernel_local_npos;
}


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
    workgroup_preferred_size = std::min(workgroup_preferred_size,
                                        CLXF_MAX_WORKGROUP_WIDTH);

    // try the whole dataset, but respecting the configured max sizes per chunk
    size_t max_npos_per_workgroup = CLXF_MAX_NPOS_PER_THREAD*workgroup_preferred_size;
    size_t max_npos_per_chunk = CLXF_MAX_NPOS_PER_CHUNK;//CLXF_MAX_NPOS_PER_THREAD*workgroup_preferred_size;
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
        // chunk_ngvec = std::max(chunk_ngvec, std::min(ngvec, compute_units));

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
                chunk_npos = 1; chunk_ngvec = 1; workgroup_preferred_size = 1;
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
    size_t tile_ngvec = 1;
    size_t tile_npos = std::min(chunk_npos, max_npos_per_workgroup);

    sizes.total_size[0] = ngvec;
    sizes.total_size[1] = npos;
    sizes.chunk_size[0] = chunk_ngvec;
    sizes.chunk_size[1] = chunk_npos;
    sizes.tile_size[0] = tile_ngvec;
    sizes.tile_size[1] = tile_npos;
    sizes.kernel_local_size[0] = kernel_local_ngvec;
    sizes.kernel_local_size[1] = kernel_local_npos;

#if 0
    printf("total size: (%zd, %zd) chunk_size: (%zd, %zd) "
           "tile_size: (%zd, %zd) kernel_local_size: (%zd, %zd)\n",
           sizes.total_size[0], sizes.total_size[1],
           sizes.chunk_size[0], sizes.chunk_size[1],
           sizes.tile_size[0], sizes.tile_size[1],
           sizes.kernel_local_size[0], sizes.kernel_local_size[1]);
#endif
}

// parameters for use in gvec_to_xy. Templatized to support both, float and
// double versions.
template <typename REAL>
struct g2xy_params
{
    cl_uint tile_size[2];
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

    /*
    hardcode_ocl_g2xy_sizes(sizes, ngvec, npos,
                            cl->device_global_mem_size(),
                            cl->kernel_preferred_workgroup_size_multiple(kernel),
                            cl->device_max_compute_units(),
                            sizeof(REAL),
                            4, 4);
    */
    params->tile_size[0] = static_cast<cl_uint>(sizes.tile_size[0]);
    params->tile_size[1] = static_cast<cl_uint>(sizes.tile_size[1]);
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
#  define FMT "hlf"
#else
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#  define REAL double
#  define REAL2 double2
#  define REAL3 double3
#  define FMT "lf"
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define LOGS 0

struct __attribute__((packed)) g2xy_params {
    uint tile_size[2];
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
    __global REAL * g_xy_result)
{
    /* 
       Each thread executes a block of pt_ngvec x pt_npos (pt -> per_thread)
       the way to infer that size is based on the local_size and the 
       work_group_chunk_size.
     */

    // local_size is the layout of the threads in the current workgroup, while
    // local_id is the id for this thread within that layout.
    // for now we are always using a (1, num_threads) layout.
    uint2 local_size = (uint2)(get_local_size(0), get_local_size(1));
    uint2 local_id = (uint2)(get_local_id(0), get_local_id(1));
if (local_size.x != 1 || local_id.x != 0)
    printf("unexpected local_size.x: %u local_id.x: %u", local_size.x, local_id.x);
 

    // the total size of the problem, in ngvec x npos
    uint2 total_size = vload2(0, params->total_size);

    // the size of the tile in ngvec x npos.
    uint2 tile_size = vload2(0, params->tile_size);

    // the offset for this kernel call, in chunks
    uint2 chunk_size = vload2(0, params->chunk_size);
    uint2 chunk_offset = (uint2)(get_global_offset(0), get_global_offset(1));
    uint2 offset = chunk_offset*chunk_size; // the actual offset in ngvec, npos

    // adjust size of the chunk to handle handle border cases
    chunk_size = min(chunk_size, total_size - offset);

    // the position for this workgroup, in tiles, relative to the whole problem
    uint2 tile_pos = (uint2)(get_group_id(0), get_group_id(1));
    uint2 pos = tile_pos*tile_size; // actual position within the chunk
    uint2 global_pos = pos + offset;

    // per-thread problem size, in ngvec x npos. ngvec should always result in 1
    uint2 pt_size = tile_size / local_size;
/*
if (tile_size.x%pt_size.x != 0 || tile_size.y%pt_size.y != 0)
    printf("unexpected pt_size (%u, %u) for local_size (%u, %u)\n",
            pt_size.x, pt_size.y, local_size.x, local_size.y);
*/
#if 0
printf("local_idx: %llu global_gvec_idx: %llu gvec_idx: %llu global_pos_idx: %llu\n"
       "\ttotal_size: %ux%u\n"
       "\tchunk_size: %ux%u\n"
       "\ttile_size: %ux%u\n"
       "\tworkgroup_pos: %llu->%llu\n",
        local_idx, global_gvec_idx, gvec_offset, global_pos_idx,
        total_size.x, total_size.y,
        chunk_size.x, chunk_size.y,
        tile_size.x, tile_size.y,
        workgroup_pos_begin, workgroup_pos_end);
#endif

    // ignore gvecs past the range
    if (pos.x < chunk_size.x)
    {
        REAL3 rMat_s0, rMat_s1, rMat_s2;
        REAL3 rMat_c0 = vload3(0, params->rMat_c),
              rMat_c1 = vload3(1, params->rMat_c),
              rMat_c2 = vload3(2, params->rMat_c);
        REAL3 rMat_d0 = vload3(0, params->rMat_d),
              rMat_d1 = vload3(1, params->rMat_d),
              rMat_d2 = vload3(2, params->rMat_d);
        REAL3 tVec_s = vload3(0, params->tVec_s);
        REAL3 tVec_d = vload3(0, params->tVec_d);

        REAL3 gVec_c = vload3(global_pos.x, g_gVec_c);
        if (g_rMat_s)
        {
            rMat_s0 = vload3(3*global_pos.x+0, g_rMat_s);
            rMat_s1 = vload3(3*global_pos.x+1, g_rMat_s);
            rMat_s2 = vload3(3*global_pos.x+2, g_rMat_s);
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


        __global REAL * restrict result_row = g_xy_result + 2*pos.x*chunk_size.y;

        // change this to change how the work is executed locally (each thread
        // working on contiguous npos or in an interleaved way).
        uint thread_start = pos.y + local_id.y*pt_size.y;
        uint thread_stop = min(thread_start+pt_size.y, chunk_size.y);
        uint thread_step = 1;
        uint npos_computed = 0;
#if 0
printf("tile: (%u, %u) tile_size: (%u, %u)\n",
       tile_pos.x, tile_pos.y, tile_size.x, tile_size.y);
#endif
#if 0
if (pos.x == 0)
printf("gvec: %u npos: %u -> %u (%u step) (%u -> %u [%u]) chunk: (%u, %u) %p\n"
       "\toffset: (%u, %u) global_pos: (%u, %u)\n",
        pos.x, thread_start, thread_stop, thread_step,
        pos.y, pos.y + chunk_size.y, local_id.y, chunk_size.x, chunk_size.y,
        result_row,
        offset.x, offset.y, global_pos.x, global_pos.y
      );
#endif
        for (uint pos_idx = thread_start; pos_idx < thread_stop; pos_idx += thread_step)
        {
            // pos_idx is the kernel-relative position index. Add the offset to have
            // the effective pos in the g_tVec_c array. 
            REAL3 tVec_c = vload3(offset.y+pos_idx, g_tVec_c);

            REAL3 ray_origin = transform_vector(tVec_s, rMat_s0, rMat_s1, rMat_s2, tVec_c);
            REAL2 projected = to_detector(rMat_d0, rMat_d1, rMat_d2, tVec_d, ray_origin, ray_vector);

            // store should be relative to the current kernel call. The result array
            // is contiguous, being in position major order. That is, the xy pair
            // for consecutive positions are consecutive. Note that the resulting
            // array is compact and with space only for the results of this kernel
            // execution, so when computing the offset, the gvec_idx used must be
            // the kernel relative version. result_npos must be the total number of
            // positions handled in this kernel. And the position index must be the
            // one relative to the position start of this kernel.
#if 0
printf("stored result (%u, %u) at %p.\n"
       " G: %9.6f, %9.6f, %9.6f\n"
       "tC: %9.6f, %9.6f, %9.6f\n"
       "xy: %9.6f, %9.6f\n",
       global_pos.x, offset.y+pos_idx, result_row+2*pos_idx,
       gVec_c.x, gVec_c.y, gVec_c.z,
       tVec_c.x, tVec_c.y, tVec_c.z,
       projected.x, projected.y);
#endif
            vstore2(projected, pos_idx, result_row);
            //vstore2(projected, 0, result_row+2*pos_idx);
            npos_computed++;
        }
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

struct cl_mem_transfer_context
{
    cl_mem_transfer_context(cl_command_queue q, cl_mem b);
    ~cl_mem_transfer_context();

    cl_mem_transfer_context(const cl_mem_transfer_context&) = delete;
    
    cl_command_queue queue;
    cl_mem staging_buffer;
    size_t staging_size;
    void* staging_mem;
};

inline
cl_mem_transfer_context::cl_mem_transfer_context(cl_command_queue q, cl_mem b):
    queue(q), staging_buffer(b)
{
    CL_LOG_CHECK(clGetMemObjectInfo(b, CL_MEM_SIZE, sizeof(staging_size),
                                    &staging_size, NULL));
    staging_mem = clEnqueueMapBuffer(queue, staging_buffer,
                                     CL_TRUE, CL_MAP_READ|CL_MAP_WRITE,
                                     0, staging_size,
                                     0, NULL, NULL, NULL);
}

inline
cl_mem_transfer_context::~cl_mem_transfer_context()
{
    CL_LOG_CHECK(clEnqueueUnmapMemObject(queue, staging_buffer,
                                         staging_mem, 0, NULL, NULL));
}

template <typename REAL>
inline array_copy_convert_error
copy_convert_to_buffer(cl_mem_transfer_context& ctx, cl_mem buffer,
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

    void *dst = clEnqueueMapBuffer(ctx.queue, buffer, CL_TRUE,
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
    clEnqueueUnmapMemObject(ctx.queue, buffer, dst, 0, NULL, NULL);

    return err;
}

// approach the copy in a very agnostic way. Just dma as much as possible and
// fill strided...
static void
sliced_buff_to_mem(cl_mem_transfer_context& ctx, cl_mem buffer, void *dst,
                   size_t inner_size, ptrdiff_t stride, size_t count)
{ TIME_SCOPE("sliced_buff_to_mem");
    //    printf("sliced_buff_to_mem - dst: %p row_size: %zd stride: %zd count: %zd\n",
    //       dst, inner_size, stride, count);
    cl_event pending[2] = { 0, 0};
    int db_cpu = 0, db_dma = 0;
    size_t slice_size = ctx.staging_size/2;
    size_t buffer_offset[2] = { 0, slice_size };
    void *buffer_ptr[2] = { ctx.staging_mem, byte_index(ctx.staging_mem, slice_size) };
    size_t copied_cpu = 0, copied_dma = 0, to_copy = inner_size*count;
    size_t in_row_offset = 0;
    if (stride == (ptrdiff_t)inner_size)
    { // this will coalesce all rows into a single one if they happen to be
      // contiguous
        inner_size = to_copy;
    }
    while (copied_cpu < to_copy)
    {
        while (copied_dma < to_copy && !pending[db_dma])
        { // if not everything has been dma'd and dma is available, enqueue
            size_t transfer_size = std::min(slice_size, to_copy-copied_dma);
            //printf("queued_dma [%d]: size: %zd\n", db_dma, transfer_size);
            CL_LOG_CHECK(clEnqueueCopyBuffer(ctx.queue, buffer, ctx.staging_buffer,
                                             copied_dma, buffer_offset[db_dma],
                                             transfer_size,
                                             0, NULL, &pending[db_dma]));
            db_dma = 1 - db_dma;
            copied_dma += transfer_size;
        }
        // at this point, wait for the dma associated with the current transfer
        // to finish and perform the memory copies
        CL_LOG_CHECK(clWaitForEvents(1, &pending[db_cpu]));
        CL_LOG_CHECK(clReleaseEvent(pending[db_cpu]));
        pending[db_cpu] = 0;
        void *src = buffer_ptr[db_cpu];
        size_t copy_size = std::min(slice_size, to_copy - copied_cpu);
        size_t copied = 0;
        while (copied < copy_size)
        {
            size_t size = std::min(inner_size - in_row_offset,
                                   copy_size - copied);
            //printf("copying mem [%d] dst: %p size: %zd.\n", db_cpu, byte_index(dst, in_row_offset), size);
            memcpy(byte_index(dst, in_row_offset),
                   byte_index(src, copied),
                   size);
            copied += size;
            in_row_offset += size;
            if (in_row_offset >= inner_size)
            { // change row. Note that == should work just as well... > would mean an error.
                dst = byte_index(dst, stride);
                in_row_offset = 0;
            }
        }
        copied_cpu += copy_size;
        db_cpu = 1 - db_cpu;  
    } 
}


template <typename REAL>
inline array_copy_convert_error
copy_convert_from_buffer(cl_mem_transfer_context& ctx, cl_mem buffer,
                         const stream_desc *stream, const size_t *pos,
                         const size_t *sz, size_t ndim)
{ TIME_SCOPE("copy_convert_from_buffer");
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

    bool needs_conversion = numpy_type<REAL>() != stream->base_type;
    int non_contiguous_dimensions = total_ndim;
    ptrdiff_t contiguous_size = sizeof(REAL);
    while (non_contiguous_dimensions &&
           strides[non_contiguous_dimensions-1] == contiguous_size)
        contiguous_size *= dims[--non_contiguous_dimensions];

    /* at this point, non_contiguous_dimensions holds the number of dimensions
       that need to be iterated upon, copying contiguous_size/sizeof(REAL)
       elements (or memcpying contiguous_size if no conversion is needed */
    if (!needs_conversion && 1 >= non_contiguous_dimensions)
    { TIME_SCOPE("copy_convert_from_buffer: memcpy optimized");
        sliced_buff_to_mem(ctx, buffer, dst, total_size/dims[0], strides[0], dims[0]);
    }
    else
    { TIME_SCOPE("copy_convert_from_buffer: generic case");
        void *src;
        { TIME_SCOPE("copy_convert_from_buffer: clEnqueueMapBuffer");
            src = clEnqueueMapBuffer(ctx.queue, buffer, CL_TRUE, CL_MAP_READ, 0,
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
        clEnqueueUnmapMemObject(ctx.queue, buffer, src, 0, NULL, NULL);
    }

    return err;
}

template <typename REAL>
static inline void
execute_g2xy_chunked(cl_command_queue queue, cl_kernel kernel,
                     cl_mem_transfer_context& ctx,
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

    // this will generate the actual get_global_id(0), get_global_id(1) inside
    // the kernel. This should be tile based, as it will generate one entry
    // per workgroup. Each workgroup will use its own indexing as well within
    // the tile.
    size_t work_size[2] = {
        round_up_divide(params.chunk_size[0],params.tile_size[0])*host_info.kernel_local_size[0],
        round_up_divide(params.chunk_size[1],params.tile_size[1])*host_info.kernel_local_size[1]
    };
    #if 0
    debug_print_dims("chunks to run", chunk_dims, 2);
    debug_print_dims("work_size", work_size, 2);
    debug_print_array("chunk_size", params.chunk_size, 2, "%u");
    debug_print_array("tile_size", params.tile_size, 2, "%u");
    debug_print_array("local_size", host_info.kernel_local_size, 2, "%zu");
    #endif
    size_t count = 0;
    do
    {
        { TIME_SCOPE("cl_gvec_to_xy - launch kernel (chunked)");

            /* enqueue next_kernel */
            CL_LOG_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffs.params));
            CL_LOG_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffs.gvec_c));
            CL_LOG_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffs.rmat_s));
            CL_LOG_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffs.tvec_c));
            CL_LOG_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffs.xy_result[0]));

            //debug_print_dims("chunk", curr_chunk, 2);
            //debug_print_dims("\tsize", work_size, 2);
            CL_LOG_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2,
                                                curr_chunk, work_size,
                                                host_info.kernel_local_size,
                                                0, NULL,
                                                NULL));
            clFinish(queue);
        }

        { TIME_SCOPE("cl_gvec_to_xy - copy results (chunked)");
            size_t chunk_offset[2] = {
                curr_chunk[0]*params.chunk_size[0],
                curr_chunk[1]*params.chunk_size[1]
            };
            size_t this_chunk_size[2] = {
                std::min<size_t>(params.chunk_size[0], params.total_size[0] - chunk_offset[0]),
                std::min<size_t>(params.chunk_size[1], params.total_size[1] - chunk_offset[1])
            };
            //debug_print_dims("chunk (mem)", curr_chunk, 2);
            //debug_print_dims("\toffset", chunk_offset,2);
            //debug_print_dims("\tsize", this_chunk_size, 2);
            copy_convert_from_buffer<REAL>(ctx, buffs.xy_result[0],
                                           &host_info.xy_out_stream,
                                           chunk_offset, this_chunk_size, 2);
            clFinish(ctx.queue);
        }
        count++;
    } while (next_element(curr_chunk, chunk_dims, 2));
    //printf("%zd chunks run.\n", count);
}

template <typename REAL>
static inline void
execute_g2xy_oneshot(cl_command_queue queue, cl_kernel kernel,
                     cl_mem_transfer_context& ctx,
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
        copy_convert_from_buffer<REAL>(ctx, buffs.xy_result[0],
                                       &host_info.xy_out_stream,
                                       chunk_offset, this_chunk_size, 2);
        clFinish(ctx.queue);
    }
}

template <typename REAL>
static int
cl_gvec_to_xy(PyArrayObject *gVec_c,
              PyArrayObject *rMat_d, PyArrayObject *rMat_s, PyArrayObject *rMat_c,
              PyArrayObject *tVec_d, PyArrayObject *tVec_s, PyArrayObject *tVec_c,
              PyArrayObject *beam_vec, // may be nullptr
              PyArrayObject *result_xy_array)
{ TIME_SCOPE("cl_gvec_to_xy");
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

        cl_mem_transfer_context ctx(cl->mem_queue, cl->staging_buffer);
        /* Right now, all inputs are assumed to fit into device memory */
        { TIME_SCOPE("cl_gvec_to_xy - copy args");
            copy_convert_to_buffer<REAL>(ctx, buffs.gvec_c,
                                         &host_info.gVec_c_stream,
                                         &zero, &ngvec, 1);
            copy_convert_to_buffer<REAL>(ctx, buffs.rmat_s,
                                         &host_info.rMat_s_stream,
                                         &zero, &ngvec, 1);
            copy_convert_to_buffer<REAL>(ctx, buffs.tvec_c,
                                         &host_info.tVec_c_stream,
                                         &zero, &npos, 1);

            clFinish(ctx.queue);
        }

        //if (params.chunk_size[0] <= params.total_size[0] ||
        //    params.chunk_size[1] <= params.total_size[1])
        //{
        execute_g2xy_chunked(queue, kernel, ctx, params, host_info, buffs);
        //}
        //else
        // {
        //execute_g2xy_oneshot(queue, kernel, ctx, params, host_info, buffs);
        //}
    }
    release_g2xy_buffs(&buffs);

    return 0;
}

XRD_PYTHON_WRAPPER PyObject *
python_cl_gvec_to_xy(PyObject *self, PyObject *args, PyObject *kwargs)
{ TIME_SCOPE("python wrapper");
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
