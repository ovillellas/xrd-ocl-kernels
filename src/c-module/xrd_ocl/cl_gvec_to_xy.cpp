#include "checks.hpp"
#include "utils.hpp"
#include "dev_help.hpp"
#include "cl_memory.hpp"
#include <algorithm> // std::max, std::min
#include <cstdint>




/* this will physically include the sources for the kernels */
#include "cl_source.cpp"

// Maximum result size. But will constraint also to 1/4 of device memory in
// any case.
#define CLXF_MAX_RESULT_SIZE ((size_t)64*1024*1024)

// maximum number of npos per thread. In a workgroup X will be handled.
#define CLXF_MAX_WORKGROUP_WIDTH ((size_t)SIZE_MAX)
#define CLXF_MAX_NPOS_PER_THREAD ((size_t)128)
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

    print_stream("gVec_c", host_info->gVec_c_stream);
    print_stream("rMat_s", host_info->rMat_s_stream);
    print_stream("tVec_c", host_info->tVec_c_stream);
    print_stream("xy_out", host_info->xy_out_stream);
}


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
    }
    else
    {
        params->beam[0] = (REAL)0.0;
        params->beam[1] = (REAL)0.0;
        params->beam[2] = (REAL)1.0;
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

    params->tile_size[0] = static_cast<cl_uint>(sizes.tile_size[0]);
    params->tile_size[1] = static_cast<cl_uint>(sizes.tile_size[1]);
    params->chunk_size[0] = static_cast<cl_uint>(sizes.chunk_size[0]);
    params->chunk_size[1] = static_cast<cl_uint>(sizes.chunk_size[1]);
    params->total_size[0] = static_cast<cl_uint>(sizes.total_size[0]);
    params->total_size[1] = static_cast<cl_uint>(sizes.total_size[1]);
    host_info->kernel_local_size[0] = sizes.kernel_local_size[0];
    host_info->kernel_local_size[1] = sizes.kernel_local_size[1];

    //print_g2xy(params, host_info);
    return 0;
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


template <typename REAL>
static inline void
execute_g2xy_chunked(cl_command_queue queue, cl_kernel kernel,
                     cl_mem_transfer_context& ctx,
                     g2xy_params<REAL>& params,
                     g2xy_host_info& host_info,
                     g2xy_buffs& buffs)
{
    /* when executing chunked... do it with different offsets and so... */
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

    cl_event pending[2] ={ 0, 0};
    int db_compute = 0, db_transfer = 0;
    size_t launched = 0, transferred = 0, total = chunk_dims[0]*chunk_dims[1];
    size_t chunk_compute[2] = {0};
    size_t chunk_transfer[2] = {0};
    CL_LOG_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffs.params));
    CL_LOG_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffs.gvec_c));
    CL_LOG_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffs.rmat_s));
    CL_LOG_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffs.tvec_c));
    while (transferred < total)
    {
        while (launched < total && !pending[db_compute])
        { TIME_SCOPE("g2xy - enqueue kernel");
            /* enqueue next_kernel */

            CL_LOG_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffs.xy_result[db_compute]));

            CL_LOG_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2,
                                                chunk_compute, work_size,
                                                host_info.kernel_local_size,
                                                0, NULL,
                                                &pending[db_compute]));
            clFlush(queue);
            ndim_next_element(chunk_compute, chunk_dims, 2);
            db_compute = 1-db_compute;
            launched++;
        }

        { TIME_SCOPE("g2xy - wait for event");
            CL_LOG_CHECK(clWaitForEvents(1, &pending[db_transfer]));
            CL_LOG_EVENT_PROFILE("g2xy kernel exec", pending[db_transfer]);
            CL_LOG_CHECK(clReleaseEvent(pending[db_transfer]));
            pending[db_transfer] = 0;
        }

        TIME_SCOPE("g2xy - copy-convert");
        size_t chunk_offset[2] = {
            chunk_transfer[0]*params.chunk_size[0],
            chunk_transfer[1]*params.chunk_size[1]
        };
        size_t this_chunk_size[2] = {
            std::min<size_t>(params.chunk_size[0], params.total_size[0] - chunk_offset[0]),
            std::min<size_t>(params.chunk_size[1], params.total_size[1] - chunk_offset[1])
        };
        //debug_print_dims("chunk (mem)", curr_chunk, 2);
        //debug_print_dims("\toffset", chunk_offset,2);
        //debug_print_dims("\tsize", this_chunk_size, 2);
        copy_convert_from_buffer<REAL>(ctx, buffs.xy_result[db_transfer],
                                           &host_info.xy_out_stream,
                                           chunk_offset, this_chunk_size, 2);
        db_transfer = 1-db_transfer;
        transferred++;
        ndim_next_element(chunk_transfer, chunk_dims, 2);
    };
    //printf("%zd chunks run.\n", count);
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

        execute_g2xy_chunked(queue, kernel, ctx, params, host_info, buffs);
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
