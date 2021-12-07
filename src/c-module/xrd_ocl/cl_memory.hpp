#ifndef CLXF_CL_MEMORY_HPP
#define CLXF_CL_MEMORY_HPP

#include <algorithm> // std::max, std::min...

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
    } while(ndim_next_element(curr_pos, dims, ndim));
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
    } while(ndim_next_element(curr_pos, dims, ndim));

#if defined(CLXF_LOG_COPY_CONVERT) && CLXF_LOG_COPY_CONVERT
    print_dims("chunk dimensions", dims, ndim);
    printf("Chunk out: %zd %s written (%s original)\n", count,
           type_to_cstr<DST_REAL>(), type_to_cstr<SRC_REAL>());
#endif
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
{ CL_LOG_CUMULATIVE_PROFILE(copy_profile, "CopyBuffer");
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
            clFlush(ctx.queue);
        }
        // at this point, wait for the dma associated with the current transfer
        // to finish and perform the memory copies
        CL_LOG_CHECK(clWaitForEvents(1, &pending[db_cpu]));
        CL_LOG_CUMULATIVE_PROFILE_ADD(copy_profile, pending[db_cpu]);
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
{
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
        // TODO: this code needs to be revised
        void *src;
        {
            src = clEnqueueMapBuffer(ctx.queue, buffer, CL_TRUE, CL_MAP_READ, 0,
                                     total_size, 0, NULL, NULL, NULL);
        }

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



#endif //CLXF_CL_MEMORY_HPP
