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
};

REAL3 diffract(REAL3 beam, REAL3 gvec);
REAL3 transform_vector(REAL3 tr, REAL3 rot0, REAL3 rot1, REAL3 rot2, REAL3 v);
REAL3 rotate_vector(REAL3 rot0, REAL3 rot1, REAL3 rot2, REAL3 v);
REAL2 to_detector(REAL3 rD0, REAL3 rD1, REAL3 rD2, REAL3 tD,
                  REAL3 ray_origin, REAL3 ray_vector);

inline REAL3
diffract(REAL3 beam, REAL3 gvec)
{
    REAL3 bm_diag  = ((REAL)2.0)*gvec*gvec - ((REAL)1.0);
    REAL3 bm_other = ((REAL)2.0)*gvec.xxy*gvec.yzz;
    REAL3 row0 = (REAL3)(bm_diag.x, bm_other.x, bm_other.y);
    REAL3 row1 = (REAL3)(bm_other.x, bm_diag.y, bm_other.z);
    REAL3 row2 = (REAL3)(bm_other.y, bm_other.z, bm_diag.z);
    return (REAL3)(dot(row0, beam), dot(row1, beam), dot(row2,beam));
}

inline REAL3
transform_vector(REAL3 tr, REAL3 rot0, REAL3 rot1, REAL3 rot2, REAL3 v)
{
    return (REAL3)(tr.x + dot(rot0, v), tr.y + dot(rot1, v), tr.z + dot(rot2, v));
}

inline REAL3
rotate_vector(REAL3 rot0, REAL3 rot1, REAL3 rot2, REAL3 v)
{
    return (REAL3)(dot(rot0, v), dot(rot1, v), dot(rot2, v));
}

inline REAL2
to_detector(REAL3 rDt0, REAL3 rDt1, REAL3 rDt2, REAL3 tD,
            REAL3 ray_origin, REAL3 ray_vector)
{
    REAL3 ray_origin_det = ray_origin - tD;

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

        REAL3 rDt0 = (REAL3)(rMat_d0.x, rMat_d1.x, rMat_d2.x),
              rDt1 = (REAL3)(rMat_d0.y, rMat_d1.y, rMat_d2.y),
              rDt2 = (REAL3)(rMat_d0.z, rMat_d1.z, rMat_d2.z);
 
        REAL3 tVec_s = vload3(0, params->tVec_s);
        REAL3 tVec_d = vload3(0, params->tVec_d);
        REAL3 beam = vload3(0, params->beam);
        
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

        REAL3 ray_vector = diffract(beam, gVec_lab);


        __global REAL * restrict result_row = g_xy_result + 2*pos.x*chunk_size.y;

        // change this to change how the work is executed locally (each thread
        // working on contiguous npos or in an interleaved way).
        uint thread_start = pos.y + local_id.y*pt_size.y;
        uint thread_stop = min(thread_start+pt_size.y, chunk_size.y);
        uint thread_step = 1;

        for (uint pos_idx = thread_start; pos_idx < thread_stop; pos_idx += thread_step)
        {
            // pos_idx is the kernel-relative position index. Add the offset to have
            // the effective pos in the g_tVec_c array. 
            REAL3 tVec_c = vload3(offset.y+pos_idx, g_tVec_c);

            REAL3 ray_origin = transform_vector(tVec_s, rMat_s0, rMat_s1, rMat_s2, tVec_c);
            REAL2 projected = to_detector(rDt0, rDt1, rDt2, tVec_d, ray_origin, ray_vector);

            // store should be relative to the current kernel call. The result array
            // is contiguous, being in position major order. That is, the xy pair
            // for consecutive positions are consecutive. Note that the resulting
            // array is compact and with space only for the results of this kernel
            // execution, so when computing the offset, the gvec_idx used must be
            // the kernel relative version. result_npos must be the total number of
            // positions handled in this kernel. And the position index must be the
            // one relative to the position start of this kernel.
            vstore2(projected, pos_idx, result_row);
        }
    }
}
