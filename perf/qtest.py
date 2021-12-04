#!/usr/bin/env python

import argparse
import numpy as np

import perf_gvec_to_xy
import xrd_transforms
import xrd_ocl_kernels as xok

def print_slice(slice):
    if slice.step is None:
        return f"{slice.start}:{slice.stop}"
    else:
        return f"{slice.start}:{slice.stop}:{slice.step}"

def print_coord(tuple_of_slices):
    return f"[{','.join(print_slice(s) for s in tuple_of_slices)}]"

def gen_chunks(global_shape, chunk_shape):
    for i in range(0, global_shape[0], chunk_shape[0]):
        first_dim = slice(i, min(i + chunk_shape[0], global_shape[0] ))
        for j in range(0, global_shape[1], chunk_shape[1]):
            second_dim = slice(j, min(j + chunk_shape[1], global_shape[1]))
            yield first_dim, second_dim


def detailed_check(orig, other, block_size=(512, 4096), level=1):
    if level < 1:
        return
    
    for block in gen_chunks(orig.shape, block_size):
        result = "PASS" if np.allclose(orig[block], other[block], equal_nan=True) else "FAIL"
        print(f"{print_coord(block)} {result}")
        if level > 1 and result != "PASS":
            print(f"orig:\n{orig[block]}\nother:\n{other[block]}")

            
def eval_test(ngvec, npos, only_double=False):
    foo = perf_gvec_to_xy.build_args(ngvec, npos, False)
    print(foo);
    gvec_to_xy_cpu = perf_gvec_to_xy.make_multi_npos_gvec_to_xy(xrd_transforms.xf_new_capi.gvec_to_xy)
    res_orig = gvec_to_xy_cpu(*foo)
    res_double = xok.gvec_to_xy(*foo)
    res_single = None if only_double else xok.gvec_to_xy(*foo, single_precision=True)
    return res_orig, res_double, res_single


def run_test(ngvec, npos, chunk_size=(512, 4096), detail=0, only_double=False):
    orig, double, single = eval_test(ngvec, npos)
    
    result = "PASS" if np.allclose(orig, double, equal_nan=True) else "FAIL"
    print(f"double precision: {result}")
    if result == "FAIL":
        detailed_check(orig, double, block_size=chunk_size, level=detail)
    if only_double:
        return
        
    result = "PASS" if np.allclose(orig, single, equal_nan=True) else "FAIL"
    print(f"single precision: {result}")
    if result == "FAIL":
        detailed_check(orig, single, block_size=chunk_size, level=detail)
    
    print(f"max difference between single and double results {np.nanmax(np.abs(double-single))}")

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngvec', type=int, default=10000,
                        help='number of G vectors')
    parser.add_argument('--npos', type=int, default=10000,
                        help='number of candidate positions')
    parser.add_argument('--detail-level', type=int, default=0,
                        help='level of detail on error information (0:none, 1: per-block, 2: full)')
    parser.add_argument('--chunk-ngvec', type=int, default=512,
                        help='chunk_size to check, ngvecs')
    parser.add_argument('--chunk-npos', type=int, default=4096,
                        help='chunk_size to check, npos')
    parser.add_argument('--only-double', action='store_true',
                        help='only run the double kernel');
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args();
    run_test(args.ngvec, args.npos, (args.chunk_ngvec, args.chunk_npos), args.detail_level, args.only_double);
