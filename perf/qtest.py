#!/usr/bin/env python

import argparse
import numpy as np

import perf_gvec_to_xy
import xrd_transforms
import xrd_ocl_kernels as xok


def run_test(ngvec, npos):
    foo = perf_gvec_to_xy.build_args(ngvec, npos, False)

    gvec_to_xy_cpu = perf_gvec_to_xy.make_multi_npos_gvec_to_xy(xrd_transforms.xf_new_capi.gvec_to_xy)

    res_orig = gvec_to_xy_cpu(*foo)

    res_new = xok.gvec_to_xy(*foo)
    result = "PASS" if np.allclose(res_orig, res_new) else "FAIL"
    print(f"double precision: {result}")

    res_single = xok.gvec_to_xy(*foo, single_precision=True)
    result = "PASS" if np.allclose(res_orig, res_single) else "FAIL"
    print(f"single precision: {result}")

    print(f"max difference between single and double results {np.max(np.abs(res_new-res_single))}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngvec', type=int, default=10000, help='number of G vectors')
    parser.add_argument('--npos', type=int, default=10000, help='number of candidate positions')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args();
    run_test(args.ngvec, args.npos);
