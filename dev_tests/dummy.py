#! /usr/bin/env ipython -i

from numpy.random import Generator, PCG64
from xrd_ocl_kernels import xrd_ocl

rng=Generator(PCG64())

args = (rng.random((10,3), dtype='d'),
        rng.random((3,3), dtype='f'),
        rng.random((10,3,3), dtype='f'),
        rng.random((3,3), dtype='d'),
        rng.random((3,), dtype='f'),
        rng.random((3,), dtype='d'),
        rng.random((20,3), dtype='f'))

res = xrd_ocl.cl_gvec_to_xy(*args)

print(args == res[:-1])
