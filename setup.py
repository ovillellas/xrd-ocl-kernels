#! /usr/bin/env python

# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import glob

import numpy

from setuptools import Command, Extension, find_packages, setup

base_dependencies = []

################################################################################
# versioneer is used to base the generate the version using the tag name in git.
# Just use a tag named xrd-transforms-v<major>.<minor>.<release>[.dev<devrel>]
################################################################################
import versioneer

cmdclass = versioneer.get_cmdclass()

versioneer.VCS = 'git'
versioneer.style = 'default'
versioneer.versionfile_source = 'src/xrd_ocl_kernels/_version.py'
versioneer.versionfile_build = 'xrd_ocl_kernels/_version.py'
versioneer.tag_prefix = 'xrd-ocl-kernels-v'
versioneer.parentdir_prefix = 'xrd-ocl-kernels-v'


################################################################################
# Dependencies
################################################################################
xrd_ocl_extension = Extension(
    'xrd_ocl_kernels.xrd_ocl',
    sources=['src/c-module/xrd_ocl/module.cpp'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=[
        '-std=c++11',
        '-fno-exceptions',
        ],
    extra_link_args=[
        '-framework', 'OpenCL'
        ],
    )

_version = versioneer.get_version()

setup(
    name = 'ocl_transforms_version',
    version = versioneer.get_version(True),
    license = 'LGPLv2',

    description = 'OpenCL implementation of some xrd kernels',
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',

    author = 'The HEXRD Development Team',
    author_email = 'praxes@googlegroups.com',
    url = 'http://xrd_transforms.readthedocs.org',

    ext_modules = [xrd_ocl_extension],
    packages = find_packages(where='src/', ),
    package_dir = { '': 'src'},

    include_package_data = True,
    zip_safe = False,
    
    classifiers = [
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],

    keywords=[
    ],

    install_requires = base_dependencies,

    entry_points = {
    },
    
    cmdclass = cmdclass,
)
