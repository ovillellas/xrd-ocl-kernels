


TRANSFORMS_CAPI_SRCS = $(wildcard src/c-module/_transforms_CAPI/*.[ch])
NEW_TRANSFORMS_CAPI_SRCS = $(wildcard src/c-module/new_transforms_capi/*.[ch])

.PHONY : all build install clean test
all: test


perf : install
	python perf/perf_angles_to_dvec.py 1000000 10000000 100000000

build : setup.py $(TRANSFORMS_CAPI_SRCS) $(NEW_TRANSFORMS_CAPI_SRCS)
	python setup.py build

install : build
	python setup.py install

clean :
	rm -rf __pycache__ build *.egg-info
	rm -rf src/*.egg-info

test : install
	python -m pytest -v
