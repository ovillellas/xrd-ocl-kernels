/*
 * %BEGIN LICENSE HEADER
 * %END LICENSE HEADER
 */


/* =============================================================================
 * THIS_MODULE_NAME is used in several places in macros to generate the module
 * entry point for Python 2 and Python 3, as well as for the internal module
 * name exported to Python.
 *
 * Having this as a define makes it easy to change as needed (only a single
 * place to modify
 *
 * Note the supporting CONCAT and STR macros...
 * =============================================================================
 */
#define THIS_MODULE_NAME xrd_ocl

#define _CONCAT(a,b) a ## b
#define CONCAT(a,b) _CONCAT(a,b)
#define _STR(a) # a
#define STR(a) _STR(a)

/* ************************************************************************** */


#include "compiler.hpp"
extern "C" {
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
}

#include <OpenCL/opencl.h>



/* =============================================================================
 * Note: As we really want to only the entry point of the module to be visible
 * in the exported library, but there is a clear convenience on having different
 * functions in their own source file, the approach will be to include use this
 * file as the compile unit and import the source code of all the functions.
 *
 * This file will also contain all the module scaffolding needed (like the init
 * function as well as the exported method list declaration.
 * =============================================================================
 */



/* =============================================================================
 * This macros configure the way the module is built.
 * All implementation files included in a single compile unit.
 * Python wrappers are to be included.
 * Visibility for all functions are wrappers is removed by turning them into
 * statics.
 * =============================================================================
 */

#define XRD_SINGLE_COMPILE_UNIT 1
#define XRD_INCLUDE_PYTHON_WRAPPERS 1
#define XRD_CFUNCTION static inline
#define XRD_PYTHON_WRAPPER static
/*
#include "transforms_types.h"
#include "transforms_utils.h"
#include "transforms_prototypes.h"
#include "checks.h"
#include "transforms_task_system.h"


#include "angles_to_gvec.c"
#include "angles_to_dvec.c"
#include "gvec_to_xy.c"
#include "xy_to_gvec.c"
#include "oscill_angles_of_HKLs.c"
#include "unit_row_vector.c"
*/
/* #include "make_detector_rmat.c" */

/*#include "make_sample_rmat.c"
#include "make_rmat_of_expmap.c"
#include "make_binary_rmat.c"
#include "make_beam_rmat.c"
#include "validate_angle_ranges.c"
#include "rotate_vecs_about_axis.c"
#include "quat_distance.c"
*/

#include "cl_support.cpp"
#include "cl_gvec_to_xy.cpp"
/* =============================================================================
 * Module initialization
 * =============================================================================
 */

#define EXPORT_METHOD_NA(name) \
    { STR(name), reinterpret_cast<PyCFunction>(CONCAT(python_, name)), METH_NOARGS, NULL } 

#define EXPORT_METHOD_VA(name) \
    { STR(name), reinterpret_cast<PyCFunction>(CONCAT(python_, name)), METH_VARARGS, NULL }

#define EXPORT_METHOD_VAKW(name) \
    { STR(name), reinterpret_cast<PyCFunction>(CONCAT(python_, name)), METH_VARARGS | METH_KEYWORDS, NULL }

#define EXPORT_METHOD_SENTINEL() \
    { NULL, NULL, 0, NULL }


static PyMethodDef _module_methods[] = {
    EXPORT_METHOD_NA(cl_get_info),
    EXPORT_METHOD_VAKW(cl_gvec_to_xy),
    EXPORT_METHOD_SENTINEL()
};

/*
 * In Python 3 the entry point for a C module changes slightly, but both can
 * be supported with little effort with some conditionals and macro magic
 */

#if PY_VERSION_HEX < 0x03000000
#  error Only Python 3+ is supported.
#endif

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        STR(THIS_MODULE_NAME),
        NULL,
        -1,
        _module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

extern "C" {
PyMODINIT_FUNC
CONCAT(PyInit_, THIS_MODULE_NAME)(void)
{
    PyObject *module;

    module = PyModule_Create(&moduledef);

    if (NULL != module)
    {
        /* add any extra initialization */    
        import_array();
    }
    return module;
}
}

