#include <stdio.h>
#include <OpenCL/opencl.h>


#define COUNT_OF(x) (sizeof(x)/sizeof(x[0]))

#define CL_CHECK(value,error_str,label) \
    if (CL_SUCCESS != value) { \
        PyErr_SetString(PyExc_RuntimeError, error_str);  \
    goto label; \
    } \

#define CL_CHECK_DEFAULT(value,label) \
    CL_CHECK(value, "CL function returned an error.", label)




typedef struct get_info_ctxt_struct
{
    void *tmp_buffer;
    size_t tmp_buffer_size;
} get_info_ctxt;

static inline void
get_info_ctxt_init(get_info_ctxt *ctxt)
{
    ctxt->tmp_buffer = NULL;
    ctxt->tmp_buffer_size = 0;
}

static inline void
get_info_ctxt_release(get_info_ctxt *ctxt)
{
    free(ctxt->tmp_buffer);
    ctxt->tmp_buffer = NULL;
    ctxt->tmp_buffer_size = 0;
}

/* tries to realloc the context to hold the new_size.
   on failure the old buffer with the old size is kept
*/
static void
get_info_ctxt_try_realloc(get_info_ctxt *ctxt, size_t new_size)
{
    const size_t BUFFER_MULTIPLE = 4096;
    new_size = BUFFER_MULTIPLE*(1+(new_size-1)/BUFFER_MULTIPLE);
    void *new_alloc = realloc(ctxt->tmp_buffer, new_size);
    if (NULL != new_alloc)
    {
        ctxt->tmp_buffer = new_alloc;
        ctxt->tmp_buffer_size = new_size;
    }

}

static inline void
get_info_ctxt_ensure_size(get_info_ctxt *ctxt, size_t requested_size)
{
    if (ctxt->tmp_buffer_size < requested_size)
    {
        get_info_ctxt_try_realloc(ctxt, requested_size);
    }
}

typedef enum cl_type_enum {
    CLT_NONE,
    CLT_NTSTRING,
    CLT_UINT,
    CLT_ULONG,
    CLT_SIZE_T,
    CLT_BOOL,
    
    CLT_SIZE_T_VECT,
    CLT_DEVICE_TYPE,
    CLT_FP_CONFIG,
    CLT_MEM_CACHE_TYPE,
    CLT_LOCAL_MEM_TYPE,
    CLT_EXEC_CAPABILITIES,
    CLT_COMMAND_QUEUE_PROPERTIES,
    CLT_PLATFORM_ID,
    CLT_DEVICE_ID,
    CLT_PARTITION_PROPERTIES,
    CLT_AFFINITY_DOMAIN,
} cl_type;

typedef struct cl_platform_info_entry_struct
{
    cl_platform_info param_id;
    const char *param_name;
    cl_type param_type;
} cl_platform_info_entry;

cl_platform_info_entry platform_params[] = {
    { CL_PLATFORM_PROFILE, "profile", CLT_NTSTRING },
    { CL_PLATFORM_VERSION, "version", CLT_NTSTRING },
    { CL_PLATFORM_NAME, "name", CLT_NTSTRING },
    { CL_PLATFORM_VENDOR, "vendor", CLT_NTSTRING },
    { CL_PLATFORM_EXTENSIONS, "extensions", CLT_NTSTRING },
};

static PyObject *
get_platform_param(cl_platform_id platform, cl_platform_info param,
                   cl_type type, get_info_ctxt *ctxt)
{
    PyObject *return_val = Py_None;
    switch(type) {
    case CLT_NTSTRING:
        {
            cl_int errval;
            size_t param_value_size;
            errval = clGetPlatformInfo(platform, param, 0, NULL,
                                       &param_value_size);
            if (CL_SUCCESS != errval)
                break;
            
            get_info_ctxt_ensure_size(ctxt, param_value_size);
            errval = clGetPlatformInfo(platform, param,
                                       ctxt->tmp_buffer_size,
                                       ctxt->tmp_buffer,
                                       NULL);
            if (CL_SUCCESS != errval)
                break;

            return_val = PyUnicode_FromString(static_cast<const char*>(ctxt->tmp_buffer));
        }
    default:
        break;
    }
    return return_val;
}

typedef struct cl_device_info_entry_struct
{
    cl_platform_info param_id;
    const char *param_name;
    cl_type param_type;
} cl_device_info_entry;

cl_device_info_entry device_params[] =
{
    { CL_DEVICE_TYPE, "type", CLT_DEVICE_TYPE },
    { CL_DEVICE_VENDOR_ID, "vendor_id", CLT_UINT },
    { CL_DEVICE_MAX_COMPUTE_UNITS, "max_compute_units", CLT_UINT },
    { CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "max_work_item_dimensions", CLT_UINT },
    { CL_DEVICE_MAX_WORK_ITEM_SIZES, "max_work_item_sizes", CLT_SIZE_T_VECT },
    { CL_DEVICE_MAX_WORK_GROUP_SIZE, "max_work_group_size", CLT_SIZE_T },

    { CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "preferred_vector_width_char", CLT_UINT },
    { CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "preferred_vector_width_short", CLT_UINT },
    { CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "preferred_vector_width_int", CLT_UINT },
    { CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "preferred_vector_width_long", CLT_UINT },
    { CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "preferred_vector_width_float", CLT_UINT },
    { CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "preferred_vector_width_double", CLT_UINT },
    { CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, "preferred_vector_width_half", CLT_UINT },

    { CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, "native_vector_width_char", CLT_UINT },
    { CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, "native_vector_width_short", CLT_UINT },
    { CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, "native_vector_width_int", CLT_UINT },
    { CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, "native_vector_width_long", CLT_UINT },
    { CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, "native_vector_width_float", CLT_UINT },
    { CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, "native_vector_width_double", CLT_UINT },
    { CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, "native_vector_width_half", CLT_UINT },

    { CL_DEVICE_MAX_CLOCK_FREQUENCY, "max_clock_frequency", CLT_UINT },

    { CL_DEVICE_ADDRESS_BITS, "address_bits", CLT_UINT },

    { CL_DEVICE_MAX_MEM_ALLOC_SIZE, "max_mem_alloc_size", CLT_ULONG },

    { CL_DEVICE_IMAGE_SUPPORT, "image_support", CLT_BOOL },

    { CL_DEVICE_MAX_READ_IMAGE_ARGS, "max_read_image_args", CLT_UINT },
    { CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "max_write_image_args", CLT_UINT },
    
    { CL_DEVICE_IMAGE2D_MAX_WIDTH, "image2d_max_width", CLT_SIZE_T },
    { CL_DEVICE_IMAGE2D_MAX_HEIGHT, "image2d_max_height", CLT_SIZE_T },
    { CL_DEVICE_IMAGE3D_MAX_WIDTH, "image3d_max_width", CLT_SIZE_T },
    { CL_DEVICE_IMAGE3D_MAX_HEIGHT, "image3d_max_height", CLT_SIZE_T },
    { CL_DEVICE_IMAGE3D_MAX_DEPTH, "image3d_max_depth", CLT_SIZE_T },

    { CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, "image_max_buffer_size", CLT_SIZE_T },
    { CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, "image_max_array_size", CLT_SIZE_T },

    { CL_DEVICE_MAX_SAMPLERS, "max_samplers", CLT_UINT },

    { CL_DEVICE_MAX_PARAMETER_SIZE, "max_parameter_size", CLT_SIZE_T },

    { CL_DEVICE_MEM_BASE_ADDR_ALIGN, "mem_base_addr_align", CLT_UINT },
    { CL_DEVICE_SINGLE_FP_CONFIG, "single_fp_config", CLT_FP_CONFIG },
    { CL_DEVICE_DOUBLE_FP_CONFIG, "double_fp_config", CLT_FP_CONFIG },

    { CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, "global_mem_cache_type", CLT_MEM_CACHE_TYPE },
    { CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "global_mem_cacheline_size", CLT_UINT },
    { CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "global_mem_cache_size", CLT_ULONG },
    { CL_DEVICE_GLOBAL_MEM_SIZE, "global_mem_size", CLT_ULONG },
    { CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "max_constant_buffer_size", CLT_ULONG },
    { CL_DEVICE_MAX_CONSTANT_ARGS, "max_constant_args", CLT_UINT },
    { CL_DEVICE_LOCAL_MEM_TYPE, "local_mem_type", CLT_LOCAL_MEM_TYPE },
    { CL_DEVICE_LOCAL_MEM_SIZE, "local_mem_size", CLT_ULONG },
    { CL_DEVICE_ERROR_CORRECTION_SUPPORT, "error_correction_support", CLT_BOOL },
    { CL_DEVICE_HOST_UNIFIED_MEMORY, "host_unified_memory", CLT_BOOL },

    /* page 68 */
    { CL_DEVICE_PROFILING_TIMER_RESOLUTION, "profiling_timer_resolution", CLT_SIZE_T },
    { CL_DEVICE_ENDIAN_LITTLE, "endian_little", CLT_BOOL },
    { CL_DEVICE_AVAILABLE, "available", CLT_BOOL },
    { CL_DEVICE_COMPILER_AVAILABLE, "compiler_available", CLT_BOOL },
    { CL_DEVICE_LINKER_AVAILABLE, "linker_available", CLT_BOOL },
    { CL_DEVICE_EXECUTION_CAPABILITIES, "execution_capabilities", CLT_EXEC_CAPABILITIES },
    { CL_DEVICE_QUEUE_PROPERTIES, "queue_properties", CLT_COMMAND_QUEUE_PROPERTIES },

    /* page 70 */
    { CL_DEVICE_BUILT_IN_KERNELS, "built_in_kernels", CLT_NTSTRING },
    { CL_DEVICE_PLATFORM, "platform", CLT_PLATFORM_ID },
    { CL_DEVICE_NAME, "name", CLT_NTSTRING },
    { CL_DEVICE_VENDOR, "vendor", CLT_NTSTRING },
    { CL_DRIVER_VERSION, "driver_version", CLT_NTSTRING },

    /* page 71 */
    { CL_DEVICE_PROFILE, "profile", CLT_NTSTRING },
    { CL_DEVICE_VERSION, "version", CLT_NTSTRING },

    /* page 72 */
    { CL_DEVICE_OPENCL_C_VERSION, "open_cl_C_version", CLT_NTSTRING },

    /* page 75 */
    { CL_DEVICE_EXTENSIONS, "extensions", CLT_NTSTRING },

    /* page 76 */
    { CL_DEVICE_PRINTF_BUFFER_SIZE, "printf_buffer_size", CLT_SIZE_T },
    { CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, "preferred_interop_user_sync", CLT_BOOL },
    { CL_DEVICE_PARENT_DEVICE, "parent_device", CLT_DEVICE_ID },
    { CL_DEVICE_PARTITION_MAX_SUB_DEVICES, "partition_max_sub_devices", CLT_UINT },
    { CL_DEVICE_PARTITION_PROPERTIES, "partition_properties", CLT_PARTITION_PROPERTIES },

    /* page 77 */
    { CL_DEVICE_PARTITION_AFFINITY_DOMAIN, "partition_affinity_domain", CLT_AFFINITY_DOMAIN },
    /* CL_DEVICE_PARTITION_TYPE ignored ... */
    { CL_DEVICE_REFERENCE_COUNT, "reference_count", CLT_UINT },
    
};


static struct {
    cl_device_fp_config bits;
    const char *name;
} fp_config_tags[] = {
    { CL_FP_DENORM, "denorm" },
    { CL_FP_INF_NAN, "inf_nan" },
    { CL_FP_ROUND_TO_NEAREST, "round_to_nearest" },
    { CL_FP_ROUND_TO_ZERO, "round_to_zero" },
    { CL_FP_ROUND_TO_INF, "round_to_inf" },
    { CL_FP_FMA, "fused multiply-add" },
    { CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, "correctly_rounded_divide_sqrt" },
    { CL_FP_SOFT_FLOAT, "soft float" },
};

static struct {
    cl_device_exec_capabilities bits;
    const char *name;
} exec_capabilities_tags[] = {
    { CL_EXEC_KERNEL, "exec_kernel" },
    { CL_EXEC_NATIVE_KERNEL, "exec_native_kernel" },
};

static struct {
    cl_command_queue_properties bits;
    const char *name;
} command_queue_properties_tags[] = {
    { CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "queue_out_of_order_exec_mode_enable" },
    { CL_QUEUE_PROFILING_ENABLE, "queue_profiling_enable" },
};

static struct {
    cl_device_affinity_domain bits;
    const char *name;
} affinities_tags[] = {
    { CL_DEVICE_AFFINITY_DOMAIN_NUMA, "numa" },
    { CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE, "l4_cache" },
    { CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE, "l3_cache" },
    { CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE, "l2_cache" },
    { CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE, "l1_cache" },
};

static PyObject *
get_device_param(cl_device_id device, cl_device_info param,
                 cl_type type, get_info_ctxt *ctxt)
{
    PyObject *return_val = Py_None;
    size_t actual_value_size;
    cl_int errval;

    errval = clGetDeviceInfo(device, param, 0, NULL, &actual_value_size);
    if (CL_SUCCESS != errval)
        goto done;
    get_info_ctxt_ensure_size(ctxt, actual_value_size);
    errval = clGetDeviceInfo(device, param, ctxt->tmp_buffer_size,
                             ctxt->tmp_buffer, NULL);
    if (CL_SUCCESS != errval)
        goto done;
    
    switch (type)
    {
    case CLT_DEVICE_TYPE:
        {
            cl_device_type the_value = *(cl_device_type *)ctxt->tmp_buffer;
            const char *device_type_name = "UNKNOWN";
            switch (the_value)
            {
            case CL_DEVICE_TYPE_CPU:
                device_type_name = "cpu";
                break;
            case CL_DEVICE_TYPE_GPU:
                device_type_name = "gpu";
                break;
            case CL_DEVICE_TYPE_ACCELERATOR:
                device_type_name = "accelerator";
                break;
            case CL_DEVICE_TYPE_CUSTOM:
                device_type_name = "custom";
                break;
            default:
                break;
            }
            return_val = PyUnicode_FromString(device_type_name);
        }
        break;

    case CLT_UINT:
        {
            cl_uint the_value = *(cl_uint *)ctxt->tmp_buffer;
            return_val = PyLong_FromSize_t((size_t)the_value);
        }
        break;

    case CLT_ULONG:
        {
            cl_ulong the_value = *(cl_ulong *)ctxt->tmp_buffer;
            return_val = PyLong_FromSize_t((size_t)the_value);
        }
        break;

    case CLT_SIZE_T:
        {
            size_t the_value = *(size_t *)ctxt->tmp_buffer;
            return_val = PyLong_FromSize_t(the_value);
        }
        break;

    case CLT_BOOL:
        {
            cl_bool the_value = *(cl_bool *)ctxt->tmp_buffer;
            return_val = PyBool_FromLong(the_value);
        }
        break;

    case CLT_SIZE_T_VECT:
        {
            size_t value_count = actual_value_size / sizeof(size_t);
            size_t *values = (size_t *) ctxt->tmp_buffer;
            size_t idx;
            
            return_val = PyTuple_New(value_count);
            for (idx = 0; idx < value_count; idx++)
            {
                PyObject *element;
                element =  PyLong_FromSize_t(values[idx]);
                PyTuple_SET_ITEM(return_val, idx, element);
            }
        }
        break;

    case CLT_NTSTRING:
        {
            return_val = PyUnicode_FromString(static_cast<const char*>(ctxt->tmp_buffer));
        }
        break;

    case CLT_PLATFORM_ID:
        {
            cl_platform_id the_value = *static_cast<cl_platform_id *>(ctxt->tmp_buffer);
            return_val = PyLong_FromSize_t(reinterpret_cast<size_t>(the_value));
        }
        break;

    case CLT_DEVICE_ID:
        {
            cl_device_id the_value = *static_cast<cl_device_id *>(ctxt->tmp_buffer);
            return_val = PyLong_FromSize_t(reinterpret_cast<size_t>(the_value));
        }
        break;

    case CLT_FP_CONFIG:
        {
            size_t tag;
            size_t tag_count = COUNT_OF(fp_config_tags);
            cl_device_fp_config fp_config = *(cl_device_fp_config *)ctxt->tmp_buffer;
            return_val = PyDict_New();
            for (tag = 0; tag < tag_count; tag++)
            {
                PyDict_SetItemString(return_val, fp_config_tags[tag].name,
                                 PyBool_FromLong(fp_config & fp_config_tags[tag].bits));
            }
        }
        break;
        
    case CLT_MEM_CACHE_TYPE:
        {
            const char *type_str = "UNKNOWN";
            cl_device_mem_cache_type type = *(cl_device_mem_cache_type *)ctxt->tmp_buffer;
            switch (type)
            {
            case CL_NONE:
                type_str = "none";
                break;
            case CL_READ_ONLY_CACHE:
                type_str = "read_only";
                break;
            case CL_READ_WRITE_CACHE:
                type_str = "read_write";
                break;
            default:
                break;
            }
            return_val = PyUnicode_FromString(type_str);
        }
        break;
        
    case CLT_LOCAL_MEM_TYPE:
        {
            const char *type_str = "UNKNOWN";
            cl_device_local_mem_type type = *(cl_device_local_mem_type *)ctxt->tmp_buffer;
            switch (type)
            {
            case CL_LOCAL:
                type_str = "local";
                break;
            case CL_GLOBAL:
                type_str = "global";
                break;
            case CL_NONE:
                type_str = "none";
                break;
            default:
                break;
            }
            return_val = PyUnicode_FromString(type_str);
        }
        break;

    case CLT_EXEC_CAPABILITIES:
        {
            size_t tag;
            size_t tag_count = COUNT_OF(exec_capabilities_tags);
            cl_device_exec_capabilities exec_capabilities =
                *(cl_device_exec_capabilities *)ctxt->tmp_buffer;
            return_val = PyDict_New();
            for (tag = 0; tag < tag_count; tag++)
            {
                PyObject *value = PyBool_FromLong(exec_capabilities &
                                                  exec_capabilities_tags[tag].bits);
                PyDict_SetItemString(return_val,
                                     exec_capabilities_tags[tag].name,
                                     value);
            }
        }
        break;

    case CLT_COMMAND_QUEUE_PROPERTIES:
        {
            size_t tag;
            size_t tag_count = COUNT_OF(command_queue_properties_tags);
            cl_command_queue_properties command_queue_properties =
                *(cl_command_queue_properties *)ctxt->tmp_buffer;
            return_val = PyDict_New();
            for (tag = 0; tag < tag_count; tag++)
            {
                PyObject *value = PyBool_FromLong(command_queue_properties &
                                                  command_queue_properties_tags[tag].bits);
                PyDict_SetItemString(return_val,
                                     command_queue_properties_tags[tag].name,
                                     value);
            }
        }
        break;
        
    case CLT_PARTITION_PROPERTIES:
        {
            size_t idx;
            size_t value_count = actual_value_size /
                sizeof(cl_device_partition_property);
            cl_device_partition_property *value_array =
                (cl_device_partition_property *) ctxt->tmp_buffer;

            return_val = PySet_New(NULL);

            for (idx = 0; idx < value_count; idx++)
            {
                const char *partition_kind;
                switch (value_array[idx]) {
                case CL_DEVICE_PARTITION_EQUALLY:
                    partition_kind = "equally";
                    break;
                case CL_DEVICE_PARTITION_BY_COUNTS:
                    partition_kind = "by_counts";
                    break;
                case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
                    partition_kind = "by_affinity_domain";
                    break;
                case 0:
                    partition_kind = NULL;
                    break;
                default:
                    partition_kind = "UNKNOWN";
                    break;
                }

                if (partition_kind)
                {
                    PyObject *value = PyUnicode_FromString(partition_kind);
                    PySet_Add(return_val, value);
                }
            }
        }
        break;
    case CLT_AFFINITY_DOMAIN:
        {
            size_t tag;
            size_t tag_count = COUNT_OF(affinities_tags);
            cl_device_affinity_domain affinity =
                *(cl_device_affinity_domain *)ctxt->tmp_buffer;
            return_val = PyDict_New();
            for (tag = 0; tag < tag_count; tag++)
            {
                PyObject *value = PyBool_FromLong(affinity &
                                                  affinities_tags[tag].bits);
                PyDict_SetItemString(return_val,
                                     affinities_tags[tag].name,
                                     value);
            }
        }
        break;
    default:
        break;
    }
  done:
    return return_val;
}

static PyObject*
describe_device(cl_device_id device, get_info_ctxt *ctxt)
{
    PyObject *device_dict = NULL;
    device_dict = PyDict_New();
    if (NULL == device_dict)
        return Py_None;

    /* set the device id */
    {
        PyObject *id_int = PyLong_FromSize_t((size_t) device);
        if (NULL == id_int)
        {
            id_int = Py_None;
        }

        PyDict_SetItemString(device_dict, "id", id_int);
        if (Py_None != id_int)
        {
            Py_DECREF(id_int);
        }
    }

    /* extract params into the dictionary */
    {
        size_t param_count = COUNT_OF(device_params);
        size_t param_idx;

        for (param_idx = 0; param_idx < param_count; param_idx++)
        {
            PyObject *param_value;
            const char *param_name = device_params[param_idx].param_name;
            cl_device_info param_id = device_params[param_idx].param_id;
            cl_type param_type = device_params[param_idx].param_type;

            param_value = get_device_param(device, param_id, param_type, ctxt);
            PyDict_SetItemString(device_dict, param_name, param_value);
            if (Py_None != param_value)
            {
                Py_DECREF(param_value);
            }
        }
    }

    return device_dict;
}

/*
 * return a dictionary describing the platform identified by id
 *
 * On any attribute for the platform that is consulted, a failure to obtain
 * a value will result in a 'None'.
 *
 * this will contain:
 * - id: the id
 * - name: the platform name
 * - version: 
 */
static PyObject *
describe_platform(cl_platform_id platform, get_info_ctxt *ctxt)
{
    PyObject *platform_dict = NULL;
    platform_dict = PyDict_New();
    if (NULL == platform_dict)
        return Py_None;

    /* set the actual platform id */
    {
        PyObject *id_int = PyLong_FromSize_t((size_t) platform);
        if (NULL == id_int)
        {
            id_int = Py_None; /* this should not happen */
        }
        
        PyDict_SetItemString(platform_dict, "id", id_int);
        if (Py_None != id_int)
        {
            Py_DECREF(id_int);
        }
    }

    /* extract params */
    {
        size_t param_count = COUNT_OF(platform_params);
        size_t param_idx;

        for (param_idx = 0; param_idx < param_count; param_idx++)
        {
            PyObject *param_value;
            const char *param_name = platform_params[param_idx].param_name;
            cl_platform_info param_id = platform_params[param_idx].param_id;
            cl_type param_type = platform_params[param_idx].param_type;
            
            param_value = get_platform_param(platform, param_id,
                                             param_type, ctxt);

            PyDict_SetItemString(platform_dict, param_name, param_value);
            if (Py_None != param_value)
            {
                Py_DECREF(param_value);
            }
        }
    }

    /* extract devices */
    {
        PyObject *device_list;
        cl_uint num_devices, device_idx;
        cl_device_id *devices;

        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        devices = static_cast<cl_device_id*>(calloc(num_devices, sizeof(cl_device_id)));
        if (NULL != devices)
        {
            /* highly unlikely that the previous allocation will fail. That would
               mean really low memory conditions. In this case just don't add the
               device list. We may 'fail' gracefully, but doom is incoming to this
               process.
            */
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
            device_list = PyList_New(num_devices);
            if (NULL != device_list)
            {
                /* Again, highly unlikely. If unable to create the list just
                   don't add device info. This code may fail gracefully, but
                   doom is impending.
                */
                for (device_idx = 0; device_idx < num_devices; device_idx++)
                {
                    PyObject *device_desc;
                    
                    device_desc = describe_device(devices[device_idx], ctxt);
                    PyList_SET_ITEM(device_list, (Py_ssize_t)device_idx, device_desc);
                    if (Py_None == device_desc)
                    {
                        Py_INCREF(device_desc);
                    }
                }                
                PyDict_SetItemString(platform_dict, "devices", device_list);
            }

            free(devices);
        }
    }
    return platform_dict;
}

/*
 * cl_get_info will return a list.
 * Each element in the list will describe a platform.
 * Each platform will be represented by a dictionary containing:
 *  ...
 * For each platform, a list of devices will be present.
 */
XRD_PYTHON_WRAPPER PyObject *
python_cl_get_info(PyObject *self)
{
    get_info_ctxt ctxt;
    cl_platform_id *platforms = NULL;
    cl_uint platform_count, platform_idx;
    cl_int cl_error = CL_SUCCESS;
    PyObject* result_list = NULL;

    get_info_ctxt_init(&ctxt);
    cl_error = clGetPlatformIDs(0, NULL, &platform_count);
    CL_CHECK_DEFAULT(cl_error, fail);

    if (platform_count > 0)
    {
        platforms = static_cast<cl_platform_id*>(calloc(platform_count, sizeof(cl_platform_id)));
        if (NULL == platforms)
        {
            PyErr_SetString(PyExc_RuntimeError, "Out of Memory?!?!");
            goto fail;
        }
        cl_error = clGetPlatformIDs(platform_count, platforms, &platform_count);
        CL_CHECK_DEFAULT(cl_error, fail);
    }

    result_list = PyList_New(platform_count);
    if (NULL == result_list)
        goto clean_up;
    
    for (platform_idx = 0; platform_idx < platform_count; platform_idx++)
    {
        PyObject *platform_description = Py_None;
        platform_description = describe_platform(platforms[platform_idx], &ctxt);
        PyList_SET_ITEM(result_list, (Py_ssize_t)platform_idx,
                        platform_description);

        if (Py_None == platform_description)
        {
            Py_INCREF(platform_description);
        }
    }

 clean_up:
    free(platforms);
    get_info_ctxt_release(&ctxt);
    
    return result_list;

 fail:
    Py_CLEAR(result_list);
    goto clean_up;
}
