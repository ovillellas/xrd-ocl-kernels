#include "cl_instance.hpp"
#include "dev_help.hpp"
#include "utils.hpp"

#define CLXF_STAGING_BUFFER_SIZE ((size_t)4*1024*1024)


cl_kernel kernel_cache[cl_instance::kernel_slot::count];
static cl_instance *the_instance;

cl_kernel
cl_instance::build_kernel(const char *kernel_name, const char *source,
                          const char *compile_options)
{
    TIME_SCOPE("build_kernel");
    cl_program program;
    cl_kernel kernel;

    program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);

    // will build errors get notified via the context callback?
    clBuildProgram(program, 0, NULL, compile_options, NULL, NULL);
    #if 0
    // this code would show the build log. Activate if it is not actually shown
    // via the callback
    if (CL_SUCCESS != err) {
        if (CL_BUILD_PROGRAM_FAILURE == err)
        {
            size_t log_size;
            clGetProgramBuildInfo(state.program, state.device,
                                  CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char*)malloc(log_size);
            if (NULL != log)
            {
                clGetProgramBuildInfo(state.program, state.device,
                                      CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                printf("Build log:\n%s\n", log);
                free(log);
            }
            else
                printf("no memory to show log.\n");
        }
    }
    #endif
    kernel = clCreateKernel(program, kernel_name, NULL);
    if (!kernel)
    {
        CL_LOG_FAIL("kernel '%s' failed to build\n", kernel_name);
    }
    else
    {
        CL_LOG_SUCCESS("kernel '%s' successfully built as %p\n",
                       kernel_name, kernel);

#if defined(CLXF_CL_LOG_KERNEL_INFO) && CLXF_CL_LOG_KERNEL_INFO
        if (kernel)
        {
            size_t global_work_size[3], compile_work_group_size[3],
                work_group_size, preferred_work_group_size_multiple;
            cl_ulong local_mem_size, private_mem_size;
            CL_LOG_CHECK(clGetKernelWorkGroupInfo(kernel, cl->device,
                                                   CL_KERNEL_GLOBAL_WORK_SIZE,
                                                   sizeof(global_work_size),
                                                   global_work_size, NULL));
            CL_LOG_CHECK(clGetKernelWorkGroupInfo(kernel, cl->device,
                                                  CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                                  sizeof(compile_work_group_size),
                                                  compile_work_group_size, NULL));
            CL_LOG_CHECK(clGetKernelWorkGroupInfo(kernel, cl->device,
                                                  CL_KERNEL_WORK_GROUP_SIZE,
                                                  sizeof(work_group_size),
                                                  &work_group_size, NULL));
            CL_LOG_CHECK(clGetKernelWorkGroupInfo(kernel, cl->device,
                                                  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                                  sizeof(preferred_work_group_size_multiple),
                                                  &preferred_work_group_size_multiple, NULL));
            CL_LOG_CHECK(clGetKernelWorkGroupInfo(kernel, cl->device,
                                                  CL_KERNEL_LOCAL_MEM_SIZE,
                                                  sizeof(local_mem_size),
                                                  &local_mem_size, NULL));
            CL_LOG_CHECK(clGetKernelWorkGroupInfo(kernel, cl->device,
                                                  CL_KERNEL_PRIVATE_MEM_SIZE,
                                                  sizeof(private_mem_size),
                                                  &private_mem_size, NULL));
            printf("Kernel built for gvec_to_xy <%s>:\n", floating_kind_name<REAL>());
            print_dims("\tglobal_work_size", global_work_size, 3);
            print_dims("\tcompile_work_group_size", compile_work_group_size, 3);
            printf("\twork_group_size - max: %zd preferred: %zd\n",
                   work_group_size, preferred_work_group_size_multiple);
            printf("\tmem_size - private: %llu local: %llu\n",
                   private_mem_size, local_mem_size);
        }
#endif
    }
    clReleaseProgram(program);

    return kernel;
}

cl_kernel
cl_instance::get_kernel(kernel_slot slot)
{
    if (slot >= 0 && slot < kernel_slot::count)
        return kernel_cache[slot];

    CL_LOG(1, "get_kernel at %d [out of range].\n", slot);
    return 0;
}

void
cl_instance::set_kernel(kernel_slot slot, cl_kernel kernel)
{
    if (slot >= 0 && slot < kernel_slot::count)
    {
        CL_LOG(2, "set_kernel at %d to %p (was %p).\n", slot, kernel_cache[slot], kernel);
        if (kernel_cache[slot])
            CL_LOG_CHECK(clReleaseKernel(kernel_cache[slot]));

        if (kernel)
            CL_LOG_CHECK(clRetainKernel(kernel));
        kernel_cache[slot] = kernel;
    }
    else
    {
        CL_LOG(1, "set_kernel at %d [out of range].\n", slot);
    }
}

cl_instance::cl_instance()
{
    zap_to_zero(*this);
}

cl_instance::~cl_instance()
{
    for (size_t i = 0; i < sizeof(kernel_cache)/sizeof(kernel_cache[0]); i++)
    {
        if (kernel_cache[i])
            CL_LOG_CHECK(clReleaseKernel(kernel_cache[i]));
    }
    if (staging_buffer)
        CL_LOG_CHECK(clReleaseMemObject(staging_buffer));
    if (mem_queue)
        CL_LOG_CHECK(clReleaseCommandQueue(mem_queue));
    if (queue)
        CL_LOG_CHECK(clReleaseCommandQueue(queue));
    if (context)
        CL_LOG_CHECK(clReleaseContext(context));

    zap_to_zero(kernel_cache);
}

static void
error_notify_callback(const char *errinfo, const void* private_info, size_t cb,
                      void *user_data)
{
    if (0 == strncmp(errinfo, "OpenCL Build Warning", 20))
    {
        // notify when logging successes and errors
        CL_LOG(2, "%s", errinfo);
    }
    else
    {
        // notify only when logging errors.
        CL_LOG(1, "CLERROR: %s\n", errinfo);
    }
}

bool cl_instance::init()
{
    TIME_SCOPE("cl_instance::init");
    cl_platform_id platform;

    clGetPlatformIDs(1, &platform, NULL); // by default, the first platform
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // first GPU

    { TIME_SCOPE("Create CL context");
        context = clCreateContext(0, 1, &device, error_notify_callback, NULL,
                                  NULL);
    }

    { TIME_SCOPE("Create CL command queue");
        queue = clCreateCommandQueue(context, device, 0, NULL);
    }

    { TIME_SCOPE("Create memory transfer channel");
        mem_queue = clCreateCommandQueue(context, device, 0, NULL);

        staging_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
                                        CLXF_STAGING_BUFFER_SIZE, NULL, NULL);
    }
    return context && queue;
}

cl_instance *cl_instance::instance()
{
    if (!the_instance) {
        the_instance = new cl_instance();
        if (!the_instance->init())
        {
            shutdown();
        }
    }

    return the_instance;
}


// shutting down will lead to the context held by this to be released.
// Everything should be ok as long as everything that could survive a shutdown
// is independently retained.
void cl_instance::shutdown()
{
    if (the_instance) {
        delete(the_instance);
        the_instance = 0;
    }
}


size_t
cl_instance::device_max_alloc_size() const
{

    cl_ulong allocsize;
    CL_LOG_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                 sizeof(allocsize), &allocsize, NULL));
    return static_cast<size_t>(allocsize);
}

size_t
cl_instance::device_global_mem_size() const
{
    cl_ulong memsize;
    CL_LOG_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                                 sizeof(memsize), &memsize, NULL));
    return static_cast<size_t>(memsize);
}

size_t
cl_instance::device_max_compute_units() const
{
    cl_uint max_cu;
    CL_LOG_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(max_cu), &max_cu, NULL));
    return static_cast<size_t>(max_cu);
}

bool
cl_instance::device_host_unified_memory() const
{
    cl_bool unifiedmemoryp;
    CL_LOG_CHECK(clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY,
                                 sizeof(unifiedmemoryp), &unifiedmemoryp, NULL));
    return unifiedmemoryp != CL_FALSE;
}

bool
cl_instance::device_compiler_available() const
{
    cl_bool compilerp;
    CL_LOG_CHECK(clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE,
                                 sizeof(compilerp), &compilerp, NULL));
    return compilerp != CL_FALSE;
}

size_t
cl_instance::kernel_preferred_workgroup_size_multiple(cl_kernel kernel) const
{
    size_t preferred_work_group_size_multiple;
    CL_LOG_CHECK(clGetKernelWorkGroupInfo(kernel, device,
                                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                          sizeof(preferred_work_group_size_multiple),
                                          &preferred_work_group_size_multiple, NULL));
    return preferred_work_group_size_multiple;
}
