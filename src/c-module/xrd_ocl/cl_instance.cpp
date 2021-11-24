#include "cl_instance.hpp"
#include "dev_help.hpp"
#include "utils.hpp"

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
    cl_device_id device;

    clGetPlatformIDs(1, &platform, NULL); // by default, the first platform
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // first GPU

    { TIME_SCOPE("clCreateContext");
        context = clCreateContext(0, 1, &device, error_notify_callback, NULL,
                                  NULL);
    }

    { TIME_SCOPE("clCreateCommandQueue");
        queue = clCreateCommandQueue(context, device, 0, NULL);
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

