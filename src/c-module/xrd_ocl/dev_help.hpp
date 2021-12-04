#ifndef XRD_OCL_DEV_HELP_HPP
#define XRD_OCL_DEV_HELP_HPP

#include <stdio.h>
// Configuration and declarations of development helper function
// Use the following defines to configure different kinds of internal logging...
//
// CLXF_LOG_COPY_CONVERT - LOGs related to copy_conversion of matrices
// CLXF_CL_LOG_LEVEL - LOG level of several opencl routines
//                     (0 - no log; 1 - on failure; 2 - always)
// CLXF_CL_LOG_KERNEL_INFO - LOG kernel info after successful build
// CLXF_LOG_TIMINGS - LOG timing information
// CLXF_ENABLE_TRACES - Enable output from TRACE macro
#if not defined(CLXF_LOG_COPY_CONVERT)
#  define CLXF_LOG_COPY_CONVERT 0
#endif

#if not defined(CLXF_CL_LOG_LEVEL)
#  define CLXF_CL_LOG_LEVEL 0
#endif

#if not defined(CLXF_CL_LOG_KERNEL_INFO)
#  define CLXF_CL_LOG_KERNEL_INFO 0
#endif

#if not defined(CLXF_LOG_TIMINGS)
#  define CLXF_LOG_TIMINGS 0
#endif

#if not defined(CLXF_ENABLE_TRACES)
#  define CLXF_ENABLE_TRACES 0
#endif


#define _CONCAT(a,b) a ## b
#define CONCAT(a,b) _CONCAT(a,b)
#define _STR(a) # a
#define STR(a) _STR(a)


// -----------------------------------------------------------------------------

#if defined(CLXF_LOG_TIMINGS) && CLXF_LOG_TIMINGS
#include <chrono>

class scope_timer
{
    typedef std::chrono::high_resolution_clock clock;
public:
    scope_timer(const char *name):tag(name), start(clock::now()) {}
    ~scope_timer()
    {
        double ellapsed =  std::chrono::duration_cast<std::chrono::microseconds>(clock::now()-start).count();
        printf("TIMER: %s: %9.3f ms ellapsed.\n", tag, 1e-3*ellapsed);
    }
private:
    const char *tag;
    clock::time_point start;
};
#  define TIME_SCOPE(name) scope_timer CONCAT(scptmr_,__LINE__)(name)

static inline void
cl_log_event_profile(const char* name, cl_event event)
{
    cl_ulong time_queued;
    cl_ulong time_submit;
    cl_ulong time_start;
    cl_ulong time_end;

    double factor = 1./1e+6;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(time_queued), &time_queued, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(time_submit), &time_submit, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    printf("TIMER (cl_event): %s: %6.3f ms (submit: %6.3f execute: %6.3f queued: %6.3f)\n",
           name, (time_end-time_submit)*factor, (time_start-time_submit)*factor,
           (time_end-time_start)*factor, (time_submit-time_queued)*factor);
}

#  define CL_LOG_EVENT_PROFILE(name, event) do { cl_log_event_profile(name, event); } while(0)
#else
#  define TIME_SCOPE(name) do {} while(0)
#  define CL_LOG_EVENT_PROFILE(...) do {} while(0)
#endif

// -----------------------------------------------------------------------------

#if defined(CLXF_ENABLE_TRACES) && CLXF_ENABLE_TRACES
#  define TRACE() \
    do { printf("%s::%d - %s\n", __FILE__, __LINE__, __FUNCTION__); } while(0)
#else
#  define TRACE() do {} while(0)
#endif

// -----------------------------------------------------------------------------

#if not defined(CLXF_CL_LOG_LEVEL) || 0 == CLXF_CL_LOG_LEVEL
#  define CL_LOG(...) do { } while(0)
#  define CL_LOG_CHECK(code) \
    do {                     \
        code;                \
    } while(0)
#else
#  define CL_LOG(level,...)                                             \
    do {                                                                \
        if (level <= CLXF_CL_LOG_LEVEL) {                               \
            printf("%s::%d: ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                        \
        }                                                               \
    } while(0)
#  define CL_LOG_CHECK(code)                                            \
          do {                                                          \
              cl_int cl_log_check_err = code;                           \
              if (CLXF_CL_LOG_LEVEL > 1 || cl_log_check_err != CL_SUCCESS) \
                  printf("%s::%d: %s - %s\n", __FILE__, __LINE__,       \
                         cl_error_to_str(cl_log_check_err), #code);     \
          } while(0)
#endif

#define CL_LOG_FAIL(...) CL_LOG(1, __VA_ARGS__)
#define CL_LOG_SUCCESS(...) CL_LOG(2, __VA_ARGS__)

#if defined(CLXF_CL_LOG_LEVEL) && 0 != CLXF_CL_LOG_LEVEL

#  define CLERR_TO_STR(x) case x: return #x
static
const char *cl_error_to_str(int cl_error)
{
    switch(cl_error)
    {
    CLERR_TO_STR(CL_SUCCESS);

    CLERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
    CLERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
    CLERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
    CLERR_TO_STR(CL_DEVICE_NOT_FOUND);
    CLERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
    CLERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
    CLERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    CLERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
    CLERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    CLERR_TO_STR(CL_INVALID_ARG_INDEX);
    CLERR_TO_STR(CL_INVALID_ARG_SIZE);
    CLERR_TO_STR(CL_INVALID_ARG_VALUE);
    CLERR_TO_STR(CL_INVALID_BINARY);
    CLERR_TO_STR(CL_INVALID_BUFFER_SIZE);
    CLERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
    CLERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
    CLERR_TO_STR(CL_INVALID_CONTEXT);
    CLERR_TO_STR(CL_INVALID_DEVICE);
    CLERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);
    CLERR_TO_STR(CL_INVALID_DEVICE_TYPE);
    CLERR_TO_STR(CL_INVALID_EVENT);
    CLERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
    CLERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
    CLERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
    CLERR_TO_STR(CL_INVALID_HOST_PTR);
    CLERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
    CLERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    CLERR_TO_STR(CL_INVALID_IMAGE_SIZE);
    CLERR_TO_STR(CL_INVALID_KERNEL);
    CLERR_TO_STR(CL_INVALID_KERNEL_ARGS);
    CLERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
    CLERR_TO_STR(CL_INVALID_KERNEL_NAME);
    CLERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
    CLERR_TO_STR(CL_INVALID_MEM_OBJECT);
    CLERR_TO_STR(CL_INVALID_OPERATION);
    CLERR_TO_STR(CL_INVALID_PLATFORM);
    CLERR_TO_STR(CL_INVALID_PROGRAM);
    CLERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
    CLERR_TO_STR(CL_INVALID_PROPERTY);
    CLERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
    CLERR_TO_STR(CL_INVALID_SAMPLER);
    CLERR_TO_STR(CL_INVALID_VALUE);
    CLERR_TO_STR(CL_INVALID_WORK_DIMENSION);
    CLERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
    CLERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
    CLERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
    CLERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
    CLERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
    CLERR_TO_STR(CL_MAP_FAILURE);
    CLERR_TO_STR(CL_MEM_COPY_OVERLAP);
    CLERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    CLERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    CLERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
    CLERR_TO_STR(CL_OUT_OF_RESOURCES);
    CLERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
    default:
        return "Unknown error code";
    }
}
#  undef CLERR_TO_STR
#endif

// -----------------------------------------------------------------------------

#endif // XRD_OCL_DEV_HELP_HPP
