#ifndef XRD_OCL_CL_INSTANCE_HPP
#define XRD_OCL_CL_INSTANCE_HPP

class cl_instance
{
public:
    enum kernel_slot {
        gvec_to_xy_f64 = 0,
        gvec_to_xy_f32,
        count,
        invalid = count
    };

    cl_kernel build_kernel(const char *kernel_name, const char *sources,
                           const char *compile_options);
    cl_kernel get_kernel(kernel_slot slot);
    void set_kernel(kernel_slot slot, cl_kernel kernel);
    
    static cl_instance *instance();
    static void shutdown();

    // this are kept as public, as there is no point on hiding handles
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
private:
    bool init();
    cl_instance();
    ~cl_instance();
};

#endif // XRD_OCL_CL_STATE_HPP
