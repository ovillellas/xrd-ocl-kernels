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


    // some helpers to get important data
    size_t device_max_alloc_size() const;
    size_t device_global_mem_size() const;
    size_t device_max_compute_units() const;
    bool   device_host_unified_memory() const;
    bool   device_compiler_available() const;
    
    //
    size_t kernel_preferred_workgroup_size_multiple(cl_kernel kernel) const;
    // this are kept as public, as there is no point on hiding handles and
    // not everything is hidden.
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_command_queue mem_queue;
    cl_mem staging_buffer;
private:
    bool init();
    cl_instance();
    ~cl_instance();
};

#endif // XRD_OCL_CL_STATE_HPP
