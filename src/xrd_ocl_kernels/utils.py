from . import xrd_ocl

_ocl_info = None

def ocl_info():
    global _ocl_info
    if _ocl_info is None:
        _ocl_info = xrd_ocl.cl_get_info()

    return _ocl_info


def list_devices():
    info = ocl_info()

    devices = []
    for pl_idx, platform in enumerate(info):
        if platform['profile'] == 'FULL_PROFILE':
            for dev_idx, device in enumerate(platform['devices']):
                typ = device['type']
                name = device['name']
                ext = device['extensions'].split()
                fp = ['single']
                if 'cl_khr_fp64' in ext:
                    fp.append('double')

                devices.append(((pl_idx, dev_idx), typ, name, tuple(fp)))

    return devices
    
