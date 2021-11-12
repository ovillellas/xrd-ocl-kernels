
from .utils import ocl_info, list_devices
from .xrd_ocl import cl_gvec_to_xy as gvec_to_xy


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
