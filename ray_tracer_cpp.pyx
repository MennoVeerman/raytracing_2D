#!python
#cython: language_level=3

import cython
import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "main.cpp":
    cdef void trace_ray(
        const float* tau, const float* ssa, const float g,
        const int* cld_mask, const int* size, const float albedo,
        const float sza_rad, const float cloud_clear_frac, const float k_null,
        const int n_photon, int* sfc_dir, int* sfc_dif)

@cython.boundscheck(True)
@cython.wraparound(False)
def trace_rays_interface(
        np.ndarray[float, ndim = 2, mode = "c"] tau not None,
        np.ndarray[float, ndim = 2, mode = "c"] ssa not None,
        float g,
        np.ndarray[int, ndim = 2, mode = "c"] cld_mask not None,
        np.ndarray[int, ndim = 1, mode = "c"] size not None,
        float albedo,
        float sza_rad,
        float cloud_clear_frac,
        float k_null,
        int n_photon,
        np.ndarray[int, ndim = 1, mode = "c"] sfc_dir not None,
        np.ndarray[int, ndim = 1, mode = "c"] sfc_dif not None):
    trace_ray(
            &tau[0,0], &ssa[0,0], g, &cld_mask[0,0], &size[0],
            albedo, sza_rad, cloud_clear_frac, k_null, n_photon,
            &sfc_dir[0], &sfc_dif[0])
    return None


