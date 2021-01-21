import datetime as dt
import numpy as np
import ray_tracer_cpp
import matplotlib.pyplot as pl
tau = np.loadtxt("tau.txt",dtype=np.float32).reshape((47,82))
ssa = np.loadtxt("ssa.txt",dtype=np.float32).reshape((47,82))
g = 0.85
cld_mask = (tau > .1).astype(np.int32)
size = np.array([47,82], dtype=np.int32)
albedo = 0.
sza_rad = np.deg2rad(40.)
cloud_clear_frac = 0.99962515
k_null = 0.8 + 3e-4

n_photon = 10000000
sfc_dir = np.zeros(82, dtype=np.int32)
sfc_dif = np.zeros(82, dtype=np.int32)
ray_tracer_cpp.trace_rays_interface(tau, ssa, np.float32(g), cld_mask, size,
                                    np.float32(albedo), np.float32(sza_rad),
                                    np.float32(cloud_clear_frac), np.float32(k_null),
                                    np.int32(n_photon), sfc_dir, sfc_dif)

#print("-------------------")
#means = []
#stdvs = []
#nphot = [10000000] # (np.arange(1,10,1) * (10**np.arange(2,7,1))[:,np.newaxis]).flatten()
#
#for n_photon in nphot:
#    print(n_photon, type(g))
#    frac_photon_surf = []
#    for i in range(1):
#        sfc_dir = np.zeros(82, dtype=np.int32)
#        sfc_dif = np.zeros(82, dtype=np.int32)
#        ray_tracer_cpp.trace_rays_interface(tau, ssa, np.float32(g), cld_mask, size,
#                                            np.float32(albedo), np.float32(sza_rad),
#                                            np.float32(cloud_clear_frac), np.float32(k_null),
#                                            np.int32(n_photon), sfc_dir, sfc_dif)
#        frac_photon_surf += [(sfc_dir+sfc_dif).sum() / n_photon]
#    means += [np.mean(frac_photon_surf)]
#    stdvs += [np.std(frac_photon_surf)]
#
#means = np.array(means)
#stdvs = np.array(stdvs)
#
#fig,ax = pl.subplots(1, figsize=(10,8))
#ax.semilogx(nphot,means,c='C0')
#ax.fill_between(nphot, means+stdvs, means-stdvs, facecolor="C0", alpha=.3)
#ax.set_xlabel("Photon count", fontsize=16)
#ax.set_ylabel("Fraction of photons reaching the surface", fontsize=16)
#
#pl.savefig("convergence_test.png")
