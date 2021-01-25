import datetime as dt
import numpy as np
import ray_tracer_cpp
import matplotlib.pyplot as pl
import multiprocessing as mp
import time
def run_raytracer(i, nphot):
    np.random.seed(i)
    tau = np.loadtxt("tau.txt",dtype=np.float32).reshape((47,82))
    ssa = np.loadtxt("ssa.txt",dtype=np.float32).reshape((47,82))
    g = 0.85
    cld_mask = (tau > .1).astype(np.int32)
    size = np.array([47,82], dtype=np.int32)
    albedo = 0.
    sza_rad = np.deg2rad(40.)
    cloud_clear_frac = 0.99962515
    k_null = 0.8 + 3e-4
    
    print("-------------------")
    frac_photon_surf = []
    for iphoton in range(len(nphot)):
        n_photon = nphot[iphoton]
        sfc_dir = np.zeros(82, dtype=np.int32)
        sfc_dif = np.zeros(82, dtype=np.int32)
        start = dt.datetime.now()
        ray_tracer_cpp.trace_rays_interface(tau, ssa, np.float32(g), cld_mask, size,
                                            np.float32(albedo), np.float32(sza_rad),
                                            np.float32(cloud_clear_frac), np.float32(k_null),
                                            np.int32(n_photon), sfc_dir, sfc_dif, i)
        frac_photon_surf += [(sfc_dir+sfc_dif).sum() / n_photon]
        end = dt.datetime.now()
        if i==0:
            print(n_photon, end-start)
    np.save("results_%s.npy"%i, np.array(frac_photon_surf))

#nphot = (np.arange(1,10,2) * (10**np.arange(2,7,1))[:,np.newaxis]).flatten()
#nphot = np.append(nphot, 10000000)
nphot=np.array([10000000])

niter = 1
procs = []
for i in range(niter):
    procs += [mp.Process(target=run_raytracer, args = (i, nphot))]
for i in procs:
    i.start()
    time.sleep(.1)
for i in procs:
    i.join()

results = np.zeros((niter, len(nphot)))
for i in range(niter):
    results[i,:] = np.load("results_%s.npy"%i)
means = np.mean(results,axis=0)    
stdvs = np.std(results,axis=0)    
fig,ax = pl.subplots(1, figsize=(10,8))
ax.semilogx(nphot, means,c='C0')
ax.fill_between(nphot, means+stdvs, means-stdvs, facecolor="C0", alpha=.3)
ax.set_xlabel("Photon count", fontsize=16)
ax.set_ylabel("Fraction of photons reaching the surface", fontsize=16)

pl.savefig("convergence_test.png")
