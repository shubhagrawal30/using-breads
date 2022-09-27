from posixpath import dirname
import sys
sys.path.append("/scr3/jruffio/shubh/breads")
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as pyfits
from glob import glob
from pathlib import Path
import multiprocessing as mp
import h5py
from scipy.interpolate import RegularGridInterpolator
import time
import arguments as args

from breads.instruments.OSIRIS import OSIRIS
from breads.fit import log_prob

import emcee
import corner

print("making subdirectories")
Path("./plots/thesis/hd148352/").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass
    
    star = "ROXs35A"
    # star = "HD148352"
    sky_calib_file = args.sky_calib_file[star]
    numthreads = 16
    # dir_name = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"
    # filename = "s210626_a032009_Kn5_020.fits"
    dir_name = args.dir_name[star]
    filename = "s210628_a009004_Kn5_020.fits"
    dataobj = OSIRIS(dir_name+filename)
    nz,ny,nx = dataobj.data.shape
    # dataobj.noise = np.sqrt(np.abs(dataobj.data))
    print("setting noise")
    dataobj.set_noise()
    # dataobj.noise = np.ones((nz,ny,nx))

    print("sky calibrating")
    dataobj.calibrate(sky_calib_file)

    spec_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
    print("Reading spectrum file", spec_file)
    with pyfits.open(spec_file) as hdulist:
        star_spectrum = hdulist[2].data
        mu_x = hdulist[3].data
        mu_y = hdulist[4].data
        sig_x = hdulist[5].data
        sig_y = hdulist[6].data
    print("setting reference position")
    dataobj.set_reference_position((np.nanmedian(mu_y), np.nanmedian(mu_x)))
    # dataobj.set_reference_position((2, 2))

    mypool = mp.Pool(processes=numthreads)

    planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-4.0-0.0a+0.0.BT-Settl.spec.7"

    # Define planet model grid from BTsettl
    minwv,maxwv= np.min(dataobj.wavelengths),np.max(dataobj.wavelengths)
    with h5py.File("/scr3/jruffio/code/BREADS_osiris_survey_scripts/bt-settl_K-band_2000-4000K_OSIRIS.hdf5", 'r') as hf:
        grid_specs = np.array(hf.get("spec"))
        grid_temps = np.array(hf.get("temps"))
        grid_loggs = np.array(hf.get("loggs"))
        grid_wvs = np.array(hf.get("wvs"))
    crop_grid = np.where((grid_wvs > minwv - 0.02) * (grid_wvs < maxwv + 0.02))
    grid_wvs = grid_wvs[crop_grid]
    grid_specs = grid_specs[:,:,crop_grid[0]]
    myinterpgrid = RegularGridInterpolator((grid_temps,grid_loggs),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
    
    # spec_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
    # print("Reading spectrum file", spec_file)
    # with pyfits.open(spec_file) as hdulist:
    #     star_spectrum = hdulist[2].data
    
    tr_dir = args.tr_dir[star]#"/scr3/jruffio/data/osiris_survey/targets/SR3/210626/first/reduced/spectra/s210626_a025"+filename[12:-13]+"_Kn5_020_spectrum.fits"
    tr_files = os.listdir(tr_dir)
    if "plots" in tr_files:
        tr_files.remove("plots")
    tr_file = tr_dir + tr_files[0]
    print("Reading transmission file", tr_file)
    with pyfits.open(tr_file) as hdulist:
        transmission = hdulist[0].data

    mypool.close()
    mypool.join()

    snr = dir_name + "planets/20220512_1700K/" + filename[:-5] + "_out.fits"
    linparas = pyfits.open(snr)[3].data
    linparas_err = pyfits.open(snr)[4].data
    val = linparas[0,:,:,0]/linparas_err[0,:,:,0]
    val[val>5] = np.nan
    val[val<0] = np.nan
    Y, X = np.unravel_index(np.nanargmax(val), val.shape)
    print(Y, X, val[Y, X])
    # plt.figure()
    # plt.imshow(val, origin="lower")
    # plt.plot(Y, X)
    # plt.show()
    # exit()

    # Definition of the (extra) parameters for fm
    from breads.fm.hc_atmgrid_hpffm import hc_atmgrid_hpffm
    fm_paras = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"transmission":transmission,"star_spectrum":star_spectrum,
                "boxw":3,"psfw":1.2,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40,"loc":(-4, 1)}
    fm_func = hc_atmgrid_hpffm
    nonlin_labels = ["Teff", "logg", "spin", "RV"]
    nonlin_paras_mins = np.array([2000, 3.5, 0, -50])
    nonlin_paras_maxs = np.array([4000, 5.5, 50, 50])

    # /!\ Optional but recommended
    # Test the forward model for a fixed value of the non linear parameter.
    # Make sure it does not crash and look the way you want
    if 1:
        nonlin_paras = [2500,4.0,0,0] # x (pix),y (pix), rv (km/s)
        # d is the data vector a the specified location
        # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
        # s is the vector of uncertainties corresponding to d
        d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)

        validpara = np.where(np.sum(M,axis=0)!=0)
        M = M[:,validpara[0]]
        d = d / s
        M = M / s[:, None]
        from scipy.optimize import lsq_linear
        paras = lsq_linear(M, d).x
        m = np.dot(M,paras)

        plt.subplot(2,1,1)
        plt.plot(d,label="data")
        plt.plot(m,label="model")
        plt.plot(paras[0]*M[:,0],label="planet model")
        plt.plot(m-paras[0]*M[:,0],label="starlight model")
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
        for k in range(M.shape[-1]-1):
            plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+1))
        plt.legend()
        plt.show()


    nwalkers = 512
    nsteps = 5
    ndim = np.size(nonlin_paras_mins)
    p0 = np.random.rand(nwalkers, ndim) * (nonlin_paras_maxs-nonlin_paras_mins)[None,:] + nonlin_paras_mins[None,:]
    print(np.nanmedian(p0, axis=0))

    def nonlin_lnprior_func(nonlin_paras):
        for p, _min, _max in zip(nonlin_paras, nonlin_paras_mins, nonlin_paras_maxs):
            if p > _max or p < _min:
                return -np.inf
        return 0

    os.environ["OMP_NUM_THREADS"] = "1"
    # Caution: Parallelization in emcee can make it much slower than sequential. You should run some tests to make sure
    # what the optimal number of processes is or if sequential is just better.
    mypool = mp.Pool(processes=4)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[dataobj, fm_func, fm_paras,nonlin_lnprior_func],pool=mypool)
    # print(log_prob(p0[0],dataobj, fm_func, fm_paras))

    # Run and time burnout
    start = time.time()
    state = sampler.run_mcmc(p0, nsteps, progress=True)
    end = time.time()
    print("burnout over // time {0}s".format(end-start))

    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)

    samples = sampler.get_chain(flat=True)
    samples_gc = samples[:, 0]
    mypool.close()
    mypool.join()

    # hdulist = pyfits.HDUList()
    # hdulist.append(pyfits.PrimaryHDU(data=samples,
    #     header=pyfits.Header(cards={"TYPE": "samples"})))                                  
    # try:
    #     hdulist.writeto("./plots/thesis/roxs35a/corner_1000_-4_1.fits", overwrite=True)
    # except TypeError:
    #     hdulist.writeto("./plots/thesis/roxs35a/corner_1000_-4_1.fits", clobber=True)
    # hdulist.close()

    figure = corner.corner(samples, labels=nonlin_labels)
    # plt.savefig("./plots/thesis/roxs35a/corner_1000_-4_1.png")
    plt.show()
