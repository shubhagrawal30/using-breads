print("starting")
import sys
from ipywidgets.widgets.widget_output import Output
sys.path.append("/scr3/jruffio/shubh/breads/")
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as pyfits
from pathlib import Path
from breads.fit import fitfm
from copy import deepcopy
import multiprocessing as mp
from itertools import repeat
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor as Pool

numthreads = 16
boxw = 3

print("Importing mkl")
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

print("Importing breads")
from breads.instruments.OSIRIS import OSIRIS
from breads.grid_search import grid_search
from breads.fm.hc_splinefm import hc_splinefm
from breads.fm.hc_mask_splinefm import hc_mask_splinefm
from breads.fm.hc_hpffm import hc_hpffm
from breads.injection import inject_planet, read_planet_info
import arguments

star = "SR3"
dir_name = arguments.dir_name[star]
tr_dir = arguments.tr_dir[star]
sky_calib_file = arguments.sky_calib_file[star]
files = os.listdir(dir_name)

subdirectory = "throughput/20210223/"
print("making subdirectories")
Path(dir_name+subdirectory+"plots/").mkdir(parents=True, exist_ok=True)

print("Reading planet file")
planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
model_wvs = arr[:, 0] / 1e4
model_spec = 10 ** (arr[:, 1] - 8)

tr_files = os.listdir(tr_dir)
if "plots" in tr_files:
    tr_files.remove("plots")
tr_counter = 0
tr_total = len(tr_files)

def one_location(args):
    dataobj, location, indices, planet_f, spec_file, transmission, flux_ratio, dat, filename = args
    if 1:
        dataobj.set_noise()
        log_prob,log_prob_H0,rchi2,linparas_b,linparas_err = grid_search([rvs,[location[0]],[location[1]]],dataobj,fm_func,fm_paras,numthreads=numthreads)
        dataobj.data = deepcopy(dat)
        # plt.figure()
        # plt.imshow(dataobj.data[0])
        inject_planet(dataobj, location, planet_f, spec_file, transmission, flux_ratio)
        # plt.figure()
        # plt.imshow(dataobj.data[0])
        # plt.show()
        dataobj.set_noise()

        if False: # Example code to test the forward model
            nonlin_paras = [0,0,0] # rv (km/s), y (pix), x (pix)
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

            print("plotting")

            plt.subplot(2,1,1)
            plt.plot(d,label="data")
            plt.plot(m,label="model")
            plt.plot(paras[0]*M[:,0],label="planet model")
            plt.plot(m-paras[0]*M[:,0],label="starlight model")
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
            for k in range(M.shape[-1]-1):
                print(k)
                plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+1))
            plt.legend()
            plt.show()
            plt.close('all')
            exit()

        print("SNR time", location)
        log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,[location[0]],[location[1]]],dataobj,fm_func,fm_paras,numthreads=numthreads)
        N_linpara = linparas.shape[-1]
        return indices, linparas[0,0,0,0] - linparas_b[0,0,0,0], linparas_err[0,0,0,0]
    # except Exception as e:
    #     print(e)
    #     print("FAILED", filename, location)
    #     return indices, np.nan, np.nan

for filename in files[:]:
    rvs = np.array([0])
    ys = np.arange(-2, 3)
    xs = np.arange(-2, 3)
    flux = np.zeros((len(ys), len(xs))) * np.nan
    noise = np.zeros((len(ys), len(xs))) * np.nan
    if ".fits" not in filename:
        print("skipping ", filename)
        continue
    print(filename)
    dataobj = OSIRIS(dir_name+filename) 
    nz,ny,nx = dataobj.data.shape

    print("sky calibrating")
    dataobj.calibrate(sky_calib_file)

    print("compute stellar PSF")
    data = dataobj.data
    nz, ny, nx = data.shape
    stamp_y, stamp_x = (boxw-1)//2, (boxw-1)//2
    img_mean = np.nanmedian(data.data, axis=0)
    star_y, star_x = np.unravel_index(np.nanargmax(img_mean), img_mean.shape)
    stamp = data[:, star_y-stamp_y:star_y+stamp_y+1, star_x-stamp_x:star_x+stamp_x+1]
    total_flux = np.sum(stamp)
    stamp = stamp/np.nansum(stamp,axis=(1,2))[:,None,None]

    spec_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
    print("Reading spectrum file", spec_file)
    with pyfits.open(spec_file) as hdulist:
        star_spectrum = hdulist[2].data
        mu_x = hdulist[3].data
        mu_y = hdulist[4].data
        # sig_x, sig_y = 1, 1
        sig_x, sig_y = np.nanmedian(hdulist[5].data), np.nanmedian(hdulist[6].data)

    print("setting reference position")
    dataobj.set_reference_position((np.nanmedian(mu_y), np.nanmedian(mu_x)))
    print(dataobj.refpos)

    # tr_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
    # SR3
    tr_counter = (tr_counter + 1) % tr_total
    # tr_file = "/scr3/jruffio/data/osiris_survey/targets/HIP73049/210628/reduced/spectra/s210628_a004" \
    #     + format(tr_counter+4, '03d') + "_Kn5_020_spectrum.fits"
    tr_file = tr_dir + tr_files[tr_counter]

    # +filename[12:-13]
    print("Reading transmission file", tr_file)
    with pyfits.open(tr_file) as hdulist:
        transmission = hdulist[0].data

    print("Removing bad pixels")
    dataobj.remove_bad_pixels(med_spec=star_spectrum)

    dat = deepcopy(dataobj.data)

    print("setting planet model")
    minwv,maxwv= np.nanmin(dataobj.wavelengths),np.nanmax(dataobj.wavelengths)
    crop_btsettl = np.where((model_wvs > minwv - 0.2) * (model_wvs < maxwv + 0.2))
    model_wvs = model_wvs[crop_btsettl]
    model_spec = model_spec[crop_btsettl]
    model_broadspec = dataobj.broaden(model_wvs,model_spec)
    planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

    fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
            "boxw":boxw,"nodes":5,"psfw":(sig_x, sig_y), "star_flux":np.nanmean(star_spectrum) * np.size(star_spectrum),
            "badpixfraction":0.75,"optimize_nodes":True, "stamp":stamp}
    fm_func = hc_mask_splinefm
    flux_ratio = 1e-2
    
    print("setting noise")
    dataobj.set_noise()
    
    args1 = []
    args2 = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            args1 += [(y, x)]
            args2 += [(j, i)]
    
    args = zip(repeat(dataobj), args1, args2, repeat(planet_f), repeat(spec_file),\
         repeat(transmission), repeat(flux_ratio), repeat(dat), repeat(filename))

    with Pool() as tpool:
        for indices, f, n in tpool.map(one_location, args):
            print(indices, f, n)
            j, i = indices
            flux[j, i], noise[j, i] = f, n

    plt.figure()
    plt.imshow(flux/noise,origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_snr.png")
    plt.figure()
    plt.imshow(flux/flux_ratio,origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("flux")
    plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_flux.png")
    plt.figure()
    plt.imshow(noise,origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("noise")
    plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_noise.png")
    plt.close('all')

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=flux,
        header=pyfits.Header(cards={"TYPE": "flux", "FILE": filename, "PLANET": planet_btsettl,\
                                        "FLUX": spec_file, "TRANS": tr_file})))    
    hdulist.append(pyfits.PrimaryHDU(data=noise,
        header=pyfits.Header(cards={"TYPE": "noise", "FILE": filename, "PLANET": planet_btsettl,\
                                        "FLUX": spec_file, "TRANS": tr_file})))   
                                
    try:
        hdulist.writeto(dir_name+subdirectory+filename[:-5]+"_out.fits", overwrite=True)
    except TypeError:
        hdulist.writeto(dir_name+subdirectory+filename[:-5]+"_out.fits", clobber=True)
    hdulist.close()

    print("DONE", filename)
    break