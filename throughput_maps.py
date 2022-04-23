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
from copy import copy
import scipy.linalg as la

numthreads = 16

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

numthreads = 8
# star = "AB_Aur"
star = sys.argv[1]
boxw = 3
dir_name = arguments.dir_name[star]
tr_dir = arguments.tr_dir[star]
sky_calib_file = arguments.sky_calib_file[star]
files = os.listdir(dir_name)

res_numbasis = 5
nodes = 5

subdirectory = "throughput/20220417/"

print("making subdirectories")
Path(dir_name+subdirectory+"plots/").mkdir(parents=True, exist_ok=True)

print("Reading planet file")
planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
model_wvs = arr[:, 0] / 1e4
model_spec = 10 ** (arr[:, 1] - 8)
planet_model_set = True

tr_files = os.listdir(tr_dir)
if "plots" in tr_files:
    tr_files.remove("plots")
tr_counter = 0
tr_total = len(tr_files)

if star in arguments.rotated_seqs.keys():
    rotated_seqs = arguments.rotated_seqs[star]
else:
    rotated_seqs = []

def one_location(args):
    dataobj, location, indices, planet_f, spec_file, transmission, flux_ratio, dat, filename = args
    try:
        # if filename[8:12] in rotated_seqs: # add not if other way TODO
        #     print("rotated 90", filename)
        #     rotate = True
        #     y, x = location
        #     y *= -1
        # else:
        #     rotate = False
        #     x, y = location 
        x, y = location 
        dataobj.set_noise()
        log_prob,log_prob_H0,rchi2,linparas_b,linparas_err = grid_search([rvs,[x],[y]],dataobj,fm_func,fm_paras,numthreads=numthreads)
        fk_dataobj = deepcopy(dataobj)
        
        inject_planet(fk_dataobj, location, planet_f, spec_file, transmission, flux_ratio)#, rotated_90=rotate)
        fk_dataobj.set_noise()
        log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,[x],[y]],fk_dataobj,fm_func,fm_paras,numthreads=None)
        print("SNR time", location)
        return indices, linparas[0,0,0,0], linparas_b[0,0,0,0], linparas_err[0,0,0,0]
    except Exception as e:
        print(e)
        print("FAILED", filename, location)
        return indices, np.nan, np.nan, np.nan

for filename in files[:]:
    rvs = np.array([0])
    ys = np.arange(-40, 40)
    xs = np.arange(0, 1)
    # ys = np.arange(-8,8)
    # xs = np.arange(-5,5)
    flux = np.zeros((len(ys), len(xs))) * np.nan
    flux_b = np.zeros((len(ys), len(xs))) * np.nan
    noise = np.zeros((len(ys), len(xs))) * np.nan
    if ".fits" not in filename:
        print("skipping ", filename)
        continue
    print(filename)
    dataobj = OSIRIS(dir_name+filename) 
    nz,ny,nx = dataobj.data.shape

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
    print(dataobj.refpos)

    print("compute stellar PSF")
    data = dataobj.data
    nz, ny, nx = data.shape
    stamp_y, stamp_x = (boxw-1)//2, (boxw-1)//2
    img_mean = np.nanmedian(data, axis=0)
    # star_y, star_x = np.unravel_index(np.nanargmax(img_mean), img_mean.shape)
    star_y, star_x = int(np.round(dataobj.refpos[1])), int(np.round(dataobj.refpos[0]))
    stamp = data[:, star_y-stamp_y:star_y+stamp_y+1, star_x-stamp_x:star_x+stamp_x+1]
    total_flux = np.sum(stamp)
    # stamp = stamp/np.nansum(stamp,axis=(1,2))[:,None,None]

    print("setting noise")
    dataobj.set_noise()

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
    dataobj.bad_pixels[0:5,:,:] = np.nan
    dataobj.bad_pixels[312:318,:,:] = np.nan
    dataobj.bad_pixels[343:349,:,:] = np.nan
    dataobj.bad_pixels[396:402,:,:] = np.nan
    dataobj.bad_pixels[418:422,:,:] = np.nan
    dataobj.bad_pixels[446::,:,:] = np.nan

    dataobj.bad_pixels[366:370,:,:] = np.nan
    dataobj.bad_pixels[373:378,:,:] = np.nan
    dataobj.bad_pixels[384:388,:,:] = np.nan

    dat = deepcopy(dataobj.data)

    if planet_model_set:
        print("setting planet model")
        minwv,maxwv= np.nanmin(dataobj.wavelengths),np.nanmax(dataobj.wavelengths)
        crop_btsettl = np.where((model_wvs > minwv - 0.2) * (model_wvs < maxwv + 0.2))
        model_wvs = model_wvs[crop_btsettl]
        model_spec = model_spec[crop_btsettl]
        model_broadspec = dataobj.broaden(model_wvs,model_spec)
        planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)
        planet_model_set = False

    for fractional_fov in ["bottom", "top"]:
        if res_numbasis != 0:
            rvs = np.array([0])
            # ys = np.arange(-11, 11)
            # xs = np.arange(-6, 6)
            if "bottom" in fractional_fov:
                ys = np.arange(1, 11)
            elif "top" in fractional_fov:
                ys = np.arange(-10, 0)
            elif "all" in fractional_fov:
                ys = np.arange(-10, 11)
            xs = np.arange(-5, 6)

            # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
            #         "boxw":3,"nodes":20,"psfw":1.2,"badpixfraction":0.75}
            # fm_func = hc_splinefm
            fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
                "boxw":boxw,"nodes":nodes,"psfw":(np.nanmedian(sig_y), np.nanmedian(sig_x)), "star_flux":total_flux,
                "badpixfraction":0.75,"optimize_nodes":True, "stamp":stamp}
            print("psfw:", np.nanmedian(sig_y), np.nanmedian(sig_x))
            fm_func = hc_mask_splinefm
            # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
            #             "boxw":3,"psfw":1.5,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40}
            # fm_func = hc_hpffm

            out_res = np.zeros((nz,np.size(ys),np.size(xs))) + np.nan
            log_prob_test = np.zeros((np.size(ys),np.size(xs))) + np.nan
            for k in range(np.size(ys)):
                for l in range(np.size(xs)):
                    # badpix_vec = copy(dataobj.bad_pixels[:,int(np.round(dataobj.refpos[1] + ys[k])),int(np.round(dataobj.refpos[0] + xs[l]))])
                    # badpix_vec[np.where(np.isnan(star_spectrum*transmission))] = np.nan
                    # myres = np.zeros(np.size(np.where(np.isfinite(badpix_vec))[0]))+np.nan
                    # log_prob_test[k,l],_,_,_,_ = fitfm([0, ys[k], xs[l]], dataobj, fm_func, fm_paras,computeH0 = False,bounds = None,
                    #                                    residuals=myres)
                    # out_res[np.where(np.isfinite(badpix_vec)),k,l] = myres
                    w = (boxw-1)//2
                    # w=0
                    _y,_x=int(np.round(dataobj.refpos[1] + ys[k])),int(np.round(dataobj.refpos[0] + xs[l]))
                    badpix_vec = copy(dataobj.bad_pixels[:,_y-w:_y+w+1,_x-w:_x+w+1])
                    noise_vec = copy(dataobj.noise[:,_y-w:_y+w+1,_x-w:_x+w+1])
                    data_vec = copy(dataobj.data[:,_y-w:_y+w+1,_x-w:_x+w+1])

                    badpix_vec[np.where(np.isnan(star_spectrum*transmission))[0],:,:] = np.nan
                    canvas_res = np.zeros(badpix_vec.shape)+np.nan
                    myres = np.zeros(np.size(np.where(np.isfinite(badpix_vec))[0]))+np.nan
                    log_prob_test[k,l],_,rchi2,_,_ = fitfm([0, ys[k], xs[l]], dataobj, fm_func, fm_paras,computeH0 = True,bounds = None,
                                                    residuals_H0=myres,residuals=None)
                    # print(rchi2)
                    canvas_res[np.where(np.isfinite(badpix_vec))] = myres
                    out_res[:,k,l] = np.nanmean(canvas_res,axis=(1,2))

            X = np.reshape(out_res,(nz,np.size(ys)*np.size(xs))).T
            X = X[np.where(np.nansum(X,axis=1)!=0)[0],:]
            X = X/np.nanstd(X,axis=1)[:,None]
            X[np.where(np.isnan(X))] = np.tile(np.nanmedian(X,axis=0)[None,:],(X.shape[0],1))[np.where(np.isnan(X))]
            X[np.where(np.isnan(X))] = 0

            # print(X.shape)
            C = np.cov(X)
            # print(C.shape)
            # exit()
            tot_basis = C.shape[0]
            tmp_res_numbasis = np.clip(np.abs(res_numbasis) - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
            max_basis = np.max(tmp_res_numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
            evals, evecs = la.eigh(C, eigvals=(tot_basis-max_basis, tot_basis-1))
            check_nans = np.any(evals <= 0) # alternatively, check_nans = evals[0] <= 0
            evals = np.copy(evals[::-1])
            evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication
            # calculate the KL basis vectors
            kl_basis = np.dot(X.T, evecs)
            res4model_kl = kl_basis * (1. / np.sqrt(evals * (nz- 1)))[None, :]  #multiply a value for each row
            print(res4model_kl.shape)

        else:
            res4model_kl = None
        


        # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
        #         "boxw":3,"nodes":20,"psfw":1.2,"badpixfraction":0.75}
        # fm_func = hc_splinefm
        # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
        #             "boxw":boxw,"nodes":5,"psfw":(np.nanmedian(sig_y), np.nanmedian(sig_x)), 
        #             "star_flux": np.nanmean(stamp) * np.size(stamp), 
        #             #"star_flux": np.nanmean(star_spectrum) * np.size(star_spectrum),
        #             "badpixfraction":0.75,"optimize_nodes":True, "stamp": stamp}
        # print("psfw:", np.nanmedian(sig_y), np.nanmedian(sig_x))
        fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
                        "boxw":boxw,"nodes":nodes,"psfw":(np.nanmedian(sig_y), np.nanmedian(sig_x)), "star_flux": total_flux,
                        "badpixfraction":0.75,"optimize_nodes":True, "stamp": stamp,"KLmodes":res4model_kl,"fit_background":False,"recalc_noise":True}
        fm_func = hc_mask_splinefm
        # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
        #             "boxw":3,"psfw":1.5,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40}
        # fm_func = hc_hpffm
        flux_ratio = 1e-2
        rvs = np.array([0])
        ys = np.arange(-40, 40)
        xs = np.arange(0, 1)
        args1 = []
        args2 = []
        if "bottom" in fractional_fov:
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    if y >= 0:
                        continue
                    args1 += [(y, x)]
                    args2 += [(j, i)]
        elif "top" in fractional_fov:
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    if y < 0:
                        continue
                    args1 += [(y, x)]
                    args2 += [(j, i)]
        elif "all" in fractional_fov:
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    args1 += [(y, x)]
                    args2 += [(j, i)]

        
        args = zip(repeat(dataobj), args1, args2, repeat(planet_f), repeat(spec_file),\
            repeat(transmission), repeat(flux_ratio), repeat(dat), repeat(filename))

        with Pool() as tpool:
            for indices, f, fb, n in tpool.map(one_location, args):
                print(indices, f, fb, n)
                j, i = indices
                flux[j, i], flux_b[j, i], noise[j, i] = f, fb, n


    # print("parsing output")
    # for i, x in enumerate(xs):
    #     for j, y in enumerate(ys):
    #         flux[j, i],noise[j, i] = outputs[i*len(ys)+j]

    # plt.figure()
    # plt.imshow(flux/noise,origin="lower")
    # cbar = plt.colorbar()
    # cbar.set_label("SNR")
    # plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_snr.png")

    # plt.figure()
    # plt.imshow(flux/flux_ratio,origin="lower")
    # cbar = plt.colorbar()
    # cbar.set_label("flux/flux_ratio")
    # plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_flux.png")

    # plt.figure()
    # plt.imshow((flux-flux_b)/flux_ratio,origin="lower")
    # cbar = plt.colorbar()
    # cbar.set_label("(flux-flux_b)/flux_ratio")
    # plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_tp.png")

    # plt.figure()
    # plt.imshow(noise,origin="lower")
    # cbar = plt.colorbar()
    # cbar.set_label("noise")
    # plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_noise.png")
    # plt.close('all')

    plt.figure()
    plt.plot(flux/noise)
    plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_snr.png")

    plt.figure()
    plt.plot(flux/flux_ratio)
    plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_flux.png")

    plt.figure()
    plt.plot((flux-flux_b)/flux_ratio)
    plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_tp.png")

    plt.figure()
    plt.plot(noise)
    plt.savefig(dir_name+subdirectory+"plots/"+filename[:-5]+"_noise.png")

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=flux,
        header=pyfits.Header(cards={"TYPE": "flux", "FILE": filename, "PLANET": planet_btsettl,\
                                        "FLUX": spec_file, "TRANS": tr_file})))    
    hdulist.append(pyfits.PrimaryHDU(data=flux_b,
        header=pyfits.Header(cards={"TYPE": "flux_b", "FILE": filename, "PLANET": planet_btsettl,\
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
    # break