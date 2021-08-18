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

print("Importing mkl")
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

print("Importing breads")
from breads.instruments.OSIRIS import OSIRIS
from breads.search_planet import search_planet
from breads.fm.hc_splinefm import hc_splinefm
from breads.fm.hc_no_splinefm import hc_no_splinefm
from breads.fm.hc_hpffm import hc_hpffm
from breads.injection import inject_planet, read_planet_info

# dir_name = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"
# dir_name = "/scr3/jruffio/data/osiris_survey/targets/SR3/210626/first/reduced/"
dir_name = "/scr3/jruffio/data/osiris_survey/targets/SR14/210628/reduced/"
files = os.listdir(dir_name)

subdirectory = "throughput/TP/"

print("making subdirectories")
Path(dir_name+subdirectory+"plots/").mkdir(parents=True, exist_ok=True)

print("Reading planet file")
planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
model_wvs = arr[:, 0] / 1e4
model_spec = 10 ** (arr[:, 1] - 8)

tr_dir = "/scr3/jruffio/data/osiris_survey/targets/SR3/210628/first/reduced/spectra/"
tr_files = os.listdir(tr_dir)
tr_counter = 0
tr_total = len(tr_files)

sky_calib_file = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210628/reduced/s210628_a002002_Kn3_020_calib.fits"

def one_location(args):
    dataobj, location, indices, planet_f, spec_file, transmission, flux_ratio, dat, filename = args
    try:
        dataobj.data = deepcopy(dat)
        inject_planet(dataobj, location, planet_f, spec_file, transmission, flux_ratio)
        print("SNR time", location)
        out = search_planet([rvs,[location[0]],[location[1]]],dataobj,fm_func,fm_paras,numthreads=numthreads)
        N_linpara = (out.shape[-1]-2)//2
        return indices, out[0,0,0,3], out[0,0,0,3+N_linpara]
    except Exception as e:
        print(e)
        print("FAILED", filename, location)
        return indices, np.nan, np.nan

for filename in files[:]:
    rvs = np.array([0])
    ys = np.arange(-40, 40)
    xs = np.arange(-20, 20)
    # ys = np.arange(-5, 5)
    # xs = np.arange(-5, 5)
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

    spec_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
    print("Reading spectrum file", spec_file)
    with pyfits.open(spec_file) as hdulist:
        star_spectrum = hdulist[2].data
        mu_x = hdulist[3].data
        mu_y = hdulist[4].data

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

    # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
    #         "boxw":3,"nodes":20,"psfw":1.2,"badpixfraction":0.75}
    # fm_func = hc_splinefm
    fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
            "boxw":3,"nodes":5,"psfw":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
            "badpixfraction":0.75,"optimize_nodes":True}
    fm_func = hc_no_splinefm
    # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
    #             "boxw":3,"psfw":1.5,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40}
    # fm_func = hc_hpffm
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


    # print("parsing output")
    # for i, x in enumerate(xs):
    #     for j, y in enumerate(ys):
    #         flux[j, i],noise[j, i] = outputs[i*len(ys)+j]

    plt.figure()
    plt.imshow(flux/noise,origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    plt.savefig(dir_name+subdirectory+filename[:-5]+"_snr.png")
    plt.figure()
    plt.imshow(flux,origin="lower")
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
