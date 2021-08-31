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

import arguments

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

star = "SR4"
# star = str(sys.argv[1])
print(star)
fol = "TP"
target = f"{fol}_{star}"
dir_name = arguments.dir_name[star]
files = os.listdir(dir_name)

subdirectory = f"temp_recover/{fol}/"

print("making subdirectories")
Path(dir_name+subdirectory).mkdir(parents=True, exist_ok=True)

tr_dir = arguments.tr_dir[star]
tr_files = os.listdir(tr_dir)
if "plots" in tr_files:
    tr_files.remove("plots")
tr_counter = 0
tr_total = len(tr_files)

sky_calib_file = arguments.sky_calib_file[star]

def one_location(args):
    dataobj, location, planet_f, spec_file, transmission, flux_ratio, dat, filename, fm_func, fm_paras = args
    try:
        dataobj.data = deepcopy(dat)
        out1 = search_planet([rvs,[location[0]],[location[1]]],dataobj,fm_func,fm_paras,numthreads=numthreads)
        inject_planet(dataobj, location, planet_f, spec_file, transmission, flux_ratio)
        print("SNR time", location)
        out2 = search_planet([rvs,[location[0]],[location[1]]],dataobj,fm_func,fm_paras,numthreads=numthreads)
        N_linpara = (out1.shape[-1]-2)//2
        return out1[0,0,0,3], out1[0,0,0,3+N_linpara], out2[0,0,0,3], out2[0,0,0,3+N_linpara] 
    except Exception as e:
        print(e)
        print("FAILED", filename, location)
        return np.nan, np.nan, np.nan, np.nan

def get_planetf_temp(args):
    dataobj, temp = args 
    strtemp = str(temp)
    if np.isclose(temp, int(temp)):
        strtemp = strtemp[:strtemp.index('.')]
    print(temp, strtemp)
    planet_btsettl = f"/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte0{strtemp}-5.0-0.0a+0.0.BT-Settl.spec.7"
    arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                    converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
    model_wvs = arr[:, 0] / 1e4
    model_spec = 10 ** (arr[:, 1] - 8)
    minwv,maxwv= np.nanmin(dataobj.wavelengths),np.nanmax(dataobj.wavelengths)
    crop_btsettl = np.where((model_wvs > minwv - 0.2) * (model_wvs < maxwv + 0.2))
    model_wvs = model_wvs[crop_btsettl]
    model_spec = model_spec[crop_btsettl]
    model_broadspec = dataobj.broaden(model_wvs,model_spec)
    return temp, interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

sep = 100
num_angles = 4
flux_ratio = 1e-2
angles = np.linspace(0, 2*np.pi, num_angles+1)[:-1]
# available temps : [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 
# 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 22.0, 28.0, 30.0, 70.0]
temperatures = np.arange(10, 20.5, 0.5)
# temperatures = np.array([10., 15., 20.])
rvs = np.array([0])
ys = sep / 20 * np.cos(angles)
xs = sep / 20 * np.sin(angles)
injected_temp = 18.0

print("Reading planet files")
try:
    if True:
        print("FORCE REDO")
        raise Exception
    planetfs = np.load('./plots/temp_recover/planetfs.npy', allow_pickle=True).item()
except:    
    print("making planet model grid")
    planetfs = {}
    for filename in files[:]:
        if ".fits" not in filename:
            continue
        dataobj = OSIRIS(dir_name+filename)
        break
    args = zip(repeat(dataobj), temperatures)
    with Pool() as tpool:
        for temp, pf in tpool.map(get_planetf_temp, args):
            print(temp)
            planetfs[temp] = pf
    np.save('./plots/temp_recover/planetfs.npy', planetfs)

b_flux, b_err_rec, t_flux, t_err_rec, bsnr = {}, {}, {}, {}, {}
for temp in temperatures:
    b_flux[temp] = np.zeros_like(angles)
    b_err_rec[temp] = np.zeros_like(angles)
    t_flux[temp] = np.zeros_like(angles)
    t_err_rec[temp] = np.zeros_like(angles)
    bsnr[temp] = {}

for filename in files[:4]:
    if ".fits" not in filename:
        print("SKIP", filename)
        continue
    print("START", filename)
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

    tr_counter = (tr_counter + 1) % tr_total
    tr_file = tr_dir + tr_files[tr_counter]
    print("Reading transmission file", tr_file)
    with pyfits.open(tr_file) as hdulist:
        transmission = hdulist[0].data

    print("Removing bad pixels")
    dataobj.remove_bad_pixels(med_spec=star_spectrum)
    dat = deepcopy(dataobj.data)

    print("setting noise")
    dataobj.set_noise()

    for temp in temperatures:
        fm_paras = {"planet_f":planetfs[temp],"transmission":transmission,"star_spectrum":star_spectrum,
                "boxw":3,"nodes":20,"psfw":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
                "badpixfraction":0.75,"optimize_nodes":True}
        fm_func = hc_no_splinefm

        args = zip(repeat(dataobj), list(zip(ys, xs)), repeat(planetfs[injected_temp]), repeat(spec_file),\
            repeat(transmission), repeat(flux_ratio), repeat(dat), repeat(filename), repeat(fm_func), repeat(fm_paras))
        bflux, bnoise, flux, noise = [], [], [], []
        with Pool() as tpool:
            for bf, bn, f, n in tpool.map(one_location, args):
                print(temp, bf, bn, f, n)
                bflux += [bf]
                bnoise += [bn]
                flux += [f]
                noise += [n]
        bflux, bnoise, flux, noise = np.array(bflux), np.array(bnoise), np.array(flux), np.array(noise) 
        b_flux[temp] += bflux / (bnoise) ** 2
        b_err_rec[temp] += 1 / bnoise ** 2
        bsnr[temp][filename] = bflux / bnoise
        t_flux[temp] += flux / (noise) ** 2
        t_err_rec[temp] += 1 / noise ** 2

snr, snr_err, noise_calib = [], [], {}
for temp in temperatures:
    noise_calib[temp] = np.nanstd(list(bsnr[temp].values()), axis=0)
    rflux = (t_flux[temp] - b_flux[temp]) / t_err_rec[temp]
    rerr = 1 / np.sqrt(t_err_rec[temp])
    snr_vals = rflux / rerr / noise_calib[temp]
    plt.figure(1)
    for val in snr_vals:
        plt.plot(temp, val, "bx")
    snr += [np.nanmean(snr_vals)]
    snr_err += [np.nanstd(snr_vals)]

plt.figure(3)
plt.errorbar(temperatures, snr, yerr=snr_err)

plt.show()

# hdulist = pyfits.HDUList()
# hdulist.append(pyfits.PrimaryHDU(data=num_nodes,
#     header=pyfits.Header(cards={"TYPE": "num_nodes", "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))    
# for num_node in num_nodes:
#     hdulist.append(pyfits.PrimaryHDU(data=t_flux[num_node],
#         header=pyfits.Header(cards={"TYPE": "t_flux", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))   
#     hdulist.append(pyfits.PrimaryHDU(data=t_err_rec[num_node],
#         header=pyfits.Header(cards={"TYPE": "t_err_rec", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))  
#     hdulist.append(pyfits.PrimaryHDU(data=b_flux[num_node],
#         header=pyfits.Header(cards={"TYPE": "b_flux", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))                          
#     hdulist.append(pyfits.PrimaryHDU(data=b_err_rec[num_node],
#         header=pyfits.Header(cards={"TYPE": "b_err_rec", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))  
#     hdulist.append(pyfits.PrimaryHDU(data=noise_calib[num_node],
#         header=pyfits.Header(cards={"TYPE": "calib", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))  
#     hdulist.append(pyfits.PrimaryHDU(data=list(bsnr[num_node].values()),
#         header=pyfits.Header(cards={"TYPE": "bsnr", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))
# hdulist.append(pyfits.PrimaryHDU(data=np.vstack((tp, tp_err)),
#     header=pyfits.Header(cards={"TYPE": "tp", "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))    
# hdulist.append(pyfits.PrimaryHDU(data=np.vstack((noi, noi_err)),
#     header=pyfits.Header(cards={"TYPE": "noi", "FILE": filename, "PLANET": planet_btsettl,\
#                                     "FLUX": spec_file, "TRANS": tr_file})))    

# try:
#     hdulist.writeto(dir_name+subdirectory+f"nodes_{target}.fits", overwrite=True)
# except TypeError:
#     hdulist.writeto(dir_name+subdirectory+f"nodes_{target}.fits", clobber=True)
# try:
#     hdulist.writeto(f"./plots/nodes-vs/nodes_{target}.fits", overwrite=True)
# except TypeError:
#     hdulist.writeto(f"./plots/nodes-vs/nodes_{target}.fits", clobber=True)
# hdulist.close()

# plt.show()
