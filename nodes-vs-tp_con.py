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
from breads.grid_search import grid_search
from breads.fm.hc_splinefm import hc_splinefm
from breads.fm.hc_mask_splinefm import hc_mask_splinefm
from breads.fm.hc_hpffm import hc_hpffm
from breads.injection import inject_planet, read_planet_info

star = "SR4"
# star = "ROXs44"
# star = "HD148352"
# star = "ROXs35A"
# star = str(sys.argv[1])
print(star)
fol = "20220512_1700K"
target = f"{fol}_{star}"
dir_name = arguments.dir_name[star]
files = os.listdir(dir_name)

subdirectory = f"nodes/{fol}/"

print("making subdirectories")
Path(dir_name+subdirectory).mkdir(parents=True, exist_ok=True)

print("Reading planet file")
planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
model_wvs = arr[:, 0] / 1e4
model_spec = 10 ** (arr[:, 1] - 8)

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
        x, y = location 
        dataobj.set_noise()
        log_prob,log_prob_H0,rchi2,linparas_b,linparas_err_b = grid_search([rvs,[x],[y]],dataobj,fm_func,fm_paras,numthreads=numthreads)
        fk_dataobj = deepcopy(dataobj)
        inject_planet(fk_dataobj, location, planet_f, spec_file, transmission, flux_ratio)#, rotated_90=rotate)
        fk_dataobj.set_noise()
        log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,[x],[y]],fk_dataobj,fm_func,fm_paras,numthreads=None)
        print("SNR time", location)
        return linparas_b[0,0,0,0], linparas_err_b[0,0,0,0], linparas[0,0,0,0], linparas_err[0,0,0,0]
    except Exception as e:
        print(e)
        print("FAILED", filename, location)
        return np.nan, np.nan, np.nan, np.nan

sep = 40
num_angles = 16
flux_ratio = 1e-2
angles = np.linspace(0, 2*np.pi, num_angles+1)[:-1]
num_nodes = np.arange(1, 21, 1)
rvs = np.array([0])
ys = sep / 20 * np.cos(angles)
xs = sep / 20 * np.sin(angles)
boxw = 3

b_flux, b_err_rec, t_flux, t_err_rec, bsnr = {}, {}, {}, {}, {}
for num_node in num_nodes:
    b_flux[num_node] = np.zeros_like(angles)
    b_err_rec[num_node] = np.zeros_like(angles)
    t_flux[num_node] = np.zeros_like(angles)
    t_err_rec[num_node] = np.zeros_like(angles)
    bsnr[num_node] = {}

for filename in files[1:12]:
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
        sig_x = hdulist[5].data
        sig_y = hdulist[6].data

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

    print("setting planet model")
    minwv,maxwv= np.nanmin(dataobj.wavelengths),np.nanmax(dataobj.wavelengths)
    crop_btsettl = np.where((model_wvs > minwv - 0.2) * (model_wvs < maxwv + 0.2))
    model_wvs = model_wvs[crop_btsettl]
    model_spec = model_spec[crop_btsettl]
    model_broadspec = dataobj.broaden(model_wvs,model_spec)
    planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

    print("setting noise")
    dataobj.set_noise()

    print("compute stellar PSF")
    data = dataobj.data
    nz, ny, nx = data.shape
    stamp_y, stamp_x = (boxw-1)//2, (boxw-1)//2
    img_mean = np.nanmedian(data, axis=0)
    # star_y, star_x = np.unravel_index(np.nanargmax(img_mean), img_mean.shape)
    star_y, star_x = int(np.round(dataobj.refpos[1])), int(np.round(dataobj.refpos[0]))
    stamp = data[:, star_y-stamp_y:star_y+stamp_y+1, star_x-stamp_x:star_x+stamp_x+1]
    total_flux = np.sum(stamp)

    for num_node in num_nodes:
        fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
                        "boxw":boxw,"nodes":int(num_node),"psfw":(np.nanmedian(sig_y), np.nanmedian(sig_x)), "star_flux": total_flux,
                        "badpixfraction":0.75,"optimize_nodes":True, "stamp": stamp,"fit_background":False,"recalc_noise":True}
        fm_func = hc_mask_splinefm

        args = zip(repeat(dataobj), list(zip(ys, xs)), repeat(planet_f), repeat(spec_file),\
            repeat(transmission), repeat(flux_ratio), repeat(dat), repeat(filename), repeat(fm_func), repeat(fm_paras))
        bflux, bnoise, flux, noise = [], [], [], []
        with Pool() as tpool:
            for bf, bn, f, n in tpool.map(one_location, args):
                print(num_node, bf, bn, f, n)
                bflux += [bf]
                bnoise += [bn]
                flux += [f]
                noise += [n]
        bflux, bnoise, flux, noise = np.array(bflux), np.array(bnoise), np.array(flux), np.array(noise) 
        b_flux[num_node] += bflux / (bnoise) ** 2
        b_err_rec[num_node] += 1 / bnoise ** 2
        bsnr[num_node][filename] = bflux / bnoise
        t_flux[num_node] += flux / (noise) ** 2
        t_err_rec[num_node] += 1 / noise ** 2

tp, tp_err, noise_calib, noi, noi_err = [], [], {}, [], []
for num_node in num_nodes:
    noise_calib[num_node] = np.nanstd(list(bsnr[num_node].values()), axis=0)
    tpvals = (t_flux[num_node] - b_flux[num_node]) / t_err_rec[num_node] / flux_ratio
    plt.figure(1)
    for val in tpvals:
        plt.plot(num_node, val, "bx")
    tp += [np.nanmean(tpvals)]
    tp_err += [np.nanstd(tpvals)]
    nvals = 1 / np.sqrt(t_err_rec[num_node]) * noise_calib[num_node] / tpvals
    plt.figure(2)
    for val in nvals:
        plt.plot(num_node, val, "bx")
    noi += [np.nanmean(nvals)]
    noi_err += [np.nanstd(nvals)]

plt.figure(1)
plt.xlabel("number of nodes")
plt.ylabel("throughput")
plt.savefig(f"./plots/nodes-vs/nodes_tp1_{target}.png")
plt.savefig(dir_name+subdirectory+f"nodes_tp1_{target}.png")

plt.figure(2)
plt.xlabel("number of nodes")
plt.ylabel("noise")
plt.savefig(f"./plots/nodes-vs/nodes_noi1_{target}.png")
plt.savefig(dir_name+subdirectory+f"nodes_noi1_{target}.png")

plt.figure(3)
plt.xlabel("number of nodes")
plt.ylabel("throughput")
plt.errorbar(num_nodes, tp, yerr=tp_err)
plt.savefig(f"./plots/nodes-vs/nodes_tp2_{target}.png")
plt.savefig(dir_name+subdirectory+f"nodes_tp2_{target}.png")

plt.figure(4)
plt.xlabel("number of nodes")
plt.ylabel("noise")
plt.errorbar(num_nodes, noi, yerr=noi_err)
plt.savefig(f"./plots/nodes-vs/nodes_noi2_{target}.png")
plt.savefig(dir_name+subdirectory+f"nodes_noi2_{target}.png")

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=num_nodes,
    header=pyfits.Header(cards={"TYPE": "num_nodes", "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))    
for num_node in num_nodes:
    hdulist.append(pyfits.PrimaryHDU(data=t_flux[num_node],
        header=pyfits.Header(cards={"TYPE": "t_flux", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))   
    hdulist.append(pyfits.PrimaryHDU(data=t_err_rec[num_node],
        header=pyfits.Header(cards={"TYPE": "t_err_rec", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))  
    hdulist.append(pyfits.PrimaryHDU(data=b_flux[num_node],
        header=pyfits.Header(cards={"TYPE": "b_flux", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))                          
    hdulist.append(pyfits.PrimaryHDU(data=b_err_rec[num_node],
        header=pyfits.Header(cards={"TYPE": "b_err_rec", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))  
    hdulist.append(pyfits.PrimaryHDU(data=noise_calib[num_node],
        header=pyfits.Header(cards={"TYPE": "calib", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))  
    hdulist.append(pyfits.PrimaryHDU(data=list(bsnr[num_node].values()),
        header=pyfits.Header(cards={"TYPE": "bsnr", "NODE": str(num_node), "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))
hdulist.append(pyfits.PrimaryHDU(data=np.vstack((tp, tp_err)),
    header=pyfits.Header(cards={"TYPE": "tp", "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))    
hdulist.append(pyfits.PrimaryHDU(data=np.vstack((noi, noi_err)),
    header=pyfits.Header(cards={"TYPE": "noi", "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))    

try:
    hdulist.writeto(dir_name+subdirectory+f"nodes_{target}.fits", overwrite=True)
except TypeError:
    hdulist.writeto(dir_name+subdirectory+f"nodes_{target}.fits", clobber=True)
try:
    hdulist.writeto(f"./plots/nodes-vs/nodes_{target}.fits", overwrite=True)
except TypeError:
    hdulist.writeto(f"./plots/nodes-vs/nodes_{target}.fits", clobber=True)
hdulist.close()

plt.show()
