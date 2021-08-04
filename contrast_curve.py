print("starting")
import sys
from scipy.sparse import data
sys.path.append("/scr3/jruffio/shubh/breads/")
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
from copy import deepcopy
import astropy.io.fits as pyfits
from pathlib import Path
from breads.fit import fitfm

print("Importing mkl")
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

print("Importing breads")
from breads.instruments.OSIRIS import OSIRIS
from breads.injection import inject_planet, read_planet_info
from breads.search_planet import search_planet
from breads.fm.hc_splinefm import hc_splinefm

numthreads = 16
dir_name = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"
fil = os.listdir(dir_name)[15]
print(fil)

planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
tr_file = "/scr3/jruffio/data/osiris_survey/targets/SR3/210626/first/reduced/spectra/s210626_a018002_Kn5_020_spectrum.fits"
spec_file = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"+"spectra/"+fil[:-5]+"_spectrum.fits"
sky_calib_file = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210626/reduced/s210626_a004002_Kn3_020_calib.fits"

dataobj = OSIRIS(dir_name+fil) 
dataobj.calibrate(sky_calib_file)
dataobj.remove_bad_pixels()

dat = deepcopy(dataobj.data)

planet_f = read_planet_info(planet_btsettl, True, True, 0.2, dataobj)

print("Reading transmission file", tr_file)
with pyfits.open(tr_file) as hdulist:
    transmission = hdulist[0].data

print("Reading spectrum file", spec_file)
with pyfits.open(spec_file) as hdulist:
    star_spectrum = hdulist[2].data
    mu_x = hdulist[3].data
    mu_y = hdulist[4].data
    psfw = (np.nanmean(hdulist[5].data) + np.nanmean(hdulist[6].data)) / 2

print("psfw:", psfw)
print("setting reference position")
dataobj.set_reference_position((np.nanmedian(mu_y), np.nanmedian(mu_x)))
print(dataobj.refpos)

boxw = 5
fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
        "boxw":boxw,"psfw":psfw,"nodes":20,"badpixfraction":0.75}
fm_func = hc_splinefm
rvs = np.array([0])
# rvs = np.array([-7])
ys = np.arange(-15, 20)
xs = np.arange(-15, 15)

snrs, flux, noise = [], [], []
distances = np.arange(12, 2, -1)

print("SNR time, no injection")
out = search_planet([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=numthreads)
N_linpara = (out.shape[-1]-2)//2
snr = out[0,:,:,3]/out[0,:,:,3+N_linpara]
arg = np.unravel_index(np.nanargmax(snr), snr.shape)
print("max snr at:", arg)
print("snr:", snr[arg[0], arg[1]])
print("flux:", out[0, arg[0], arg[1], 3])
print("noise:", out[0, arg[0], arg[1], 3+N_linpara])
plt.figure()
plt.imshow(snr,origin="lower")
cbar = plt.colorbar()
cbar.set_label("SNR after injection")
plt.show()
exit()

flux_ratio = 1e-2
# flux_ratio = 5e-3
for distance in distances:
    location = (distance, 0)
    inject_planet(dataobj, location, planet_f, spec_file, transmission, flux_ratio)
    print("SNR time", distance)
    out = search_planet([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=numthreads)
    N_linpara = (out.shape[-1]-2)//2
    snr = out[0,:,:,3]/out[0,:,:,3+N_linpara]
    arg = np.unravel_index(np.nanargmax(snr), snr.shape)
    print("max snr at:", arg)
    arg = (distance + 15, 15)
    print(arg) 
    print(out.shape)
    plt.figure()
    plt.imshow(snr,origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("SNR after injection")
    plt.title(distance)
    print(distance)
    print(np.nanmax(out[:,:,:,3]), out[0, arg[0], arg[1], 3])
    print(out[0, arg[0], arg[1], 3+N_linpara])
    print(np.nanmax(snr), snr[arg[0], arg[1]])

    snrs += [snr[arg[0], arg[1]]]
    flux += [out[0, arg[0], arg[1], 3]]
    noise += [out[0, arg[0], arg[1], 3+N_linpara]]

    print("reverting injection")
    dataobj.data = deepcopy(dat)
    plt.savefig(f"./plots/contrast/HD148352/{fil[:-5]}/SNR_{flux_ratio}_{location}_sym_{boxw}.png")
    # plt.show()

print(distances)
print(snrs)
print(flux)
print(noise)

plt.figure()
plt.title("SNRs")
plt.plot(distances, snrs)
plt.savefig(f"./plots/contrast/HD148352/{fil[:-5]}/SNRs_{flux_ratio}_{boxw}.png")
plt.figure()
plt.title("flux")
plt.plot(distances, flux)
plt.savefig(f"./plots/contrast/HD148352/{fil[:-5]}/flux_{flux_ratio}_{boxw}.png")
plt.figure()
plt.title("noise")
plt.plot(distances, noise)
plt.savefig(f"./plots/contrast/HD148352/{fil[:-5]}/noise_{flux_ratio}_{boxw}.png")

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=distances, header=pyfits.Header(cards={"TYPE": "distances", \
    "PLANET": planet_btsettl, "FLUX": spec_file, "TRANS": tr_file, "CALIB": sky_calib_file})))  
hdulist.append(pyfits.PrimaryHDU(data=snrs, header=pyfits.Header(cards={"TYPE": "snrs"})))  
hdulist.append(pyfits.PrimaryHDU(data=flux, header=pyfits.Header(cards={"TYPE": "flux"})))  
hdulist.append(pyfits.PrimaryHDU(data=noise, header=pyfits.Header(cards={"TYPE": "noise"})))                                  
try:
    hdulist.writeto(f"./plots/contrast/HD148352/{fil[:-5]}/contrast_{flux_ratio}_{boxw}.fits", overwrite=True)
except TypeError:
    hdulist.writeto(f"./plots/contrast/HD148352/{fil[:-5]}/contrast_{flux_ratio}_{boxw}.fits", clobber=True)
hdulist.close()

plt.show()
