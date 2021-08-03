print("starting")
import sys
from scipy.sparse import data
sys.path.append("/scr3/jruffio/shubh/breads/")
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
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

planet_f = read_planet_info(planet_btsettl, True, True, 0.2, dataobj)

print("Reading transmission file", tr_file)
with pyfits.open(tr_file) as hdulist:
    transmission = hdulist[0].data

print("Reading spectrum file", spec_file)
with pyfits.open(spec_file) as hdulist:
    star_spectrum = hdulist[2].data
    mu_x = hdulist[3].data
    mu_y = hdulist[4].data

print("setting reference position")
dataobj.set_reference_position((np.nanmedian(mu_y), np.nanmedian(mu_x)))
print(dataobj.refpos)

fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
        "boxw":3,"nodes":4,"psfw":1.2,"nodes":5,"badpixfraction":0.75}
fm_func = hc_splinefm
rvs = np.array([0])
ys = np.arange(-15, 15)
xs = np.arange(-15, 15)

#####################################3
print("plotting image before injection")
plt.figure()
plt.title("before injection (one slice)")
plt.imshow(dataobj.data[0], origin="lower")
cbar = plt.colorbar()
plt.savefig(f"./plots/injection/HD148352/before_slice.png")

plt.figure()
plt.title("before injection (median)")
plt.imshow(np.nanmedian(dataobj.data, axis=0), origin="lower")
cbar = plt.colorbar()
plt.savefig(f"./plots/injection/HD148352/before_median.png")

# print("SNR time")
# out = search_planet([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=numthreads)
# N_linpara = (out.shape[-1]-2)//2
# print(out.shape)
# plt.figure()
# plt.imshow(out[0,:,:,3]/out[0,:,:,3+N_linpara],origin="lower")
# cbar = plt.colorbar()
# cbar.set_label("SNR before injection")
# plt.savefig(f"./plots/injection/HD148352/before_SNR.png")
# # plt.show()

# hdulist = pyfits.HDUList()
# hdulist.append(pyfits.PrimaryHDU(data=dataobj.data, \
#     header=pyfits.Header(cards={"TYPE": "data", "FILE": fil, "CALIB": sky_calib_file})))  
# hdulist.append(pyfits.PrimaryHDU(data=out, header=pyfits.Header(cards={"TYPE": "data", \
#     "PLANET": planet_btsettl, "FLUX": spec_file, "TRANS": tr_file})))                                  
# try:
#     hdulist.writeto("./plots/injection/HD148352/before.fits", overwrite=True)
# except TypeError:
#     hdulist.writeto("./plots/injection/HD148352/before.fits", clobber=True)
# hdulist.close()


########################
flux_ratio = 1e-2
location = (10, 0)
inject_planet(dataobj, location, planet_f, spec_file, transmission, flux_ratio)

print("plotting image after injection")
plt.figure()
plt.imshow(dataobj.data[0], origin="lower")
plt.title("after injection (one slice)")
cbar = plt.colorbar()
plt.savefig(f"./plots/injection/HD148352/after_slice_{flux_ratio}_{location}.png")

plt.figure()
plt.title("after injection (median)")
plt.imshow(np.nanmedian(dataobj.data, axis=0), origin="lower")
cbar = plt.colorbar()
plt.savefig(f"./plots/injection/HD148352/after_median_{flux_ratio}_{location}.png")

print("SNR time")
out = search_planet([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=numthreads)
N_linpara = (out.shape[-1]-2)//2
print(out.shape)
plt.figure()
plt.imshow(out[0,:,:,3]/out[0,:,:,3+N_linpara],origin="lower")
cbar = plt.colorbar()
cbar.set_label("SNR after injection")
plt.savefig(f"./plots/injection/HD148352/after_SNR_{flux_ratio}_{location}.png")

print(np.nanmax(out[:,:,:,3]))
print(np.nanmax(out[:,15+location[0]-2:15+location[0]+3,15+location[1]-2:15+location[1]+3,3]))

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=dataobj.data, header=pyfits.Header(cards={"TYPE": "data"})))  
hdulist.append(pyfits.PrimaryHDU(data=out, header=pyfits.Header(cards={"TYPE": "data", \
    "PLANET": planet_btsettl, "FLUX": spec_file, "TRANS": tr_file, "CALIB": sky_calib_file})))                                  
try:
    hdulist.writeto(f"./plots/injection/HD148352/after_{flux_ratio}_{location}.fits", overwrite=True)
except TypeError:
    hdulist.writeto(f"./plots/injection/HD148352/after_{flux_ratio}_{location}.fits", clobber=True)
hdulist.close()

plt.show()