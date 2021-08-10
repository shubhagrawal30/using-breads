print("starting")
import sys
sys.path.append("/scr3/jruffio/shubh/breads/")
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as pyfits
from pathlib import Path

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

numthreads = 8
# dir_name = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"
dir_name = "/scr3/jruffio/data/osiris_survey/targets/ROXs4/210627/reduced/"

print("making subdirectories")
Path(dir_name+"planets/").mkdir(parents=True, exist_ok=True)

print("Reading planet file")
planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
model_wvs = arr[:, 0] / 1e4
model_spec = 10 ** (arr[:, 1] - 8)

# filename = "s210626_a032002_Kn5_020.fits"
# filename = "s210627_a046004_Kn5_020.fits"
filename = "s210627_a048005_Kn5_020.fits"

print(filename)
dataobj = OSIRIS(dir_name+filename) 
nz,ny,nx = dataobj.data.shape
dataobj.noise = np.ones((nz,ny,nx))

# sky_calib_file = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210626/reduced/s210626_a003002_Kn3_020_calib.fits"
sky_calib_file = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210627/reduced/s210627_a003002_Kn3_020_calib.fits"
dataobj.calibrate(sky_calib_file)

spec_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
print("Reading spectrum file", spec_file)
with pyfits.open(spec_file) as hdulist:
    star_spectrum = hdulist[2].data
    mu_x = hdulist[3].data
    mu_y = hdulist[4].data

print("setting reference position")
dataobj.set_reference_position((np.nanmedian(mu_y), np.nanmedian(mu_x)))
# dataobj.set_reference_position((2, 2))
print(dataobj.refpos)

# tr_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
# SR3
# tr_file = "/scr3/jruffio/data/osiris_survey/targets/SR3/210626/first/reduced/spectra/s210626_a025"+filename[12:-13]+"_Kn5_020_spectrum.fits"
tr_file = "/scr3/jruffio/data/osiris_survey/targets/SR3/210627/second/reduced/spectra/s210627_a053002_Kn5_020_spectrum.fits"

print("Reading transmission file", tr_file)
with pyfits.open(tr_file) as hdulist:
    transmission = hdulist[0].data

print("Removing bad pixels")
dataobj.remove_bad_pixels(med_spec=star_spectrum)

print("setting planet model")
minwv,maxwv= np.nanmin(dataobj.wavelengths),np.nanmax(dataobj.wavelengths)
crop_btsettl = np.where((model_wvs > minwv - 0.2) * (model_wvs < maxwv + 0.2))
model_wvs = model_wvs[crop_btsettl]
model_spec = model_spec[crop_btsettl]
model_broadspec = dataobj.broaden(model_wvs,model_spec)
planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
            "boxw":3,"nodes":5,"psfw":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
            "badpixfraction":0.75,"optimize_nodes":True}
fm_func = hc_no_splinefm

# fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
#         "boxw":1,"nodes":20,"psfw":1.5,"badpixfraction":0.75}
# fm_func = hc_splinefm
# from breads.fm.hc_hpffm import hc_hpffm
# fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
#             "boxw":1,"psfw":1.2,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40}
# fm_func = hc_hpffm
rvs = np.linspace(-2000,2000,2001)
# rvs = [0]
# ys = [43]
# xs = [44]
ys = [2]
xs = [10]

if False: # Example code to test the forward model
    nonlin_paras = [0,2,10] # rv (km/s), y (pix), x (pix)
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

print("RV time")
out = search_planet([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=numthreads)
N_linpara = (out.shape[-1]-2)//2
print(out.shape)

# print("plotting")
# plt.figure()
# plt.plot(rvs,np.exp(out[:,0,0,0]-np.max(out[:,0,0,0])))
# plt.ylabel("RV posterior")
# plt.xlabel("RV (km/s)")
# plt.title(rvs[np.nanargmax(np.exp(out[:,0,0,0]-np.max(out[:,0,0,0])))])
# plt.show()

plt.figure()
plt.plot(rvs,out[:,0,0,3]/out[:,0,0,3+N_linpara])
plt.axvline(x=0)
plt.ylabel("SNR")
plt.xlabel("RV (km/s)")
plt.show()

# plt.figure()
# plt.imshow(out[0,:,:,3]/out[0,:,:,3+N_linpara],origin="lower")
# cbar = plt.colorbar()
# cbar.set_label("SNR")
# plt.show()
# plt.close()