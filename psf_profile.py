print("starting")
import sys
sys.path.append("/scr3/jruffio/shubh/breads/")
import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.io.fits as pyfits
from photutils.aperture import CircularAperture, aperture_photometry
from pathlib import Path

dr = 0.01

print("Importing mkl")
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

print("Importing breads")
from breads.instruments.OSIRIS import OSIRIS

numthreads = 8
dir_name = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"
files = os.listdir(dir_name)
sky_calib_file = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210626/reduced/s210626_a003002_Kn3_020_calib.fits"

print("making subdirectories")
Path(dir_name+"psf/").mkdir(parents=True, exist_ok=True)

profiles = {}
distances = {}
max_r = np.inf

plt.figure()
for filename in files:
    if ".fits" not in filename:
        print("skipping ", filename)
        continue

    print("DOING", filename)
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

    print("setting reference position")
    dataobj.set_reference_position((np.nanmedian(mu_y), np.nanmedian(mu_x)))
    print(dataobj.refpos)

    print("Removing bad pixels")
    dataobj.remove_bad_pixels(med_spec=star_spectrum)

    img_med = np.nanmedian(dataobj.data, axis=0)
    center = dataobj.refpos
    r = 0
    profile = []
    distance = []
    int_sum = 0
    while True:
        r += dr
        if r ** 2 > ((dataobj.refpos[0] - ny) ** 2 + (dataobj.refpos[1] - nx) ** 2):
            if r < max_r:
                max_r = r
            break
        aper = CircularAperture(center, r)
        val = aperture_photometry(img_med, aper)['aperture_sum'][0]
        profile += [val - int_sum]
        int_sum = val
        distance += [(r+dr/2)*20]

    profiles[filename] = profile
    distances[filename] = distance
    plt.plot(distance, profile, label=filename)

plt.legend()
plt.savefig(dir_name+"psf/profiles.png")

ind = np.argmax(np.array(distance) > (max_r+dr/2)*20) - 1 
med_profile = np.zeros_like(distance[:ind])
for profile in profiles.values():
    print(np.array(profile[:ind]).shape)
    med_profile += np.array(profile[:ind])
med_profile /= len(profiles.values())
print(med_profile.shape)
plt.figure()
plt.plot(distance[:ind], med_profile)
plt.savefig(dir_name+"psf/med_profile.png")


hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=med_profile,
    header=pyfits.Header(cards={"TYPE": "psf", "DIR": dir_name}))) 
hdulist.append(pyfits.PrimaryHDU(data=distance[:ind],
    header=pyfits.Header(cards={"TYPE": "distance", "DIR": dir_name})))
for filename in profiles.keys():
    hdulist.append(pyfits.PrimaryHDU(data=profiles[filename],
        header=pyfits.Header(cards={"TYPE": "profiles", "FILE": filename})))
    hdulist.append(pyfits.PrimaryHDU(data=distances[filename],
        header=pyfits.Header(cards={"TYPE": "distances", "FILE": filename})))
                             
try:
    hdulist.writeto(dir_name+"psf/psf.fits", overwrite=True)
except TypeError:
    hdulist.writeto(dir_name+"psf/psf.fits", clobber=True)
hdulist.close()