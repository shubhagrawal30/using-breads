print("starting")
import sys
sys.path.append("/scr3/jruffio/shubh/breads/")
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os
from breads.instruments.OSIRIS import OSIRIS
from pathlib import Path

# date = "210626"
# star = "HD148352"
# star = "SR21A"

# date = "210626/first"
# star = "SR3"

# date = "210627"
# star = "SR4"
# star = "ROXs44"
# star = "ROXs8"
# star = "ROXs4"

date = "210628"
star = "ROXs35A"
# star = "SR14"
# star = "ROXs43B"
# star = "SR9"

fol = "TP"
target = f"{fol}_{star}"

throughput_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/throughput/{fol}/"
frames_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/planets/{fol}/"
psf_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/"
fr_files = os.listdir(frames_dir)
th_files = os.listdir(throughput_dir)
psf_files = os.listdir(psf_dir)

print("making subdirectories")
Path(psf_dir+"psf/").mkdir(parents=True, exist_ok=True)

snrs = {}
to_flux = 0
t_err_rec = 0

for fil in fr_files:
    if "_out.fits" not in fil:
        print("fr skipping", fil)
        continue
    print("fr DOING", fil)
    with pyfits.open(frames_dir + fil) as hdulist:
        out = hdulist[0].data
    N_linpara = (out.shape[-1]-2)//2
    snrs[fil] = out[0,:,:,3] / out[0,:,:,3+N_linpara]
    to_flux += out[0,:,:,3] / (out[0,:,:,3+N_linpara])**2
    t_err_rec += 1 / out[0,:,:,3+N_linpara] ** 2
    # if len(snrs.values()) == 4:
    #     break

to_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

noise_calib = np.nanstd(list(snrs.values()), axis=0)

snr = to_flux / t_err / noise_calib
print("frames combined: ", len(snrs.keys()))
print(np.nanmax(to_flux))
print(np.unravel_index(np.nanargmax(to_flux), to_flux.shape))
xD, yD = np.unravel_index(np.nanargmax(snr), snr.shape)
detection_snr = np.nanmax(snr)
detection_flux = to_flux[xD, yD]
print(detection_snr, (xD, yD), detection_flux)
# exit()

# plt.figure()
# plt.imshow(noise_calib, origin="lower")
# cbar = plt.colorbar()
# cbar.set_label("noise calib")

flux_ratio = 1e-2
threshold = 5

fluxs = {}
noises = {}
t_flux = 0
t_err_rec = 0

for fil in th_files:
    if "_out.fits" not in fil:
        print("th skipping", fil)
        continue
    print("th DOING", fil)
    with pyfits.open(throughput_dir + fil) as hdulist:
        flux = hdulist[0].data
        noise = hdulist[1].data
    fluxs[fil] = flux
    noises[fil] = noise
    t_flux += flux / (noise) ** 2
    t_err_rec += 1 / noise ** 2

t_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

throughput = (t_flux-to_flux) / flux_ratio

calibrated_err_combined = t_err * noise_calib / throughput

print("frames combined: ", len(fluxs.keys()))

assert(len(fluxs.keys()) == len(snrs.values())), "frames combined not same size"

# plt.figure()
# plt.imshow(throughput, origin="lower")
# cbar = plt.colorbar()
# cbar.set_label("throughput")

# plt.figure()
# plt.imshow(calibrated_err_combined, origin="lower")
# cbar = plt.colorbar()
# cbar.set_label("noise")


distances = []
values = []
nx, ny = calibrated_err_combined.shape
xS, yS = nx / 2, ny / 2
for x in range(nx):
    for y in range(ny):
        distances += [np.sqrt((y - yS) ** 2 + (x - xS) ** 2) * 20]
        values += [calibrated_err_combined[x, y]]

# with pyfits.open(psf_dir + "psf.fits") as hdulist:
#     profile = hdulist[0].data
#     psf_dist = hdulist[1].data

psf_dist = []
psf_profile = []
for fil in psf_files:
    if ".fits" not in fil:
        print("psf skipping", fil)
        continue
    print("psf DOING", fil)
    data = np.nanmedian(OSIRIS(psf_dir + fil).data, axis=0)
    with pyfits.open(psf_dir+"spectra/"+fil[:-5]+"_spectrum.fits") as hdulist:
        mu_x = hdulist[3].data
        mu_y = hdulist[4].data
    xS, yS = np.nanmedian(mu_x), np.nanmedian(mu_y)    
    nx, ny = data.shape
    for x in range(nx):
        for y in range(ny):
            psf_dist += [np.sqrt((y - yS) ** 2 + (x - xS) ** 2) * 20]
            psf_profile += [data[x, y]]

distances = np.array(distances)

plt.figure()
plt.scatter(psf_dist, np.array(psf_profile) / np.nanmax(psf_profile), marker=",", color="black", label="scaled PSF profile", alpha = 0.01)
plt.scatter(distances, threshold * np.array(values), marker=",", alpha = 0.2, label="snr 5")
plt.scatter(distances, detection_snr * np.array(values), marker=",", alpha = 0.2, label=f"detection snr {detection_snr}")
plt.plot(np.sqrt((yD - yS) ** 2 + (xD - xS) ** 2) * 20, detection_flux / (t_flux/flux_ratio)[xD, yD], 'rX', label="detection")
# plt.plot(psf_dist, profile / np.nanmean(profile) * np.nanmean(threshold * np.array(values)), "black", label="scaled PSF profile")
plt.title(f"{target}")
plt.xlim([np.nanmin(distances[distances > 0]) * 0.8, np.nanmax(distances) / 0.8])
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig(psf_dir+"psf/scatter.png")
plt.savefig(f"./plots/scatter/scatter_{target}.png")
plt.show()

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=calibrated_err_combined,
    header=pyfits.Header(cards={"TYPE": "noise", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=throughput,
    header=pyfits.Header(cards={"TYPE": "throughput", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=noise_calib,
    header=pyfits.Header(cards={"TYPE": "noise_calib", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=snr,
    header=pyfits.Header(cards={"TYPE": "snr", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=np.vstack((psf_dist, psf_profile)),
    header=pyfits.Header(cards={"TYPE": "dist_psf", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=np.vstack((distances, values)),
    header=pyfits.Header(cards={"TYPE": "dist_noise", "DIR": frames_dir})))

                             
try:
    hdulist.writeto(psf_dir+"psf/scatter.fits", overwrite=True)
except TypeError:
    hdulist.writeto(psf_dir+"psf/scatter.fits", clobber=True)
hdulist.close()

try:
    hdulist.writeto(f"./plots/scatter/scatter_{target}.fits", overwrite=True)
except TypeError:
    hdulist.writeto(f"./plots/scatter/scatter_{target}.fits", clobber=True)
hdulist.close()
