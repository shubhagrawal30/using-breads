print("starting")
import sys
sys.path.append("/scr3/jruffio/shubh/breads/")
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os
from breads.instruments.OSIRIS import OSIRIS
from pathlib import Path
import arguments as args

# star = "HD148352"
star = "AB_Aur"

date = args.dates[star] + "/1"
fol = "20220410"
# fol = "09232021"
target = f"{fol}_{star}"

throughput_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/throughput/{fol}/"
frames_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/planets/{fol}/"
out_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/psf/{fol}/"
psf_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/"
fr_files = os.listdir(frames_dir)
th_files = os.listdir(throughput_dir)
psf_files = os.listdir(psf_dir)

print("making subdirectories")
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(f"./plots/scatter/{fol}/").mkdir(parents=True, exist_ok=True)

snrs = {}
to_flux = 0
t_err_rec = 0
num = 65

if star in args.rotated_seqs.keys():
    rotated_seqs = args.rotated_seqs[star]
else:
    rotated_seqs = []

for fil in fr_files:
    if "_out.fits" not in fil:
        continue
    print("setting size")
    with pyfits.open(frames_dir + fil) as hdulist:
        linparas = hdulist[3].data
    nx, ny = linparas[0,:,:,0].shape
    n = max(nx, ny)
    pad = (n - min(nx, ny)) // 2
    if nx > ny:
        padding=((0,0),(pad,pad))
    else:
        padding=((pad,pad),(0,0))
    # do not know if this works with odd size maps
    to_flux = np.zeros((n, n))
    t_err_rec = np.zeros((n, n))
    break

for fil in fr_files:
    if "_out.fits" not in fil:
        print("fr skipping", fil)
        continue
    print("fr DOING", fil)
    with pyfits.open(frames_dir + fil) as hdulist:
        linparas = hdulist[3].data
        linparas_err = hdulist[4].data
    
    flux = np.pad(linparas[0,:,:,0], padding, constant_values=np.nan)
    err = np.pad(linparas_err[0,:,:,0], padding, constant_values=np.nan)
    if fil[8:12] in rotated_seqs: # add not if other way TODO
        # continue
        print("rotated 90", fil)
        flux = np.rot90(flux, 1, (0, 1))
        err = np.rot90(err, 1, (0, 1))
    f = flux / (err)**2
    e = 1 / err ** 2
    nan_locs = np.logical_or(np.isnan(f), np.isnan(e))
    f[nan_locs] = 0
    e[nan_locs] = 0
    to_flux += f
    t_err_rec += e
    snrs[fil] = flux / err
    if len(snrs.values()) == num:
        break

to_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

noise_calib = np.nanstd(list(snrs.values()), axis=0)
snr = to_flux / t_err / noise_calib
snr[np.isinf(snr)] = np.nan
# snr[np.nanpercentile(snr, 5) > snr] = np.nan
# snr[np.nanpercentile(snr, 95) < snr] = np.nan
# print(np.nanpercentile(snr, 5), np.nanpercentile(snr, 95))

# snr[snr < -50] = np.nan
snr[snr > 80] = np.nan

print("frames combined: ", len(snrs.keys()))
print(np.nanmax(to_flux))
print(np.unravel_index(np.nanargmax(to_flux), to_flux.shape))
xD, yD = np.unravel_index(np.nanargmax(snr), snr.shape)
detection_snr = np.nanmax(snr)
detection_flux = to_flux[xD, yD]
print(detection_snr, (xD, yD), detection_flux)
# exit()

plt.figure()
plt.imshow(snr, origin="lower")#, vmin=0, vmax=60)
plt.plot(yD, xD, "rx")
cbar = plt.colorbar()
# plt.show()

plt.figure()
plt.imshow(to_flux, origin="lower", vmin=0, vmax=0.01)
cbar = plt.colorbar()

# plt.show()

flux_ratio = 1e-2
threshold = 5

fluxs = {}
noises = {}
t_flux = 0
t_err_rec = 0

for fil in th_files:
    if "_out.fits" not in fil:
        continue
    print("setting size")
    with pyfits.open(throughput_dir + fil) as hdulist:
        flux = hdulist[0].data
    nx, ny = flux.shape
    n = max(nx, ny)
    pad = (n - min(nx, ny)) // 2
    if nx > ny:
        padding=((0,0),(pad,pad))
    else:
        padding=((pad,pad),(0,0))
    # do not know if this works with odd size maps
    t_flux = np.zeros((n, n))
    t_err_rec = np.zeros((n, n))
    break

for fil in th_files:
    if "_out.fits" not in fil:
        print("th skipping", fil)
        continue
    print("th DOING", fil)
    with pyfits.open(throughput_dir + fil) as hdulist:
        flux = hdulist[0].data
        noise = hdulist[1].data
    flux = np.pad(flux, padding, constant_values=np.nan)
    err = np.pad(noise, padding, constant_values=np.nan)
    # if fil[8:12] in rotated_seqs: # add not if other way TODO
    #     # continue
    #     print("rotated 90", fil)
    #     flux = np.rot90(flux, 1, (0, 1))
    #     err = np.rot90(err, 1, (0, 1))
    fluxs[fil] = flux; noises[fil] = err
    f = flux / (err)**2
    e = 1 / err ** 2
    nan_locs = np.logical_or(np.isnan(f), np.isnan(e))
    f[nan_locs] = 0
    e[nan_locs] = 0
    t_flux += f
    t_err_rec += e
    if len(fluxs.values()) == num:
        break

t_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

plt.figure()
plt.imshow(t_flux, origin="lower", vmin=0, vmax=0.01)
cbar = plt.colorbar()

plt.figure()
plt.imshow(t_err, origin="lower")
cbar = plt.colorbar()

plt.figure()
plt.imshow(noise_calib, origin="lower")
cbar = plt.colorbar()

plt.figure()
plt.imshow((t_flux - to_flux)/flux_ratio, origin="lower")
cbar = plt.colorbar()

plt.show()

# padding assumes star is at center of every frame
# nf, _ = t_flux.shape
# ni, _ = to_flux.shape # supposed to be square
# if nf > ni:
#     padding = ((nf-ni)//2, (nf-ni)//2), ((nf-ni)//2, (nf-ni)//2)
#     to_flux = np.pad(to_flux, padding, constant_values=np.nan) 
#     noise_calib = np.pad(noise_calib, padding, constant_values=np.nan)
# else:
#     padding = ((ni-nf)//2, (ni-nf)//2), ((ni-nf)//2, (ni-nf)//2)
#     t_flux = np.pad(t_flux, padding, constant_values=np.nan) 
#     t_err = np.pad(t_err, padding, constant_values=np.nan) 


throughput = (t_flux-to_flux) / flux_ratio

calibrated_err_combined = t_err * noise_calib / throughput

print("frames combined: ", len(fluxs.keys()))

assert(len(fluxs.keys()) == len(snrs.values())), f"frames combined not same size, {len(fluxs.keys())}, {len(snrs.values())}"

plt.figure()
plt.imshow(throughput, origin="lower", vmin=0, vmax=1)
cbar = plt.colorbar()
cbar.set_label("throughput")

plt.figure()
plt.imshow(calibrated_err_combined, origin="lower")
cbar = plt.colorbar()
cbar.set_label("noise")

plt.figure()
plt.imshow(np.log10(calibrated_err_combined), origin="lower")
cbar = plt.colorbar()
cbar.set_label("log10(noise)")


# plt.show()
# exit()


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
count = 0
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
    count += 1
    if count == num:
        break
distances = np.array(distances)

plt.figure()
plt.scatter(psf_dist, np.array(psf_profile) / np.nanmax(psf_profile), marker=",", color="black", label="scaled PSF profile", alpha = 0.01)
plt.scatter(distances, threshold * np.array(values), marker=",", alpha = 0.2, label="snr 5")
plt.scatter(distances, detection_snr * np.array(values), marker=",", alpha = 0.2, label=f"detection snr {detection_snr}")
plt.plot(np.sqrt((yD - yS) ** 2 + (xD - xS) ** 2) * 20, detection_flux / (t_flux/flux_ratio)[xD, yD], 'rX', label="detection")
# plt.plot(psf_dist, profile / np.nanmean(profile) * np.nanmean(threshold * np.array(values)), "black", label="scaled PSF profile")
plt.title(f"{target}")
plt.xlim([np.nanmin(distances[distances > 0]) * 0.8, np.nanmax(distances) / 0.8])
print("xlim", [np.nanmin(distances[distances > 0]) * 0.8, np.nanmax(distances) / 0.8])
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
# plt.savefig(out_dir+"scatter.png")
# plt.savefig(f"./plots/scatter/{fol}/scatter_{target}.png")
plt.show()

# hdulist = pyfits.HDUList()
# hdulist.append(pyfits.PrimaryHDU(data=calibrated_err_combined,
#     header=pyfits.Header(cards={"TYPE": "noise", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=throughput,
#     header=pyfits.Header(cards={"TYPE": "throughput", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=noise_calib,
#     header=pyfits.Header(cards={"TYPE": "noise_calib", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=snr,
#     header=pyfits.Header(cards={"TYPE": "snr", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=np.vstack((psf_dist, psf_profile)),
#     header=pyfits.Header(cards={"TYPE": "dist_psf", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=np.vstack((distances, values)),
#     header=pyfits.Header(cards={"TYPE": "dist_noise", "DIR": frames_dir})))

                             
# try:
#     hdulist.writeto(out_dir+"scatter.fits", overwrite=True)
# except TypeError:
#     hdulist.writeto(out_dir+"scatter.fits", clobber=True)
# hdulist.close()

# try:
#     hdulist.writeto(f"./plots/scatter/{fol}/scatter_{target}.fits", overwrite=True)
# except TypeError:
#     hdulist.writeto(f"./plots/scatter/{fol}/scatter_{target}.fits", clobber=True)
# hdulist.close()
