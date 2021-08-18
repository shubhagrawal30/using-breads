import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os

star = "HD148352"
fol = "TP"
target = f"{fol}_{star}"

throughput_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/210626/reduced/throughput/{fol}/"
frames_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/210626/reduced/planets/{fol}/"
fr_files = os.listdir(frames_dir)
th_files = os.listdir(throughput_dir)

snrs = {}
t_flux = 0
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
    t_flux += out[0,:,:,3] / (out[0,:,:,3+N_linpara])**2
    t_err_rec += 1 / out[0,:,:,3+N_linpara] ** 2
    # if len(snrs.values()) == 55:
    #     break

t_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

noise_calib = np.nanstd(list(snrs.values()), axis=0)

snr = t_flux / t_err / noise_calib
print("frames combined: ", len(snrs.keys()))
print(np.nanmax(t_flux))
print(np.unravel_index(np.nanargmax(t_flux), t_flux.shape))
xD, yD = np.unravel_index(np.nanargmax(snr), snr.shape)
detection_snr = np.nanmax(snr)
detection_flux = t_flux[xD, yD]
print(detection_snr, (xD, yD), detection_flux)
# exit()


plt.figure()
plt.imshow(noise_calib, origin="lower")
cbar = plt.colorbar()

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
    t_flux += flux / (noise) **2
    t_err_rec += 1 / noise ** 2

t_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

calibrated_err_combined = t_err*noise_calib/(t_flux/flux_ratio)

print("frames combined: ", len(fluxs.keys()))

plt.figure()
plt.imshow(calibrated_err_combined, origin="lower")
cbar = plt.colorbar()


distances = []
values = []
nx, ny = calibrated_err_combined.shape
xS, yS = nx / 2, ny / 2
for x in range(nx):
    for y in range(ny):
        distances += [np.sqrt((y - yS) ** 2 + (x - xS) ** 2) * 20]
        values += [calibrated_err_combined[x, y]]

plt.figure()
plt.scatter(distances, threshold * np.array(values), alpha = 0.2, label="snr 5")
# plt.scatter(distances, detection_snr * np.array(values), alpha = 0.2, label=f"detection snr {detection_snr}")
plt.plot(np.sqrt((yD - yS) ** 2 + (xD - xS) ** 2) * 20, detection_flux / (t_flux/flux_ratio)[xD, yD], 'rX', label="detection")
plt.title(f"{target}")
plt.legend()
plt.grid()
plt.show()



# hdulist = pyfits.HDUList()
# hdulist.append(pyfits.PrimaryHDU(data=calibrated_err_combined,
#     header=pyfits.Header(cards={"TYPE": "noise", "DIR": frames_dir})))
                             
# try:
#     hdulist.writeto(f"./plots/combined_{target}.fits", overwrite=True)
# except TypeError:
#     hdulist.writeto(f"./plots/combined_{target}.fits", clobber=True)
# hdulist.close()
