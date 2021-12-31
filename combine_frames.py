import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os

star = "SR14"
fol = "09232021_wide"

frames_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/210628/reduced/planets/{fol}/"
target = f"{fol}_{star}"
files = os.listdir(frames_dir)

fluxs = {}
errs = {}
snrs = {}

t_flux = 0
t_err_rec = 0

for fil in files:
    if "_out.fits" not in fil:
        print("skipping", fil)
        continue
    # if "a049" not in fil and "a049" not in fil:
    #     print("SKIP", fil)
    #     continue
    print("DOING", fil)
    with pyfits.open(frames_dir + fil) as hdulist:
        out = hdulist[0].data
    N_linpara = (out.shape[-1]-2)//2
    fluxs[fil] = out[0,:,:,3]
    errs[fil] = out[0,:,:,3+N_linpara]
    snrs[fil] = out[0,:,:,3] / out[0,:,:,3+N_linpara]
    f = out[0,:,:,3] / (out[0,:,:,3+N_linpara])**2
    e = 1 / out[0,:,:,3+N_linpara] ** 2
    nan_locs = np.logical_or(np.isnan(f), np.isnan(e))
    f[nan_locs] = 0
    e[nan_locs] = 0
    t_flux += f
    t_err_rec += e

t_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

noise_calib = np.nanstd(list(snrs.values()), axis=0)

plt.figure()
plt.imshow(noise_calib, origin="lower", vmin=-2, vmax=2)
cbar = plt.colorbar()

# snr = t_flux / t_err / noise_calib
snr = t_flux / t_err 

# snr[snr < -50] = np.nan
# snr[snr > 50] = np.nan

print("frames combined: ", len(fluxs.keys()))
print("max SNR: ", np.nanmax(snr))
x, y = np.unravel_index(np.nanargmax(snr), snr.shape)
xS, yS = np.array(snr.shape) / 2
print("relative position: ", (y - yS, x - xS))

plt.figure()
plt.imshow(snr, origin="lower")
# plt.imshow(snr, origin="lower")
plt.plot(yS, xS, "rX")
plt.plot(y, x, "b.")
cbar = plt.colorbar()
cbar.set_label("SNR")
plt.title(f"{target}, {(y - yS, x - xS)}, {np.nanmax(snr)}, {len(fluxs.keys())} frames")
plt.savefig(frames_dir+"combined.png")
plt.savefig(f"./plots/combined/combined_{target}.png")

snr = snr / noise_calib

print("frames combined: ", len(fluxs.keys()))
print("max SNR: ", np.nanmax(snr))
x, y = np.unravel_index(np.nanargmax(snr), snr.shape)
xS, yS = np.array(snr.shape) / 2
print("relative position: ", (y - yS, x - xS))

plt.figure()
plt.imshow(snr, origin="lower")
# plt.imshow(snr, origin="lower")
plt.plot(yS, xS, "rX")
plt.plot(y, x, "b.")
cbar = plt.colorbar()
cbar.set_label("SNR noise calib")
plt.title(f"{target}, {(y - yS, x - xS)}, {np.nanmax(snr)}, {len(fluxs.keys())} frames")
plt.savefig(frames_dir+"combined_noisecalib.png")
plt.savefig(f"./plots/combined/combined_noisecalib_{target}.png")

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=snr,
    header=pyfits.Header(cards={"TYPE": "snr", "DIR": frames_dir}))) 
hdulist.append(pyfits.PrimaryHDU(data=t_flux,
    header=pyfits.Header(cards={"TYPE": "total flux", "DIR": frames_dir})))    
hdulist.append(pyfits.PrimaryHDU(data=t_err,
    header=pyfits.Header(cards={"TYPE": "total err", "DIR": frames_dir})))   
                             
try:
    hdulist.writeto(f"./plots/combined/combined_{target}.fits", overwrite=True)
except TypeError:
    hdulist.writeto(f"./plots/combined/combined_{target}.fits", clobber=True)
hdulist.close()

plt.show()
plt.close()