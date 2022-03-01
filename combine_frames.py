import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os
import arguments as args

star = "HD148352"
date = args.dates[star]
fol = "20220301"

frames_dir = args.dir_name[star] + f"planets/{fol}/"
target = f"{fol}_{star}"
files = os.listdir(frames_dir)

fluxs = {}
errs = {}
snrs = {}

if star in args.rotated_seqs.keys():
    rotated_seqs = args.rotated_seqs[star]
else:
    rotated_seqs = []

for fil in files:
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
    t_flux = np.zeros((n, n))
    t_err_rec = np.zeros((n, n))
    break

for fil in files:
    if "_out.fits" not in fil:
        print("skipping", fil)
        continue
    print("DOING", fil)
    with pyfits.open(frames_dir + fil) as hdulist:
        linparas = hdulist[3].data
        linparas_err = hdulist[4].data
    flux = np.pad(linparas[0,:,:,0], padding, constant_values=np.nan)
    err = np.pad(linparas_err[0,:,:,0], padding, constant_values=np.nan)
    if fil[8:12] in rotated_seqs: # add not if other way TODO
        print("rotated 90", fil)
        flux = np.swapaxes(flux, 0, 1)
        err = np.swapaxes(err, 0, 1)
    fluxs[fil] = flux; errs[fil] = err; snrs[fil] = flux / err
    f = flux / (err)**2
    e = 1 / err ** 2
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
# plt.imshow(snr, origin="lower")
plt.imshow(snr, origin="lower", vmin=-15, vmax=15)
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
# plt.imshow(snr, origin="lower")
plt.imshow(snr, origin="lower", vmin=-15, vmax=15)
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