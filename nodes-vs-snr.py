print("starting")
import sys
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as pyfits 

star = str(sys.argv[1])
print(star)
fol = "TP"
target = f"{fol}_{star}"
nodes_file = f"./plots/nodes-vs/nodes_{target}.fits"

b_flux, b_err_rec, t_flux, t_err_rec, bsnr = {}, {}, {}, {}, {}
tp, tp_err, noise_calib, noi, noi_err = [], [], {}, [], []
with pyfits.open(nodes_file) as hdulist:
    num_nodes = hdulist[0].data
    for num_node in num_nodes:



for temp in temperatures:
    noise_calib[temp] = np.nanstd(list(bsnr[temp].values()), axis=0)
    rflux = (t_flux[temp] - b_flux[temp]) / t_err_rec[temp]
    rerr = 1 / np.sqrt(t_err_rec[temp])
    snr_vals = rflux / rerr / noise_calib[temp]
    plt.figure(1)
    for val in snr_vals:
        plt.plot(temp, val, "bx")
    snr += [np.nanmean(snr_vals)]
    snr_err += [np.nanstd(snr_vals)]

plt.figure(1)
plt.xlabel("temp / 100")
plt.ylabel("SNR")
plt.axvline(x=injected_temp, color="black", ls='dotted')
plt.savefig(f"./plots/temp_recover/{injected_temp}/snr1_{target}.png")
plt.savefig(dir_name+subdirectory+f"snr1_{target}.png")

plt.figure(4)
plt.errorbar(temperatures, snr, yerr=snr_err)
plt.xlabel("temp / 100")
plt.ylabel("SNR")
plt.axvline(x=injected_temp, color="black", ls='dotted')
plt.savefig(f"./plots/temp_recover/{injected_temp}/snr2_{target}.png")
plt.savefig(dir_name+subdirectory+f"snr2_{target}.png")