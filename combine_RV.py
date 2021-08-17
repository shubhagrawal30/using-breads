import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os

frames_dir = "/scr3/jruffio/data/osiris_data/kap_And/20210809/reduced/planets/rv/"
target = "kappa And b"
files = os.listdir(frames_dir)
sequence = "a004"

fluxs = {}
errs = {}

t_flux = 0
t_err_rec = 0

for fil in files:
    if "_out.fits" not in fil:
        print("skipping", fil)
        continue
    if sequence not in fil:
        print("SKIP", fil)
        continue
    print("DOING", fil)
    with pyfits.open(frames_dir + fil) as hdulist:
        out = hdulist[0].data
    N_linpara = (out.shape[-1]-2)//2
    fluxs[fil] = out[:,0,0,3]
    errs[fil] = out[:,0,0,3+N_linpara]
    t_flux += out[:,0,0,3] / (out[:,0,0,3+N_linpara])**2
    t_err_rec += 1 / out[:,0,0,3+N_linpara] ** 2

t_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

snr = t_flux / t_err

# snr[snr < -20] = np.nan
# snr[snr > 25] = np.nan
rvs = np.linspace(-4000,4000,4001) 
print("frames combined: ", len(fluxs.keys()))
print("max SNR: ", np.nanmax(snr))
rv = rvs[np.nanargmax(snr)]
# xS, yS = np.array(snr.shape) / 2
# print("relative position: ", (y - yS, x - xS))

plt.figure()
plt.plot(rvs, snr)
plt.ylabel("SNR")
plt.xlabel("RV (km/s)")
plt.title(f"{target}, {sequence}, {rv}, {np.nanmax(snr)}, {len(fluxs.keys())} frames")
plt.savefig(frames_dir+f"combined_{sequence}.png")
plt.savefig(f"./plots/combined_{target}_{sequence}_rv.png")
plt.show()
plt.close()
hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=snr,
    header=pyfits.Header(cards={"TYPE": "snr", "DIR": frames_dir}))) 
hdulist.append(pyfits.PrimaryHDU(data=t_flux,
    header=pyfits.Header(cards={"TYPE": "total flux", "DIR": frames_dir})))    
hdulist.append(pyfits.PrimaryHDU(data=t_err,
    header=pyfits.Header(cards={"TYPE": "total err", "DIR": frames_dir})))   
                             
try:
    hdulist.writeto(f"./plots/combined_{target}_{sequence}_rv.fits", overwrite=True)
except TypeError:
    hdulist.writeto(f"./plots/combined_{target}_{sequence}_rv.fits", clobber=True)
hdulist.close()



    


