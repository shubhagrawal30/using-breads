import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os

# frames_dir = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/planets/REF/"
frames_dir = "/scr3/jruffio/data/osiris_survey/targets/SR21A/210626/reduced/planets/REF/"
files = os.listdir(frames_dir)

fluxs = {}
errs = {}

t_flux = 0
t_err_rec = 0

for fil in files:
    if "_out.fits" not in fil:
        print("skipping", fil)
        continue
    print(fil)
    with pyfits.open(frames_dir + fil) as hdulist:
        out = hdulist[0].data
    N_linpara = (out.shape[-1]-2)//2
    fluxs[fil] = out[0,:,:,3]
    errs[fil] = out[0,:,:,3+N_linpara]
    t_flux += out[0,:,:,3] / (out[0,:,:,3+N_linpara])**2
    t_err_rec += 1 / out[0,:,:,3+N_linpara] ** 2

t_flux /= t_err_rec
t_err = 1 / np.sqrt(t_err_rec)

snr = t_flux / t_err

print("frames combined: ", len(fluxs.keys()))
print("max SNR: ", np.nanmax(snr))
x, y = np.unravel_index(np.nanargmax(snr), snr.shape)
xS, yS = np.array(snr.shape) / 2
print("relative position: ", (y - yS, x - xS))

plt.figure()
plt.imshow(snr,origin="lower", vmin=0, vmax=5)
plt.plot(yS, xS, "rX")
plt.plot(y, x, "b.")
cbar = plt.colorbar()
cbar.set_label("SNR")
plt.savefig(frames_dir+"combined.png")
plt.savefig("./plots/combined_SR21A.png")
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
    hdulist.writeto("./plots/combined_SR21A.fits", overwrite=True)
except TypeError:
    hdulist.writeto("./plots/combined_SR21A.fits", clobber=True)
hdulist.close()



    


