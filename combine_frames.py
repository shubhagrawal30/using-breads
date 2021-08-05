import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os

# frames_dir = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/planets/REF/"
# frames_dir = "/scr3/jruffio/data/osiris_survey/targets/SR21A/210626/reduced/planets/REF/"
# frames_dir = "/scr3/jruffio/data/osiris_survey/targets/ROXs35A/210628/reduced/planets/REF/"
# frames_dir = "/scr3/jruffio/data/osiris_survey/targets/SR14/210628/reduced/planets/REF/"
# frames_dir = "/scr3/jruffio/data/osiris_survey/targets/ROXs43B/210628/reduced/planets/REF/"
# frames_dir = "/scr3/jruffio/data/osiris_survey/targets/SR9/210628/reduced/planets/REF/"
# frames_dir = "/scr3/jruffio/data/osiris_survey/targets/SR4/210627/reduced/planets/REF/"
frames_dir = "/scr3/jruffio/data/osiris_survey/targets/ROXs44/210627/reduced/planets/REF/"
target = "HD148352"
files = os.listdir(frames_dir)

fluxs = {}
errs = {}

t_flux = 0
t_err_rec = 0

for fil in files:
    if "_out.fits" not in fil:
        print("skipping", fil)
        continue
    # if "a015" not in fil and "a015" not in fil:
    #     print("SKIP", fil)
    #     continue
    print("DOING", fil)
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

snr[snr < -30] = np.nan
# snr[snr > 400] = np.nan

print("frames combined: ", len(fluxs.keys()))
print("max SNR: ", np.nanmax(snr))
x, y = np.unravel_index(np.nanargmax(snr), snr.shape)
xS, yS = np.array(snr.shape) / 2
print("relative position: ", (y - yS, x - xS))

plt.figure()
# plt.imshow(snr, origin="lower", vmin=-20, vmax=25)
plt.imshow(snr, origin="lower")
plt.plot(yS, xS, "rX")
plt.plot(y, x, "b.")
cbar = plt.colorbar()
cbar.set_label("SNR")
plt.title(f"{target}, {(y - yS, x - xS)}, {np.nanmax(snr)}, {len(fluxs.keys())} frames")
plt.savefig(frames_dir+"combined.png")
plt.savefig(f"./plots/combined_{target}.png")
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
    hdulist.writeto(f"./plots/combined_{target}.fits", overwrite=True)
except TypeError:
    hdulist.writeto(f"./plots/combined_{target}.fits", clobber=True)
hdulist.close()



    


