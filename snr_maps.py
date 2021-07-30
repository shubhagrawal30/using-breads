import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

candidate = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"

frames = ["s210626_a032009_Kn5_020.fits", "s210626_a033003_Kn5_020.fits", "s210626_a034004_Kn5_020.fits"]
for frame in frames:
    target = candidate + frame
    snr = candidate + "planets/" + frame[:-5] + "_out.fits"
    star = pyfits.open(target)[0].data
    out = pyfits.open(snr)[0].data
    N_linpara = (out.shape[-1]-2)//2
    val = out[0,:,:,3]/out[0,:,:,3+N_linpara]
    x, y = np.unravel_index(np.nanargmax(star[:, :, 200]), star[:,:, 200].shape)
    Y, X = np.unravel_index(np.nanargmax(val), val.shape)
    print(x, y, X, Y)
    plt.figure()
    plt.title(frame[:-5])
    plt.imshow(val, origin="lower")
    plt.plot(x, y, "rX")
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    plt.savefig("./plots/"+frame[:-5]+"_snr.png")
    plt.close()
    print(np.sqrt((x-X)**2 + (y-Y)**2))
    print(np.arctan((y-X)/(x-Y)))
