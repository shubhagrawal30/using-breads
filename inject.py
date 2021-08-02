print("starting")
import sys
from scipy.sparse import data
sys.path.append("/scr3/jruffio/shubh/breads/")
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as pyfits


print("Importing mkl")
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

print("Importing breads")
from breads.instruments.OSIRIS import OSIRIS
from breads.injection import inject_planet

numthreads = 4
dir_name = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"
fil = os.listdir(dir_name)[1]

planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
tr_file = "/scr3/jruffio/data/osiris_survey/targets/SR3/210626/first/reduced/spectra/s210626_a018002_Kn5_020_spectrum.fits"
spec_file = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"+"spectra/"+fil[:-5]+"_spectrum.fits"

dataobj = OSIRIS(dir_name+fil) 
dataobj.remove_bad_pixels()

print("plotting image before injection")
plt.figure()
plt.title("before injection (one slice)")
plt.imshow(dataobj.data[0], origin="lower")
cbar = plt.colorbar()

plt.figure()
plt.title("before injection (median)")
plt.imshow(np.nanmedian(dataobj.data, axis=0), origin="lower")
cbar = plt.colorbar()

inject_planet(dataobj, (-10, -10), planet_btsettl, spec_file, tr_file, 1)

print("plotting image after injection")
plt.figure()
plt.imshow(dataobj.data[0], origin="lower")
plt.title("after injection (one slice)")
cbar = plt.colorbar()

plt.figure()
plt.title("after injection (median)")
plt.imshow(np.nanmedian(dataobj.data, axis=0), origin="lower")
cbar = plt.colorbar()

plt.show()