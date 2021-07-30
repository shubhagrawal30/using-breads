print("importing modules")
import sys, os
sys.path.append("/scr3/jruffio/shubh/breads/")
import breads.calibration as cal
import breads.instruments.OSIRIS as o
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

dir_name = "/scr3/jruffio/data/osiris_survey/targets/ROXs4/210627/reduced/"
files = os.listdir(dir_name)

print("making subdirectories")
Path(dir_name+"spectra/plots/").mkdir(parents=True, exist_ok=True)

print("starting")
for fil in files:
    if ".fits" not in fil:
        continue
    print(fil)
    data = o.OSIRIS(dir_name+fil)
    _ = data.remove_bad_pixels(med_spec="transmission")
    print("extracting")
    obj = cal.extract_star_spectrum(data, verbose=True, calib_filename=dir_name+"spectra/"+fil[:-5]+"_spectrum.fits")
    print("plotting")
    plt.figure()
    plt.plot(obj.read_wavelengths, obj.fluxs)
    plt.savefig(dir_name+"spectra/plots/"+fil[:-5]+"_spectrum.png")
    plt.close()
