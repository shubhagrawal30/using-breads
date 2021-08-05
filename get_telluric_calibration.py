import sys, os
sys.path.append("/scr3/jruffio/shubh/breads/")
import breads.calibration as cal
import breads.instruments.OSIRIS as o
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

# dir_name = "/scr3/jruffio/data/osiris_survey/targets/SR3/210628/second/reduced/"
dir_name = "/scr3/jruffio/data/osiris_survey/targets/HIP73049/210628/reduced/"
files = os.listdir(dir_name)

print("making subdirectories")
Path(dir_name+"spectra/plots").mkdir(parents=True, exist_ok=True)

wvs_phoenix = "/scr3/jruffio/data/kpic/models/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
A0_phoenix = "/scr3/jruffio/models/phoenix/fitting/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte09000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
star_spectrum = (wvs_phoenix, A0_phoenix)

for fil in files:
    if ".fits" not in fil:
        continue
    print(fil)
    data = o.OSIRIS(dir_name+fil)
    _ = data.remove_bad_pixels(med_spec="transmission")
    obj = cal.telluric_calibration(data, star_spectrum, mask=True, verbose=True, calib_filename=dir_name+"spectra/"+fil[:-5]+"_spectrum.fits")
    print("plotting")
    plt.figure()
    plt.plot(obj.read_wavelengths, obj.transmission)
    plt.savefig(dir_name+"spectra/plots/"+fil[:-5]+"_transmission.png")
    plt.close()
    plt.figure()
    plt.plot(obj.read_wavelengths, obj.fluxs)
    plt.savefig(dir_name+"spectra/plots/"+fil[:-5]+"_flux.png")
    plt.close()
