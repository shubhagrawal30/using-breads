import sys, os
sys.path.append("/scr3/jruffio/shubh/breads/")
import breads.calibration as cal
import breads.instruments.OSIRIS as o

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

# dir_name = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210626/reduced/"
# dir_name = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210627/reduced/"
dir_name = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210628/reduced/"
# dir_name = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/211020/reduced/"
# dir_name = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/220625/reduced/"

files = os.listdir(dir_name)

for fil in files:
    if ".fits" not in fil or "calib" in fil:
        continue
    print(fil)
    data = o.OSIRIS(dir_name+fil)
    data.remove_bad_pixels()
    SkyCalibrationObj = cal.sky_calibration(data, zero_order=True, \
        calib_filename=dir_name+fil[:-5]+"_calib.fits")
