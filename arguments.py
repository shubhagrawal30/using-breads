dates = {}
dir_name = {}
tr_dir = {}
sky_calib_file = {}
rotated_seqs = {}

star = "HD148352"
dates[star] = "210626"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/first/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"

star = "SR3"
dates[star] = "210626"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/first/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/first/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"

star = "SR21A"
dates[star] = "210626"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/first/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"

star = "SR4"
dates[star] = "210627"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/HIP73049/{dates[star]}/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"

star = "ROXs44"
dates[star] = "210627"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/first/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"

star = "ROXs8"
dates[star] = "210627"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/first/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"

star = "ROXs4"
dates[star] = "210627"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/second/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"

star = "ROXs35A"
dates[star] = "210628"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/HIP73049/{dates[star]}/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a002002_Kn3_020_calib.fits"

star = "SR14"
dates[star] = "210628"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/first/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a002002_Kn3_020_calib.fits"

star = "ROXs43B"
dates[star] = "210628"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/first/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a002002_Kn3_020_calib.fits"

star = "SR9"
dates[star] = "210628"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/SR3/{dates[star]}/second/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a002002_Kn3_020_calib.fits"

star = "AB_Aur"
dates[star] = "211018"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/1/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/1/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a008002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a013", "a014", "a015", "a016", "a017", "a018"]

# star = "AB_Aur"
# dates[star] = "211018"
# dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/2/reduced/"
# tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/2/reduced/spectra/"
# sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a008002_Kn3_020_calib.fits"
# rotated_seqs[star] = ["a013", "a014", "a015", "a016", "a017", "a018"]

star = "CW_Tau"
dates[star] = "211018"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/1/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a008002_Kn3_020_calib.fits"

star = "DS_Tau"
dates[star] = "211018"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/2/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a008002_Kn3_020_calib.fits"

star = "LkCa15"
dates[star] = "211018"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/3/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a008002_Kn3_020_calib.fits"
