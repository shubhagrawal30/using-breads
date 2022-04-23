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
rotated_seqs[star] = ["a025", "a026", "a027", "a028"]

star = "DS_Tau"
dates[star] = "211018"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/2/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a008002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a035", "a036", "a037", "a038"]

star = "LkCa15"
dates[star] = "211018"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/3/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a008002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a052", "a053", "a054", "a055", "a056"]

star = "LkCa19"
dates[star] = "211019"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/1/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a005002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a016", "a017", "a018", "a019", "a020"]

star = "HBC388"
dates[star] = "211019"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/2/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a005002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a022", "a023", "a024", "a025", "a026", "a027"]

star = "GM_Aur"
dates[star] = "211019"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/2/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a005002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a038", "a039", "a040", "a041", "a042"]

star = "HN_Tau"
dates[star] = "211019"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/3/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a005002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a056", "a057", "a058", "a059", "a060", "a061", "a062", "a063", "a064"]

star = "HBC354"
dates[star] = "211020"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/1/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a056", "a057", "a058", "a059"]

star = "HBC392"
dates[star] = "211020"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/2/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a060", "a061", "a062", "a063", "a064", "a065"]

star = "HBC372"
dates[star] = "211020"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/2/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a080", "a081", "a082", "a083", "a084"]

star = "HBC353"
dates[star] = "211020"
dir_name[star] = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{dates[star]}/reduced/"
tr_dir[star] = f"/scr3/jruffio/data/osiris_survey/targets/AB_Aur/{dates[star]}/2/reduced/spectra/"
sky_calib_file[star] = f"/scr3/jruffio/data/osiris_survey/targets/calibration_skys/{dates[star]}/reduced/s{dates[star]}_a003002_Kn3_020_calib.fits"
rotated_seqs[star] = ["a098", "a099", "a100", "a101", "a102"]