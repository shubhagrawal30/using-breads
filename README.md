# using-breads

`./plots/` : some example outputs from the code in this repo.

`get_sky_calibration.py` : do OH line calibration using long exposure sky frames (for wavelength and resolution calibration).

`get_star_calibration.py` : get star spectrum and location from data frames.

`get_telluric_calibration.py` : get star spectrum, transmission, and location from data frames of standard stars.

`get_planet.py` : generate SNR maps for a given data set.

`analyze_planet.py` : once a candidate has been detected, code here can analyze that location and try to constrain characteristics like effective temperature and surface gravity.

`SNR_maps.py` : plot some single frame SNR maps together to validate detections when not using reference position (say, when star is not in the FOV of frame).

`get_RV.py` : generate RV posteriors and SNR in RV direction once possible candidate has been located.

`checks.py` : code used to check the forward models made and the data taken from OSIRIS.

`combine_frames.py` : combine X-Y SNR frames once generated and stored in a directory.

`combine_RV.py` : combine SNR data in the RV direction to get CCF for all data combined.

