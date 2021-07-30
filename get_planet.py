print("starting")
import sys
sys.path.append("/scr3/jruffio/shubh/breads/")
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as pyfits
from pathlib import Path
from breads.fit import fitfm


print("Importing mkl")
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

print("Importing breads")
from breads.instruments.OSIRIS import OSIRIS
from breads.search_planet import search_planet
from breads.fm.hc_splinefm import hc_splinefm

numthreads = 4
# dir_name = "/scr3/jruffio/data/osiris_survey/targets/HD148352/210626/reduced/"
dir_name = "/scr3/jruffio/data/osiris_survey/targets/SR21A/210626/reduced/"
files = os.listdir(dir_name)

# print(files.index("s210626_a033011_Kn5_020.fits"))
# exit()

print("making subdirectories")
Path(dir_name+"planets/REF/").mkdir(parents=True, exist_ok=True)

print("Reading planet file")
planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
model_wvs = arr[:, 0] / 1e4
model_spec = 10 ** (arr[:, 1] - 8)

tr_counter = 0
tr_total = 6

for filename in files[11:]:
    if ".fits" not in filename:
        print("skipping ", filename)
        continue
    print(filename)
    dataobj = OSIRIS(dir_name+filename) 
    nz,ny,nx = dataobj.data.shape
    dataobj.noise = np.ones((nz,ny,nx))

    sky_calib_file = "/scr3/jruffio/data/osiris_survey/targets/calibration_skys/210626/reduced/s210626_a004002_Kn3_020_calib.fits"
    dataobj.calibrate(sky_calib_file)

    spec_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
    print("Reading spectrum file", spec_file)
    with pyfits.open(spec_file) as hdulist:
        star_spectrum = hdulist[2].data
        mu_x = hdulist[3].data
        mu_y = hdulist[4].data

    print("setting reference position")
    dataobj.set_reference_position((np.nanmedian(mu_y), np.nanmedian(mu_x)))
    # dataobj.set_reference_position((2, 2))
    print(dataobj.refpos)

    # tr_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
    # SR3
    tr_counter = (tr_counter + 1) % tr_total
    tr_file = "/scr3/jruffio/data/osiris_survey/targets/SR3/210626/second/reduced/spectra/s210626_a037" \
        + format(tr_counter+1, '03d') + "_Kn5_020_spectrum.fits"
    
    # +filename[12:-13]
    print("Reading transmission file", tr_file)
    with pyfits.open(tr_file) as hdulist:
        transmission = hdulist[0].data

    print("Removing bad pixels")
    dataobj.remove_bad_pixels(med_spec=star_spectrum)

    print("setting planet model")
    minwv,maxwv= np.nanmin(dataobj.wavelengths),np.nanmax(dataobj.wavelengths)
    crop_btsettl = np.where((model_wvs > minwv - 0.2) * (model_wvs < maxwv + 0.2))
    model_wvs = model_wvs[crop_btsettl]
    model_spec = model_spec[crop_btsettl]
    model_broadspec = dataobj.broaden(model_wvs,model_spec)
    planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

    fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
            "boxw":3,"nodes":4,"psfw":1.2,"nodes":5,"badpixfraction":0.75}
    fm_func = hc_splinefm
    rvs = np.array([0])
    ys = np.arange(-30, 30)
    xs = np.arange(-30, 30)

    print(dataobj.data.shape, dataobj.wavelengths.shape, transmission.shape, star_spectrum.shape)

    if True: # Example code to test the forward model
        nonlin_paras = [0, -17, 4] # rv (km/s), y (pix), x (pix)
        # d is the data vector a the specified location
        # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
        # s is the vector of uncertainties corresponding to d

        log_prob,log_prob_H0,rchi2,linparas,linparas_err = fitfm(nonlin_paras,dataobj,fm_func,fm_paras)
        print(log_prob,log_prob_H0,rchi2,linparas,linparas_err)
        d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)

        validpara = np.where(np.sum(M,axis=0)!=0)
        M = M[:,validpara[0]]
        d = d / s
        M = M / s[:, None]
        print(M.shape, d.shape, s.shape)
        from scipy.optimize import lsq_linear
        paras = lsq_linear(M, d).x
        m = np.dot(M,paras)

        print("plotting")

        plt.subplot(2,1,1)
        plt.plot(d,label="data")
        plt.plot(m,label="model")
        plt.plot(paras[0]*M[:,0],label="planet model")
        plt.plot(m-paras[0]*M[:,0],label="starlight model")
        plt.subplot(2,1,2)
        plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
        for k in range(M.shape[-1]-1):
            print(k)
            plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+1))
        plt.legend()
        plt.show()
        plt.close('all')
        exit()

    print("SNR time")
    out = search_planet([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=None)
    N_linpara = (out.shape[-1]-2)//2
    print(out.shape)

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=out,
        header=pyfits.Header(cards={"TYPE": "output", "FILE": filename, "PLANET": planet_btsettl,\
                                    "FLUX": spec_file, "TRANS": tr_file})))                                  
    try:
        hdulist.writeto(dir_name+"planets/REF/"+filename[:-5]+"_out.fits", overwrite=True)
    except TypeError:
        hdulist.writeto(dir_name+"planets/REF/"+filename[:-5]+"_out.fits", clobber=True)
    hdulist.close()

    plt.figure()
    plt.imshow(out[0,:,:,3]/out[0,:,:,3+N_linpara],origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    plt.savefig(dir_name+"planets/REF/"+filename[:-5]+"_snr.png")
    plt.close()

