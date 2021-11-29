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
from breads.fm.hc_no_splinefm import hc_no_splinefm
from breads.fm.hc_hpffm import hc_hpffm
import arguments

numthreads = 8
star = "AB_Aur"
dir_name = arguments.dir_name[star]
tr_dir = arguments.tr_dir[star]
sky_calib_file = arguments.sky_calib_file[star]
files = os.listdir(dir_name)

subdirectory = "planets/11292021/"

print("making subdirectories")
Path(dir_name+subdirectory).mkdir(parents=True, exist_ok=True)

print("Reading planet file")
planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
model_wvs = arr[:, 0] / 1e4
model_spec = 10 ** (arr[:, 1] - 8)

tr_files = os.listdir(tr_dir)
if "plots" in tr_files:
    tr_files.remove("plots")
tr_counter = 0
tr_total = len(tr_files)

for filename in files[:]:
    try:
        if ".fits" not in filename:
            print("skipping ", filename)
            continue
        print(filename)
        dataobj = OSIRIS(dir_name+filename) 
        nz,ny,nx = dataobj.data.shape
        # dataobj.noise = np.sqrt(np.abs(dataobj.data))
        print("setting noise")
        dataobj.set_noise()
        # dataobj.noise = np.ones((nz,ny,nx))

        print("sky calibrating")
        dataobj.calibrate(sky_calib_file)


        spec_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
        print("Reading spectrum file", spec_file)
        with pyfits.open(spec_file) as hdulist:
            star_spectrum = hdulist[2].data
            mu_x = hdulist[3].data
            mu_y = hdulist[4].data
            sig_x = hdulist[5].data
            sig_y = hdulist[6].data

        print("setting reference position")
        dataobj.set_reference_position((np.nanmedian(mu_y), np.nanmedian(mu_x)))
        # dataobj.set_reference_position((2, 2))
        print(dataobj.refpos)

        # tr_file = dir_name+"spectra/"+filename[:-5]+"_spectrum.fits"
        # SR3
        tr_counter = (tr_counter + 1) % tr_total
        # tr_file = "/scr3/jruffio/data/osiris_survey/targets/HIP73049/210628/reduced/spectra/s210628_a004" \
        #     + format(tr_counter+4, '03d') + "_Kn5_020_spectrum.fits"
        tr_file = tr_dir + tr_files[tr_counter]

        # +filename[12:-13]
        print("Reading transmission file", tr_file)
        with pyfits.open(tr_file) as hdulist:
            transmission = hdulist[0].data

        print("Removing bad pixels")
        dataobj.remove_bad_pixels(med_spec=star_spectrum)

        # dataobj.trim_data(10)

        # print(dataobj.data.shape, dataobj.wavelengths.shape, transmission.shape, star_spectrum.shape)
        # print(dataobj.wavelengths)

        print("setting planet model")
        minwv,maxwv= np.nanmin(dataobj.wavelengths),np.nanmax(dataobj.wavelengths)
        crop_btsettl = np.where((model_wvs > minwv - 0.2) * (model_wvs < maxwv + 0.2))
        model_wvs = model_wvs[crop_btsettl]
        model_spec = model_spec[crop_btsettl]
        model_broadspec = dataobj.broaden(model_wvs,model_spec)
        planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(dataobj.read_wavelengths, planet_f(dataobj.read_wavelengths) / np.nanmax(planet_f(dataobj.read_wavelengths)), label="planet model")
        plt.plot(dataobj.read_wavelengths, star_spectrum / transmission / np.nanmax(star_spectrum / transmission), label="starlight model")
        plt.xlabel("wavelength")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(dataobj.read_wavelengths, planet_f(dataobj.read_wavelengths) / np.nanmax(planet_f(dataobj.read_wavelengths)), label="planet model")
        plt.plot(dataobj.read_wavelengths, np.abs(star_spectrum / transmission) / np.nanmax(np.abs(star_spectrum / transmission)), label="starlight model")
        plt.xlabel("wavelength")
        plt.legend()
        plt.grid()
        plt.savefig("./plots/TEMP5.png")
        exit()

        # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
        #         "boxw":3,"nodes":20,"psfw":1.2,"badpixfraction":0.75}
        # fm_func = hc_splinefm
        fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
                "boxw":1,"nodes":5,"psfw":(np.nanmedian(sig_y), np.nanmedian(sig_x)), "star_flux":None, # np.nanmean(star_spectrum) * np.size(star_spectrum)
                "badpixfraction":0.75,"optimize_nodes":True}
        print("psfw:", np.nanmedian(sig_y), np.nanmedian(sig_x))
        fm_func = hc_no_splinefm
        # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
        #             "boxw":3,"psfw":1.5,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40}
        # fm_func = hc_hpffm
        rvs = np.array([0])
        ys = np.arange(-40, 40)
        xs = np.arange(-20, 20)

        if True: # Example code to test the forward model
            nonlin_paras = [0, -5, -5] # rv (km/s), y (pix), x (pix)
            # nonlin_paras = [0, 0, 0] # rv (km/s), y (pix), x (pix)
            # d is the data vector a the specified location
            # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
            # s is the vector of uncertainties corresponding to d
            
            # log_prob,log_prob_H0,rchi2,linparas,linparas_err = fitfm(nonlin_paras,dataobj,fm_func,fm_paras)
            # print(log_prob,log_prob_H0,rchi2,linparas,linparas_err)
            d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)

            validpara = np.where(np.sum(M,axis=0)!=0)
            M = M[:,validpara[0]]
            d = d / s
            M = M / s[:, None]
            print(M.shape, d.shape, s.shape, star_spectrum.shape)
            from scipy.optimize import lsq_linear
            paras = lsq_linear(M, d).x
            m = np.dot(M,paras)

            print("plotting")

            plt.figure()
            plt.subplot(2, 1, 1)
            for k in range(M.shape[-1]-1):
                print(k)
                plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="model {0}".format(k+1))
            plt.legend()
            plt.grid()
            plt.subplot(2, 1, 2)
            plt.plot(d,label="data", alpha=0.5)
            plt.plot(paras[0]*M[:,0],label="planet model", alpha=0.5)
            plt.plot(m-paras[0]*M[:,0],label="starlight model")
            plt.legend()
            plt.grid()
            plt.xlabel("wavelength/index")
            plt.savefig("./plots/TEMP4.png")
            
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(dataobj.read_wavelengths, star_spectrum / np.nanmedian(star_spectrum), label= r"star-spectrum $\times$ transmission")
            plt.legend()
            plt.grid()
            plt.subplot(2,1,2)
            plt.plot(dataobj.read_wavelengths, planet_f(dataobj.read_wavelengths) / np.nanmedian(planet_f(dataobj.read_wavelengths)), label= r"planet-spectrum $\times$ transmission")
            plt.legend()
            plt.grid()
            plt.xlabel("wavelength")
            plt.savefig("./plots/TEMP1.png")
            
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(m-paras[0]*M[:,0],label="starlight model")
            plt.plot(paras[0]*M[:,0],label="planet model")
            plt.legend()
            plt.grid()
            plt.subplot(2,1,2)
            plt.plot(d,label="data")
            plt.plot(m,label="combined model")
            plt.legend()
            plt.grid()
            plt.xlabel("wavelength")
            plt.savefig("./plots/TEMP2.png")
            exit()

            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(d,label="data")
            plt.plot(m,label="model")
            plt.plot(paras[0]*M[:,0],label="planet model")
            plt.plot(m-paras[0]*M[:,0],label="starlight model")
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
            for k in range(M.shape[-1]-1):
                print(k)
                plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+1))
            plt.legend()

            paras_H0 = lsq_linear(M[:,1::], d).x
            m_H0 = np.dot(M[:,1::] , paras_H0)
            r_H0 = d  - m_H0
            r = d - m

            plt.figure()
            plt.plot(np.cumsum((r) ** 2),label="Residuals")
            plt.plot(np.cumsum((r_H0) ** 2),label="Residuals H0")
            plt.plot(np.cumsum((r_H0) ** 2 - (r) ** 2),label="Residuals H0 - H1")
            plt.legend()
            plt.show()
            plt.close('all')
            exit()

        print("SNR time")
        out = search_planet([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=numthreads)
        N_linpara = (out.shape[-1]-2)//2
        print(out.shape)

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=out,
            header=pyfits.Header(cards={"TYPE": "output", "FILE": filename, "PLANET": planet_btsettl,\
                                        "FLUX": spec_file, "TRANS": tr_file})))                                  
        try:
            hdulist.writeto(dir_name+subdirectory+filename[:-5]+"_out.fits", overwrite=True)
        except TypeError:
            hdulist.writeto(dir_name+subdirectory+filename[:-5]+"_out.fits", clobber=True)
        hdulist.close()

        plt.figure()
        plt.imshow(out[0,:,:,3]/out[0,:,:,3+N_linpara],origin="lower", vmin=-10, vmax=10)
        cbar = plt.colorbar()
        cbar.set_label("SNR")
        # plt.show()
        plt.savefig(dir_name+subdirectory+filename[:-5]+"_snr.png")
        plt.close()
        print("DONE", filename)
        # break
    except Exception as e:
        print(e)
        print("FAILED", filename)
        # break