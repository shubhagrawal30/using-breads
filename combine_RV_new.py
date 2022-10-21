print("starting")
import sys
sys.path.append("/scr3/jruffio/shubh/breads/")
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os
from breads.instruments.OSIRIS import OSIRIS
from pathlib import Path
import arguments as args
from scipy.interpolate import interp1d
from scipy.signal import correlate2d, correlate
from scipy.interpolate import interp1d
from scipy.stats import t
from pathlib import Path
from glob import glob

# not rotation friendly

# star = "CW_Tau"
# star = "LkCa19"
# star = "DS_Tau"
# star = "GM_Aur"
# star = "HBC353"
# star = "HBC372"
# star = "HBC388"
# star = "HBC392"
star = "HD148352"
# star = "HN_Tau"
# star = "LkCa15"
# star = "LkCa19"
# star = "ROXs4"
# star = "ROXs8"
# star = "ROXs35A"
# star = "ROXs43B"
# star = "ROXs44"
# star = "SR3"
# star = "SR4"

# star = "HN_Tau"
# star = "ROXs8"
# star = sys.argv[1]

# dates = ["210626/first", "210626/second", "210627/first", "210627/second", "210628/first", "210628/second"]

dates = [args.dates[star]]# + "/1"# + "/first"
th_fol = "20220512_1700K"
fr_fol = "20220512_1700K"
# fr_fol = "20220512"
fol = "20220512_1700K"

rvs = np.linspace(-4000,4000,41)

# fr_fol = "20220409"
# fol = "20220409"
target = f"{fr_fol}_{th_fol}_{star}"

fr_files, th_files, psf_files = [], [], []

for date in dates:
    throughput_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/throughput/{th_fol}/"
    frames_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/planets/{fr_fol}/"
    psf_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/"
    fr_files += os.listdir(frames_dir)
    th_files += os.listdir(throughput_dir)
    psf_files += os.listdir(psf_dir)

# flux_ratio = 1e-2
# threshold = 5

main_out_dir = f"./plots/paper/hd148352/"

print("making subdirectories")
Path(main_out_dir).mkdir(parents=True, exist_ok=True)

snrs = {}
# num = 40

if star in args.rotated_seqs.keys():
    rotated_seqs = args.rotated_seqs[star]
else:
    rotated_seqs = []

for filename in fr_files:
    fil = os.path.basename(filename)
    if "_out.fits" not in fil:
        continue
    print("setting size")
    print(fil)
    with pyfits.open(frames_dir + fil) as hdulist:
        linparas = hdulist[3].data
    # print(linparas.shape)
    # exit()
    ny,nx  = linparas[0,:,:,0].shape
    n = max(ny,nx)
    pad1 = (n - min(nx, ny)) // 2
    pad0 = (n - min(nx, ny) -pad1)
    print(pad0,pad1)
    if nx > ny:
        if ny%2 == 0:
            padding=((pad1,pad0),(0,0))
        else:
            padding=((pad0,pad1),(0,0))
    else:
        if nx%2 == 0:
            padding=((0,0),(pad1,pad0))
        else:
            padding=((0,0),(pad0,pad1))
    break

# distances_pixels = np.zeros((n, n))
# star_loc = n // 2
# for y in range(n):
#     for x in range(n):
#         distances_pixels[y, x] += [np.sqrt((y - star_loc) ** 2 + (x - star_loc) ** 2)]
yS, xS = n // 2,n // 2
xvec = np.arange(0,n)-xS
yvec = np.arange(0,n)-yS
x_grid,y_grid = np.meshgrid(xvec,yvec,indexing="xy")
r_grid = np.sqrt(x_grid**2+y_grid**2)
# planety, planetx = 40+1,40-4
planety, planetx = 40-4,20-4

for planety in [planety]:#[40-5,40-4,40-3]:
    for planetx in [planetx]:#[20-5,20-4,20-3]:
        flux_arr = np.zeros((len(fr_files),len(rvs)))+np.nan
        err_arr = np.zeros((len(fr_files),len(rvs)))+np.nan
        snr_arr = np.zeros((len(fr_files),len(rvs)))+np.nan
        flux_arr_spa = np.zeros((len(fr_files),n,n))+np.nan
        err_arr_spa = np.zeros((len(fr_files),n,n))+np.nan
        snr_arr_spa = np.zeros((len(fr_files),n,n))+np.nan
        starflux_arr = np.zeros((len(fr_files),len(rvs)))+np.nan
        N_files = 0
        for fid,filename in enumerate(fr_files):
            fil = os.path.basename(filename)
            if "_out.fits" not in fil:
                continue
            # if fil[8:12] in rotated_seqs: # add not if other way TODO
            #     continue
            print("DOING", filename)
            N_files+=1
            if 0:
                with pyfits.open(frames_dir + fil) as hdulist:
                    linparas = hdulist[3].data
                    linparas_err = hdulist[4].data
            else:
                with pyfits.open(frames_dir + fil.replace("_out.fits","_out_withRVs.fits")) as hdulist:
                    linparas = hdulist[3].data
                    linparas_err = hdulist[4].data
                where_rvnull = np.argmin(np.abs(rvs))
                linparas[:,:,:,0] = linparas[:,:,:,0]-np.nanmedian(linparas[:,:,:,0],axis=0)[None,:,:]
                ccfsnr = linparas/linparas_err
                ccfstd = np.nanstd(np.concatenate([ccfsnr[0:where_rvnull-5,:,:,:],ccfsnr[where_rvnull+6::,:,:,:]],axis=0),axis=0)
                # plt.plot(rvs,linparas[:,36-padding[0][0],41-padding[1][0],0]/linparas_err[:,36-padding[0][0],41-padding[1][0],0])
                # # plt.show()
                # continue
                print(ccfsnr.shape)
                # plt.plot(ccfsnr[:,40-4,20-4,0])
                # plt.show()
                # exit()
                err = (linparas_err[:,:,:,:]*ccfstd[None,:,:,:])[:, planety, planetx, 0]
                flux = (linparas[:,:,:,:])[:, planety, planetx, 0]
                flux_spa = (linparas_err[where_rvnull:where_rvnull+1,:,:,:]*ccfstd[None,:,:,:])
                err_spa = (linparas[where_rvnull:where_rvnull+1,:,:,:])
                
                flux_spa = np.pad(flux_spa[0,:,:,0], padding, constant_values=np.nan)
                err_spa = np.pad(err_spa[0,:,:,0], padding, constant_values=np.nan)
                # plt.plot(linparas_err)
                # plt.show()
                # plt.plot(linparas)
                # plt.show()
            nan_mask_boxsize = 3
            where_to_mask = np.where(np.isnan(correlate(flux,np.ones((nan_mask_boxsize)),mode="same")))
            flux[where_to_mask] = np.nan
            err[where_to_mask] = np.nan

            where_to_mask = np.where(np.isnan(correlate2d(flux_spa,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))
            flux_spa[where_to_mask] = np.nan
            err_spa[where_to_mask] = np.nan

            flux_arr[fid] = flux
            err_arr[fid] = err
            snr_arr[fid] = flux / err
            flux_arr_spa[fid] = flux_spa
            err_arr_spa[fid] = err_spa
            snr_arr_spa[fid] = flux_spa / err_spa

        for im_id in range(snr_arr.shape[0]):
            flux_arr[im_id] = (flux_arr[im_id]-np.nanmedian(flux_arr[im_id]))

        t_flux = np.nansum(flux_arr/err_arr**2,axis=0)/np.nansum(1/err_arr**2,axis=0)
        t_err = 1/np.sqrt(np.nansum(1/err_arr**2,axis=0))
        snr0 = t_flux / t_err
        noise_calib = np.nanstd(snr_arr, axis=0)
        noise_calib[np.where(noise_calib==0)] = np.nan

        t_flux_spa = np.nansum(flux_arr_spa/err_arr_spa**2,axis=0)/np.nansum(1/err_arr_spa**2,axis=0)
        t_err_spa = 1/np.sqrt(np.nansum(1/err_arr_spa**2,axis=0))
        snr0_spa = t_flux_spa / t_err_spa
        noise_calib_spa = np.nanstd(snr_arr_spa, axis=0)
        noise_calib_spa[np.where(noise_calib_spa==0)] = np.nan

        seppix_list = np.arange(2,100)
        ann_corr = np.zeros_like(seppix_list,dtype=np.float)
        for sepid,seppix in enumerate(seppix_list):
            where_annulus = np.where((r_grid>seppix-1)*(r_grid<seppix+1))
            if np.size(where_annulus[0]) <5:
                ann_corr[sepid] = np.nan
            else:
                ann_corr[sepid] = np.nanstd((snr0_spa/noise_calib_spa)[where_annulus])

        PSF_FWHM=2
        smallsamp_1sig_ann_corr = np.array([t.ppf(0.841345,(2*np.pi*sep)/PSF_FWHM-1,scale=std)*np.sqrt(1+1./((2*np.pi*sep)/PSF_FWHM)) for sep,std in zip(seppix_list,ann_corr)])
        # smallsamp_5sig_ann_corr = np.array([t.ppf(0.99999971334,(2*np.pi*sep)/PSF_FWHM-1,scale=std)*np.sqrt(1+1./((2*np.pi*sep)/PSF_FWHM)) for sep,std in zip(seppix_list,ann_corr)])

        ann_corr_map_1sig = interp1d(seppix_list,smallsamp_1sig_ann_corr,fill_value=np.nan,bounds_error=False)(r_grid)
        snr_calib = snr0/noise_calib#/ann_corr_map_1sig[planety, 20+planetx]
        
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=snr_calib,
                                        header=pyfits.Header(cards={"TYPE": "snr calib", "DIR": frames_dir})))
        hdulist.append(pyfits.PrimaryHDU(data=snr0,
                                        header=pyfits.Header(cards={"TYPE": "snr0", "DIR": frames_dir})))

        try:
            hdulist.writeto(main_out_dir+"rvccf.fits", overwrite=True)
        except TypeError:
            hdulist.writeto(main_out_dir+"rvccf.fits", clobber=True)
        # break

        plt.figure()
        plt.title(r"RV CCF, $\Delta y = \Delta x = - 4$")
        plt.plot(snr0, label = "no calib")
        plt.plot(snr_calib, label = "calib")
        plt.legend()
        plt.grid()
        # plt.show()
        # plt.savefig("./plots/thesis/hd148352/rvccf.png")

# plt.show()
plt.close()