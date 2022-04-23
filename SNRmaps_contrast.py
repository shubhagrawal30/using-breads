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
from scipy.signal import correlate2d
from scipy.interpolate import interp1d
from scipy.stats import t
from pathlib import Path
from glob import glob

# star = "HD148352"
star = sys.argv[1]

date = args.dates[star]# + "/first"
th_fol = "20220417"
fr_fol = "20220417"
fol = "20220417"

# fr_fol = "20220409"
# fol = "20220409"
target = f"{fr_fol}_{th_fol}_{star}"

throughput_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/throughput/{th_fol}/"
frames_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/planets/{fr_fol}/"
out_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/contrast/{fol}/"
psf_dir = f"/scr3/jruffio/data/osiris_survey/targets/{star}/{date}/reduced/"
fr_files = os.listdir(frames_dir)
th_files = os.listdir(throughput_dir)
psf_files = os.listdir(psf_dir)

flux_ratio = 1e-2
threshold = 5

print("making subdirectories")
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(f"./plots/scatter/{fol}/").mkdir(parents=True, exist_ok=True)

snrs = {}
num = 40

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
xS, yS = n // 2,n // 2
xvec = np.arange(0,n)-yS
yvec = np.arange(0,n)-xS
x_grid,y_grid = np.meshgrid(xvec,yvec,indexing="xy")
r_grid = np.sqrt(x_grid**2+y_grid**2)

flux_arr = np.zeros((len(fr_files),n,n))+np.nan
err_arr = np.zeros((len(fr_files),n,n))+np.nan
snr_arr = np.zeros((len(fr_files),n,n))+np.nan
starflux_arr = np.zeros((len(fr_files),n,n))+np.nan
for fid,filename in enumerate(fr_files):
    fil = os.path.basename(filename)
    if "_out.fits" not in fil:
        continue
    # if fil[8:12] in rotated_seqs: # add not if other way TODO
    #     continue
    print("DOING", filename)
    with pyfits.open(frames_dir + fil) as hdulist:
        linparas = hdulist[3].data
        linparas_err = hdulist[4].data
    flux = np.pad(linparas[0,:,:,0], padding, constant_values=np.nan)

    N_star_paras = np.pad(np.sum(np.isfinite(linparas[0,:,:,1::]),axis=2).astype(np.float),padding, constant_values=np.nan)
    star_flux = linparas[0,:,:,1::]
    star_flux = np.pad(np.nanmedian(star_flux,axis=2), padding, constant_values=np.nan)
    # star_flux[np.where(N_star_paras<(linparas.shape[3]-1))] = np.nan

    err = np.pad(linparas_err[0,:,:,0], padding, constant_values=np.nan)

    if len(glob(os.path.join(throughput_dir,fil))) !=0:
        print("found throughput file",os.path.join(throughput_dir,fil))
        with pyfits.open(os.path.join(throughput_dir,fil)) as hdulist:
            flux_a = hdulist[0].data
            loc = flux_a.shape[0] // 2
            flux_b = hdulist[1].data

        tp = ((flux_a - flux_b) / flux_ratio)[:, 0]
        temp = np.vstack((np.append(tp[loc:], [np.nan]), np.append([np.nan], tp[:loc][::-1])))
        tp = np.nanmean(temp, axis=0)
        dists = np.abs(np.arange(0, loc+1))[~np.isnan(tp)]
        tp = tp[~np.isnan(tp)]
        dist_func = interp1d(dists, tp, kind="quadratic", fill_value="extrapolate")

        throughput = dist_func(r_grid)

        flux /= throughput
        err /= throughput
    else:
        pass
        # raise Exception("Missing throughput")

    if fil[8:12] in rotated_seqs: # add not if other way TODO
        print("rotated 90", fil)
        flux = np.rot90(flux, 1, (0, 1))
        err = np.rot90(err, 1, (0, 1))

    flux_arr[fid,:,:] = flux
    err_arr[fid,:,:] = err
    snr_arr[fid,:,:] = flux / err
    starflux_arr[fid,:,:] = star_flux


from scipy.stats import median_abs_deviation
for yid in range(snr_arr.shape[1]):
    for xid in range(snr_arr.shape[2]):
        if np.nansum(np.isfinite(snr_arr[:,yid,xid]))==0:
            continue
        snrvec_mad = median_abs_deviation((snr_arr[:,yid,xid])[np.where(np.isfinite(snr_arr[:,yid,xid]))])
        wherebad = np.where(np.abs((snr_arr[:,yid,xid]-np.nanmedian(snr_arr[:,yid,xid]))/snrvec_mad)>5)

        flux_arr[wherebad[0],yid,xid] = np.nan
        err_arr[wherebad[0],yid,xid] = np.nan
        snr_arr[wherebad[0],yid,xid] = np.nan

for im_id in range(snr_arr.shape[0]):
    snrim_mad = median_abs_deviation((snr_arr[im_id,:,:])[np.where(np.isfinite(snr_arr[im_id,:,:]))])
    snr_arr[im_id,:,:] = snr_arr[im_id,:,:]/snrim_mad
    err_arr[im_id,:,:] = err_arr[im_id,:,:]*snrim_mad

t_flux = np.nansum(flux_arr/err_arr**2,axis=0)/np.nansum(1/err_arr**2,axis=0)
t_err = 1/np.sqrt(np.nansum(1/err_arr**2,axis=0))
snr0 = t_flux / t_err
print("max SNR calib: ", np.nanmax(snr0))
x, y = np.unravel_index(np.nanargmax(snr0), snr0.shape)
detection_dist = np.sqrt((y - yS) ** 2 + (x - xS) ** 2) * 0.02
detection_snr = np.nanmax(snr0)
detection_flux = t_flux[x, y]

noise_calib = np.nanstd(snr_arr, axis=0)
noise_calib[np.where(noise_calib==0)] = np.nan


x, y = np.unravel_index(np.nanargmax(snr0), snr0.shape)
print("relative position: ", (y - yS, x - xS))


seppix_list = np.arange(2,100)
ann_corr = np.zeros_like(seppix_list,dtype=np.float)
for sepid,seppix in enumerate(seppix_list):
    where_annulus = np.where((r_grid>seppix-1)*(r_grid<seppix+1))
    if np.size(where_annulus[0]) <5:
        ann_corr[sepid] = np.nan
    else:
        ann_corr[sepid] = np.nanstd((snr0/noise_calib)[where_annulus])

PSF_FWHM=2
smallsamp_1sig_ann_corr = np.array([t.ppf(0.841345,(2*np.pi*sep)/PSF_FWHM-1,scale=std)*np.sqrt(1+1./((2*np.pi*sep)/PSF_FWHM)) for sep,std in zip(seppix_list,ann_corr)])
# smallsamp_5sig_ann_corr = np.array([t.ppf(0.99999971334,(2*np.pi*sep)/PSF_FWHM-1,scale=std)*np.sqrt(1+1./((2*np.pi*sep)/PSF_FWHM)) for sep,std in zip(seppix_list,ann_corr)])

ann_corr_map_1sig = interp1d(seppix_list,smallsamp_1sig_ann_corr,fill_value=np.nan,bounds_error=False)(r_grid)
snr_calib = snr0/noise_calib/ann_corr_map_1sig
print("max SNR calib: ", np.nanmax(snr_calib))
x, y = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
print("relative position: ", (y - yS, x - xS))
t_err_1sig_calib = t_err*noise_calib*ann_corr_map_1sig
detection_dist_calib = np.sqrt((y - yS) ** 2 + (x - xS) ** 2) * 0.02
detection_snr_calib = np.nanmax(snr_calib)
detection_flux_calib = t_flux[x, y]

plt.figure()
plt.title("SNR calibration map")
vlim = np.max(noise_calib*ann_corr_map_1sig)
plt.imshow(noise_calib*ann_corr_map_1sig, origin="lower", vmin=-vlim, vmax=vlim, cmap='cividis')
cbar = plt.colorbar()
plt.savefig(out_dir+"calibration_map_for_snr.png")
# plt.savefig(f"./plots/combined/calibration_map_for_snr_{target}.png")

# snr[snr < -50] = np.nan
# snr[snr > 50] = np.nan

plt.figure()
plt.title("1sig calib err map")
plt.imshow(np.log10(t_err_1sig_calib), origin="lower", cmap='cividis')
plt.savefig(out_dir+"contrast_map_1sigma.png")
# plt.savefig(f"./plots/combined/contrast_map_1sigma_{target}.png")

# psf_dist = []
# psf_profile = []
# count = 0
# from breads.instruments.OSIRIS import OSIRIS
# for fil in psf_files:
#     if ".fits" not in fil:
#         print("psf skipping", fil)
#         continue
#     print("psf DOING", fil)
#     data = np.nanmedian(OSIRIS(psf_dir + fil).data, axis=0)
#     with pyfits.open(psf_dir+"spectra/"+fil[:-5]+"_spectrum.fits") as hdulist:
#         mu_x = hdulist[3].data
#         mu_y = hdulist[4].data
#     xS, yS = np.nanmedian(mu_x), np.nanmedian(mu_y)
#     nx, ny = data.shape
#     for x in range(nx):
#         for y in range(ny):
#             psf_dist += [np.sqrt((y - yS) ** 2 + (x - xS) ** 2)]
#             psf_profile += [data[x, y]]
#     count += 1
#     if count == num:
#         break

plt.figure()
plt.title("Sensitivity/contrast curves")
star_flux = np.nansum(starflux_arr,axis=0)
ny,nx = star_flux.shape
plt.scatter(r_grid*0.02,star_flux/star_flux[ny//2,nx//2],label="psf profile", alpha=0.1)
# plt.scatter(np.array(psf_dist)*0.02, np.array(psf_profile) / np.nanmax(psf_profile), marker=".", color="black", label="scaled PSF profile", alpha =1)
plt.scatter(r_grid*0.02,threshold*t_err,label="raw 5sig contrast", alpha=0.1)
plt.scatter(r_grid*0.02,threshold*t_err_1sig_calib,label="calib 5sig contrast", alpha=0.1)
# plt.scatter(r_grid*0.02,detection_snr_calib*t_err_1sig_calib,label="detection calib snr contrast", alpha=0.1)
# plt.scatter(r_grid*0.02,detection_snr*t_err,label="detection snr contrast", alpha=0.1)
plt.plot(detection_dist_calib, detection_flux_calib, 'bX', label=f"detection at snr {detection_snr_calib:.2f}")
# plt.plot(detection_dist, detection_flux, 'bX', label="detection")
plt.xlim([1e-2,1])
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig(out_dir+"contrast_curves.png")

plt.figure()
plt.title("Sensitivity/contrast/detection curves")
star_flux = np.nansum(starflux_arr,axis=0)
ny,nx = star_flux.shape
plt.scatter(r_grid*0.02,star_flux/star_flux[ny//2,nx//2],label="psf profile", alpha=0.1)
# plt.scatter(np.array(psf_dist)*0.02, np.array(psf_profile) / np.nanmax(psf_profile), marker=".", color="black", label="scaled PSF profile", alpha =1)
plt.scatter(r_grid*0.02,threshold*t_err,label="raw 5sig contrast", alpha=0.1)
plt.scatter(r_grid*0.02,threshold*t_err_1sig_calib,label="calib 5sig contrast", alpha=0.1)
plt.scatter(r_grid*0.02,detection_snr_calib*t_err_1sig_calib,label="detection calib snr contrast", alpha=0.1)
plt.scatter(r_grid*0.02,detection_snr*t_err,label="detection snr contrast", alpha=0.1)
plt.plot(detection_dist_calib, detection_flux_calib, 'bX', label=f"detection at snr {detection_snr_calib:.2f}")
plt.plot(detection_dist, detection_flux, 'rX', label=f"detect calib at snr {detection_snr:.2f}")
plt.xlim([1e-2,1])
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig(out_dir+"contrast_detection_curves.png")

plt.figure()
snr_scale = 20
# plt.imshow(snr, origin="lower")
plt.imshow(snr0, origin="lower", vmin=-snr_scale, vmax=snr_scale, cmap='cividis')
plt.plot(yS, xS, "rX")
plt.plot(y, x, "b.")
cbar = plt.colorbar()
cbar.set_label("SNR")
plt.title(f"{target}, {(y - yS, x - xS)}, {np.nanmax(snr0)}, {len(fr_files)} frames")
plt.savefig(out_dir+"combined_snr_nocalib.png")
# plt.savefig(f"./plots/combined/combined_snr_nocalib_{target}.png")

snr_scale = 5
plt.figure()
plt.imshow(snr_calib, origin="lower", vmin=-snr_scale, vmax=snr_scale, cmap='cividis')
plt.plot(yS, xS, "rX")
plt.plot(y, x, "b.")
cbar = plt.colorbar()
cbar.set_label("SNR noise calib")
plt.title(f"{target}, {(y - yS, x - xS)}, {np.nanmax(snr_calib)}, {len(fr_files)} frames")
plt.savefig(out_dir+"combined_snr_calib.png")
# plt.savefig(f"./plots/combined/combined_snr_calib_{target}.png")

text_font = 12
bin_w = 0.25
bins = np.arange(-6,6,0.25)
bin_centers = np.array([(r1+r2)/2. for r1,r2 in zip(bins[0:-1],bins[1:])])
# image_selec = np.where(np.isfinite(snr_calib)*(r_grid>IWA)*(r_grid<OWA))
# H,xedges = np.histogram(snr_calib[image_selec],bins = bins,density=1)
H,xedges = np.histogram(snr_calib,bins = bins,density=1)

plt.figure()
plt.title("Residual SNR", fontsize=text_font,color="black")
plt.plot(bin_centers,H,"-",linewidth=2,color="black")
plt.plot(bin_centers,1./(np.sqrt(2*np.pi))*np.exp(-(bin_centers**2)/2.),linestyle = "--",linewidth = 2,color="grey")
plt.xlim([np.min(bins),np.max(bins)])
plt.ylim([1e-5,1])
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel("SNR", fontsize=text_font,color="black")
plt.ylabel("PDF", fontsize=text_font,color="black")
ax.tick_params(axis='x', labelsize=text_font,colors="black")
ax.tick_params(axis='y', labelsize=text_font,colors="black")
# [i.set_color("white") for i in iter(ax.spines.values())]
# ax.set_facecolor("black")
ax.grid(which='major',axis="both",linestyle="-",color="grey")

plt.savefig(out_dir+"snr_hist.png")
# plt.savefig(f"./plots/combined/snr_hist_{target}.png")

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=snr_calib,
                                 header=pyfits.Header(cards={"TYPE": "snr calib", "DIR": frames_dir})))
try:
    hdulist.writeto(out_dir+"snr_map_calib.fits", overwrite=True)
except TypeError:
    hdulist.writeto(out_dir+"snr_map_calib.fits", clobber=True)
hdulist.close()

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=t_flux,
                                 header=pyfits.Header(cards={"TYPE": "flux ratio calib", "DIR": frames_dir})))
try:
    hdulist.writeto(out_dir+"flux_ratio_map_calib.fits", overwrite=True)
except TypeError:
    hdulist.writeto(out_dir+"flux_ratio_map_calib.fits", clobber=True)
hdulist.close()

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=t_err_1sig_calib,
                                 header=pyfits.Header(cards={"TYPE": "err map 1sig", "DIR": frames_dir})))
try:
    hdulist.writeto(out_dir+"err_1sig_map_calib.fits", overwrite=True)
except TypeError:
    hdulist.writeto(out_dir+"err_1sig_map_calib.fits", clobber=True)
hdulist.close()

hdulist = pyfits.HDUList()
hdulist.append(pyfits.PrimaryHDU(data=snr_calib,
                                 header=pyfits.Header(cards={"TYPE": "snr calib", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=t_flux,
                                 header=pyfits.Header(cards={"TYPE": "flux ratio calib", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=t_err_1sig_calib,
                                 header=pyfits.Header(cards={"TYPE": "err map 1sig", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=snr0,
                                 header=pyfits.Header(cards={"TYPE": "snr no calib", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=r_grid,
                                 header=pyfits.Header(cards={"TYPE": "distance to center", "DIR": frames_dir})))
hdulist.append(pyfits.PrimaryHDU(data=star_flux/star_flux[ny//2,nx//2],
                                 header=pyfits.Header(cards={"TYPE": "raw psf map", "DIR": frames_dir})))

try:
    hdulist.writeto(out_dir+"alloutputs.fits", overwrite=True)
except TypeError:
    hdulist.writeto(out_dir+"alloutputs.fits", clobber=True)
hdulist.close()

plt.show()
plt.close()
