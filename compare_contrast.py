import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

def line_from_scatter(xvals, yvals, num, equal_bins=False, log_space=True):
    if equal_bins:
        xvals, yvals = np.sort(xvals), np.array(yvals)[np.argsort(xvals)]
        rxs, rys, xerrs, yerrs = [], [], [], []
        if log_space:
            if np.nanmin(xvals) == 0:
                lower = 0
            else:
                lower = np.log10(np.nanmin(xvals))
            bin_lims = np.logspace(lower, np.log10(np.nanmax(xvals)), num+1)
        else:
            bin_lims = np.linspace(np.nanmin(xvals), np.nanmax(xvals), num+1)
        ind = 0
        for i in range(num):
            xs, ys = [], []
            while True:
                if ind >= len(xvals) or xvals[ind] >= bin_lims[i+1]:
                    break
                xs += [xvals[ind]]
                ys += [yvals[ind]]  
                ind += 1
            if len(xs) == 0:
                continue
            rxs += [np.nanmedian(xs)]
            rys += [np.nanmedian(ys)]
            xerrs += [np.nanstd(xs)]
            yerrs += [np.nanstd(ys)]
        rxs = np.array(rxs)
        rys = np.array(rys)
        xerrs = np.array(xerrs)
        yerrs = np.array(yerrs)
        return rxs, rys, xerrs, yerrs
    else:
        num = len(xvals) // num
        xvals, yvals = np.sort(xvals), np.array(yvals)[np.argsort(xvals)]
        rxs, rys, xerrs, yerrs = [], [], [], []
        for ind in range(0, len(xvals), num):
            rxs += [np.nanmean(xvals[ind:ind+num])]
            rys += [np.nanmean(yvals[ind:ind+num])]
            xerrs += [np.nanstd(xvals[ind:ind+num])]
            yerrs += [np.nanstd(yvals[ind:ind+num])]
        rxs = np.array(rxs)
        rys = np.array(rys)
        xerrs = np.array(xerrs)
        yerrs = np.array(yerrs)
        return rxs, rys, xerrs, yerrs



out_fold = "/scr3/jruffio/shubh/using-breads/plots/paper/contrast/"

fig = plt.figure(1,figsize=(6,4))
# if 0:
    # targets = ["HD148352", "SR3", "SR14", "ROXs44", "ROXs43B", "SR9", "SR4", "ROXs8", "ROXs35A", "SR21A", "ROXs4"]
    # Kmags = np.array([6.511, 6.504, 8.878, 7.61, 7.09, 7.207, 7.518, 6.227, 8.531, 6.719, 8.331])
    # spec_types = ["F2V", "A0", "G5", "K3e", "K3e", "K5e", "K0:Ve", "K0", "K1IV", "G1", "K5.5"]
    # Teffs = [6700, 4000, 5400, 4700, 4000, 3900, 4300, 5200, 4700, 4500, 4100]
    # threshold_snr = 5
    #
    # norm = mpl.colors.Normalize(vmin=Kmags.min(), vmax=Kmags.max())
    # cmap = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    # cmap.set_array([])
    #
    # # targets_to_plot = ["HD148352", "ROXs4", "SR14"]
    #
    # # tp60, tp100_err, con60, con60_err = [], [], [], []
    # tp100, tp100_err, con100, con100_err = [], [], [], []
    # # fol = "09232021"
    # fol = "TP"
    #
    # for tid,(target, Kmag, spec_type) in enumerate(zip(targets, Kmags, spec_types)):
    #     # if target not in targets_to_plot:
    #     #     continue
    #     print(target)
    #     targetf = f"{fol}_{target}"
    #     label = f"{target}: {Kmag}, {spec_type}"
    #     with pyfits.open(f"/scr3/jruffio/shubh/using-breads/plots/scatter/{fol}/scatter_{targetf}.fits") as hdulist:
    #         calibrated_err_combined, throughput = hdulist[0].data, hdulist[1].data
    #     distances = []
    #     noise = []
    #     tp = []
    #     nx, ny = calibrated_err_combined.shape
    #     xS, yS = nx / 2, ny / 2
    #     for x in range(nx):
    #         for y in range(ny):
    #             distances += [np.sqrt((y - yS) ** 2 + (x - xS) ** 2) * 20]
    #             noise += [threshold_snr * calibrated_err_combined[x, y]]
    #             tp += [throughput[x, y]]
    #
    #     rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, 100, True)
    #     rxs = np.array(rxs)
    #     rys = np.array(rys)
    #     rys[np.where(rxs<40)] = np.nan
    #     if tid == 0:
    #         plt.plot(rxs / 1000, rys, alpha=1, color=cmap.to_rgba(Kmag),label="OSIRIS (Agrawal S. et al.)")
    #     else:
    #         plt.plot(rxs/1000, rys, alpha=1, color=cmap.to_rgba(Kmag))

targets = ["LkCa15", "SR3", "AB_Aur", "ROXs43B"]#["AB_Aur"]#,"SR21A","SR9"]
# dates = [""]#["211018"]#,"210626","210628"]
# fol = "20220417"
fol = "20220512_1700K"
threshold_snr = 5
for tid,target in enumerate(targets):
    # if target not in targets_to_plot:
    #     continue
    print(target)
    targetf = f"{fol}_{target}"
    label = f"{target}"
    #
# hdulist = pyfits.HDUList()
# hdulist.append(pyfits.PrimaryHDU(data=snr_calib,
#                                  header=pyfits.Header(cards={"TYPE": "snr calib", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=t_flux,
#                                  header=pyfits.Header(cards={"TYPE": "flux ratio calib", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=t_err_1sig_calib,
#                                  header=pyfits.Header(cards={"TYPE": "err map 1sig", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=snr0,
#                                  header=pyfits.Header(cards={"TYPE": "snr no calib", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=r_grid,
#                                  header=pyfits.Header(cards={"TYPE": "distance to center", "DIR": frames_dir})))
# hdulist.append(pyfits.PrimaryHDU(data=star_flux/star_flux[ny//2,nx//2],
#                                  header=pyfits.Header(cards={"TYPE": "raw psf map", "DIR": frames_dir})))
    # if "AB_Aur" in target:
    #     # with pyfits.open(f"/scr3/jruffio/data/osiris_survey/targets/{target}/{target_date}/1/reduced/contrast/{fol}/alloutputs.fits") as hdulist:
    #     with pyfits.open(f"/scr3/jruffio/data/osiris_survey/targets/{target}/{target_date}/1/reduced/planets/{fol}/alloutputs.fits") as hdulist:
    #         calibrated_err_combined = hdulist[2].data
    #         err_combined = hdulist[2].data*hdulist[0].data/hdulist[3].data
    #         r_grid = hdulist[4].data
    # else:
    #     with pyfits.open(f"/scr3/jruffio/data/osiris_survey/targets/{target}/{target_date}/reduced/contrast/{fol}/alloutputs.fits") as hdulist:
    #     # with pyfits.open(f"/scr3/jruffio/data/osiris_survey/targets/{target}/{target_date}/reduced/planets/{fol}/alloutputs.fits") as hdulist:
    #         calibrated_err_combined = hdulist[2].data
    #         err_combined = hdulist[2].data*hdulist[0].data/hdulist[3].data
    #         r_grid = hdulist[4].data
    with pyfits.open(f"/scr3/jruffio/code/BREADS_osiris_survey_scripts/plots/SNRmaps_contrast/20220512_1700K/{target}/alloutputs.fits") as hdulist:
        calibrated_err_combined = hdulist[2].data
        err_combined = hdulist[2].data*hdulist[0].data/hdulist[3].data
        r_grid = hdulist[4].data

    bins = 50
    rxs, rys, xerrs, yerrs = line_from_scatter(np.ravel(r_grid), np.ravel(calibrated_err_combined), bins, True)
    rxs_nocalib, rys_nocalib, _, _ = line_from_scatter(np.ravel(r_grid), np.ravel(err_combined), bins, True)
    rys[np.where(rxs>0.6/0.02)] = np.nan
    rys_nocalib[np.where(rxs_nocalib>0.6/0.02)] = np.nan
    
    # if True:
    # plt.plot(rxs*0.02, threshold_snr*rys, alpha=1,linewidth=3,label=f"Keck/OSIRIS: {target}", marker="1", linestyle="--")
    # plt.plot(rxs_nocalib*0.02, threshold_snr*rys_nocalib, alpha=1,linewidth=3,label=f"Keck/OSIRIS: {target}", marker="1", linestyle="--")
    # if tid == 0:
    plt.plot(rxs*0.02, threshold_snr*rys, alpha=1,linewidth=3,label=f"OSIRIS (Agrawal S. et al.): {target}")
        # plt.plot(rxs*0.02, threshold_snr*rys, alpha=1, color="#ff9900",linewidth=3,label=f"OSIRIS (Agrawal S. et al.)")
        # plt.plot(rxs_nocalib*0.02, threshold_snr*rys_nocalib, alpha=1, linestyle="--",linewidth=1,color="grey",label="OSIRIS Goal")
    # else:
    #     plt.plot(rxs*0.02,threshold_snr*rys, alpha=1,linewidth=1,label=f"Keck/OSIRIS: {target}")
    # plt.scatter(np.ravel(r_grid)*0.02, threshold_snr*np.ravel(calibrated_err_combined))

#GPI contrast
from glob import glob
# GPIcont_list = glob("/scr3/jruffio/data/osiris_survey/figures/*.txt")
GPIcont_list = ['/scr3/jruffio/data/osiris_survey/figures/contrast-S20151105-K1.txt',
                  '/scr3/jruffio/data/osiris_survey/figures/contrast-S20151218-K2.txt',
                  '/scr3/jruffio/data/osiris_survey/figures/contrast-S20160128-K1.txt',
                  '/scr3/jruffio/data/osiris_survey/figures/contrast-S20160125-K1.txt',
                  '/scr3/jruffio/data/osiris_survey/figures/contrast-S20151106-K1.txt']
from scipy.interpolate import interp1d
gpicont_list = []
for fid,filename in enumerate(GPIcont_list):
    with open(filename,"r") as f:
        lines = f.readlines()
        for line in lines:
            if "InputFiles" in line:
                tint = float(len(line.split(".fits")))
    data = np.loadtxt(filename,comments="#",delimiter=" ",skiprows=24)
    gpiseps = data[:,0]
    gpicont = data[:,1]
    print(np.sqrt(tint/30.))

    print(np.logspace(-2,0,100))
    gpicont_list.append(interp1d(gpiseps,gpicont*np.sqrt(tint/30.), bounds_error=False, fill_value=np.nan)(np.logspace(-2,0,100)))
    if fid == 2:
        plt.plot(gpiseps,gpicont*np.sqrt(tint/30.),linestyle = ":",color="#ff99cc",linewidth=3,label="GPI (scaled to 30 min exposures)")
    else:
        pass
        # plt.plot(gpiseps,gpicont*np.sqrt(tint/30.),linestyle = "--",linewidth=2,label="{0}".format(fid))#,color="#6600ff"

# plt.plot(np.logspace(-2,0,100),np.nanmedian(gpicont_list,axis=0),linestyle = "-",color="black",linewidth=3,label="GPI (scaled to 30 min exposures)")

idps=np.array([0.3128491620111731, 6.870838881491345,
0.3812849162011174, 7.350199733688415,
0.49860335195530725, 7.882822902796272,
0.6159217877094971, 8.415446071904128,
0.6941340782122905, 8.894806924101198,
0.9287709497206704, 9.933422103861517,
1.2709497206703912, 11.078561917443409,
1.4958100558659218, 11.797603195739013,
1.7500000000000004, 12.276964047936085,
2.1801675977653634, 12.80958721704394,
2.6201117318435756, 13.129161118508655,
3.2067039106145248, 13.342210386151798,
4.135474860335195, 13.36884154460719,
5.181564245810057, 13.581890812250332,
6.110335195530727, 13.55525965379494,
6.980446927374302, 13.262316910785618])
idpsseps,idpscont = idps[0::2],10**(idps[1::2]/-2.5)

plt.plot(idpsseps,idpscont,linestyle = ":",color="#6600ff",linewidth=3,label="IPDS median sensitivity")

NRM = np.array([0.006164383561643824, 0.9999999999999785,
0.00650684931506848, 1.6113537117903747,
0.008219178082191775, 2.3449781659388496,
0.00993150684931507, 3.4556040756914017,
0.014383561643835613, 4.831149927219793,
0.023972602739726026, 6.318777292576423,
0.03150684931506849, 6.970887918486179,
0.040753424657534246, 7.215429403202337,
0.05, 7.164483260553138,
0.05890410958904112, 7.0625909752547384,
0.06404109589041097, 7.042212518195059,
0.07294520547945209, 7.093158660844257,
0.08356164383561644, 7.144104803493458,
0.09486301369863014, 7.184861717612818,
0.10684931506849317, 7.093158660844257,
0.11849315068493153, 7.001455604075699,
0.12808219178082197, 7.001455604075699,
0.136986301369863, 7.113537117903937,
0.14828767123287673, 7.235807860262017,
0.1589041095890411, 7.195050946142657,
0.1671232876712329, 7.093158660844257,
0.17636986301369867, 7.093158660844257,
0.18904109589041096, 7.0625909752547384,
0.20102739726027397, 7.0727802037845775])
nrmseps,nrmcont = NRM[0::2],10**(NRM[1::2]/-2.5)

plt.plot(nrmseps,nrmcont,linestyle = "-.",color="silver",linewidth=3,label="NRM Sallum et al. 2019")

NRM_Cheetham = np.array([[39.70177073625349, 3.643028846153846],
[36.980428704566634, 3.6850961538461537],
[32.730661696178935, 3.752403846153846],
[28.74184529356943, 3.8197115384615383],
[23.336439888164023, 3.7103365384615383],
[19.27306616961789, 3.5841346153846154],
[15.694315004659831, 3.457932692307692],
[11.742777260018638, 3.424278846153846],
[8.052190121155638, 3.3401442307692304],
[4.436160298229264, 3.079326923076923],
[3.727865796831314, 2.877403846153846],
[2.982292637465051, 2.456730769230769]])
nrmseps,nrmcont = NRM_Cheetham[:,0]/40*0.3,10**(NRM_Cheetham[:,1]/-2.5)

# plt.plot(nrmseps,nrmcont,linestyle = "--",color="black",linewidth=3,label="NRM Cheetham et al. 2015")

plt.xscale('log')
plt.yscale('log')
# plt.xticks(fontsize= 12)
# plt.yticks(fontsize= 12)
plt.ylim([1e-5,1e-2])
plt.xlim([1e-2,1e0])
plt.grid()
plt.tight_layout()
# cbar = fig.colorbar(cmap)
# cbar.set_label('Stellar K mag', fontsize=12)
plt.legend(loc="lower left", fontsize=8)
plt.xlabel("separation (arcseconds)")
plt.ylabel("flux ratio")
# plt.legend()
ytick_locations = [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2]
xtick_locations = [1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0]
plt.xticks(ticks=xtick_locations, labels=xtick_locations)
plt.yticks(ticks=ytick_locations, labels=ytick_locations)
plt.title("Sensitivity Curves")
# plt.savefig("/scr3/jruffio/shubh/using-breads/plots/thesis/compare_contrast/compare_contrast_potential.png", bbox_inches='tight')

plt.savefig(out_fold+f"compare_contrast.png", bbox_inches='tight')
plt.savefig(out_fold+f"compare_contrast.pdf", bbox_inches='tight')
plt.savefig(out_fold+f"compare_contrast.eps", bbox_inches='tight')

plt.close()
