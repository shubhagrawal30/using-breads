print("starting")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import astropy.io.fits as pyfits
import arguments as args
from glob import glob
import corner
import matplotlib.gridspec as gridspec

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
            rxs += [np.nanmean(xs)]
            rys += [np.nanmean(ys)]
            xerrs += [np.nanstd(xs)]
            yerrs += [np.nanstd(ys)]
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
        return rxs, rys, xerrs, yerrs

def get_calib_snr(fold, out_fold):
    target_list = os.listdir(fold)
    target_list = ["ROXs35A"]

    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)

    snr_scale = 5
    for target in target_list:
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            snr_calib = hdulist[0].data
        y_calib, x_calib = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
        ny, nx = snr_calib.shape
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(snr_calib, origin="lower", vmin=-snr_scale, vmax=snr_scale, cmap='cividis')
        ax.plot(ny // 2, nx // 2, "rX")
        ax.plot(x_calib, y_calib, "b.")
        cbar = fig.colorbar(img)
        cbar.set_label("SNR")
        ax.set_xticks(np.arange(0, 81, 10))
        ax.set_yticks(np.arange(0, 81, 10))
        ax.set_xticklabels(np.arange(-40, 41, 10))
        ax.set_yticklabels(np.arange(-40, 41, 10))
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$\Delta y$")
        ax.grid(linestyle=':', linewidth=1.5)
        plt.title(target)
        plt.savefig(out_fold+f"{target}.png")
        plt.show()
        plt.close()

def get_no_calib_snr():
    fold = "/scr3/jruffio/shubh/using-breads/plots/SNRmaps_contrast/20220417/"
    out_fold = "/scr3/jruffio/shubh/using-breads/plots/thesis/snrmaps_nocalib/"
    target_list = os.listdir(fold)

    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)

    snr_scale = 25
    for target in target_list:
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            snr_calib = hdulist[3].data
        y_calib, x_calib = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
        ny, nx = snr_calib.shape
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(snr_calib, origin="lower", vmin=-snr_scale, vmax=snr_scale, cmap='cividis')
        ax.plot(ny // 2, nx // 2, "rX")
        ax.plot(x_calib, y_calib, "b.")
        cbar = fig.colorbar(img)
        cbar.set_label("SNR")
        ax.set_xticks(np.arange(0, 81, 10))
        ax.set_yticks(np.arange(0, 81, 10))
        ax.set_xticklabels(np.arange(-40, 41, 10))
        ax.set_yticklabels(np.arange(-40, 41, 10))
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$\Delta y$")
        ax.grid(linestyle=':', linewidth=1.5)
        plt.title(target)
        plt.savefig(out_fold+f"{target}.png")
        plt.close()

def get_all_snr_hist_nocalib():
    fold = "/scr3/jruffio/shubh/using-breads/plots/SNRmaps_contrast/20220417/"
    out_fold = "/scr3/jruffio/shubh/using-breads/plots/thesis/snrhist_nocalib/"
    target_list = os.listdir(fold)

    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)

    for target in target_list:
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            snr_calib = hdulist[3].data
            text_font = 12
        bin_w = 0.25
        bins = np.arange(-6,6,0.25)
        bin_centers = np.array([(r1+r2)/2. for r1,r2 in zip(bins[0:-1],bins[1:])])
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

        plt.savefig(out_fold+f"{target}.png")
        plt.close()

def get_all_snr_hist_calib():
    # fold = "/scr3/jruffio/shubh/using-breads/plots/SNRmaps_contrast/20220417/"
    # out_fold = "/scr3/jruffio/shubh/using-breads/plots/thesis/snrhist_calib/"
    target_list = os.listdir(fold)

    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)

    for target in target_list:
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            snr_calib = hdulist[0].data
            text_font = 12
        bin_w = 0.25
        bins = np.arange(-6,6,0.25)
        bin_centers = np.array([(r1+r2)/2. for r1,r2 in zip(bins[0:-1],bins[1:])])
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

        # plt.savefig(out_fold+f"{target}.png")
        plt.show()
        plt.close()

def get_combined_snr_hist_calib(fold, out_fold):
    target_list = os.listdir(fold)
    text_font = 12
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title("Residual signal-to-noise ratio", fontsize=text_font,color="black")
    bins = np.arange(-6,6,0.25)
    bin_centers = np.array([(r1+r2)/2. for r1,r2 in zip(bins[0:-1],bins[1:])])
    for ind,target in enumerate(target_list):
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            snr_calib = hdulist[0].data
        if target == "SR9":
            continue
        H,xedges = np.histogram(snr_calib,bins = bins,density=1)
        plt.plot(bin_centers,H,"-.",linewidth=0.8,label=f"{target}",marker=["^", ".", "*"][ind%3],alpha=0.5)
    plt.plot(bin_centers,1./(np.sqrt(2*np.pi))*np.exp(-(bin_centers**2)/2.),linestyle = "--",linewidth = 2,color="grey")
    plt.xlim([np.nanmin(bins),np.nanmax(bins)])
    plt.ylim([1e-4,2])
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel("SNR", fontsize=text_font,color="black")
    plt.ylabel("PDF", fontsize=text_font,color="black")
    ax.tick_params(axis='x', labelsize=text_font,colors="black")
    ax.tick_params(axis='y', labelsize=text_font,colors="black")
    # [i.set_color("white") for i in iter(ax.spines.values())]
    # ax.set_facecolor("black")
    ax.grid()
    ax.legend(loc='center left', ncol = 1, bbox_to_anchor=(1, 0, 1, 1))
    # plt.legend(loc='upper center', bbox_to_anchor=(-0.1,-0.35,1.1,0.2), ncol=4, mode='expand', fontsize='x-small')    
    plt.savefig(out_fold+f"combined_snr_hist_calib.png", bbox_inches='tight')
    plt.savefig(out_fold+f"combined_snr_hist_calib.pdf", bbox_inches='tight')
    plt.savefig(out_fold+f"combined_snr_hist_calib.eps", bbox_inches='tight')
    plt.show()
    plt.close()

def get_combined_snr_hist_nocalib(fold, out_fold):
    target_list = os.listdir(fold)
    text_font = 12
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title("Residual SNR without noise calibration", fontsize=text_font,color="black")
    bins = np.arange(-6,6,0.25)
    bin_centers = np.array([(r1+r2)/2. for r1,r2 in zip(bins[0:-1],bins[1:])])
    for target in target_list:
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            snr_calib = hdulist[3].data
        H,xedges = np.histogram(snr_calib,bins = bins,density=1)
        plt.plot(bin_centers,H,"-.",linewidth=0.8,label=f"{target}")
    plt.plot(bin_centers,1./(np.sqrt(2*np.pi))*np.exp(-(bin_centers**2)/2.),linestyle = "--",linewidth = 2,color="grey")
    plt.xlim([np.min(bins),np.max(bins)])
    plt.ylim([1e-4,2])
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel("SNR", fontsize=text_font,color="black")
    plt.ylabel("PDF", fontsize=text_font,color="black")
    ax.tick_params(axis='x', labelsize=text_font,colors="black")
    ax.tick_params(axis='y', labelsize=text_font,colors="black")
    # [i.set_color("white") for i in iter(ax.spines.values())]
    # ax.set_facecolor("black")
    ax.grid()
    ax.legend(loc='center left', ncol = 1, bbox_to_anchor=(1, 0, 1, 1))
    # plt.legend(loc='upper center', bbox_to_anchor=(-0.1,-0.35,1.1,0.2), ncol=4, mode='expand', fontsize='x-small')    
    plt.savefig(out_fold+f"combined_snr_hist_nocalib.png", bbox_inches='tight')
    plt.close()

def get_target_hists(target, fold, out_fold):
    target_list = os.listdir(fold)
    text_font = 12
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title(f"Residual SNR for {target}", fontsize=text_font,color="black")
    bins = np.arange(-6,6,0.25)
    bin_centers = np.array([(r1+r2)/2. for r1,r2 in zip(bins[0:-1],bins[1:])])
    print(target)
    folder = fold + target
    with pyfits.open(folder + "/alloutputs.fits") as hdulist:
        snr_calib = hdulist[0].data
        snr_nocalib = hdulist[3].data
    H,xedges = np.histogram(snr_calib,bins = bins,density=1)
    plt.plot(bin_centers,H,"-",linewidth=2,label=f"calib")
    H,xedges = np.histogram(snr_nocalib,bins = bins,density=1)
    plt.plot(bin_centers,H,"-",linewidth=2,label=f"no calib")
    plt.plot(bin_centers,1./(np.sqrt(2*np.pi))*np.exp(-(bin_centers**2)/2.),linestyle = "--",linewidth = 2,color="grey")
    plt.xlim([np.min(bins),np.max(bins)])
    plt.ylim([1e-4,1])
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel("SNR", fontsize=text_font,color="black")
    plt.ylabel("PDF", fontsize=text_font,color="black")
    ax.tick_params(axis='x', labelsize=text_font,colors="black")
    ax.tick_params(axis='y', labelsize=text_font,colors="black")
    # [i.set_color("white") for i in iter(ax.spines.values())]
    # ax.set_facecolor("black")
    ax.grid()
    ax.legend(loc='center left', ncol = 1, bbox_to_anchor=(1, 0, 1, 1))
    # plt.legend(loc='upper center', bbox_to_anchor=(-0.1,-0.35,1.1,0.2), ncol=4, mode='expand', fontsize='x-small')    
    plt.savefig(out_fold+f"hist_{target}.png", bbox_inches='tight')
    plt.close()

def get_target_contrast_curve(target, fold, out_fold):
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    print(target)
    folder = fold + target
    threshold = 5
    with pyfits.open(folder + "/alloutputs.fits") as hdulist:
        t_err_1sig_calib = hdulist[2].data
        r_grid = hdulist[4].data
        psf_profile = hdulist[5].data
        snr_calib, snr0 = hdulist[0].data, hdulist[3].data
        t_err = t_err_1sig_calib / snr0 * snr_calib 
        # t_err = hdulist[6].data

    img, ax = plt.subplots(1,1)
    plt.title(f"{target}")
    plt.scatter(r_grid*0.02,psf_profile,label="PSF Profile", alpha=0.2, marker=".")
    plt.scatter(r_grid*0.02,threshold*t_err,label=r"5$\sigma$ contrast (w/o)", alpha=0.2, marker="1")
    plt.scatter(r_grid*0.02,threshold*t_err_1sig_calib,label=r"5$\sigma$ contrast (with)", alpha=0.2, marker="2")
    # plt.plot(detection_dist_calib, detection_flux_calib, 'bX', label=f"detection at snr {detection_snr_calib:.2f}")
    # plt.plot(detection_dist, detection_flux, 'bX', label="detection")
    plt.xlim([2e-2,1])
    plt.ylim([5e-5,1])
    # ax.set_xticks(np.logspace(np.log10(2e-2), 0))
    # ax.set_xticklabels(np.logspace(np.log10(2e-2), 0))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("separation (arcseconds)")
    plt.ylabel("contrast")
    plt.legend()
    plt.grid()
    tick_locations = [3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1]
    plt.xticks(ticks=tick_locations, labels=tick_locations)
    plt.savefig(out_fold+f"{target}.png")

def get_combined_contrast_curve_nocalib(fold, out_fold):
    target_list = os.listdir(fold)
    threshold = 5
    bins = 40

    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    for target in target_list:
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            t_err_1sig_calib = hdulist[2].data
            r_grid = hdulist[4].data
            psf_profile = hdulist[5].data
            snr_calib, snr0 = hdulist[0].data, hdulist[3].data
            t_err = t_err_1sig_calib / snr0 * snr_calib
        ny, nx = r_grid.shape
        distances, noise = [], []
        for y in range(ny):
            for x in range(nx):
                distances += [r_grid[y, x]*20]
                noise += [threshold * t_err[y, x]]

        # plt.figure(0)
        # plt.scatter(distances, noise, alpha=0.1)
        # plt.figure(1)
        # rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, bins, False)
        # plt.scatter(rxs, rys, alpha=0.6)
        # # plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, alpha=0.6, ls='none')
        plt.figure(1)
        rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, bins, True)
        rxs = np.array(rxs) / 1000
        # plt.scatter(rxs, rys, marker="1", alpha=0.6, ls="dotted", linewidth=0.1)
        plt.plot(rxs, rys, marker="1", label=f"{target}", linestyle=":", linewidth=1)
        # plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, alpha=0.6, ls='dotted')
        # for ind in range(1, 2):
            # plt.figure(ind)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([3e-2,1])
    plt.ylim([1e-4,5e-1])
    # ax.set_xticks(np.logspace(np.log10(2e-2), 0))
    # ax.set_xticklabels(np.logspace(np.log10(2e-2), 0))
    plt.xlabel("separation (arcseconds)")
    plt.ylabel("contrast")
    # plt.legend()
    plt.grid()
    ytick_locations = [5e-1, 3e-1, 1e-1, 5e-2, 3e-2, 1e-2, 3e-3, 1e-3, 3e-3, 3e-4]
    xtick_locations = [3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1]
    plt.xticks(ticks=xtick_locations, labels=xtick_locations)
    plt.yticks(ticks=ytick_locations, labels=ytick_locations)
    ax = plt.gca()
    ax.legend(loc='center left', ncol = 1, bbox_to_anchor=(1, 0, 1, 1))
    plt.title("Potential Sensitivity")
    plt.savefig(out_fold+f"combined_contrast_nocalib.png", bbox_inches='tight')
    # plt.show()
    plt.close()

def get_combined_contrast_curve_calib(fold, out_fold):
    target_list = os.listdir(fold)
    threshold = 5
    bins = 30

    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    for target in target_list:
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            t_err_1sig_calib = hdulist[2].data
            r_grid = hdulist[4].data
            psf_profile = hdulist[5].data
            snr_calib, snr0 = hdulist[0].data, hdulist[3].data
            t_err = t_err_1sig_calib / snr0 * snr_calib
        ny, nx = r_grid.shape
        distances, noise = [], []
        for y in range(ny):
            for x in range(nx):
                distances += [r_grid[y, x]*20]
                noise += [threshold * t_err_1sig_calib[y, x]]

        # plt.figure(0)
        # plt.scatter(distances, noise, alpha=0.1)
        # plt.figure(1)
        # rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, bins, False)
        # # plt.scatter(rxs, rys, alpha=0.6)
        # plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, alpha=0.6, ls='none')
        plt.figure(1)
        rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, bins, True)
        rxs = np.array(rxs) / 1000
        # plt.scatter(rxs, rys, marker="1", alpha=0.6, ls="dotted", linewidth=0.1)
        plt.plot(rxs, rys, marker="1", label=f"{target}", linestyle=":", linewidth=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([3e-2,1])
    plt.ylim([1e-4,5e-1])
    # ax.set_xticks(np.logspace(np.log10(2e-2), 0))
    # ax.set_xticklabels(np.logspace(np.log10(2e-2), 0))
    plt.xlabel("separation (arcseconds)")
    plt.ylabel("contrast")
    # plt.legend()
    plt.grid()
    ytick_locations = [5e-1, 3e-1, 1e-1, 5e-2, 3e-2, 1e-2, 3e-3, 1e-3, 3e-3, 3e-4]
    xtick_locations = [3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1]
    plt.xticks(ticks=xtick_locations, labels=xtick_locations)
    plt.yticks(ticks=ytick_locations, labels=ytick_locations)
    ax = plt.gca()
    ax.legend(loc='center left', ncol = 1, bbox_to_anchor=(1, 0, 1, 1))
    plt.title("Sensitivity without RV normalization")
    plt.savefig(out_fold+f"combined_contrast_calib.png", bbox_inches='tight')
    plt.close()

def get_combined_contrast_curve_calibwithrv(fold, out_fold):
    target_list = os.listdir(fold)
    threshold = 5
    bins = 25

    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    for ind, target in enumerate(target_list):
        # if target == "LkCa15":
        #     continue
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            t_err_1sig_calib = hdulist[2].data
            r_grid = hdulist[4].data
            psf_profile = hdulist[5].data
            snr_calib, snr0 = hdulist[0].data, hdulist[3].data
            t_err = t_err_1sig_calib / snr0 * snr_calib
        ny, nx = r_grid.shape
        # distances, noise = [], []
        # for y in range(ny):
        #     for x in range(nx):
        #         distances += [r_grid[y, x]]
        #         noise += [threshold * t_err_1sig_calib[y, x]]

        # plt.figure(0)
        # plt.scatter(distances, noise, alpha=0.1)
        # plt.figure(1)
        # rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, bins, False)
        # # plt.scatter(rxs, rys, alpha=0.6)
        # plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, alpha=0.6, ls='none')
        plt.figure(1)
        rxs, rys, xerrs, yerrs = line_from_scatter(np.ravel(r_grid), threshold*np.ravel(t_err), bins, True)
        rxs, rys = np.array(rxs), np.array(rys)
        rys[np.where(rxs>0.6/0.02)] = np.nan
        rys[np.where(rxs<4e-2/0.02)] = np.nan
        # rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, bins, True)
        rxs = np.array(rxs) * 0.02
        # plt.scatter(rxs, rys, marker="1", alpha=0.6, ls="dotted", linewidth=0.1)
        rxs[np.abs(rxs-np.nanmean(rxs))>3*np.nanstd(rxs)] = np.nan
        rys[np.abs(rys-np.nanmean(rys))>3*np.nanstd(rys)] = np.nan

        plt.plot(rxs, rys, marker=["^", ".", "*"][ind%3], label=f"{target}", linestyle=":", linewidth=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-4,2e-2])
    plt.xlim([4e-2,6e-1])
    # ax.set_xticks(np.logspace(np.log10(2e-2), 0))
    # ax.set_xticklabels(np.logspace(np.log10(2e-2), 0))
    plt.xlabel("separation (arcseconds)")
    plt.ylabel("flux ratio")
    # plt.legend()
    plt.grid()
    ytick_locations = [2e-2, 1e-2, 3e-3, 1e-3, 3e-3, 3e-4]
    xtick_locations = [4e-2, 5e-2, 1e-1, 3e-1, 0.4]
    plt.xticks(ticks=xtick_locations, labels=xtick_locations)
    plt.yticks(ticks=ytick_locations, labels=ytick_locations)
    ax = plt.gca()
    ax.legend(loc='center left', ncol = 1, bbox_to_anchor=(1, 0, 1, 1))
    plt.title("Sensitivity curves")
    plt.savefig(out_fold+f"combined_contrast_calib_withrv.png", bbox_inches='tight')
    plt.savefig(out_fold+f"combined_contrast_calib_withrv.pdf", bbox_inches='tight')
    plt.savefig(out_fold+f"combined_contrast_calib_withrv.eps", bbox_inches='tight')
    plt.close()
    # plt.show()

def get_HD148352_snr():
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    target = "HD148352"
    snr_scale = 5
    print(target)
    folder = fold + target
    with pyfits.open(folder + "/alloutputs.fits") as hdulist:
        snr_calib = hdulist[0].data
    y_calib, x_calib = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
    ny, nx = snr_calib.shape
    fig, (ax) = plt.subplots(1,1)
    img = ax.imshow(snr_calib, origin="lower", vmin=-snr_scale, vmax=30, cmap='cividis')
    ax.plot(ny // 2, nx // 2, "rX", label="HD 148352")
    ax.plot(x_calib, y_calib, "b.", 
        label=r"SNR {:.2f}, $\Delta y$ = {}, $\Delta x$ = {}".format(snr_calib[y_calib, x_calib], y_calib-ny//2, x_calib-nx//2))
    ax.plot(x_calib-1, y_calib, ".", color="purple", 
        label=r"SNR {:.2f}, $\Delta y$ = {}, $\Delta x$ = {}".format(snr_calib[y_calib, x_calib-1], y_calib-ny//2, x_calib-nx//2-1))
    cbar = fig.colorbar(img)
    cbar.set_label("SNR")
    ax.set_xlim([-10+40, 2+40])
    ax.set_ylim([-10+40, 2+40])
    ax.set_xticks(np.arange(-10+40, 2+40, 2))
    ax.set_yticks(np.arange(-10+40, 2+40, 2))
    ax.set_xticklabels(np.arange(-10, 2, 2))
    ax.set_yticklabels(np.arange(-10, 2, 2))
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.grid(linestyle=':', linewidth=1)
    ax.set_title(target)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0, -1.11, 1, 1), fontsize='small')
    plt.savefig(out_fold+"snr_map.png", bbox_inches='tight')
    plt.savefig(out_fold+"snr_map.pdf", bbox_inches='tight')
    plt.savefig(out_fold+"snr_map.eps", bbox_inches='tight')
    # plt.show()
    plt.close()
    # plt.figure()
    # # plt.imshow(snr_calib[y_calib-2:y_calib+3, x_calib-3:x_calib+3], vmin=-snr_scale, vmax=30, cmap='cividis', origin="lower")
    # plt.plot(snr_calib[y_calib])
    # plt.show()

def get_HD148352_snr_larger():
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    target = "HD148352"
    snr_scale = 5
    print(target)
    folder = fold + target
    with pyfits.open(folder + "/alloutputs.fits") as hdulist:
        snr_calib = hdulist[0].data
    y_calib, x_calib = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
    ny, nx = snr_calib.shape
    fig, (ax) = plt.subplots(1,1)
    img = ax.imshow(snr_calib, origin="lower", vmin=-snr_scale, vmax=30, cmap='cividis')
    ax.plot(ny // 2, nx // 2, "rX", label="HD 148352")
    ax.plot(x_calib, y_calib, "b.", 
        label=r"SNR {:.2f}, $\Delta y$ = {}, $\Delta x$ = {}".format(snr_calib[y_calib, x_calib], y_calib-ny//2, x_calib-nx//2))
    ax.plot(x_calib-1, y_calib, ".", color="purple", 
        label=r"SNR {:.2f}, $\Delta y$ = {}, $\Delta x$ = {}".format(snr_calib[y_calib, x_calib-1], y_calib-ny//2, x_calib-nx//2-1))
    cbar = fig.colorbar(img)
    cbar.set_label("SNR")
    ax.set_xlim([-22+40, 17+40])
    ax.set_ylim([-42+40, 27+40])
    ax.set_xticks(np.arange(-20+40, 17+40, 5))
    ax.set_yticks(np.arange(-40+40, 27+40, 5))
    ax.set_xticklabels(np.arange(-20, 17, 5))
    ax.set_yticklabels(np.arange(-40, 27, 5))
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.grid(linestyle=':', linewidth=1)
    ax.set_title(target)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0, -1.11, 1, 1), fontsize='small')
    # plt.savefig(out_fold+"snr_map.png", bbox_inches='tight')
    plt.savefig(out_fold+"snr_map_large.png", bbox_inches='tight')
    plt.savefig(out_fold+"snr_map_large.pdf", bbox_inches='tight')
    plt.savefig(out_fold+"snr_map_large.eps", bbox_inches='tight')
    # plt.show()
    plt.close()
    # plt.figure()
    # # plt.imshow(snr_calib[y_calib-2:y_calib+3, x_calib-3:x_calib+3], vmin=-snr_scale, vmax=30, cmap='cividis', origin="lower")
    # plt.plot(snr_calib[y_calib])
    # plt.show()


def get_HD148352_contrast():
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    target = "HD148352"
    print(target)
    folder = fold + target
    threshold = 5
    with pyfits.open(folder + "/alloutputs.fits") as hdulist:
        t_err_1sig_calib = hdulist[2].data
        r_grid = hdulist[4].data
        psf_profile = hdulist[5].data
        snr_calib, snr0 = hdulist[0].data, hdulist[3].data
        flux = snr_calib * t_err_1sig_calib
        t_err = t_err_1sig_calib / snr0 * snr_calib 
        # t_err = hdulist[6].data
    y_calib, x_calib = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
    detection_flux = flux[y_calib, x_calib]
    ny, nx = snr_calib.shape
    detection_distance = np.sqrt((y_calib-ny//2)**2 + (x_calib-nx//2)**2) * 0.02
    ny, nx = snr_calib.shape
    img, ax = plt.subplots(1,1)
    plt.title(f"{target}")
    plt.scatter(r_grid*0.02,psf_profile,label="PSF Profile", alpha=0.2, marker=".")
    # plt.scatter(r_grid*0.02,threshold*t_err,label=r"5$\sigma$ contrast (w/o)", alpha=0.2, marker="1")
    plt.scatter(r_grid*0.02,threshold*t_err_1sig_calib,label=r"5$\sigma$ sensitivity", alpha=0.2, marker="2")
    plt.plot(detection_distance, detection_flux, 'rX', label=f"detection")
    plt.axhline(detection_flux, color="red")
    plt.axvline(detection_distance, color="red")
    # plt.plot(detection_dist_calib, detection_flux_calib, 'bX', label=f"detection at snr {detection_snr_calib:.2f}")
    # plt.plot(detection_dist, detection_flux, 'bX', label="detection")
    plt.xlim([3e-2,1])
    plt.ylim([1e-4,5e-1])
    # ax.set_xticks(np.logspace(np.log10(2e-2), 0))
    # ax.set_xticklabels(np.logspace(np.log10(2e-2), 0))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("separation (arcseconds)")
    plt.ylabel("flux ratio")
    plt.legend()
    plt.grid()
    ytick_locations = [5e-1, 3e-1, 1e-1, 5e-2, 3e-2, 1e-2, 1e-3, 3e-4, round(detection_flux, 5)]
    xtick_locations = [3e-2, 5e-2, 3e-1, 5e-1, 1, round(detection_distance, 5)]
    plt.xticks(ticks=xtick_locations, labels=xtick_locations)
    plt.yticks(ticks=ytick_locations, labels=ytick_locations)
    plt.savefig(out_fold+"hd_contrast.png", bbox_inches='tight')
    plt.savefig(out_fold+"hd_contrast.pdf", bbox_inches='tight')
    plt.savefig(out_fold+"hd_contrast.eps", bbox_inches='tight')
    plt.show()

def get_ROXs35A_snr():
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    target = "ROXs35A"
    snr_scale = 5
    print(target)
    folder = fold + target
    with pyfits.open(folder + "/alloutputs.fits") as hdulist:
        snr_calib = hdulist[0].data
    ny, nx = snr_calib.shape
    snr_calib[ny//2-1:ny//2+2, nx//2-1:nx//2+2] = np.nan
    snr_calib = snr_calib[40-3:40+5, 40-7:40+1]
    nyc, nxc = snr_calib.shape
    print(nyc, nxc)
    y_calib, x_calib = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(snr_calib, origin="lower", vmin=-snr_scale, vmax=snr_scale, cmap='cividis')
    ax.plot(7, 3, "rX", label="ROXs35A")
    ax.plot(x_calib, y_calib, "b.", 
        label=r"SNR {:.2f}, $\Delta y$ = {}, $\Delta x$ = {}".format(snr_calib[y_calib, x_calib], y_calib-3, x_calib-7))
    ax.plot(x_calib, y_calib-1, ".", color="purple", 
        label=r"SNR {:.2f}, $\Delta y$ = {}, $\Delta x$ = {}".format(snr_calib[y_calib-1, x_calib], y_calib-3-1, x_calib-7))
    cbar = fig.colorbar(img)
    cbar.set_label("SNR")
    # ax.set_xlim([-10+40, 2+40])
    # ax.set_ylim([-10+40, 2+40])
    ax.set_yticks(np.arange(0, nyc, 1))
    ax.set_xticks(np.arange(0, nxc, 1))
    ax.set_xticklabels(np.arange(-7, 1, 1))
    ax.set_yticklabels(np.arange(-3, 5, 1))
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.grid(linestyle=':', linewidth=1)
    ax.set_title(target)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0, -1.11, 1, 1), fontsize='small')
    plt.savefig(out_fold+"snr_map.png", bbox_inches='tight')
    plt.show()
    plt.close()
    # plt.figure()
    # # plt.imshow(snr_calib[y_calib-2:y_calib+3, x_calib-3:x_calib+3], vmin=-snr_scale, vmax=30, cmap='cividis', origin="lower")
    # plt.plot(snr_calib[y_calib])
    # plt.show()

def get_roxs35a_contrast():
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    target = "ROXs35A"
    print(target)
    folder = fold + target
    threshold = 5
    with pyfits.open(folder + "/alloutputs.fits") as hdulist:
        t_err_1sig_calib = hdulist[2].data
        r_grid = hdulist[4].data
        psf_profile = hdulist[5].data
        snr_calib, snr0 = hdulist[0].data, hdulist[3].data
        flux = snr_calib * t_err_1sig_calib
        t_err = t_err_1sig_calib / snr0 * snr_calib 
        # t_err = hdulist[6].data
    # y_calib, x_calib = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
    y_calib, x_calib = 40+1, 40-4
    detection_flux = flux[y_calib, x_calib]
    ny, nx = snr_calib.shape
    detection_distance = np.sqrt((y_calib-ny//2)**2 + (x_calib-nx//2)**2) * 0.02
    ny, nx = snr_calib.shape
    img, ax = plt.subplots(1,1)
    plt.title(f"{target}")
    plt.scatter(r_grid*0.02,psf_profile,label="PSF Profile", alpha=0.2, marker=".")
    plt.scatter(r_grid*0.02,threshold*t_err,label=r"5$\sigma$ contrast (w/o)", alpha=0.2, marker="1")
    plt.scatter(r_grid*0.02,threshold*t_err_1sig_calib,label=r"5$\sigma$ contrast (with)", alpha=0.2, marker="2")
    plt.plot(detection_distance, detection_flux, 'rX', label=f"detection")
    plt.axhline(detection_flux, color="red")
    plt.axvline(detection_distance, color="red")
    # plt.plot(detection_dist_calib, detection_flux_calib, 'bX', label=f"detection at snr {detection_snr_calib:.2f}")
    # plt.plot(detection_dist, detection_flux, 'bX', label="detection")
    plt.xlim([3e-2,1])
    plt.ylim([1e-4,5e-1])
    # ax.set_xticks(np.logspace(np.log10(2e-2), 0))
    # ax.set_xticklabels(np.logspace(np.log10(2e-2), 0))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("separation (arcseconds)")
    plt.ylabel("contrast")
    plt.legend()
    plt.grid()
    ytick_locations = [5e-1, 3e-1, 1e-1, 5e-2, 3e-2, 1e-2, 1e-3, 3e-4, round(detection_flux, 5)]
    xtick_locations = [3e-2, 5e-2, 3e-1, 5e-1, 1, round(detection_distance, 5)]
    plt.xticks(ticks=xtick_locations, labels=xtick_locations)
    plt.yticks(ticks=ytick_locations, labels=ytick_locations)
    plt.savefig(out_fold+"contrast.png", bbox_inches='tight')
    plt.show()

def get_nodes_vs():
    fold = "/scr3/jruffio/shubh/using-breads/plots/nodes-vs/"
    targets = ["HD148352", "ROXs44", "ROXs35A", "SR4"]
    for target in targets:
        fil = fold + f"nodes_20220512_1700K_{target}.fits"
        print(target, fil)
        with pyfits.open(fil) as hdulist:
            noi = hdulist[-1].data[0, :]
            tp = hdulist[-2].data[0, :]
            num_nodes = hdulist[0].data
        plt.figure(0)
        plt.plot(num_nodes[1:], tp[1:], marker="1", label=target)
        plt.figure(1)
        plt.plot(num_nodes[1:], noi[1:], marker="1", label=target)
    plt.figure(0)
    plt.legend()
    plt.grid()
    plt.title("Throughput at 60 mas")
    plt.xlabel("Number of Spline Nodes")
    plt.ylabel("Throughput")
    plt.savefig(out_fold + "tp.png")
    plt.figure(1)
    plt.yscale("log")
    plt.grid()
    plt.xlabel("Number of Spline Nodes")
    plt.ylabel("Noise")
    plt.title("Noise at 60 mas")
    plt.legend()
    plt.savefig(out_fold + "noise.png")
    plt.show()

def get_temp_recover():
    fold = "/scr3/jruffio/shubh/using-breads/plots/temp_recover/"
    targets = ["11.0", "13.0", "15.0", "17.0", "19.0"]
    labels= [1100, 1300, 1500, 1700, 1900]
    for ind, target in enumerate(targets):
        fil = fold + f"{target}/temp_recover_20220523_SR4.fits"
        print(target, fil)
        with pyfits.open(fil) as hdulist:
            snr = hdulist[-3].data[0, :]
            temperatures = hdulist[0].data
            if ind == 0:
                norm = np.nanmax(snr)
        plt.figure(0)
        plt.plot(temperatures * 100, snr/norm, marker=".", label=r"$T_{inject}$"+f" = {labels[ind]} K")
    plt.figure(0)
    plt.legend()
    plt.grid()
    plt.title("Relative signal-to-noise ratio at 100 milliarcseconds")
    plt.xlabel(r"Effective temperature $T_{eff}$ for recovery (K)")
    plt.ylabel("unnormalized SNR")
    plt.savefig(out_fold + "temp_recover.png")
    plt.savefig(out_fold + "temp_recover.pdf")
    plt.savefig(out_fold + "temp_recover.eps")
    plt.show()

def get_tp():
    fold = "/scr3/jruffio/data/osiris_survey/targets/"
    target_list = os.listdir(fold)
    plt.figure()
    for ind, target in enumerate(target_list):
        if "calib" in target or "HIP" in target or "HBC" in target:
            continue
        if "SR3" in target:
            fil = fold + f"{target}/{args.dates[target]}/first/reduced/throughput/20220417/"
        elif "AB_Aur" in target:
            fil = fold + f"{target}/{args.dates[target]}/1/reduced/throughput/20220417/"
        else:
            fil = fold + f"{target}/{args.dates[target]}/reduced/throughput/20220417/"
        fil = fil + os.listdir(fil)[2]
        print(target, fil)
        with pyfits.open(fil) as hdulist:
            flux = hdulist[0].data
            flux_b = hdulist[1].data
            flux_ratio = 1e-2
        plt.plot(np.arange(-40, 40, 1), (flux-flux_b)/flux_ratio, marker="1", label=target_list[ind], alpha=0.5, ls=":")
    plt.legend(loc='center left', ncol = 1, bbox_to_anchor=(1, 0, 1, 1))
    plt.grid()
    plt.title("Single Frame Throughput")
    plt.xlabel("Separation in pixels")
    plt.ylabel("Throughput")
    plt.savefig(out_fold + "tp.png", bbox_inches='tight')

def get_HD148352_rvccf():
    rvs = np.linspace(-4000,4000,41)
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    folder = "/scr3/jruffio/shubh/using-breads/plots/thesis/hd148352/rvccf.fits"
    with pyfits.open(folder) as hdulist:
        snr_calib = hdulist[0].data
        snr0 = hdulist[1].data
    plt.figure()
    plt.title(r"RV CCF, $\Delta y = \Delta x = - 4$")
    plt.plot(rvs, snr_calib, marker='1', linestyle='solid', label = "signal-to-noise ratio")
    # plt.legend()
    plt.grid()
    plt.xlabel(r"radial velocity $RV$ (km/s)")
    plt.ylabel(r"signal-to-noise ratio $SNR$")
    plt.savefig(out_fold+f"rvccf.png", bbox_inches='tight')
    plt.savefig(out_fold+f"rvccf.pdf", bbox_inches='tight')
    plt.savefig(out_fold+f"rvccf.eps", bbox_inches='tight')
    plt.close()
    # plt.show()


def get_HD148352_emcee():
    rvs = np.linspace(-4000,4000,41)
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    folder = "/scr3/jruffio/shubh/using-breads/plots/thesis/hd148352/corner_1000.fits"
    with pyfits.open(folder) as hdulist:
        samples = hdulist[0].data
    nonlin_labels = [r"$T_{eff}$", r"$\log g$", "spin", r"$RV$"]
    figure = corner.corner(samples, labels=nonlin_labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    # figure.set_title(r"MCMC Posteriors at $\Delta y = \Delta x = - 4$")
    plt.savefig(out_fold+f"corner.png", bbox_inches='tight')
    plt.savefig(out_fold+f"corner.pdf", bbox_inches='tight')
    plt.savefig(out_fold+f"corner.eps", bbox_inches='tight')
    plt.show()

def get_large_snr_plot():
    target_list = os.listdir(fold)
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    gs = gridspec.GridSpec(5, 5)
    fig = plt.figure(figsize=(22, 22))
    snr_scale = 5
    for ind,target in enumerate(target_list):
        print(ind,target)
        if ind//5 == 4:
            ax = plt.subplot(gs[ind//5, (ind%5)+1:(ind%5) + 2])
        else:
            ax = plt.subplot(gs[ind//5, (ind%5):(ind%5) + 1])
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            snr_calib = hdulist[0].data
        # y_calib, x_calib = np.unravel_index(np.nanargmax(snr_calib), snr_calib.shape)
        ny, nx = snr_calib.shape
        img = ax.imshow(snr_calib, origin="lower", vmin=-snr_scale, vmax=snr_scale, cmap='cividis')
        ax.plot(ny // 2, nx // 2, "rX")
        # ax.plot(x_calib, y_calib, "b.")
        ax.set_xticks(np.arange(0, 81, 10))
        ax.set_yticks(np.arange(0, 81, 10))
        ax.set_xticklabels(np.arange(-40, 41, 10))
        ax.set_yticklabels(np.arange(-40, 41, 10))
        if (ind+1)//5 == 4:
            ax.set_xlabel(r"$\Delta x$")
        if ind%5 == 0:
            ax.set_ylabel(r"$\Delta y$")
        ax.grid(linestyle=':', linewidth=1.5)
        ax.set_title(target)
        # plt.close()
    cbar = fig.colorbar(img, cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]))
    cbar.set_label("SNR")
    plt.savefig(out_fold+f"SNRmaps.png", dpi=250)
    plt.savefig(out_fold+f"SNRmaps.pdf", dpi=250)
    plt.savefig(out_fold+f"SNRmaps.eps", dpi=250)
    # plt.show()
    plt.close()

def forward_model_components():
    if 0:
        import sys
        sys.path.append("/scr3/jruffio/shubh/breads/")
        sys.path.append("/scr3/jruffio/code/BREADS_osiris_survey_scripts/")
        from breads.instruments.OSIRIS import OSIRIS
        from breads.grid_search import grid_search
        from breads.fm.hc_splinefm import hc_splinefm
        from breads.fm.hc_mask_splinefm import hc_mask_splinefm
        from breads.fm.hc_hpffm import hc_hpffm
        from breads.fit import fitfm
        import arguments_20220731 as arguments
        from copy import copy
        import multiprocessing as mp
        from scipy.interpolate import interp1d
        import scipy.linalg as la
        RV = -8.59
        numthreads = 16
        boxw = 1
        nodes = 5
        res_numbasis=5
        just_tellurics=True
        recalc_noise = True
        fit_background = False
        optimize_nodes = False
        star = "HD148352"
        dir_name = arguments.dir_name[star]
        tr_dir = arguments.tr_dir[star]
        sky_calib_file = arguments.sky_calib_file[star]
        files = glob(os.path.join(dir_name,"*.fits"))
        files.sort()
        my_pool = mp.Pool(processes=numthreads)

        # subdirectory = "planets/20220731_bleedoff/"
        # subdirectory = "planets/20220512_1700K/"
        subdirectory = "planets/20220926_flux/"

        print("making subdirectories")
        Path(dir_name+subdirectory).mkdir(parents=True, exist_ok=True)

        print("Reading planet file")
        # planet_btsettl="/scr3/jruffio/models/BT-Settl/BT-Settl-15/lte017-3.5-0.0a+0.0.BT-Settl.spec.7" #5Mjup at 2Myr
        planet_btsettl="/scr3/jruffio/models/BT-Settl/BT-Settl-15/lte032-5.0-0.0a+0.0.BT-Settl.spec.7" # HD Candidate
        # planet_btsettl="/scr3/jruffio/models/BT-Settl/BT-Settl-15/lte010-4.0-0.0a+0.0.BT-Settl.spec.7" #5Mjup at 20Myr
        print(planet_btsettl)
        try:
            arr = np.genfromtxt(planet_btsettl, delimiter=[13,13], dtype=np.float,
                                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
            model_wvs = arr[:, 0] / 1e4
            model_spec = 10 ** (arr[:, 1] - 8)
        except:
            arr = np.genfromtxt(planet_btsettl, delimiter=[13,12], dtype=np.float,
                                converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
            model_wvs = arr[:, 0] / 1e4
            model_spec = 10 ** (arr[:, 1] - 8)
        planet_model_set = True

        tr_files = os.listdir(tr_dir)
        if "plots" in tr_files:
            tr_files.remove("plots")
        tr_counter = 0
        tr_total = len(tr_files)
        fullfilename = files[5]
        filename = os.path.basename(fullfilename)
        print(filename)
        dataobj = OSIRIS(dir_name+filename) 
        nz,ny,nx = dataobj.data.shape
        # dataobj.noise = np.sqrt(np.abs(dataobj.data))
        print("setting noise")
        dataobj.set_noise(method="cont")
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

        print("compute stellar PSF")
        data = dataobj.data
        nz, ny, nx = data.shape
        stamp_y, stamp_x = (boxw-1)//2, (boxw-1)//2
        img_mean = np.nanmedian(data, axis=0)
        # star_y, star_x = np.unravel_index(np.nanargmax(img_mean), img_mean.shape)
        star_y, star_x = int(np.round(dataobj.refpos[1])), int(np.round(dataobj.refpos[0]))
        stamp = data[:, star_y-stamp_y:star_y+stamp_y+1, star_x-stamp_x:star_x+stamp_x+1]
        total_flux = np.nanmean(stamp) * np.size(stamp)
        # stamp = stamp/np.nansum(stamp,axis=(1,2))[:,None,None]

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
        dataobj.remove_bad_pixels(med_spec=star_spectrum,mypool=my_pool,threshold=5)
        dataobj.bad_pixels[0:5,:,:] = np.nan
        dataobj.bad_pixels[312:318,:,:] = np.nan
        dataobj.bad_pixels[343:349,:,:] = np.nan
        dataobj.bad_pixels[396:402,:,:] = np.nan
        dataobj.bad_pixels[418:422,:,:] = np.nan
        dataobj.bad_pixels[446::,:,:] = np.nan

        dataobj.bad_pixels[366:370,:,:] = np.nan
        dataobj.bad_pixels[373:378,:,:] = np.nan
        dataobj.bad_pixels[384:388,:,:] = np.nan

        # dataobj.trim_data(10)

        # print(dataobj.data.shape, dataobj.wavelengths.shape, transmission.shape, star_spectrum.shape)
        # print(dataobj.wavelengths)

        if planet_model_set:
            print("setting planet model")
            minwv,maxwv= np.nanmin(dataobj.wavelengths),np.nanmax(dataobj.wavelengths)
            crop_btsettl = np.where((model_wvs > minwv - 0.2) * (model_wvs < maxwv + 0.2))
            model_wvs = model_wvs[crop_btsettl]
            model_spec = model_spec[crop_btsettl]
            model_broadspec = dataobj.broaden(model_wvs,model_spec)
            planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)
            planet_model_set = False

        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(dataobj.read_wavelengths, planet_f(dataobj.read_wavelengths) / np.nanmax(planet_f(dataobj.read_wavelengths)), label="planet model")
        # plt.plot(dataobj.read_wavelengths, star_spectrum / transmission / np.nanmax(star_spectrum / transmission), label="starlight model")
        # plt.xlabel("wavelength")
        # plt.legend()
        # plt.grid()
        # plt.subplot(2, 1, 2)
        # plt.plot(dataobj.read_wavelengths, planet_f(dataobj.read_wavelengths) / np.nanmax(planet_f(dataobj.read_wavelengths)), label="planet model")
        # plt.plot(dataobj.read_wavelengths, np.abs(star_spectrum / transmission) / np.nanmax(np.abs(star_spectrum / transmission)), label="starlight model")
        # plt.xlabel("wavelength")
        # plt.legend()
        # plt.grid()
        # plt.savefig("./plots/TEMP5.png")
        # exit()
        for fractional_fov in ["bottom","top"]:
            if res_numbasis != 0:
                rvs = np.array([RV])
                # ys = np.arange(-11, 11)
                # xs = np.arange(-6, 6)
                if "bottom" in fractional_fov:
                    ys = np.arange(1, 11)
                elif "top" in fractional_fov:
                    ys = np.arange(-10, 0)
                elif "all" in fractional_fov:
                    ys = np.arange(-10, 11)
                xs = np.arange(-5, 6)

                # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
                #         "boxw":3,"nodes":20,"psfw":1.2,"badpixfraction":0.75}
                # fm_func = hc_splinefm
                fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
                    "boxw":boxw,"nodes":nodes,"psfw":(np.nanmedian(sig_y), np.nanmedian(sig_x)), "star_flux":total_flux,
                    "badpixfraction":0.75,"optimize_nodes":True, "stamp":stamp,"fit_background":fit_background,"recalc_noise":recalc_noise,"just_tellurics":just_tellurics}
                print("psfw:", np.nanmedian(sig_y), np.nanmedian(sig_x))
                fm_func = hc_mask_splinefm
                # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
                #             "boxw":3,"psfw":1.5,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40}
                # fm_func = hc_hpffm

                out_res = np.zeros((nz,np.size(ys),np.size(xs))) + np.nan
                log_prob_test = np.zeros((np.size(ys),np.size(xs))) + np.nan
                for k in range(np.size(ys)):
                    for l in range(np.size(xs)):
                        # badpix_vec = copy(dataobj.bad_pixels[:,int(np.round(dataobj.refpos[1] + ys[k])),int(np.round(dataobj.refpos[0] + xs[l]))])
                        # badpix_vec[np.where(np.isnan(star_spectrum*transmission))] = np.nan
                        # myres = np.zeros(np.size(np.where(np.isfinite(badpix_vec))[0]))+np.nan
                        # log_prob_test[k,l],_,_,_,_ = fitfm([0, ys[k], xs[l]], dataobj, fm_func, fm_paras,computeH0 = False,bounds = None,
                        #                                    residuals=myres)
                        # out_res[np.where(np.isfinite(badpix_vec)),k,l] = myres
                        w = (boxw-1)//2
                        # w=0
                        _y,_x=int(np.round(dataobj.refpos[1] + ys[k])),int(np.round(dataobj.refpos[0] + xs[l]))
                        badpix_vec = copy(dataobj.bad_pixels[:,_y-w:_y+w+1,_x-w:_x+w+1])
                        noise_vec = copy(dataobj.noise[:,_y-w:_y+w+1,_x-w:_x+w+1])
                        data_vec = copy(dataobj.data[:,_y-w:_y+w+1,_x-w:_x+w+1])

                        badpix_vec[np.where(np.isnan(star_spectrum*transmission))[0],:,:] = np.nan
                        canvas_res = np.zeros(badpix_vec.shape)+np.nan
                        myres = np.zeros(np.size(np.where(np.isfinite(badpix_vec))[0]))+np.nan
                        log_prob_test[k,l],_,rchi2,_,_ = fitfm([0, ys[k], xs[l]], dataobj, fm_func, fm_paras,computeH0 = True,bounds = None,
                                                           residuals_H0=myres,residuals=None)
                        # print(rchi2)
                        canvas_res[np.where(np.isfinite(badpix_vec))] = myres
                        out_res[:,k,l] = np.nanmean(canvas_res,axis=(1,2))

                X = np.reshape(out_res,(nz,np.size(ys)*np.size(xs))).T
                X = X[np.where(np.nansum(X,axis=1)!=0)[0],:]
                X = X/np.nanstd(X,axis=1)[:,None]
                X[np.where(np.isnan(X))] = np.tile(np.nanmedian(X,axis=0)[None,:],(X.shape[0],1))[np.where(np.isnan(X))]
                X[np.where(np.isnan(X))] = 0

                # print(X.shape)
                C = np.cov(X)
                # print(C.shape)
                # exit()
                tot_basis = C.shape[0]
                tmp_res_numbasis = np.clip(np.abs(res_numbasis) - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
                max_basis = np.max(tmp_res_numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
                evals, evecs = la.eigh(C, eigvals=(tot_basis-max_basis, tot_basis-1))
                check_nans = np.any(evals <= 0) # alternatively, check_nans = evals[0] <= 0
                evals = np.copy(evals[::-1])
                evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication
                # calculate the KL basis vectors
                kl_basis = np.dot(X.T, evecs)
                res4model_kl = kl_basis * (1. / np.sqrt(evals * (nz- 1)))[None, :]  #multiply a value for each row
                print(res4model_kl.shape)

            else:
                res4model_kl = None

            fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
                    "boxw":boxw,"nodes":nodes,"psfw":(np.nanmedian(sig_y), np.nanmedian(sig_x)), "star_flux": total_flux,
                    "badpixfraction":0.75,"optimize_nodes":optimize_nodes, "stamp": stamp,"KLmodes":res4model_kl,"fit_background":fit_background,"recalc_noise":recalc_noise,"just_tellurics":just_tellurics}
            # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":None, "star_loc":(np.nanmedian(mu_y), np.nanmedian(mu_x)),
            #             "boxw":1,"nodes":5,"psfw":(np.nanmedian(sig_y), np.nanmedian(sig_x)), "star_flux": np.nanmean(star_spectrum) * np.size(star_spectrum),
            #             "badpixfraction":0.75,"optimize_nodes":True, "stamp": stamp[:,stamp_y:stamp_y+1,stamp_x:stamp_x+1],"fit_background":False}
            print("psfw:", np.nanmedian(sig_y), np.nanmedian(sig_x))
            fm_func = hc_mask_splinefm
            # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
            #             "boxw":3,"psfw":1.5,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40}
            # fm_func = hc_hpffm
            # rvs = np.array([0])
            rvs = np.linspace(-4000,4000,41)
            if "bottom" in fractional_fov:
                ys = np.arange(-40, 0)
            elif "top" in fractional_fov:
                ys = np.arange(0, 40)
            elif "all" in fractional_fov:
                ys = np.arange(-40, 40)
            xs = np.arange(-20, 20)
        nonlin_paras = [-8.59, -4, -4] # rv (km/s), y (pix), x (pix)
        # nonlin_paras = [0, 0, 0] # rv (km/s), y (pix), x (pix)
        # d is the data vector a the specified location
        # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
        # s is the vector of uncertainties corresponding to d
        
        # log_prob,log_prob_H0,rchi2,linparas,linparas_err = fitfm(nonlin_paras,dataobj,fm_func,fm_paras)
        # print(log_prob,log_prob_H0,rchi2,linparas,linparas_err)
        d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)
        s = np.ones_like(s)

        np.savez(out_fold+"temp.npz", d=d, M=M, s=s)
    else:
        npzfile = np.load(out_fold+"temp.npz")
        d, M, s = npzfile["d"], npzfile["M"], npzfile["s"]

    validpara = np.where(np.sum(M,axis=0)!=0)
    M = M[:,validpara[0]]
    d = d / s
    M = M / s[:, None]
    # print(M.shape, d.shape, s.shape, star_spectrum.shape)
    from scipy.optimize import lsq_linear
    paras = lsq_linear(M, d).x
    m = np.dot(M,paras)

    print("plotting")
    # plt.subplot(2,1,1)
    # plt.plot(d,label="data")
    # plt.plot(m,label="model")
    # plt.plot(paras[0]*M[:,0],label="planet model")
    # plt.plot(m-paras[0]*M[:,0],label="starlight model")
    # plt.legend()
    # plt.subplot(2,1,2)
    # plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
    # plt.plot(M[:,1]/np.max(M[:,1]),label="telluric model")
    # for k in range(nodes * boxw):
    #     print(k)
    #     plt.plot(M[:,k+2]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+2))
    # for k in range(M.shape[-1]-(nodes*boxw)):
    #     print(k)
    #     plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="PCA model {0}".format(k+1))
    # plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(M[:, 7:]+np.array([-2,-1,0,1,2])*0.5)
    plt.show()



out_fold = "/scr3/jruffio/shubh/using-breads/plots/paper/firstdraft/"
fold = "/scr3/jruffio/code/BREADS_osiris_survey_scripts/plots/SNRmaps_contrast/20220512_1700K/"

# fold = "/scr3/jruffio/shubh/using-breads/plots/SNRmaps_contrast/20220417/"
# fold = "/scr3/jruffio/code/BREADS_osiris_survey_scripts/plots/SNRmaps_contrast/20220512_1700K/"
# out_fold = "/scr3/jruffio/shubh/using-breads/plots/thesis/throughput/"
# out_fold = "/scr3/jruffio/shubh/using-breads/plots/thesis/hist_with_rv/"
# get_calib_snr(fold, out_fold)
# get_no_calib_snr()

# get_all_snr_hist_nocalib()
# get_all_snr_hist_calib()

# get_combined_snr_hist_calib(fold, out_fold)
# get_combined_snr_hist_nocalib(fold, out_fold)
# get_target_hists("HD148352", fold, out_fold)
# get_target_hists("AB_Aur", fold, out_fold)

# get_target_contrast_curve("HD148352", fold, out_fold)
# get_target_contrast_curve("CW_Tau", fold, out_fold)
# get_target_contrast_curve("ROXs35A", fold, out_fold)

# get_combined_contrast_curve_nocalib(fold, out_fold)
# get_combined_contrast_curve_calib(fold, out_fold)

# get_combined_contrast_curve_nocalib(fold, out_fold)
# get_combined_contrast_curve_calibwithrv(fold, out_fold)

# get_HD148352_snr()
# get_HD148352_snr_larger()

# get_HD148352_contrast()

# get_ROXs35A_snr()
# get_roxs35a_contrast()

# get_nodes_vs()
# get_temp_recover()

# get_tp()

# get_HD148352_rvccf()
# get_HD148352_emcee()

# get_large_snr_plot()

forward_model_components()