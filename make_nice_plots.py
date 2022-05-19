print("starting")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import astropy.io.fits as pyfits
import arguments as args
from glob import glob

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
    fold = "/scr3/jruffio/shubh/using-breads/plots/SNRmaps_contrast/20220417/"
    out_fold = "/scr3/jruffio/shubh/using-breads/plots/thesis/snrhist_calib/"
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

        plt.savefig(out_fold+f"{target}.png")
        plt.close()

def get_combined_snr_hist_calib(fold, out_fold):
    target_list = os.listdir(fold)
    text_font = 12
    print("making subdirectories")
    Path(out_fold).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title("Residual SNR with noise+RV calibration", fontsize=text_font,color="black")
    bins = np.arange(-6,6,0.25)
    bin_centers = np.array([(r1+r2)/2. for r1,r2 in zip(bins[0:-1],bins[1:])])
    for target in target_list:
        print(target)
        folder = fold + target
        with pyfits.open(folder + "/alloutputs.fits") as hdulist:
            snr_calib = hdulist[0].data
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
    plt.savefig(out_fold+f"combined_snr_hist_calib.png", bbox_inches='tight')
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
    plt.xlim([2e-2,0.4])
    plt.ylim([3e-4,3e-2])
    # ax.set_xticks(np.logspace(np.log10(2e-2), 0))
    # ax.set_xticklabels(np.logspace(np.log10(2e-2), 0))
    plt.xlabel("separation (arcseconds)")
    plt.ylabel("contrast")
    # plt.legend()
    plt.grid()
    ytick_locations = [3e-2, 1e-2, 3e-3, 1e-3, 3e-3, 3e-4]
    xtick_locations = [2e-2, 3e-2, 5e-2, 1e-1, 3e-1, 0.4]
    plt.xticks(ticks=xtick_locations, labels=xtick_locations)
    plt.yticks(ticks=ytick_locations, labels=ytick_locations)
    ax = plt.gca()
    ax.legend(loc='center left', ncol = 1, bbox_to_anchor=(1, 0, 1, 1))
    plt.title("Sensitivity with RV normalization")
    # plt.savefig(out_fold+f"combined_contrast_calib_withrv.png", bbox_inches='tight')
    # plt.close()
    plt.show()

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
    fig, (ax, ax1) = plt.subplots(2,1)
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
    # plt.savefig(out_fold+"snr_map.png", bbox_inches='tight')
    plt.show()
    plt.close()
    plt.figure()
    # plt.imshow(snr_calib[y_calib-2:y_calib+3, x_calib-3:x_calib+3], vmin=-snr_scale, vmax=30, cmap='cividis', origin="lower")
    plt.plot(snr_calib[y_calib])
    plt.show()

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
    # plt.show()

# fold = "/scr3/jruffio/shubh/using-breads/plots/SNRmaps_contrast/20220417/"
fold = "/scr3/jruffio/code/BREADS_osiris_survey_scripts/plots/SNRmaps_contrast/20220512_1700K/"
out_fold = "/scr3/jruffio/shubh/using-breads/plots/thesis/hd148352/"
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
fold = "/scr3/jruffio/code/BREADS_osiris_survey_scripts/plots/SNRmaps_contrast/20220512_1700K/"
# get_combined_contrast_curve_calibwithrv(fold, out_fold)

# get_HD148352_snr()
get_HD148352_contrast()