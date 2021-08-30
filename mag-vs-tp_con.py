import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

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


targets = ["HD148352", "SR3", "SR14", "ROXs44", "ROXs43B", "SR9", "SR4", "ROXs8", "ROXs35A", "SR21A"]
Kmags = [6.511, 6.504, 8.878, 7.61, 7.09, 7.207, 7.518, 6.227, 8.531, 6.719]
spec_types = ["F2V", "A0", "G5", "K3e", "K3e", "K5e", "K0:Ve", "K0", "K1IV", "G1"]
threshold_snr = 5

for target, Kmag, spec_type in zip(targets, Kmags, spec_types):
    print(target)
    targetf = f"TP_{target}"
    label = f"{target}: {Kmag}, {spec_type}"
    with pyfits.open(f"./plots/scatter/scatter_{targetf}.fits") as hdulist:
        calibrated_err_combined, throughput = hdulist[0].data, hdulist[1].data
    distances = []
    noise = []
    tp = []
    nx, ny = calibrated_err_combined.shape
    xS, yS = nx / 2, ny / 2
    for x in range(nx):
        for y in range(ny):
            distances += [np.sqrt((y - yS) ** 2 + (x - xS) ** 2) * 20]
            noise += [threshold_snr * calibrated_err_combined[x, y]]
            tp += [throughput[x, y]]
    plt.figure(1)
    plt.scatter(distances, noise, label=label, alpha=0.1)
    plt.figure(2)
    plt.scatter(distances, tp, label=label, alpha=0.1)
    plt.figure(3)
    rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, 100, False)
    plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, label=label, alpha=0.6, ls='none')
    plt.figure(4)
    rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, 100, True)
    plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, label=label, alpha=0.6, ls='dotted')

for ind in [1, 2, 3, 4]:
    print("plotting", ind)
    plt.figure(ind)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.14,-0.32,1.16,0.2), ncol=5, mode='expand', fontsize='x-small')
    plt.xlabel("separation in mas")
    if ind == 2:
        plt.ylabel("recovered / injected flux")
        plt.title("throughputs")
    else:
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel("flux ratio")
        plt.ylim([1e-4, 1e-2])
        plt.title(f"contrast curves (SNR {threshold_snr})")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./plots/mag_con_tp/{ind}.png")

plt.show()

