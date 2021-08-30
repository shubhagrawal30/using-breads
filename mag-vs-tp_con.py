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


targets = ["HD148352", "SR3", "SR14", "ROXs44", "ROXs43B", "SR9", "SR4", "ROXs8", "ROXs35A"]
Kmags = [6.511, 6.504, 8.878, 7.61, 7.09, 7.207, 7.518, 6.227, 8.531]
spec_types = ["F2V", "A0", "G5", "K3e", "K3e", "K5e", "K0:Ve", "K0", "K1IV"]
threshold_snr = 5

for target, Kmag, spec_type in zip(targets[:-1], Kmags[:-1], spec_types[:-1]):
    targetf = f"TP_{target}"
    label = f"{target}: {Kmag}, {spec_type}"
    with pyfits.open(f"./plots/scatter_{targetf}.fits") as hdulist:
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
    rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, 500, False)
    plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, label=label, alpha=0.8, ls='none')
    plt.figure(4)
    rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, 500, True)
    plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, label=label, alpha=0.8, ls='none')

plt.figure(1)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.xscale('log')
plt.yscale('log')
plt.xlabel("separation in mas")
plt.ylabel("flux ratio")
plt.title("contrast curves")
plt.grid()
plt.tight_layout()

plt.figure(2)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.xlabel("separation in mas")
plt.ylabel("recovered / injected flux")
plt.title("throughputs")
plt.grid()
plt.tight_layout()

plt.figure(3)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.xscale('log')
plt.yscale('log')
plt.xlabel("separation in mas")
plt.ylabel("flux ratio")
plt.title("contrast curves")
plt.grid()
plt.tight_layout()

plt.figure(4)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.xscale('log')
plt.yscale('log')
plt.xlabel("separation in mas")
plt.ylabel("flux ratio")
plt.title("contrast curves")
plt.grid()
plt.tight_layout()

plt.show()

