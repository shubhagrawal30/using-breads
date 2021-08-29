import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

def line_from_scatter(xvals, yvals, num, equal_bins=False):
    if equal_bins:
        xvals, yvals = np.sort(xvals), np.array(yvals)[np.argsort(xvals)]
        rxs, rys, xerrs, yerrs = [], [], [], []
        bin_lims = np.linspace( )
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


targets = ["TP_HD148352", "TP_SR3", "TP_SR14", "TP_ROXs44"]
infos = ["6.511", "6.504", "8.878", "7.61"]

for target, info in zip(targets, infos):
    infof = f": {info}"
    with pyfits.open(f"./plots/scatter_{target}.fits") as hdulist:
        calibrated_err_combined, throughput = hdulist[0].data, hdulist[1].data
    distances = []
    noise = []
    tp = []
    nx, ny = calibrated_err_combined.shape
    xS, yS = nx / 2, ny / 2
    for x in range(nx):
        for y in range(ny):
            distances += [np.sqrt((y - yS) ** 2 + (x - xS) ** 2) * 20]
            noise += [calibrated_err_combined[x, y]]
            tp += [throughput[x, y]]
    plt.figure(1)
    plt.scatter(distances, noise, label=target[3:]+infof, alpha=0.1)
    plt.figure(2)
    plt.scatter(distances, tp, label=target[3:]+infof, alpha=0.1)
    plt.figure(3)
    rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, 100, False)
    plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, label=target[3:]+infof, alpha=0.5, ls=None)
    # plt.figure(4)
    # rxs, rys, xerrs, yerrs = line_from_scatter(distances, noise, 100, True)
    # plt.errorbar(rxs, rys, yerr=yerrs, xerr=xerrs, label=target[3:]+infof, alpha=0.5, ls=None)

plt.figure(1)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.figure(2)
plt.legend()
plt.figure(3)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()
# plt.figure(4)
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.show()

