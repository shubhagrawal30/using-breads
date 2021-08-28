import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

nodes = 20
psfw = 2.1244
boxw = 3

# fil = "s210626_a032009_Kn5_020"
fil = "s210626_a032007_Kn5_020"
fil1 = f"./plots/contrast/{fil}/HD148352/contrast_0.01_{boxw}.fits"
fil2 = f"./plots/contrast/{fil}/HD148352/contrast_0.005_{boxw}.fits"

# 009, boxw 5
flux = 0.0017156623959581566
dHD = np.sqrt(4 ** 2 + 3 ** 2) * 20
snr = 12.494889242380125
noise = 0.0001373091319720529

# 009, boxw 5
# flux = 0.0017156623959581566
# dHD = np.sqrt(4 ** 2 + 3 ** 2) * 20
# snr = 12.494889242380125
# noise = 0.0001373091319720529

# 009, boxw 3
# flux = 0.00398953
# dHD = np.sqrt(3 ** 2 + 3 ** 2) * 20
# snr = 16.198
# noise = 0.00024629

with pyfits.open(fil1) as a:
    with pyfits.open(fil2) as b:
        noiseP = (a[3].data + b[3].data) / 2
        scaling = (b[2].data/5e-3 + a[2].data/1e-2) / 2
        distances = a[0].data * 20
        plt.figure()
        plt.plot(distances, a[3].data, label="1e-2")
        plt.plot(distances, b[3].data, label="5e-3")
        plt.legend()
        plt.title(f"{fil}, predicted noise")
        plt.ylabel("noise")
        plt.xlabel("separation in mas")

        plt.figure()
        plt.title(f"{fil}, recovered flux")
        plt.ylabel("predicted flux / injected flux")
        plt.xlabel("separation in mas")
        plt.plot(a[0].data, a[2].data/1e-2, label="1e-2")
        plt.plot(b[0].data, b[2].data/5e-3, label="5e-3")
        plt.legend()


fluxP = flux / interp1d(distances, scaling, kind='cubic')(dHD)

plt.figure()
plt.plot([dHD], [fluxP], "rx", label="calibated detection")
plt.title(f"{fil}, {nodes} nodes, psfw = {psfw}, boxw={boxw}, cubic scaling")
plt.plot(distances, 5*noiseP/scaling, label = f"snr = 5")
plt.plot(distances, snr*noiseP/scaling, label = f"snr = {snr}")
plt.ylabel("calibrated flux")
plt.xlabel("separation in mas")
plt.grid()
plt.legend()

plt.figure()
plt.plot([dHD], [flux], "rx", label="detection")
plt.title(f"{fil}, {nodes} nodes, psfw = {psfw}, boxw={boxw}")
plt.plot(distances, 5*noiseP, label = f"snr = 5")
plt.plot(distances, snr*noiseP, label = f"snr = {snr}")
plt.ylabel("detected flux")
plt.xlabel("separation in mas")
plt.grid()
plt.legend()
plt.show()
