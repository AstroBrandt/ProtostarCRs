# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:10:35 2017

@author: gache_000
"""

import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sys

effArr = np.logspace(-2, -0.698970, 4) 
nclusters = np.logspace(0.5, 3.5, 6)
lnclusters = np.log10(nclusters)

def func((eff, Nstar), slopeE, slopeN, offset):
    return slopeE*eff + slopeN*Nstar + offset

datMean = np.loadtxt("data_CR_mean.txt")
datStd = np.loadtxt("data_CR_std.txt")

zarr = []
parr = []
slopes = []
offsets = []
for k in range(len(effArr)):
    ydat = []
    xdat = []
    for i in range(len(nclusters)):
        Mgas = 0.2*10**lnclusters[i]/effArr[k]
        ydat.append(datMean[k][i])
        xdat.append(lnclusters[i])
    z = np.polyfit(xdat, ydat, 1)
    p = np.poly1d(z)
    zarr.append(z)
    parr.append(p)
    slopes.append(z[0])
    offsets.append(z[1])
    
fitOffsets =  np.polyfit(np.log10(effArr), offsets, 1)
lngrid, legrid = np.meshgrid(np.log10(nclusters), np.log10(effArr))
ydata = datMean.ravel()
sigma = datStd.ravel()
xdata = np.vstack((legrid.ravel(), lngrid.ravel()))   

print datMean.shape, legrid.shape, lngrid.shape

'''from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(legrid, lngrid, datMean)
ax.set_xlabel("$\epsilon_g$", fontsize=18)
ax.set_ylabel("N$_*$", fontsize=18)
ax.set_zlabel("$\log \zeta$ (s$^{-1}$)")
plt.show()'''

popt, pcov = sopt.curve_fit(func, xdata, ydata, maxfev=int(1E4))
print popt


figG = plt.figure(figsize=(10,10))
axG = figG.add_subplot(111)
axG.grid(alpha=0.25)
axG.set_xlabel(r"$\log$ M$_{\rm gas}$", fontsize=24)
axG.set_ylabel(r"$\log\zeta_{p,{\rm attenuated}}$ (s$^{-1}$)", fontsize=24)
PuBu = plt.get_cmap('viridis')
cNormG = colors.Normalize(vmin=np.log10(effArr[0]), vmax=np.log10(effArr[-1]))
scalarMapG = cmx.ScalarMappable(norm=cNormG, cmap=PuBu)
scalarMapG._A = []
for k in range(len(effArr)):
    colorG = scalarMapG.to_rgba(np.log10(effArr[k]))
    for i in range(len(nclusters)):
        Mgas = 0.2*10**lnclusters[i]/effArr[k]
        axG.errorbar(np.log10(Mgas), datMean[k][i], markersize=5.+lnclusters[i]**1.5, color=colorG, yerr = 2.*datStd[k][i], fmt='o')
    lnspace = np.logspace(0, 4, 20)
    mgasspace = 0.2*lnspace/effArr[k]
    #axG.plot(np.log10(mgasspace), parr[k](np.log10(lnspace)), color=colorG, linestyle=':', label=r"$\alpha$ = %.2f"%slopes[k])
    #axG.plot(np.log10(mgasspace), parr[k](np.log10(mgasspace)), color=colorG, linestyle=':', label=r"$\alpha$ = %.2f"%slopes[k])         
nplotarr = np.logspace(np.log10(nclusters[0]), np.log10(nclusters[-1]), 40)
eplotarr = np.logspace(np.log10(effArr[0]), np.log10(effArr[-1]), 40)
for k in range(len(effArr)):
    colorG = scalarMapG.to_rgba(np.log10(effArr[k]))
    xp = np.log10(np.vstack((np.ones(40)*effArr[k], nplotarr)))
    yp = func(xp, *popt)
    mgasspace = 0.2*nplotarr/effArr[k]
    axG.plot(np.log10(mgasspace), yp, color=colorG, linestyle=':')
for k in range(len(nclusters)):
    colorG = scalarMapG.to_rgba(np.log10(eplotarr))
    xp = np.log10(np.vstack((eplotarr, nclusters[k]*np.ones(40))))
    yp = func(xp, *popt)
    mgasspace = 0.2*nclusters[k]/eplotarr
    axG.scatter(np.log10(mgasspace), yp, color=colorG, s=2)
nclustann = np.array([1, 10, 100, 1000, 5000])
dx = np.array([0.01, 0.035, 0.06, 0.11, 0.12])
for k in range(len(nclustann)):
    marr = np.logspace(0, 5, 50)
    earr = 0.2*nclustann[k]/marr
    xp = np.log10(np.vstack((earr, nclustann[k]*np.ones(50))))
    yp = func(xp, *popt)
    axG.scatter(np.log10(marr), yp, color='black', s=2, alpha=0.7)
    if nclustann[k] <= 100:
        indx = np.where(np.log10(marr) >= 4)[0][0]
        axG.annotate(r"N$_*$ = %d"%nclustann[k], xy=(np.log10(marr[indx]), yp[indx]+dx[k]), xytext=(np.log10(marr[indx]), yp[indx]+dx[k]), rotation=22, fontsize=18)
    else:
        indx = np.where(np.log10(marr) >= 1)[0][0]
        axG.annotate(r"N$_*$ = %d"%nclustann[k], xy=(np.log10(marr[indx]), yp[indx]+dx[k]), xytext=(np.log10(marr[indx]), yp[indx]+dx[k]), rotation=22, fontsize=18)


axG.annotate(r"$\log\,\zeta \approx$ %.2f $\log \epsilon_g$ + %.2f $\log \, N_*$ - %.2f"%(popt[0], popt[1], np.fabs(popt[2])),
             xy=(0.1, -15.15), xytext=(0.1, -15.15), fontsize=16)
axG.axhline(y=np.log10(3E-17), linewidth=2, color='#d3d3d3')
c1 = plt.colorbar(scalarMapG)
c1.set_label(r"$\log \, \epsilon_g$", fontsize=24)
axG.set_xlim(0, 5); axG.set_ylim(-19.5, -15)
plt.legend(loc=2)
plt.tight_layout()
figG.savefig("BIG_RESULT_GENERAL_small.png")
plt.show()
    