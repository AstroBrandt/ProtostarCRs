# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 15:50:08 2017

@author: gache_000
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sint
import scipy.interpolate as si
import sys
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy.random as nrnd
import CR_physics
import cosmicSpec
label_size=16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size


nrnd.seed()
#PARAMETERS
RSUN = 6.957E10
LSUN = 3.848E33
ALPHA =  0.39704170
BETA  =  8.52762600
GAMMA =  0.00025546
DELTA =  5.43288900
EPSILON = 5.56357900
ZETA  =  0.78866060
ETA   =  0.00586685 
THETA =  1.71535900
IOTA  =  6.59778800
KAPPA = 10.08855000
LAMBDA = 1.01249500
MU    =  0.07490166
NU    =  0.01077422
XI    =  3.08223400
UPSILON = 17.84778000
PI    =  0.00022582

mh = 1.67372E-24
me = 9.109383E-28
kb = 1.380648E-16 #erg/K
kb_eV = 8.617330E-5 #eV/K
RSUN = 6.957E10
MSUN = 1.988E33
G = 6.67259E-8
planck_cgs = 6.6260755E-27 #erg s
cl = 2.998E10 #cm/s
gamma_ad = 5./3.
shock_eff = 0
GeVc_to_cgs = 5.3442859E-14

def lZAMS(m):
     LZAMS = (ALPHA*m**5.5 + BETA*m**11)/(GAMMA + m**3 + DELTA*m**5 + EPSILON*m**7 
             + ZETA*m**8 + ETA*m**9.5)
     return LZAMS
def rZAMS(m):
    RZAMS = (THETA*m**2.5 + IOTA*m**6.5 + KAPPA*m**11 + LAMBDA*m**19 + 
        MU*m**19.5)/(NU + XI*m**2 + UPSILON*m**8.5 + m**18.5 + PI*m**19.5)
    return RZAMS

deltan1 = 1.
#For TC
j = 0.5
jf = 0.75
Sigma = 1.0 #0.06
m0 = 3.6E-5*Sigma**(0.75)
######
ml = 0.033
mmax = 100.
r = 2.5 #in Solar units
facc = 0.75 #For feedback
L1 = 31.3*facc*(m0/1.0E-6)
#Read in the r(m, mf)
dat = np.loadtxt("TC_m_mf_r.txt")
ms = dat[:,0]; mfs = dat[:,1]; rs = dat[:,2] # 2.5*np.ones(len(dat[:,0])) #
mp = ms.reshape((69,5001)); mfp = mfs.reshape((69, 5001)); rp = rs.reshape((69, 5001))
for i in range(69):
	ir = np.where(mp[i] >= mfp[i])[0][0]
	rval = rp[i][ir]
	rp[i][np.where(mp[i] >= mfp[i])] = rval
ms = mp.flatten(); mfs = mfp.flatten(); rs = rp.flatten()
dum0, dum1 = np.mgrid[np.amin(ms):np.amax(ms):250j, np.amin(mfs):np.amax(mfs):250j]
grid_r = griddata((ms, mfs), rs, (dum0, dum1), method='nearest')
rspl = si.RectBivariateSpline(dum0[:,0], dum1[0,:], grid_r, s = 2, kx=1, ky=1)
#delete variables not used to clear memory
del dat; del ms; del mfs; del rs; del dum0; del dum1; del grid_r;  del mp;
del mfp; del rp;

def interpPhip2_mf(marr, mfarr, psip2, mi):
    if mi <= marr[0]:
        indx = 0
    elif mi >= marr[-1]:
        indx = len(marr)-1
    else:
        try:
            indx = np.where(mi <= marr)[0][0]-1
        except:
            print "ERROR: mi = ", mi
            print "mmax = ", marr[-1]
            print "mmin = ", marr[0]
            sys.exit()
    mat = psip2[:, indx:indx+2].T
    newfunc = []
    for i in range(len(mfarr)):
        slp = (mat[1,i] - mat[0, i])/(marr[indx+1]-marr[indx])
        b = mat[0, i]
        dx = mi-marr[indx]
        newfunc.append(slp*dx + b)
    return np.array(newfunc)

def Chabrier05(m):
	bi = 0.740741*(1. - mmax**(-27./20.))
	A1 = 1./(2.851 + bi*0.44956)
	A2 = 0.445956*A1
	if m < 1.0:
		return A1*np.exp(-(np.log10(m) - np.log10(0.2))**2/(2*0.55**2))
	else:
		return A2* m ** (-1.35)
       
def IMFArr(IMF, m):
    return np.array([IMF(mi) for mi in m])
    

def CIMF(IMF, ML=ml, MU=mmax):
	marr = np.logspace(np.log10(ml), np.log10(mmax), 1E4)
	integrand = (IMFArr(IMF, marr)/marr)
	cdist = sint.cumtrapz(integrand, x=marr, initial=0)
	f = si.interp1d(cdist, marr)
	return f

def acc(m, mf):
    return m0*(m/mf)**j*mf**jf*(1. - deltan1*(m/mf)**(1.-j))**(0.5)
    
def tm(mf):
    return (mf**(1.-jf)/((1.-j)*m0))*(1+deltan1)
    
def tacc(m, mf):
    return (1.-j)*(m/mf)**(1.-j)**(1. - deltan1*(m/mf)**(1.-j))**(-0.5)*tm(mf)

def tmav(IMF, ML, MU):
    mfs = np.linspace(ML, MU, 1E3)
    tmi = tm(mfs)
    imfi = np.array([IMF(mi) for mi in mfs])
    integrand = tmi*imfi/mfs
    return sint.trapz(integrand, x = mfs)
def PhiInvertSample(IMF, N=100):
    #Calculate the analytic bivariate on a massres^2 grid
    massres = 512
    mi = np.logspace(np.log10(ml), np.log10(mmax), massres)
    mfi = np.logspace(np.log10(ml), np.log10(mmax), massres)
    MS, MFS = np.meshgrid(mi, mfi)
    psip2arr = np.zeros((massres, massres), 'float64')
    tav = tmav(IMF, ml, mmax)
    for jj in range(0, massres):
        for ii in range(0, massres):
            if mi[ii] < mfi[jj]:
                psip2arr[jj,ii] = (IMF(mfi[jj])*tacc(mi[ii], mfi[jj]))/(tav)

    #Calculate the marginalization over final masses (essentially the PMF) and
    #ensure its normalized to 1
    psip2ymarg = sint.trapz(psip2arr, x=np.log(mfi), axis=0)
    psip2ymarg /= sint.trapz(psip2ymarg, x=np.log(mfi))
    #Interpolation function. GY gives the probabilty of a protostar with mass m
    GX = si.interp1d(mi, psip2ymarg)
    #Calculate the cumulative and setup the interpolator
    CPSIX = sint.cumtrapz(psip2ymarg, x=np.log(mi), initial=0)
    CX = si.interp1d(CPSIX, mi)
    
    mis = []
    mfs = []
    #N = number of stars to sample
    for ip in range(N):
        #Pull random value, use it to get the current mass
        rnd1 = nrnd.uniform(0,1)
        mrnd = CX(rnd1)
        #Given the current mass, calculate the conditional probabilty for the final mass
        #This is a sharp function that has a discontinuity at m = mf
        #The mask filters out the unphysical discontinuity, then it has to be rescaled to
        #ensure it integrates to 1. The rescaling needs to be done because of the finite grid
        #size making the initial discontinuity not as sharp
        psix = interpPhip2_mf(mi, mfi, psip2arr, mrnd)/GX(mrnd)
        indx0 = np.where(psix > 0)[0][0]
        mfi_n = mfi[indx0:]
        psix_n = psix[indx0:]
        psix_n /= sint.trapz(psix_n, x=np.log(mfi_n))
        CPSIY = sint.cumtrapz(psix_n, x=np.log(mfi_n), initial=0)
        #Setup the interpolator. The try is needed because, given the finite mass
        #resolution in the grid, if the pulled mass is within the very last bin
        #it cant interpolate with only 1 point. If this happens, it just continues
        #In general, this happens only a few millionths of the time
        try:
            CY = si.interp1d(CPSIY, mfi_n)
        except ValueError:
            print "Error line 775 - CY interp1d", len(CPSIY), len(mfi_n), mrnd
            continue            
        rnd2 = nrnd.uniform(0,1)
        #Try to do the interpolation. If there has been a problem with the scaling
        #this can throw and error, and in this case the program is killed
        try:
            mfrnd = CY(rnd2)
        except ValueError:
            print rnd2, CPSIY[0], CPSIY[-1]
            sys.exit()
        mis.append(mrnd)
        mfs.append(mfrnd)
    
    return np.array(mis), np.array(mfs)


B0 = 5#1E3
mu = 0.6


massres = 128
marr = np.logspace(np.log10(ml), np.log10(mmax), massres)
mfarr = np.logspace(np.log10(ml), np.log10(mmax), massres)
mgrid, mfgrid = np.meshgrid(marr, mfarr)
ngrid = np.zeros(mgrid.shape); rmgrid = np.zeros(mgrid.shape)
vgrid = np.zeros(mgrid.shape); tgrid = np.zeros(mgrid.shape)
xgrid = np.zeros(mgrid.shape)

Emaxgrid = np.zeros(mgrid.shape); Emingrid = np.zeros(mgrid.shape)
Egrid = np.zeros((massres, massres, cosmicSpec.ERes)); jEgrid = np.zeros((massres, massres, cosmicSpec.ERes))

NCOL = 3.073E23*(Sigma)
CRSPEC = cosmicSpec.CRSpectrum()
CRSPEC.add_attenuator("LE_loss_p.txt")
CRSPEC.set_column(NCOL)

for ii in range(massres):
    print "(%d/%d)"%(ii+1, massres)
    for jj in range(massres):
        if marr[ii] <= mfarr[jj]:
            ri = rspl.ev(marr[ii],mfarr[jj])
            magdip = B0*(ri*RSUN)**3
            mdot = max(acc(marr[ii], mfarr[jj]), 1E-20)*6.305286E25 #to g/s
            rmi = 0.7*(magdip**4/(4*G*marr[ii]*MSUN*mdot**2))**(1./7.)
            vffi = ((2.*G*marr[ii]*MSUN)/(ri*RSUN))**(0.5)
            tsi = (3./16.)*((mu*mh)/(kb))*vffi**2
            ni = (1./(mu*mh))*mdot*(1./(0.1*4.*np.pi*(ri*RSUN)**2*vffi))
            ngrid[ii,jj] = ni; rmgrid[ii,jj] = ri #rmi/RSUN;
            vgrid[ii,jj] = vffi; tgrid[ii,jj] = tsi
            eparams = CR_physics.calcSpecParams(vffi/1E5, tsi, ni, B0, ri*(0.00465))
            #espace, je = CR_physics.calcSpec(vffi/1E5, tsi, ni, B0, ri*(0.00465))
            CRSPEC.gen_spectrum(eparams['pmin'], eparams['pmax'], eparams['q'], eparams['f0'])
            rcore = 0.057*(Sigma)**(-0.5)*(mfarr[jj]/30.)**(0.5)*3.086E18
            CRSPEC.attenuate_spec()
            CRSPEC.spatial_dilute_spec(ri*RSUN, rcore, p = 2)
            espace, je = CRSPEC.get_dilutedSpectrum(retType='raw')
            #espace, je = CRSPEC.get_nullSpectrum(retType='raw')
            Emaxgrid[ii,jj] = eparams['emax']; Emingrid[ii,jj] = eparams['emin']
            #Emaxgrid[ii,jj] = espace[-1]; Emingrid[ii,jj] = espace[0]
            Egrid[ii,jj] = espace; jEgrid[ii,jj] = je
            CRSPEC.clearMem()     
                 
print "Done with grid"

fig = plt.figure(figsize=(8,8))
c = plt.pcolor(mgrid, mfgrid, rmgrid, cmap=plt.cm.viridis)
#c = plt.imshow(rmgrid.T, extent=[ml, mmax, ml, mmax],  cmap=plt.cm.viridis)
plt.xlabel("m", fontsize=20)
plt.ylabel(r"m$_f$", fontsize=20)
l = plt.colorbar(c)
l.set_label("r", fontsize=20)
plt.xscale('log')
plt.yscale('log')
fig.savefig("rm_grid.pdf")
plt.show()

                 
fig = plt.figure(figsize=(8,8))
c = plt.pcolor(mgrid, mfgrid, np.ma.masked_where(np.isinf(np.fabs(np.log10(ngrid))), np.log10(ngrid)), cmap = plt.cm.viridis)
#c = plt.imshow(np.log10(ngrid.T), extent=[ml, mmax, ml, mmax], cmap=plt.cm.viridis)
plt.xlabel("m", fontsize=20)
plt.ylabel(r"m$_f$", fontsize=20)
l = plt.colorbar(c)
l.set_label("n", fontsize=20)
plt.xscale('log')
plt.yscale('log')
fig.savefig("n_grid.pdf")
plt.show()

fig = plt.figure(figsize=(8,8))
c = plt.pcolor(mgrid, mfgrid, np.ma.masked_where(np.isinf(np.fabs(np.log10(vgrid/1.0E5))), np.log10(vgrid/1E5)), cmap = plt.cm.viridis)
#c = plt.imshow(np.log10(vgrid.T/1E5), extent=[ml, mmax, ml, mmax], cmap=plt.cm.viridis)
plt.xlabel("m", fontsize=20)
plt.ylabel(r"m$_f$", fontsize=20)
l = plt.colorbar(c)
l.set_label("v", fontsize=20)
plt.xscale('log')
plt.yscale('log')
fig.savefig("vkms_grid.pdf")
plt.show()

fig = plt.figure(figsize=(8,8))
c = plt.pcolor(mgrid, mfgrid, np.ma.masked_where(np.isinf(np.fabs(np.log10(tgrid))), np.log10(tgrid)), cmap = plt.cm.gist_heat)
#c = plt.imshow(np.log10(tgrid.T), extent=[ml, mmax, ml, mmax], cmap=plt.cm.viridis)
plt.xlabel("m", fontsize=20)
plt.ylabel(r"m$_f$", fontsize=20)
l = plt.colorbar(c)
l.set_label("T", fontsize=20)
plt.xscale('log')
plt.yscale('log')
fig.savefig("T_grid.pdf")
plt.show()

fig = plt.figure(figsize=(8,8))
c = plt.pcolor(mgrid, mfgrid, np.ma.masked_where(np.isinf(np.fabs(np.log10(Emaxgrid))), Emaxgrid), cmap = plt.cm.magma)
#c = plt.imshow(Emaxgrid.T, extent=[ml, mmax, ml, mmax], cmap=plt.cm.magma)
plt.xlabel("m", fontsize=20)
plt.ylabel(r"m$_f$", fontsize=20)
l = plt.colorbar(c)
l.set_label("Emax (GeV)", fontsize=20)
plt.xscale('log')
plt.yscale('log')
fig.savefig("emax_grid.pdf")
plt.show()

fig = plt.figure(figsize=(8,8))
c = plt.pcolor(mgrid, mfgrid, np.ma.masked_where(np.isinf(np.fabs(np.log10(Emingrid))), np.log10(Emingrid)), cmap = plt.cm.BuPu_r)
#c = plt.imshow(Emingrid.T, extent=[ml, mmax, ml, mmax], cmap=plt.cm.viridis)
plt.xlabel("m", fontsize=20)
plt.ylabel(r"m$_f$", fontsize=20)
l = plt.colorbar(c)
l.set_label("Emin (GeV)", fontsize=20)
plt.xscale('log')
plt.yscale('log')
fig.savefig("emin_grid.pdf")
plt.show()

'''fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
for ii in range(massres):
    for jj in range(massres):
        if ii <= jj:
            ax.loglog(Egrid[ii,jj]*1E9, jEgrid[ii,jj]*6.24E11, '-', c=plt.cm.viridis(np.log10(marr[ii])))
plt.xlabel("E (eV)")
plt.ylabel("j(E) (particles eV$^{-1}$ s$^{-1}$ cm$^{-2}$ sr$^{-1}$)")
plt.show()'''      
        
nclusters = np.logspace(0.5, 3.5, 12)
NSTAT = 20
CRIRT = []
CRIRstd = []
Nene = 400
earr = np.logspace(2, np.log10(1E10), Nene)
#Cross section WITHOUT relativistic correction
a0 = 5.29177211E-9 #cm
memp = 5.44617E-4; A = 0.71; B = 1.63; C = 0.51; D = 1.24;
xarr = memp*(earr/13.598)
sigmal = 4.*np.pi*a0**2*C*xarr**D
sigmah = 4.*np.pi*a0**2*(A*np.log(1+xarr) + B)*(1./xarr)
sigmap = 1./((1./sigmal) + (1./sigmah))

fig = plt.figure(figsize=(8,8))
plt.loglog(earr, sigmap, 'k-')
plt.xlabel("E (eV)", fontsize=20)
plt.ylabel(r"$\sigma_p$ (cm$^{-2}$)", fontsize=20)
fig.savefig("cross_section.pdf")
plt.show()

print "Generating distribution"

nc = int(1E5)
mis, mfs = PhiInvertSample(Chabrier05, N=int(nc))
CRIRdist = []
for i in range(int(nc)):
    ii = np.where(mis[i] < marr)[0][0] - 1
    jj = np.where(mfs[i] < mfarr)[0][0] - 1
    if (ii == jj):
        jj += 1
    ei = Egrid[ii,jj]; ji = jEgrid[ii,jj]
    if len(np.where(ji <= 0)[0] != 0):
        print "ERROR: ji flux = 0. m, mf = ", mis[i], mfs[i], ii, jj
        sys.exit()
    ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
    jnew = 10**ji_interp(np.log10(earr))
    integrand = jnew*sigmap
    CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
    CRIRdist.append(CRIRi)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.hist(np.log10(CRIRdist), bins=25, normed=True, facecolor='blue', alpha=0.6)
ax.set_xlabel(r"$\log\zeta_{p,{\rm unattenuated}}$ (s$^{-1}$)", fontsize=20)
ax.set_ylabel(r"\Psi(\zeta)")
fig.savefig("CRdist.pdf")
plt.show()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.grid(alpha=0.25)
import matplotlib.colors as colors
import matplotlib.cm as cmx
magma = plt.get_cmap('magma_r')
cNorm = colors.Normalize(vmin=np.log10(nclusters[0]), vmax=np.log10(nclusters[-1]))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
scalarMap._A = []


f = open("Cluster_CR_data_TTC.txt", 'w+')
for nc in nclusters:
    print "On ", int(nc)
    Jtotal_arr = []
    CRIR_arr = []
    MTOT_arr = []
    for kk in range(NSTAT):
        mis, mfs = PhiInvertSample(Chabrier05, N=int(nc))
        Jtotal = np.zeros(Nene, 'float64')
        MTOT_arr.append(np.log10(np.nansum(mis)))
        for i in range(int(nc)):
            ii = np.where(mis[i] < marr)[0][0] - 1
            jj = np.where(mfs[i] < mfarr)[0][0] - 1
            if (ii == jj):
                jj += 1
            ei = Egrid[ii,jj]; ji = jEgrid[ii,jj]
            if len(np.where(ji <= 0)[0] != 0):
                print "ERROR: ji flux = 0. m, mf = ", mis[i], mfs[i], ii, jj
                sys.exit()
            ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
            jnew = 10**ji_interp(np.log10(earr))
            Jtotal += jnew
        integrand = Jtotal*sigmap
        CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
        Jtotal_arr.append(Jtotal)
        CRIR_arr.append(CRIRi)
    CRIRT.append(np.mean(np.log10(CRIR_arr)))
    CRIRstd.append(np.std(np.log10(CRIR_arr)))
    color = scalarMap.to_rgba(np.log10(nc))
    Jmean = 10**np.mean(np.log10(Jtotal_arr), axis=0)
    f.write("%d\t%.3e\t%.3e\n"%(nc, CRIRT[-1], np.mean(MTOT_arr)))
    ax.loglog(earr, 10**np.mean(np.log10(Jtotal_arr), axis=0), '-', color=color)
    #f = open("%d_spec.txt"%int(nc), 'w+')
    #for jj in range(len(Jtotal)):
    #    f.write("%e\t%e\n"%(earr[jj], Jmean[jj]))
    #f.close()
CRIRT = np.array(CRIRT)
CRIRstd = np.array(CRIRstd)
ax.set_xlabel("E (eV)", fontsize=20)
ax.set_ylabel("j(E) (particles eV$^{-1}$ s$^{-1}$ cm$^{-2}$ sr$^{-1}$)", fontsize=20)
c1 = plt.colorbar(scalarMap)
c1.set_label(r"$\log \, N_*$")
fig.savefig("Cluster_Spec.pdf")
plt.show()


fig = plt.figure(figsize=(8,8))
plt.errorbar(nclusters, CRIRT, yerr = CRIRstd, color='b', markersize=10)
plt.xscale('log')
plt.xlabel("N$_*$", fontsize=20)
plt.ylabel(r"$\log\zeta_{p,{\rm surface}}$ (s$^{-1}$)", fontsize=20)
fig.savefig("CRIRcluster.pdf")
plt.show()

import matplotlib.patches as patches

lnclusters = np.log10(nclusters)
dx = np.fabs((lnclusters[:-1] - lnclusters[1:]))[0]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
for i in range(len(CRIRT)):
    ax.add_patch(patches.Rectangle((lnclusters[i] - 0.5*dx, CRIRT[i] - CRIRstd[i]), dx, 2*CRIRstd[i], facecolor='#808080', alpha=0.25))
    ax.add_patch(patches.Rectangle((lnclusters[i] - 0.5*dx, CRIRT[i] - 0.5*CRIRstd[i]), dx, CRIRstd[i], facecolor='blue', alpha=0.5))
ax.plot(np.log10(nclusters), CRIRT, 'k-')
ax.plot(np.log10(nclusters), CRIRT, 'k.')
ax.set_xlabel(r"$\log N_*$", fontsize=20)
ax.set_ylabel(r"$\log\zeta_{p,{\rm unattenuated}}$ (s$^{-1}$)", fontsize=20)
plt.tight_layout()
fig.savefig("BIG_RESULT.pdf")
plt.show()
