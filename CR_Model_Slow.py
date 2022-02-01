0# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:53:43 2017

@author: gache_000
"""

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
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx
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
mpc2 = 0.938 #GeV
mpc = 0.938 #GeV/c
ev_to_ergs = 1.602E-12
rsun_to_au = 0.00465
Msunyr_to_gs = 6.305286E25
au_to_cm = 1.496E13

def pfuncE(E):
    return np.sqrt((E + mpc2)**2 - mpc2**2)

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
#facc = 0.5
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

def getTemperature(r):
    #Disk parameters
    T0 = 200. #K
    r0 = 1.0 #AU
    q = 0.5
    T_r = T0*(r/r0)**(-q)
    return T_r

'''def get_spec(mi, mf, CRSPEC, atten=True, fullOut = False, ruse = None, atrans = 3.0, NCOL=None):
    a2 = 3. # dipole moment, stellar field
    a8 = 7. # octpole moment, stellar field
    a10 = 9. # higher order momentsm, stellar field
    fmacc = 0.1 #0.9
    ri = rspl.ev(mi ,mf)
    magdip = B0*(ri*RSUN)**3
    mdot = max(acc(mi, mf), 1E-20)*6.305286E25 #to g/s
    rmi = 0.7*(magdip**4/(4*G*mi*MSUN*mdot**2))**(1./7.)
    vffi = ((2.*G*mi*MSUN)/(ri*RSUN))**(0.5)
    tsi = (3./16.)*((mu*mh)/(kb))*vffi**2
    ni = (1./(mu*mh))*mdot*(1./(fmacc*4.*np.pi*(ri*RSUN)**2*vffi))

    rmi = max(4.*ri*RSUN, rmi)

    eparams = CR_physics.calcSpecParams(vffi/1E5, tsi, ni, B0, ri*(0.00465))
    CRSPEC.gen_spectrum(eparams['pmin'], eparams['pmax'], eparams['q'], eparams['f0'])
    ei, ji = CRSPEC.get_nullSpectrum(retType = 'raw')
    #eatten, jatten = CRSPEC.attenuate_provided_v2(ei, ji, NCOLcol)
    #print(jatten)
    magattenfac = (0.5*((ri*RSUN)/rmi)**(a2)+ 0.3*((ri*RSUN)/rmi)**(a8)+0.1*((ri*RSUN)/rmi)**(a10))/0.9
    jatten_1 = ji#*magattenfac

    if ruse == None:
        rcore = 0.057*(Sigma)**(-0.5)*(mf/30.)**(0.5)*3.086E18
    else:
        rcore = ruse
    if atten == True:
        eatten2, jatten2 = CRSPEC.attenuate_provided_v2(ei, jatten_1, NCOL)
        jatten2_f = jatten2 * ((rmi)/rcore)**atrans
        #CRSPEC.spatial_dilute_spec(ri*RSUN, rcore, p = atrans)
        #espace, je = CRSPEC.get_dilutedSpectrum(retType='raw')
        espace = eatten2
        je = jatten2_f
    else:
        espace, je = CRSPEC.get_nullSpectrum(retType = 'raw')
    if fullOut:
        return espace, je, eparams
    else:
        return espace, je, eparams['emax']'''

def get_spec(mi, mf, CRSPEC, atten=True, fullOut = False, ruse = None, atrans = 2, NCOL=None):
    fmacc = 0.1
    ri = rspl.ev(mi ,mf)
    #magdip = B0*(ri*RSUN)**3
    mdot = max(acc(mi, mf), 1E-20)*6.305286E25 #to g/s
    #rmi = 0.7*(magdip**4/(4*G*mi*MSUN*mdot**2))**(1./7.)
    vffi = ((2.*G*mi*MSUN)/(ri*RSUN))**(0.5)
    tsi = (3./16.)*((mu*mh)/(kb))*vffi**2
    ni = (1./(mu*mh))*mdot*(1./(fmacc*4.*np.pi*(ri*RSUN)**2*vffi))

    eparams = CR_physics.calcSpecParams(vffi/1E5, tsi, ni, B0, ri*(0.00465))
    CRSPEC.gen_spectrum(eparams['pmin'], eparams['pmax'], eparams['q'], eparams['f0'])
    if ruse == None:
        rcore = 0.057*(Sigma)**(-0.5)*(mf/30.)**(0.5)*3.086E18
    else:
        rcore = ruse
    if atten == True:
        CRSPEC.attenuate_spec_v2(NCOL)
        #CRSPEC.attenuate_spec()
        CRSPEC.spatial_dilute_spec(ri*RSUN, rcore, p = atrans)
        espace, je = CRSPEC.get_dilutedSpectrum(retType='raw')
    else:
        espace, je = CRSPEC.get_nullSpectrum(retType = 'raw')
    if fullOut:
        return espace, je, eparams
    else:
        return espace, je, eparams['emax']

def get_shock_vals(mi, mf):
    fmacc = 0.1
    ri = rspl.ev(mi, mf)
    mdot = max(acc(mi, mf), 1E-20)*6.305286E25 #to g/s
    vffi = ((2.*G*mi*MSUN)/(ri*RSUN))**(0.5)
    tsi = (3./16.)*((mu*mh)/(kb))*vffi**2
    ni = (1./(mu*mh))*mdot*(1./(fmacc*4.*np.pi*(ri*RSUN)**2*vffi))
    return {'mdot':mdot, 'vff':vffi, 'ts':tsi, 'n':ni, 'r':ri}
def check_conditions(mi, mf):
    fmacc = 0.1
    ri = rspl.ev(mi ,mf)
    gamma_ad = 5./3.
    mdot = max(acc(mi, mf), 1E-20)*6.305286E25 #to g/s
    vffi = ((2.*G*mi*MSUN)/(ri*RSUN))**(0.5)
    tsi = (3./16.)*((mu*mh)/(kb))*vffi**2
    ni = (1./(mu*mh))*mdot*(1./(fmacc*4.*np.pi*(ri*RSUN)**2*vffi))
    cs = np.sqrt((gamma_ad*kb*tsi)/(mu*mh))/1E5
    va = B0/np.sqrt(4.*np.pi*ni*mh)/1E5
    #cond1 = ((vffi/1E5)/1E2) > max(cs/1E2, 2.2E-4*(ni/1E6)**(-0.5)*(B0/(10*1E-6)))
    cond1 = ((vffi/1E5)/1E2) > max(cs/1E2, va/1E2)
    return int(cond1)

test = 0
def get_secondary_electrons(ei, ji, lossFile):
    Nene = 400
    earr = np.logspace(2, 10, Nene)
    #Cross section WITHOUT relativistic correction
    a0 = 5.29177211E-9 #cm
    memp = 5.44617E-4; A = 0.71; B = 1.63; C = 0.51; D = 1.24;
    xarr = memp*(earr/13.598)
    sigmal = 4.*np.pi*a0**2*C*xarr**D
    sigmah = 4.*np.pi*a0**2*(A*np.log(1+xarr) + B)*(1./xarr)
    sigmap = 1./((1./sigmal) + (1./sigmah))
    ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
    jnew = 10**ji_interp(np.log10(earr))
    dat = np.loadtxt(lossFile, delimiter=',')
    eSpace = dat[1:,0]
    LE = dat[1:,1]*1E-16
    LossFunc = si.interp1d(np.log10(eSpace), np.log10(LE), kind='quadratic', fill_value='extrapolate', bounds_error=False)
    je = []
    for e in earr[:-1]:
        fac = e/10**LossFunc(np.log10(e))
        epspace = np.logspace(np.log10(e + 15.6), 10, Nene)
        jnew = 10**ji_interp(np.log10(epspace))
        xarr = memp*(epspace/13.598)
        sigmal = 4.*np.pi*a0**2*C*xarr**D
        sigmah = 4.*np.pi*a0**2*(A*np.log(1+xarr) + B)*(1./xarr)
        sigmap1 = 1./((1./sigmal) + (1./sigmah))
        J = 8.0
        X = (e - 15.603)/(2.*J)
        sigmap = ((sigmap1/J)*(1./(np.arctan(X))))/(1. + (e/J)**2) #From Glassgold & langer 1973
        integrand = jnew*sigmap
        integral = fac*sint.trapz(integrand, x = epspace)

        if integral < 0:
            print "ERROR! IN SECONDARY ELECTRONS"
            for i in range(len(ei)):
                print ei[i], ji[i]
            sys.exit()
        je.append(integral)
    je.append(je[-1])

    return earr, np.array(je)
def get_electron_cross(E):
    n = 2.4; A1 = 0.74; A2 = 0.87; A3 = -0.6; N = 2
    IH = 13.598
    IH2 = 15.603
    a0 = 5.29177211E-9;
    fac = 4*np.pi*a0**2*N*(IH/IH2)**2
    t = E/IH2
    Ft = (1 - t**(1.-n))/(n - 1) - (2/(1+t))**(n/2.)*((1 - t**(1. - (n/2.)))/(n - 2.))
    Gt = (1./t)*(A1*np.log(t) + A2 + A3/t)
    sigmaE = fac*Ft*Gt
    return sigmaE


B0 = 1E1
mu = 0.6

massres = 64
marr = np.logspace(np.log10(ml), np.log10(mmax), massres)
mfarr = np.logspace(np.log10(ml), np.log10(mmax), massres)
mgrid, mfgrid = np.meshgrid(marr, mfarr)
if 0:
    Sigma = 1.0
    m0 = 3.6E-5*Sigma**(0.75)
    #NCOL = 1.22*(Sigma/(1.4*mh))
    NCOL = 1.22*(Sigma/(2.8*mh))
    RMAT = np.zeros((massres, massres), 'float64')
    NMAT = np.zeros((massres, massres), 'float64')
    TMAT = np.zeros((massres, massres), 'float64')
    UMAT = np.zeros((massres, massres), 'float64')
    VAMAT = np.zeros((massres, massres), 'float64')
    mset = marr[5]
    mfprob = mfarr[9]
    mftest = mfarr[np.where(marr > mset)]
    for jj in range(massres):
            for ii in range(massres):
                if marr[ii] < mfarr[jj]:
                    vals = get_shock_vals(marr[ii], mfarr[jj])
                    VAi = B0/(np.sqrt(4*np.pi*mu*mh*vals['n']))
                    RMAT[jj,ii] = vals['r']
                    NMAT[jj,ii] = vals['n']
                    TMAT[jj,ii] = vals['ts']
                    UMAT[jj,ii] = vals['vff']
                    VAMAT[jj,ii] = VAi

    '''lR = np.ma.masked_where(RMAT == 0, np.log10(RMAT))
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    p1 = ax.pcolor(marr, mfarr, lR, cmap=plt.cm.viridis)
    ax.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax.set_xscale('log'); ax.set_yscale('log')
    c1 = plt.colorbar(p1)
    c1.set_label("R", fontsize=20)

    lN = np.ma.masked_where(NMAT == 0, np.log10(NMAT))
    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)
    p2 = ax2.pcolor(marr, mfarr, lN, cmap=plt.cm.viridis)
    ax2.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax2.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    c2 = plt.colorbar(p2)
    c2.set_label("Density", fontsize=20)

    lT = np.ma.masked_where(TMAT == 0, np.log10(TMAT))
    fig3 = plt.figure(figsize=(8,8))
    ax3 = fig3.add_subplot(111)
    p3 = ax3.pcolor(marr, mfarr, lT, cmap=plt.cm.viridis)
    ax3.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax3.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax3.set_xscale('log'); ax3.set_yscale('log')
    c3 = plt.colorbar(p3)
    c3.set_label("Temp", fontsize=20)

    lV = np.ma.masked_where(UMAT == 0, np.log10(UMAT))
    fig4 = plt.figure(figsize=(8,8))
    ax4 = fig4.add_subplot(111)
    p4 = ax4.pcolor(marr, mfarr, lV, cmap=plt.cm.viridis)
    ax4.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax4.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax4.set_xscale('log'); ax4.set_yscale('log')
    c4 = plt.colorbar(p4)
    c4.set_label("v$_s$", fontsize=20)

    lVA = np.ma.masked_where(VAMAT == 0, np.log10(VAMAT))
    fig5 = plt.figure(figsize=(8,8))
    ax5 = fig5.add_subplot(111)
    p5 = ax5.pcolor(marr, mfarr, lVA, cmap=plt.cm.viridis)
    ax5.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax5.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax5.set_xscale('log'); ax5.set_yscale('log')
    c5 = plt.colorbar(p5)
    c5.set_label("V$_A$", fontsize=20)

    #plt.show()'''

#NCOL = 3.073E23*(Sigma)
CRSPEC = cosmicSpec.CRSpectrum()
CRSPEC.add_range("LE_loss_p.txt")

nclusters = np.logspace(0.5, 3.5, 6)
NSTAT = 10
Nene = 400
earr = np.logspace(2, np.log10(1E10), Nene)
#Cross section WITHOUT relativistic correction
a0 = 5.29177211E-9 #cm
I = 13.598
memp = 5.44617E-4; A = 0.71; B = 1.63; C = 0.51; D = 1.24;
xarr = memp*(earr/13.598)
sigmal = 4.*np.pi*a0**2*C*xarr**D
sigmah = 4.*np.pi*a0**2*(A*np.log(1+xarr) + B)*(1./xarr)
sigmap = 1./((1./sigmal) + (1./sigmah))

'''#Cross section WITH relativistic correction
vrel2 = (1. - (mpc2*1E9/(earr + mpc2*1E9))**2)*cl**2
xarrp = (mh/2.)*(me/(mh*I*ev_to_ergs))*vrel2
sigmal2 = 4.*np.pi*a0**2*C*xarrp**D
sigmah2 = 4.*np.pi*a0**2*(A*np.log(1+xarrp) + B)*(1./xarrp)
sigmap = 1./((1./sigmal2) + (1./sigmah2))'''


if 0:
    Sigma = 1.0
    NCORE = 1.22*(Sigma/(2.8*mh))
    m0 = 3.6E-5*Sigma**(0.75)
    NCOLARR = np.logspace(18, np.log10(0.99*NCORE), 50)
    m_test = 0.5; mf_test = 1.0
    zeta_mat = []
    pcr_mat = []
    flux_mat = []
    for NCOL in NCOLARR:
        print "On NCOL = ", NCOL
        CRSPEC.set_column(NCOL)
        rcore_temp = 0.057*(Sigma)**(-0.5)*(mf_test/30.)**(0.5)*3.086E18
        ns = 1.11E6*(mf_test/30.)**(-0.5)*Sigma**(1.5)
        rui = ((2.*ns*rcore_temp**(1.5))/(NCORE + 2.*ns*rcore_temp - NCOL))**2
        print "\t Corresponding r/Rcore = ", rui/rcore_temp
        ei, ji, emax = get_spec(m_test, mf_test, CRSPEC, atten=True, ruse=rui)
        ee, je = get_secondary_electrons(ei, ji, "LE_loss_p.txt")
        ei = ei[:-1]; ji = ji[:-1]
        pi = pfuncE(ei/1E9)*(5.3442859E-14) #turn p from eV -> eV/c -> cgs
        PCR = (4.*np.pi/3.)*sint.trapz(pi*ji, x = ei) #Double check units here
        ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
        jnew = 10**ji_interp(np.log10(earr))
        integrand = jnew*sigmap
        integrand_e = je*get_electron_cross(ee)
        CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
        CRIRe = 2.*np.pi*sint.trapz(integrand_e, x = ee)
        zeta_mat.append(CRIRi + CRIRe)

    zeta_mat_2 = []
    for NCOL in NCOLARR:
        print "On NCOL = ", NCOL
        CRSPEC.set_column(NCOL)
        rcore_temp = 0.057*(Sigma)**(-0.5)*(mf_test/30.)**(0.5)*3.086E18
        ns = 1.11E6*(mf_test/30.)**(-0.5)*Sigma**(1.5)
        rui = ((2.*ns*rcore_temp**(1.5))/(NCORE + 2.*ns*rcore_temp - NCOL))**2
        print "\t Corresponding r/Rcore = ", rui/rcore_temp
        ei, ji, emax = get_spec(m_test, mf_test, CRSPEC, atten=True, ruse=rui, atrans=1)
        ee, je = get_secondary_electrons(ei, ji, "LE_loss_p.txt")
        ei = ei[:-1]; ji = ji[:-1]
        ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
        jnew = 10**ji_interp(np.log10(earr))
        integrand = jnew*sigmap
        integrand_e = je*get_electron_cross(ee)
        CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
        CRIRe = 2.*np.pi*sint.trapz(integrand_e, x = ee)
        zeta_mat_2.append(CRIRi + CRIRe)

    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(111)
    ax1.loglog(NCOLARR, zeta_mat, 'k-', linewidth=3., label=r"$a = 2$")
    ax1.loglog(NCOLARR, zeta_mat_2, 'r-.', linewidth=3., label=r"$a = 1$")
    ax1.legend(fontsize=14)
    ax1.set_xlabel(r"$N({\rm H_2})$ (cm$^{-2}$)", fontsize=20)
    ax1.set_ylabel(r"$\zeta$ (s$^{-1}$)", fontsize=20)
    ax1.set_xlim(np.min(NCOLARR), np.max(NCOLARR))
    plt.tight_layout()
    #fig1.savefig("zeta_vs_col_transport.pdf")
    #plt.show()

    #sys.exit()
#FOR OMC-2 FIR4 comparison
if 1:
    Sigma = 1.0
    NCORE = 1.22*(Sigma/(2.8*mh))
    #Sigma = 7.9
    m0 = 3.6E-5*Sigma**(0.75)
    NCOLARR_1 = np.logspace(18, np.log10(0.99*NCORE), 50, endpoint=True)
    m_test = 0.5; mf_test = 1.0
    #m_test = 0.3; mf_test = 1.33
    #m_test = 10.; mf_test = 20.;
    zeta_mat = []
    pcr_mat = []
    flux_mat = []
    for NCOL in NCOLARR_1:
        print "On NCOL = ", NCOL
        #CRSPEC.set_column(NCOL)
        rcore_temp = 0.057*(Sigma)**(-0.5)*(mf_test/30.)**(0.5)*3.086E18
        ns = 1.11E6*(mf_test/30.)**(-0.5)*Sigma**(1.5)
        rui = ((2.*ns*rcore_temp**(1.5))/(NCORE + 2.*ns*rcore_temp - NCOL))**2
        print "\t Corresponding r/Rcore = ", rui/rcore_temp
        ei, ji, emax = get_spec(m_test, mf_test, CRSPEC, atten=True, ruse=rui, NCOL=NCOL)
        ee, je = get_secondary_electrons(ei, ji, "LE_loss_p.txt")
        ei = ei[:-1]; ji = ji[:-1]
        pi = pfuncE(ei/1E9)*(5.3442859E-14) #turn p from eV -> eV/c -> cgs
        PCR = (4.*np.pi/3.)*sint.trapz(pi*ji, x = ei) #Double check units here
        ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
        jnew = 10**ji_interp(np.log10(earr))
        integrand = jnew*sigmap
        integrand_e = je*get_electron_cross(ee)
        CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
        CRIRe = 2.*np.pi*sint.trapz(integrand_e, x = ee)
        zeta_mat.append(CRIRi + CRIRe)
        pcr_mat.append(PCR)
        flux_mat.append(sint.trapz(ji, x = ei))
        print "\t\t Zeta = ", CRIRi + CRIRe

    Sigma = 7.9
    NCORE = 1.22*(Sigma/(2.8*mh))
    m0 = 3.6E-5*Sigma**(0.75)
    NCOLARR_2 = np.logspace(18, np.log10(0.99*NCORE), 50)
    m_test = 10.; mf_test = 20.;
    zeta_mat_2 = []
    for NCOL in NCOLARR_2:
        #print "On NCOL = ", NCOL
        #CRSPEC.set_column(NCOL)
        rcore_temp = 0.057*(Sigma)**(-0.5)*(mf_test/30.)**(0.5)*3.086E18
        ns = 1.11E6*(mf_test/30.)**(-0.5)*Sigma**(1.5)
        rui = ((2.*ns*rcore_temp**(1.5))/(NCORE + 2.*ns*rcore_temp - NCOL))**2
        #print "\t Corresponding r/Rcore = ", rui/rcore_temp
        ei, ji, emax = get_spec(m_test, mf_test, CRSPEC, atten=True, ruse=rui, NCOL=NCOL)
        ee, je = get_secondary_electrons(ei, ji, "LE_loss_p.txt")
        ei = ei[:-1]; ji = ji[:-1]
        ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
        jnew = 10**ji_interp(np.log10(earr))
        integrand = jnew*sigmap
        integrand_e = je*get_electron_cross(ee)
        CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
        CRIRe = 2.*np.pi*sint.trapz(integrand_e, x = ee)
        #print "\t\t Zeta: ", np.log10(CRIRi)+np.log10(CRIRe)
        zeta_mat_2.append(CRIRi + CRIRe)

    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(111)
    ax1.fill_between(NCOLARR_2, 1E-14, 1E-12, facecolor='gray', edgecolor='none', alpha=0.3)
    ax1.loglog(NCOLARR_1, zeta_mat, 'k-', linewidth=3., label=r"Fiducial: m = 0.5 M$_{\odot}$, m$_f$ = 1 M$_{\odot}$")
    ax1.loglog(NCOLARR_2, zeta_mat_2, 'r-.', linewidth=3., label=r"OMC-2 FIR-4: m = 10 M$_{\odot}$, m$_f$ = 20 M$_{\odot}$")
    ax1.legend(fontsize=14)
    ax1.set_xlabel(r"$N({\rm H_2})$ (cm$^{-2}$)", fontsize=20)
    ax1.set_ylabel(r"$\zeta$ (s$^{-1}$)", fontsize=20)
    ax1.set_xlim(np.min(NCOLARR_1), 1.25*max(np.max(NCOLARR_2), np.max(NCOLARR_1)))
    plt.tight_layout()
    fig1.savefig("figures/zeta_vs_col.pdf")
    #plt.show()

if 0:
    Sigma = 1.0
    NCORE = 1.22*(Sigma/(2.8*mh))
    #Sigma = 7.9
    m0 = 3.6E-5*Sigma**(0.75)
    NCOLARR_1 = np.logspace(18, np.log10(NCORE), 24)
    xarr = np.logspace(-4, 0, 24)
    m_test = 0.5; mf_test = 1.0
    zeta_mat = np.zeros((24,24), 'float64')
    rcore_1 = 0.057*(Sigma)**(-0.5)*(mf_test/30.)**(0.5)*3.086E18
    for jj, NCOL in enumerate(NCOLARR_1):
        print "On NCOL = ", NCOL
        CRSPEC.set_column(NCOL)
        for ii, xui in enumerate(xarr):
            rui = xui*rcore_1
            #ns = 1.11E6*(mf_test/30.)**(-0.5)*Sigma**(1.5)
            #rui = ((2.*ns*rcore_1**(1.5))/(NCORE + 2.*ns*rcore_1 - NCOL))**2
            ei, ji, emax = get_spec(m_test, mf_test, CRSPEC, atten=True, ruse=rui)
            ee, je = get_secondary_electrons(ei, ji, "LE_loss_p.txt")
            ei = ei[:-1]; ji = ji[:-1]
            ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
            jnew = 10**ji_interp(np.log10(earr))
            integrand = jnew*sigmap
            integrand_e = je*get_electron_cross(ee)
            CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
            CRIRe = 2.*np.pi*sint.trapz(integrand_e, x = ee)
            zeta_mat[jj,ii] = CRIRi + CRIRe

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    pcax = ax.pcolor(xarr, NCOLARR_1, np.log10(zeta_mat), cmap=plt.cm.viridis)
    ccax = ax.contour(xarr, NCOLARR_1, np.log10(zeta_mat), [-14., -13, -12, -11], linewidth=4, colors='white')
    cfax = ax.contourf(xarr, NCOLARR_1, np.log10(zeta_mat), [-14., -12], colors='white', alpha=0.15)
    ax.axvline(x = 0.36)
    for c in ccax.collections:
        c.set_linestyle('solid')
    ax.set_xlabel(r"$R/R_{\rm core}$", fontsize=20); ax.set_ylabel("Column density (cm$^{-2}$)", fontsize=20)
    ax.set_xscale('log'), ax.set_yscale('log')
    plt.clabel(ccax, inline=1, fontsize=14, fmt='%d')
    cb1 = fig.colorbar(pcax)
    cb1.set_label(r"$\log \zeta$ (s$^{-1}$)", fontsize=16)
    #fig.savefig('zeta_col_solar_img.pdf')
    #plt.show()

    zeta_mat_2 = np.zeros((24,24), 'float64')
    rcore = 0.057*(Sigma)**(-0.5)*(mf_test/30.)**(0.5)*3.086E18
    for jj, NCOL in enumerate(NCOLARR_1):
        print "On NCOL = ", NCOL
        CRSPEC.set_column(NCOL)
        ns = 1.11E6*(mf_test/30.)**(-0.5)*Sigma**(1.5)
        rui = ((2.*ns*rcore_1**(1.5))/(NCORE + 2.*ns*rcore_1 - NCOL))**2
        ei, ji, emax = get_spec(m_test, mf_test, CRSPEC, atten=True, ruse=rui)
        ei = ei[:-1]; ji = ji[:-1]
        for ii, xui in enumerate(xarr):
            rui2 = xui*rcore_1
            ji_x = ji*(rui/rui2)**2
            ee, je = get_secondary_electrons(ei, ji_x, "LE_loss_p.txt")
            ji_interp = si.interp1d(np.log10(ei), np.log10(ji_x), bounds_error=False,fill_value=-1E6)
            jnew = 10**ji_interp(np.log10(earr))
            integrand = jnew*sigmap
            integrand_e = je*get_electron_cross(ee)
            CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
            CRIRe = 2.*np.pi*sint.trapz(integrand_e, x = ee)
            zeta_mat_2[jj,ii] = CRIRi + CRIRe
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    pcax = ax.pcolor(xarr, NCOLARR_1, np.log10(zeta_mat_2), cmap=plt.cm.viridis)
    ax.set_xlabel(r"$R/R_{\rm core}$", fontsize=20); ax.set_ylabel("Column density (cm$^{-2}$)", fontsize=20)
    ax.set_xscale('log'), ax.set_yscale('log')
    cb1 = fig.colorbar(pcax)
    cb1.set_label(r"$\log \zeta$ (s$^{-1}$)", fontsize=16)
    #plt.show()
    fig.clf()

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    pcax = ax.pcolor(xarr, NCOLARR_1, np.log10(zeta_mat)-np.log10(zeta_mat_2), cmap=plt.cm.viridis)
    ax.set_xlabel(r"$R/R_{\rm core}$", fontsize=20); ax.set_ylabel("Column density (cm$^{-2}$)", fontsize=20)
    ax.set_xscale('log'), ax.set_yscale('log')
    cb1 = fig.colorbar(pcax)
    cb1.set_label(r"$\log \zeta$ (s$^{-1}$)", fontsize=16)
    #fig.savefig('zeta_col_solar_img.pdf')
    #plt.show()

    sys.exit()

if 1:
    Sigma = 1.0
    m0 = 3.6E-5*Sigma**(0.75)
    #NCOL = 1.22*(Sigma/(1.4*mh))
    NCOL = 1.22*(Sigma/(2.8*mh))
    print "Sigm = %e\tNCOL = %e\n"%(Sigma, NCOL)

    print "Generating 1D spectra plots"
    flux_atten_mat = []
    flux_unatten_mat = []
    ei_atten_mat = []
    ei_unatten_mat = []
    flux_atten_mat_e = []
    flux_unatten_mat_e = []
    ei_atten_mat_e = []
    ei_unatten_mat_e = []
    slopes_a = []
    offsets_a = []
    slopes_u1 = []
    offsets_u1 = []
    slopes_u2 = []
    offsets_u2 = []
    MISET = 0.5
    #MISET = 0.3
    print "Setting mass to: ", MISET
    mfplots = np.logspace(np.log10(MISET), np.log10(mmax), 16)
    magma = plt.get_cmap('magma_r')
    cNorm = colors.Normalize(vmin=np.log10(mfplots[0]), vmax=np.log10(mfplots[-1]))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
    scalarMap._A = []

    XION = np.logspace(np.log10(0.1E9), 9, 12)
    ylu = np.ones(12)*1E-5; yuu = np.ones(12)*1E13
    yla = np.ones(12)*1E-16; yua = np.ones(12)*(1E-8)
    E1 = np.logspace(5,8,12)
    E2 = np.logspace(np.log10(5E8),10.2,12)
    E3 = np.logspace(8.5,9.8,12)
    color_atten = []
    color_unatten = []
    for attenP in [True, False]:
        for mfip in mfplots:
            color = scalarMap.to_rgba(np.log10(mfip))
            if mfip > MISET:
                ei, ji, emax = get_spec(MISET, mfip, CRSPEC, atten=attenP, NCOL=NCOL)
                ee, je = get_secondary_electrons(ei, ji, "LE_loss_p.txt")
                if attenP:
                    #ind1 = np.where(ei >= 1E8)[0][0]; ind2 = -14
                    #z1 = np.polyfit(np.log10(ei[ind1:ind2]), np.log10(ji[ind1:ind2]), 1)
                    #slopes_a.append(z1[0]); offsets_a.append(z1[1])
                    flux_atten_mat.append(ji)
                    ei_atten_mat.append(ei)
                    color_atten.append(color)
                    flux_atten_mat_e.append(je); ei_atten_mat_e.append(ee)
                    #axSA.loglog(ei[:-14], ji[:-14], '-', color=color, zorder=2)
                else:
                    #ind1 = np.where(ei >= 1E5)[0][0]; ind2 = np.where(ei >= 1E8)[0][0];
                    #if emax*1E9 > 8E8:
                #        ind3 = np.where(ei >= 8E8)[0][0]
                    #z1 = np.polyfit(np.log10(ei[ind1:ind2]), np.log10(ji[ind1:ind2]), 1)
                    #if emax*1E9 > 8E8:
                    #    z2 = np.polyfit(np.log10(ei[ind3:]), np.log10(ji[ind3:]), 1)
                    #slopes_u1.append(z1[0]); offsets_u1.append(z1[1])
                    #if emax*1E9 > 8E8:
                #        slopes_u2.append(z2[0]); offsets_u2.append(z2[1])
                    flux_unatten_mat.append(ji)
                    ei_unatten_mat.append(ei)
                    color_unatten.append(color)
                    flux_unatten_mat_e.append(je); ei_unatten_mat_e.append(ee)
                    #axSU.loglog(ei, ji, '-', color=color, zorder=2)
    #avgSA = np.mean(np.array(slopes_a))
    #maxOA = max(offsets_a)
    #avgSU1 = np.mean(np.array(slopes_u1)); maxOU1 = max(offsets_u1)
    #avgSU2 = np.mean(np.array(slopes_u2)); maxOU2 = max(offsets_u2)

    #p3 = np.poly1d(np.array([avgSA, maxOA]))
    #p1 = np.poly1d(np.array([avgSU1, maxOU1]))
    #p2 = np.poly1d(np.array([avgSU2, maxOU2]))

    fig, (axU, axA) = plt.subplots(2, sharex=True,  figsize=(8,16))
    fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0);
    axA.axvline(x=1E3, color='k', linestyle=':', linewidth=2, ymax=0.2); axA.axvline(x=1E6, color='k', linestyle=':', linewidth=2, ymax=0.2); axA.axvline(x=1E9, color='k', linestyle=':', linewidth=2, ymax=0.2);
    axA.annotate("keV", xy=(1E3, 2E-15), xytext=(1E3, 2E-15), rotation=-90, fontsize=14); axA.annotate("MeV", xy=(1E6, 2E-15), xytext=(1E6, 2E-15), rotation=-90, fontsize=14);
    axA.annotate("GeV", xy=(1E9, 2E-15), xytext=(1E9, 2E-15), rotation=-90, fontsize=14)
    axU.axvline(x=1E3, color='k', linestyle=':', linewidth=2, ymax=0.2); axU.axvline(x=1E6, color='k', linestyle=':', linewidth=2, ymax=0.2); axU.axvline(x=1E9, color='k', linestyle=':', linewidth=2, ymax=0.2);
    axU.annotate("keV", xy=(1E3, 2E-3), xytext=(1E3, 2E-3), rotation=-90, fontsize=14); axU.annotate("MeV", xy=(1E6, 2E-3), xytext=(1E6, 2E-3), rotation=-90, fontsize=14);
    axU.annotate("GeV", xy=(1E9, 2E-3), xytext=(1E9, 2E-3), rotation=-90, fontsize=14)

    axEA = axA.twinx()
    axEU = axU.twinx()

    axU.fill_between(XION, ylu, yuu, color='gray', alpha=0.2)
    axA.fill_between(XION, yla, yua, color='gray', alpha=0.2)

    ii = 0
    for mfp in mfplots[1:]:
        if mfp > MISET:
            color = scalarMap.to_rgba(np.log10(mfp))
            axA.loglog(ei_atten_mat[ii], flux_atten_mat[ii], color=color, linewidth=2)
            axU.loglog(ei_unatten_mat[ii], flux_unatten_mat[ii], color=color, linewidth=2)
            axEU.loglog(ei_unatten_mat_e[ii], flux_unatten_mat_e[ii], ':', linewidth=1.5, alpha=0.5, color=color)
            axEA.loglog(ei_atten_mat_e[ii], flux_atten_mat_e[ii], ':', linewidth=1.5, alpha=0.5, color=color)
            ii += 1
    #axA.loglog(E3, 10**p3(np.log10(E3))*4., 'k--', linewidth=2)
    #axU.loglog(E1, 10**p1(np.log10(E1))*4., 'k--', linewidth=2)
    #axU.loglog(E2, 10**p2(np.log10(E2))*1.4, 'k--', linewidth=2)

    axEU.set_ylabel(r"J$_{\rm se}$(E) (particles s$^{-1}$ cm$^{-2}$ eV$^{-1}$)", fontsize=20)
    axEA.set_ylabel(r"J$_{\rm se}$(E) (particles s$^{-1}$ cm$^{-2}$ eV$^{-1}$)", fontsize=20)
    axA.set_xlabel("E (eV)", fontsize=20)
    axU.set_ylabel(r"J$_{\rm p}$(E) (particles s$^{-1}$ cm$^{-2}$ eV$^{-1}$)", fontsize=20)
    axA.set_ylabel(r"J$_{\rm p}$(E) (particles s$^{-1}$ cm$^{-2}$ eV$^{-1}$)", fontsize=20)
    axU.set_ylim(1E-4, 2E14); axA.set_ylim(5E-16, 4E-6)
    axEU.set_ylim(1E-7, 1E14)

    #axA.annotate("j(E) $\propto$ E$^{%.1f}$"%avgSA, xy=(1E9, 0.5E-10), xytext=(1E9, 0.5E-10), rotation=-75, fontsize=16)
    #axU.annotate("j(E) $\propto$ E$^{%.1f}$"%avgSU1, xy=(1E6, 5E7), xytext=(1E6, 5E7), rotation=-55, fontsize=16)
    #axU.annotate("j(E) $\propto$ E$^{%.1f}$"%avgSU2, xy=(1E9, 0.7E2), xytext=(1E9, 0.7E2), rotation=-66, fontsize=16)

    fig.subplots_adjust(top=0.90, hspace=0.00)
    cax2 = fig.add_axes([0.125, 0.90, 0.775, 0.02])
    c1 = fig.colorbar(scalarMap, cax = cax2, orientation='horizontal')
    #c1.ax.tick_params(axis='x',direction='in',labeltop='on')
    c1.ax.xaxis.set_ticks_position('top')
    c1.ax.xaxis.set_label_position('top')
    c1.set_label(r"$\log \, m_f$ (M$_{\odot}$)", fontsize=20)#, labelpad=-75)
    #plt.tight_layout()
    fig.savefig("figures/Half_solar_spectra_vert.pdf", bbox_inches='tight')
    #plt.show()

    #sys.exit()
if 1:
    Sigma = 1.0
    m0 = 3.6E-5*Sigma**(0.75)
    #NCOL = 1.22*(Sigma/(1.4*mh))
    NCOL = 1.22*(Sigma/(2.8*mh))
    print "Sigm = %e\tNCOL = %e\n"%(Sigma, NCOL)
    #CRSPEC.set_column(NCOL)

    print "Generating 2D plots"
    zeta_mat = np.zeros((massres, massres), 'float64')
    FLUX_mat = np.zeros((massres, massres), 'float64')
    PCR_mat = np.zeros((massres, massres), 'float64')
    PRA_mat = np.zeros((massres, massres), 'float64')
    EMAX_mat = np.zeros((massres, massres), 'float64')
    LOSS_type = np.zeros((massres, massres), 'float64')
    LOSS_type.astype(int)
    attenF = True
    file_end = "_atten.pdf"
    if attenF:
        file_end = "_atten.pdf"
    else:
        file_end = "_unatten.pdf"
    for jj in range(massres):
        for ii in range(massres):
            if marr[ii] < mfarr[jj]:
                ei, ji, eparams = get_spec(marr[ii], mfarr[jj], CRSPEC, atten=attenF, NCOL=NCOL, fullOut=True)
                if len(ji) == 0:
                    print "Flooring values for m = %.3f, mf = %.3f"%(marr[ii], mfarr[jj])
                    PCR = 1E-18; CRIRi = 1E-20; CRIRe = 1E-21; FLUXi = 1E-4
                else:
                    ei = np.ma.masked_where(ji == 0., ei).compressed()
                    ji = np.ma.masked_where(ji == 0., ji).compressed()
                    ee, je = get_secondary_electrons(ei, ji, "LE_loss_p.txt")
                    ei = ei[:-1]; ji = ji[:-1]
                    pi = pfuncE(ei/1E9)*(5.3442859E-14) #turn p from eV -> eV/c -> cgs
                    PCR = (4.*np.pi/3.)*sint.trapz(pi*ji, x = ei) #Double check units here
                    ji_interp = si.interp1d(np.log10(ei), np.log10(ji), bounds_error=False,fill_value=-1E6)
                    jnew = 10**ji_interp(np.log10(earr))
                    integrand = jnew*sigmap
                    integrand_e = je*get_electron_cross(ee)
                    CRIRi = 2.*np.pi*sint.trapz(integrand, x = earr)
                    CRIRe = 2.*np.pi*sint.trapz(integrand_e, x = ee)
                    FLUXi = sint.trapz(ji, x = ei)
                    if np.isnan(CRIRi):
                        print "NAN CRIR: "
                        print "m, mf = ", marr[ii], mfarr[jj]
                        print "min/max ji = ", min(ji), max(ji)
                        print "Emax/Losstyle = ", eparams['emax'], eparams['losstype']
                        print "CRIRi/CRIRe = ", CRIRi, CRIRe
                if attenF:
                    #kinP = (1000.)*(2.1)*(mh)*(1E5)**2
                    kinP = 2.0*0.88*G*Sigma**2 #From McKee & Tan 2003, eq 13
                else:
                    sVals = get_shock_vals(marr[ii], mfarr[jj])
                    kinP = (sVals['n']*(mh)*0.6)*sVals['vff']**2
                zeta_mat[jj,ii] = CRIRi+CRIRe
                FLUX_mat[jj,ii] = FLUXi
                PCR_mat[jj,ii] = PCR
                PRA_mat[jj,ii] = PCR/kinP
                EMAX_mat[jj,ii] = eparams['emax']
                LOSS_type[jj,ii] = int(eparams['losstype']+1)


    lZ = np.ma.masked_where(zeta_mat == 0, np.log10(zeta_mat))
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    p1 = ax.pcolor(marr, mfarr, lZ, cmap=plt.cm.viridis)
    con1 = ax.contour(marr, mfarr, lZ, 12, colors='k')
    ax.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax.set_xscale('log'); ax.set_yscale('log')
    c1 = plt.colorbar(p1)
    plt.clabel(con1, fontsize=12, inline=1, inline_spacing=12, fmt="%.2f")
    c1.set_label("$\log \zeta$ (s$^{-1}$)", fontsize=20)
    fig.savefig("figures/zi"+file_end, bbox_inches='tight')

    lF = np.ma.masked_where(FLUX_mat == 0, np.log10(FLUX_mat))
    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)
    p2 = ax2.pcolor(marr, mfarr, lF, cmap=plt.cm.viridis)
    con2 = ax2.contour(marr, mfarr, lF, 12, colors='k')
    ax2.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax2.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    plt.clabel(con2, fontsize=12, inline=1, inline_spacing=12, fmt="%.2f")
    c2 = fig2.colorbar(p2)
    c2.set_label(r"$F_{\rm CR}$ (particles cm$^{-2}$ s$^{-1}$)", fontsize=20)
    fig2.savefig("figures/flux"+file_end, bbox_inches='tight')
    #plt.show()

    lP = np.ma.masked_where(PCR_mat == 0, np.log10(PCR_mat))
    fig3 = plt.figure(figsize=(8,8))
    ax3 = fig3.add_subplot(111)
    p3 = ax3.pcolor(marr, mfarr, lP, cmap=plt.cm.viridis)
    con3 = ax3.contour(marr, mfarr, lP, 12, colors='k')
    ax3.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax3.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax3.set_xscale('log'); ax3.set_yscale('log')
    plt.clabel(con3, fontsize=12, inline=1, inline_spacing=12, fmt="%.2f")
    c3 = fig3.colorbar(p3)
    c3.set_label(r"$\log P_{\rm CR}$ (dyne cm$^{-2}$)", fontsize=20)
    fig3.savefig("figures/pcr"+file_end, bbox_inches='tight')
    #plt.show()

    lR = np.ma.masked_where(PRA_mat == 0, np.log10(PRA_mat))
    fig4 = plt.figure(figsize=(8,8))
    ax4 = fig4.add_subplot(111)
    p4 = ax4.pcolor(marr, mfarr, lR, cmap=plt.cm.viridis)
    con4 = ax4.contour(marr, mfarr, lR, 12, colors='k')
    ax4.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax4.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax4.set_xscale('log'); ax4.set_yscale('log')
    plt.clabel(con4, fontsize=12, inline=1, inline_spacing=12, fmt="%.2f")
    c4 = fig4.colorbar(p4)
    c4.set_label(r"$\log P_{\rm CR}$/$P_{\rm kin}$", fontsize=20)
    fig4.savefig("figures/prat"+file_end, bbox_inches='tight')
    #plt.show()

    from matplotlib.colors import LogNorm
    lE = np.ma.masked_where(EMAX_mat == 0, EMAX_mat)
    fig5 = plt.figure(figsize=(8,8))
    ax5 = fig5.add_subplot(111)
    p5 = ax5.pcolor(marr, mfarr, lE, norm = LogNorm(), cmap=plt.cm.viridis)
    con5 = ax5.contour(marr, mfarr, lE, 12, colors='k')
    ax5.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax5.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax5.set_xscale('log'); ax5.set_yscale('log')
    plt.clabel(con5, fontsize=12, inline=1, inline_spacing=12, fmt="%.2f")
    c5 = fig5.colorbar(p5)
    c5.set_label(r"E$_{\rm max}$ (GeV)", fontsize=20)
    fig5.savefig("figures/emax"+file_end)
    #plt.show()

    #2,3,4
    import matplotlib.colors as colors
    #bounds = np.array([1, 2, 3, 4, 5])
    lT = np.ma.masked_where(LOSS_type == 0, LOSS_type)
    bounds = np.array([1.5, 2.5, 3.5, 4.5])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    fig6 = plt.figure(figsize=(8,8))
    ax6 = fig6.add_subplot(111)
    p6 = ax6.pcolor(marr, mfarr, lT, norm = norm, vmin = 2, vmax = 4, cmap=plt.cm.viridis)
    ax6.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax6.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax6.set_xscale('log'); ax6.set_yscale('log')
    c6 = fig6.colorbar(p6, ticks=[2,3,4])
    c6.ax.set_yticklabels(["Collisions", "Dampening", "Esc. Up"])
    fig6.savefig("figures/losstype"+file_end, bbox_inches='tight')
    #plt.show()


    '''lR = np.ma.masked_where(LOSS_type != 1, np.log10(RMAT))
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    p1 = ax.pcolor(marr, mfarr, lR, cmap=plt.cm.viridis)
    ax.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax.set_xscale('log'); ax.set_yscale('log')
    c1 = plt.colorbar(p1)
    c1.set_label("R", fontsize=20)

    lN = np.ma.masked_where(LOSS_type != 1, np.log10(NMAT))
    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)
    p2 = ax2.pcolor(marr, mfarr, lN, cmap=plt.cm.viridis)
    ax2.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax2.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    c2 = plt.colorbar(p2)
    c2.set_label("Density", fontsize=20)

    lT = np.ma.masked_where(LOSS_type != 1, np.log10(TMAT))
    fig3 = plt.figure(figsize=(8,8))
    ax3 = fig3.add_subplot(111)
    p3 = ax3.pcolor(marr, mfarr, lT, cmap=plt.cm.viridis)
    ax3.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax3.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax3.set_xscale('log'); ax3.set_yscale('log')
    c3 = plt.colorbar(p3)
    c3.set_label("Temp", fontsize=20)

    lV = np.ma.masked_where(LOSS_type != 1, np.log10(UMAT))
    fig4 = plt.figure(figsize=(8,8))
    ax4 = fig4.add_subplot(111)
    p4 = ax4.pcolor(marr, mfarr, lV, cmap=plt.cm.viridis)
    ax4.set_xlabel("m (M$_{\odot}$)", fontsize=20); ax4.set_ylabel("$m_f$ (M$_{\odot}$)", fontsize=20)
    ax4.set_xscale('log'); ax4.set_yscale('log')
    c4 = plt.colorbar(p4)
    c4.set_label("v$_s$", fontsize=20)'''

    sys.exit()
