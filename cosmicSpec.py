# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:54:01 2017

@author: gache_000
"""
#Classes/objects for creating and attenuating a CR spectrum


import numpy as np
import scipy.interpolate as si
import scipy.integrate as sint
#import matplotlib._cntr as cntr

mpc2 = 0.938 #GeV
mpc = 0.938 #GeV/c
mu = 0.6 #0.65 #1.4
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
GeVc_to_cgs = 5.3442859E-14
GeV_to_erg = 0.0016021766

#For all of these funcs, its assumed E = [GeV] and p = [GeV/c]
def pfuncE(E):
    return np.sqrt((E + mpc2)**2 - mpc2**2)
def dpdE(E):
    return (E + mpc2)/np.sqrt(E*(2.*mpc2 + E))
def EfuncP(p):
    return np.sqrt(p**2 + mpc**2) - mpc2
def vfuncE(E):
    return np.sqrt(E/mpc2)*np.sqrt((E/mpc2) + 2)/((E/mpc2) + 1)

ERes = 256

class CRSpectrum():
    EspecRes = ERes
    lgEmax = 10.5
    lgE0 = 2
    lgEmin=-2
    def __init__(self):
        foo = 1
        del foo
    def gen_spectrum(self, pmin, pmax, q, f0):
        #print "Generating spectrum..."
        self.pspace = np.logspace(np.log10(pmin), np.log10(pmax), self.EspecRes)
        self.Espace = EfuncP(self.pspace)
        xspace = self.pspace/(pmin)
        f = f0*xspace**(-q)
        self.NE = 4.*np.pi*f*pfuncE(self.Espace)**2*dpdE(self.Espace)*(GeVc_to_cgs)**3/(GeV_to_erg)
        self.jE0 = (cl*vfuncE(EfuncP(self.pspace)))*self.NE/(4.*np.pi) * 1.602E-12 #1/erg to 1/eV
        self.Espace *= 1E9 #Cast into eV
        if len(np.where(self.jE0 == 0)[0] == len(self.Espace)):
            print "FLUX IS ZERO!"
            print f0,q,pmin
            import sys
            sys.exit()

    def get_nullSpectrum(self, retType = 'interp'):
        if retType == 'interp':
            return si.interp1d(np.log10(self.Espace), np.log10(self.jE0), bounds_error=False,fill_value=-1E6)
        elif retType == 'raw':
            return self.Espace, self.jE0
        else:
            print "Type = interp or raw."
            return None

    def add_range(self, lossFile): #Loss file provides the loss in terms of 1E-16 eV cm^2
        #print "Creating the attenuating functions...."
        dat = np.loadtxt(lossFile, delimiter=',')
        eSpace = dat[1:,0]
        LE = dat[1:,1]*1E-16
        self.LossFunc = si.interp1d(np.log10(eSpace), np.log10(LE), kind='quadratic', fill_value='extrapolate', bounds_error=False)
        REk = []
        eRange = np.logspace(self.lgE0, self.lgEmax, 2056)
        for ei in eRange:
            dE = np.logspace(-1, np.log10(ei), 256)
            integrand = 1./(10**self.LossFunc(np.log10(dE)))
            REi = sint.trapz(integrand, x = dE)
            REk.append(REi)
        REk = np.array(REk)
        self.RangeFunc = si.interp1d(np.log10(eRange), np.log10(REk), kind='linear', fill_value='extrapolate', bounds_error=False)
        self.iRangeFunc = si.interp1d(np.log10(REk), np.log10(eRange), kind='linear', fill_value='extrapolate', bounds_error=False)

    def getRangeFunc(self):
        return self.RangeFunc

    def getiRangeFunc(self):
        return self.iRangeFunc

    '''def add_attenuator(self, lossFile): #Loss file provides the loss in terms of 1E-16 eV cm^2
        #print "Creating the attenuating functions...."
        dat = np.loadtxt(lossFile, delimiter=',')
        eSpace = dat[1:,0]
        LE = dat[1:,1]*1E-16
        self.LossFunc = si.interp1d(np.log10(eSpace), np.log10(LE), kind='quadratic', fill_value='extrapolate', bounds_error=False)
        REk = []
        eRange = np.logspace(self.lgE0, self.lgEmax, 2056)
        for ei in eRange:
            dE = np.logspace(-1, np.log10(ei), 256)
            integrand = 1./(10**self.LossFunc(np.log10(dE)))
            REi = sint.trapz(integrand, x = dE)
            REk.append(REi)
        REk = np.array(REk)
        self.RangeFunc = si.interp1d(np.log10(eRange), np.log10(REk), kind='linear', fill_value='extrapolate', bounds_error=False)
        ek0Grid, ekGrid = np.meshgrid(eRange, eRange)
        NH2 = np.zeros(ek0Grid.shape, 'float64')
        for i in range(len(ekGrid)):
            for j in range(len(ek0Grid)):
                NH2[i,j] = REk[j] - REk[i]

        self.NH2_conts = cntr.Cntr(ek0Grid, ekGrid, np.log10(NH2))'''

    '''def set_column(self, N): #N = column density in cm^-2
        #print "Setting the column density for the attenuation...."
        res = self.NH2_conts.trace(np.log10(N))
        nseg = len(res) // 2
        segs = res[:nseg]
        eK = segs[0][:,1]
        eK0 = segs[0][:,0]
        self.ColumnFunc = si.interp1d(np.log10(eK), np.log10(eK0), kind='linear', fill_value='extrapolate', bounds_error=False)
    '''
    def attenuate_spec_v2(self, N):
        #print "Attenuating spectrum...."
        specN = []
        self.earrN = np.logspace(self.lgE0, np.log10(self.Espace[-1]), self.EspecRes)
        for ee in self.earrN:
            Ri = 10**self.RangeFunc(np.log10(ee))
            dR = N + Ri
            e0 = 10**self.iRangeFunc(np.log10(dR))
            #try:
            #    e0 = 10**self.ColumnFunc(np.log10(ee))
            #except:
            #    specN.append(0.)
            #    continue
            L1 = 10**self.LossFunc(np.log10(ee))
            L2 = 10**self.LossFunc(np.log10(e0))
            fraci = L2/L1
            try:
                ind = np.where(e0 <=self.Espace)[0][0] - 1
                jei = self.jE0[ind]*fraci
                specN.append(jei)
            except IndexError:
                specN.append(0.)
        self.jEN = np.array(specN)
        self.earrN = np.ma.masked_where(self.jEN == 0., self.earrN).compressed()
        self.jEN = np.ma.masked_where(self.jEN == 0., self.jEN).compressed()

    def attenuate_provided_v2(self, espec, jspec, N, ERES = None):
        if ERES == None:
            ERES = self.EspecRes
        specN = []
        earrN = np.logspace(self.lgE0, np.log10(espec[-1]), ERES)
        for ee in earrN:
            Ri = 10**self.RangeFunc(np.log10(ee))
            dR = N + Ri
            e0 = 10**self.iRangeFunc(np.log10(dR))
            L1 = 10**self.LossFunc(np.log10(ee))
            L2 = 10**self.LossFunc(np.log10(e0))
            fraci = L2/L1
            try:
                ind = np.where(e0 <= espec)[0][0] - 1
                jei = jspec[ind]*fraci
                specN.append(jei)
            except IndexError:
                specN.append(0.)
        jEN = np.array(specN)
        earrN = np.ma.masked_where(jEN == 0., earrN).compressed()
        jEN = np.ma.masked_where(jEN == 0., jEN).compressed()
        return earrN, jEN
    '''
    def attenuate_spec(self):
        #print "Attenuating spectrum...."
        specN = []
        self.earrN = np.logspace(self.lgE0, np.log10(self.Espace[-1]), self.EspecRes)
        for ee in self.earrN:
            try:
                e0 = 10**self.ColumnFunc(np.log10(ee))
            except:
                specN.append(0.)
                continue
            L1 = 10**self.LossFunc(np.log10(ee))
            L2 = 10**self.LossFunc(np.log10(e0))
            fraci = L2/L1
            try:
                ind = np.where(e0 <=self.Espace)[0][0] - 1
                jei = self.jE0[ind]*fraci
                specN.append(jei)
            except IndexError:
                specN.append(0.)
        self.jEN = np.array(specN)
        self.earrN = np.ma.masked_where(self.jEN == 0., self.earrN).compressed()
        self.jEN = np.ma.masked_where(self.jEN == 0., self.jEN).compressed()
    '''
    def spatial_dilute_spec(self, r0, r1, which = 'attenuated', p = 1): #p = index of dilution. 1 = diffusion, 2 = geometric
       if which == 'attenuated':
           self.earrG = self.earrN
           self.jEG = self.jEN*(r0/(r0 + r1))**p
       else:
           self.earrG = self.Espace
           self.jEG = self.jE0*(r0/(r0 + r1))**p

    def get_attenSpectrum(self, retType='interp'):
        if retType == 'interp':
            return si.interp1d(np.log10(self.earrN), np.log10(self.jEN), bounds_error=False,fill_value=-1E6)
        elif retType == 'raw':
            return self.earrN, self.jEN
        else:
            print("Type = interp or raw.")
            return None

    def get_dilutedSpectrum(self, retType='interp'):
        if retType == 'interp':
            return si.interp1d(np.log10(self.earrG), np.log10(self.jEG), bounds_error=False,fill_value=-1E6)
        elif retType == 'raw':
            return self.earrG, self.jEG
        else:
            print( "Type = interp or raw.")
            return None

    def get_interp_spec(self, retType='attenuated'):
        if retType[0] == 'a':
            return si.interp1d(np.log10(self.earrN), np.log10(self.jEN), bounds_error=False,fill_value=-1E6)
        elif retType[0] == 'd':
            return si.interp1d(np.log10(self.earrG), np.log10(self.jEG), bounds_error=False,fill_value=-1E6)
        elif retType[0] == 'n':
            return si.interp1d(np.log10(self.Espace), np.log10(self.jE0), bounds_error=False,fill_value=-1E6)
        else:
            print( "Need to specify type: ")
            print( "types = attenuated, diluted or null")
            return None
    def clearMem(self):
        try:
            del self.Espace; del self.jE0
        except AttributeError:
            pass
        try:
            del self.earrN; del self.jEN
        except AttributeError:
            pass
        try:
            del self.earrG; del self.jEG
        except AttributeError:
            pass
