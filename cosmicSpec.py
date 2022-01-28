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

#Here is the class to define a spectrum, attenuated it,
#or spatially dilute the spectrum and return the raw spectrum
#or an interpolator for the spectrum
class CRSpectrum():
    EspecRes = 256
    lgEmax = 10.5
    lgE0 = 2
    lgEmin=-2
    def __init__(self):
        foo = 1
        del foo

    #For all of these funcs, its assumed E = [GeV] and p = [GeV/c]
    #Further, these functions all assume a proton mass
    def pfuncE(self, E):
        mpc2 = 0.938 #GeV
        return np.sqrt((E + mpc2)**2 - mpc2**2)
    def dpdE(self, E):
        mpc2 = 0.938 #GeV
        return (E + mpc2)/np.sqrt(E*(2.*mpc2 + E))
    def EfuncP(self, p):
        mpc2 = 0.938 #GeV
        return np.sqrt(p**2 + mpc**2) - mpc2
    def vfuncE(self, E):
        mpc2 = 0.938 #GeV
        return np.sqrt(E/mpc2)*np.sqrt((E/mpc2) + 2)/((E/mpc2) + 1)

    #This spectrum allows one to define a power-law spectrum, defined in momentum space
    #With a power-law index of q, normalization factor of f0 between pmin and pmax, in GeV/c
    #The spectrum it generates is in particles/s/cm^2/eV
    def gen_spectrum(self, pmin, pmax, q, f0):
        self.pspace = np.logspace(np.log10(pmin), np.log10(pmax), self.EspecRes)
        #Convert p array to energy array, in GeV
        self.Espace = self.EfuncP(self.pspace)
        xspace = self.pspace/(pmin)
        #Define the distribution function
        f = f0*xspace**(-q)
        #This returns the number spectrum in cgs units
        self.NE = 4.*np.pi*f*self.pfuncE(self.Espace)**2*self.dpdE(self.Espace)*(GeVc_to_cgs)**3/(GeV_to_erg)
        #Defines the spectrum first in cgs, e.g., particles/s/cm^2/erg and converts to per eV
        self.jE0 = (cl*self.vfuncE(self.EfuncP(self.pspace)))*self.NE/(4.*np.pi) * 1.602E-12 #1/erg to 1/eV
        #Make the energy array from GeV -> eV
        self.Espace *= 1E9 #Cast into eV
        #Just double check that there are no zeros in the array.
        if len(np.where(self.jE0 == 0)[0] == len(self.Espace)):
            print("FLUX IS ZERO!")
            print(f0,q,pmin)
            import sys
            sys.exit()

    #Returns the "null" spectrum, e.g. the power law. Can be returned as the array, or
    #as an scipy interpolate object
    def get_nullSpectrum(self, retType = 'interp'):
        if retType == 'interp':
            return si.interp1d(np.log10(self.Espace), np.log10(self.jE0), bounds_error=False,fill_value=-1E6)
        elif retType == 'raw':
            return self.Espace, self.jE0
        else:
            print("Type = interp or raw.")
            return None

    #Add a range function from a provided loss function. This is done to perform an updating more optimzied
    #Computation than presented in the Padovani+2009 paper. The original method was to solve the equation:
    #N ~ n * (R(E_0) - R(E))
    #For a given N using contours in E_0 - E space. This is very slow. Instead, can calculate the energy, E,
    #for a given E_0 and N using an inverse numerical solution of R(E)
    #Since R(E) == 1/n * \int_0^E dE/L(E), the #n# variables cancel out
    #and you get a solution only in terms on the function \int_0^E dE/L(E)
    #Therefore, the "range" here is actually n(H2)*R (e.g. the stopping column)
    def add_range(self, lossFile): #Loss file provides the loss in terms of 1E-16 eV cm^2
        dat = np.loadtxt(lossFile, delimiter=',')
        eSpace = dat[1:,0]
        LE = dat[1:,1]*1E-16
        #Read in the loss function, given in units from Fig. 7 in Padovani+2009
        self.LossFunc = si.interp1d(np.log10(eSpace), np.log10(LE), kind='quadratic', fill_value='extrapolate', bounds_error=False)
        REk = []
        eRange = np.logspace(self.lgE0, self.lgEmax, 2056)
        #Compute the function \int_0^E dE/L(E)
        for ei in eRange:
            dE = np.logspace(-1, np.log10(ei), 256)
            integrand = 1./(10**self.LossFunc(np.log10(dE)))
            REi = sint.trapz(integrand, x = dE)
            REk.append(REi)
        REk = np.array(REk)
        #Produce a numerical interpolator and inverse interpolation function
        self.RangeFunc = si.interp1d(np.log10(eRange), np.log10(REk), kind='linear', fill_value='extrapolate', bounds_error=False)
        self.iRangeFunc = si.interp1d(np.log10(REk), np.log10(eRange), kind='linear', fill_value='extrapolate', bounds_error=False)

    #Return these functions, if you wish!
    def getRangeFunc(self):
        return self.RangeFunc

    def getiRangeFunc(self):
        return self.iRangeFunc

    #Attenuates the function assuming the free-streaming approximation
    #Here, j(E,N) = j(E, 0)*L(E_0)/L(E) where E is the "new" energy of
    #of a CR with initial energy, E_0, propagating through column N
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

    #Same as above, BUT, this time it allows you to define your own
    #spectrum and return an attenuated version back
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

    #Spatially dilutes the internal spectrum via ~(r_0/r)**p
    def spatial_dilute_spec(self, r0, r1, which = 'attenuated', p = 1): #p = index of dilution. 1 = diffusion, 2 = geometric
       if which == 'attenuated':
           self.earrG = self.earrN
           self.jEG = self.jEN*(r0/(r0 + r1))**p
       else:
           self.earrG = self.Espace
           self.jEG = self.jE0*(r0/(r0 + r1))**p

    #These spectra return all the above internal spectrum
    #Can return either the raw arrays or a scipy interpolation
    #function. The default is to return an interpolation function
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
