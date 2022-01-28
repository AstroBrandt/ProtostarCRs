# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 12:22:04 2017

@author: gache_000
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:31:56 2017

@author: gache_000
"""

import numpy as np
import scipy.optimize as sopt
import scipy.integrate as sint
import scipy.interpolate as si

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

eArr = np.logspace(-2, 11, 256)
dat = np.loadtxt("LE_loss_p.txt", delimiter=',')
eSpace = dat[1:,0]
LE = (dat[1:,1]*1E-16)/1E9 #to get in GeV/cm^2
k = 1
LE1 = si.interp1d(np.log10(eSpace), np.log10(LE), kind='quadratic', fill_value='extrapolate', bounds_error=False)

def calcSpecParams(U, T, nh, B, Rsh, Rp = 0, eta=1.0E-5):
    betath = 2E-3*(T/1E4)**(0.5)
    x = 0.9
    ku = 1; kd = ku;
    cs = np.sqrt((gamma_ad*kb*T)/(mu*mh))/1E5 #put into km/s
    Ms = U/cs
    r = ((gamma_ad + 1) * Ms**2)/((gamma_ad - 1)*Ms**2 + 2)
    q = (3*r)/(r-1.)
    alpha = -1 #parallel
    #eta = 1.0E-5
    def gamma(beta):
        return 1./np.sqrt(1. - beta**2)
    def Efncbeta_p(beta):
        return gamma(beta)*mpc2
    def eta_func(lamb):
        return eta - (4./(3.*np.sqrt(np.pi)))*(r - 1)*lamb**3*np.exp(-lamb**2)
    def LEcp(beta): #in 1E-25 GeV cm^2
          return LE1((gamma(beta)-1.)*mpc2)#0.1*((x*beta)/(betath**3 + beta**3))
    def Elossfunc(beta):
        Up = U/1E2 #U/1E2
        nhp = nh/1E6 #n/1E6
        Bp = B/(10*1E-6) # B/10 microGauss
        fac = 3.4*(ku**alpha*(r-1.))/(r*(1. + r*(kd/ku)**alpha)) * Up**2 * (1./nhp) * Bp
        return beta*LEcp(beta) - fac
    def damp_func(beta):
        TH = (B/(1E-5))**4 + 1.4E12*gamma(beta)**2*beta**2*x**2*(T/1E4)**(0.8)*(nh/1E6)**3
        fac = 8.8E-5*TH*(1-x)**(-1)*(U/1E2)**3*(T/1E4)**(-0.4)*(nh/1E6)**(-0.5)*(B/1E-5)**(-4)*(Ptwidle_CR/1E-2)
        return gamma(beta)*beta**2 - fac
    def Eescu_func(beta):
        if Rp != 0:
            MM = min((0.1*Rsh/1E2), Rp/1E2)
        else:
            MM = (0.1*(Rsh/1E2))
        fac = 4.8*MM*ku**alpha*(U/1E2)*(B/1E-5)
        return gamma(beta)*beta**2-fac
    def Eescd_func(beta):
        Cc = 1.
        fac = 2.1*Cc*(((ku*kd)**alpha*(r-1.))/(r*(1 + r*(kd/ku)**alpha)))*(U/1E2)**2*(B/1E-5)**2*(Rp/1E2)**2
        return gamma(beta)*beta**2*(gamma(beta)-1) - fac
    def age_func():
        gamma_age = 1 + 3.2E2*((ku**alpha*(r-1))/(r*(1 + r*(kd/ku)**alpha)))*(U/1E2)**2*(B/(10*1E-6))*(1E5/1E3)
        return gamma_age
    def pfuncE(E):
        return np.sqrt((E + mpc2)**2 - mpc2**2)
    def dpdE(E):
        return (E + mpc2)/np.sqrt(E*(2.*mpc2 + E))
    def EfuncP(p):
        return np.sqrt(p**2 + mpc**2) - mpc2
    def vfuncE(E):
        return np.sqrt(E/mpc2)*np.sqrt((E/mpc2) + 2)/((E/mpc2) + 1)

    E_age = (age_func()-1)*mpc2
    failCount = 0
    try:
        lamb = sopt.bisect(eta_func, 3.0, 5.0)
    except ValueError:
        lamb = 3.5
        print 'USING DEFAULT LAMB = ', lamb
    pinj = lamb*mh*(U*1E5/r)*np.sqrt(gamma_ad*(r-1))
    pinj_MeVc = 1.8711574E16*pinj
    a = 3./(r - 1.); b1 = (2*r-5.)/(r-1.); b2 = (r-4.)/(r-1.)
    ptwidle = pinj/(mh*cl)
    Ptwidle_CR = eta*r*(cl/(U*1E5))**2*ptwidle**a * ((1 - ptwidle**b1)/(2*r-5.))
    #print "Eta:", eta, r, cl, ptwidle
    ku = 4E-2*(U/1E2)**(-1)*(nh/1E6)**(-0.5)*(B/(10*1E-6))*(Ptwidle_CR/1E-2)**(-1)
    kd = ku
    #for _|_ shock
    if Rp != 0:
        kd = (1./r)*ku; alpha=1;
    try:
        beta_test = sopt.bisect(Elossfunc, betath, 1.0)
    except ValueError:
        print "bisect failed - loss"
        failCount += 1
        beta_test = 0.9999999999999999
    gamma_test = gamma(beta_test)
    E_loss = mpc2*(gamma_test-1)

    try:
        beta_test = sopt.bisect(damp_func, betath, 1.0)
        gamma_test = gamma(beta_test)
        E_damp = mpc2*(gamma_test-1)
    except ValueError:
        print "bisect failed - damp"
        failCount += 1
        E_damp = 1E50

    try:
        beta_test = sopt.bisect(Eescu_func, betath, 1.0)
        gamma_test = gamma(beta_test)
        E_escu = mpc2*(gamma_test-1)
    except ValueError:
        print "bisect failed - esc_u"
        failCount += 1
        E_escu = 1E50

    if Rp != 0.:
        beta_test = sopt.bisect(Eescd_func, betath, 1.0)
        gamma_test = gamma(beta_test)
        E_escd = mpc2*(gamma_test-1)
    else:
        failCount += 1
        E_escd = 1E50

    E_max = min([E_age, E_loss, E_damp, E_escu, E_escd])
    emaxtype = [E_age, E_loss, E_damp, E_escu, E_escd].index(E_max)
    p_max = pfuncE(E_max)
    p_max_twidle = (p_max*5.3442859E-14)/(mh*cl)

    ptwidlespace = np.logspace(np.log10(ptwidle), np.log10(p_max_twidle), 100)
    F2 = sint.trapz(ptwidlespace**(4.-q)/(np.sqrt(ptwidlespace**2 + 1)), x = ptwidlespace)
    f0 = (3./(4.*np.pi))*F2**(-1)*(mh*cl)**(-4)*cl**(-1)*ptwidle**(-q)*Ptwidle_CR*nh*mh*(U*1E5)**2
    PCR = (4.*np.pi/3.)*F2*(mh*cl)**4*cl*f0*(pinj/(mh*cl))**(q)
    return {"pmin":pinj_MeVc*1E-3, "pmax":p_max, "q":q, "f0":f0, 'emax':E_max, 'emin':EfuncP(pinj_MeVc*1E-3), "pcr":PCR, 'losstype':emaxtype}

def calcSpecParamsLE(U, T, nh, B, Rsh, Rp = 0):
    betath = 2E-3*(T/1E4)**(0.5)
    x = 0.9
    ku = 1; kd = ku;
    cs = np.sqrt((gamma_ad*kb*T)/(mu*mh))/1E5 #put into km/s
    Ms = U/cs
    r = ((gamma_ad + 1) * Ms**2)/((gamma_ad - 1)*Ms**2 + 2)
    q = (3*r)/(r-1.)
    alpha = -1 #parallel
    eta = 1.0E-5
    def gamma(beta):
        return 1./np.sqrt(1. - beta**2)
    def Efncbeta_p(beta):
        return gamma(beta)*mpc2
    def eta_func(lamb):
        return eta - (4./(3.*np.sqrt(np.pi)))*(r - 1)*lamb**3*np.exp(-lamb**2)
    def LEcp(beta): #in 1E-25 GeV cm^2
          return LE1((gamma(beta)-1.)*mpc2)#0.1*((x*beta)/(betath**3 + beta**3))
    def Elossfunc(beta):
        Up = U/1E2 #U/1E2
        nhp = nh/1E6 #n/1E6
        Bp = B/(10*1E-6) # B/10 microGauss
        fac = 3.4*(ku**alpha*(r-1.))/(r*(1. + r*(kd/ku)**alpha)) * Up**2 * (1./nhp) * Bp
        return beta*LEcp(beta) - fac
    def pfuncE(E):
        return np.sqrt((E + mpc2)**2 - mpc2**2)
    def dpdE(E):
        return (E + mpc2)/np.sqrt(E*(2.*mpc2 + E))
    def EfuncP(p):
        return np.sqrt(p**2 + mpc**2) - mpc2
    def vfuncE(E):
        return np.sqrt(E/mpc2)*np.sqrt((E/mpc2) + 2)/((E/mpc2) + 1)


    lamb = sopt.bisect(eta_func, 3.5, 4.5)
    pinj = lamb*mh*(U*1E5/r)*np.sqrt(gamma_ad*(r-1))
    pinj_MeVc = 1.8711574E16*pinj
    a = 3./(r - 1.); b1 = (2*r-5.)/(r-1.); b2 = (r-4.)/(r-1.)
    ptwidle = pinj/(mh*cl)
    Ptwidle_CR = eta*r*(cl/(U*1E5))**2*ptwidle**a * ((1 - ptwidle**b1)/(2*r-5.))
    #print "Eta:", eta, r, cl, ptwidle
    ku = 4E-2*(U/1E2)**(-1)*(nh/1E6)**(-0.5)*(B/(10*1E-6))*(Ptwidle_CR/1E-2)**(-1)
    kd = ku
    #for _|_ shock
    if Rp != 0:
        kd = (1./r)*ku; alpha=1;
    try:
        beta_test = sopt.bisect(Elossfunc, betath, 1.0)
    except ValueError:
        print "Using fminbound"
        beta_test = sopt.fminbound(Elossfunc, betath, 1.0)
    gamma_test = gamma(beta_test)
    E_loss = min(mpc2*(gamma_test-1), 50.)

    E_max = E_loss
    p_max = pfuncE(E_max)
    p_max_twidle = (p_max*5.3442859E-14)/(mh*cl)

    ptwidlespace = np.logspace(np.log10(ptwidle), np.log10(p_max_twidle), 100)
    F2 = sint.trapz(ptwidlespace**(4.-q)/(np.sqrt(ptwidlespace**2 + 1)), x = ptwidlespace)
    f0 = (3./(4.*np.pi))*F2**(-1)*(mh*cl)**(-4)*cl**(-1)*ptwidle**(-q)*Ptwidle_CR*nh*mh*(U*1E5)**2
    PCR = (4.*np.pi/3.)*F2*(mh*cl)**4*cl*f0*(pinj/(mh*cl))**(q)
    return {"pmin":pinj_MeVc*1E-3, "pmax":p_max, "q":q, "f0":f0, 'emax':E_max, 'emin':EfuncP(pinj_MeVc*1E-3), "pcr":PCR}
