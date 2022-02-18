# ProtostarCRs
Python scripts to compute the CR acceleration by protostars and their transport through the cloud. This consists of two major classes:

### ProtostarCR
This class computes the resulting CR acceleration "parameters" to determine the DSA power-law for a given shock. To it, you pass a loss function, in units 1E-16 eV cm^2 (following the Padovani+2009 plots) for your acceleration region. To do this, you use the function
`def addLossFunction(self, lfile):`
where lfile is in the formation of (Energy (eV), L(E)). This reads in the file and computes a scipy interpolated out of it. The primary funciton is 
`def calcSpecParams(self, U, T, nh, B, Rsh, Rp = 0, eta=1.0E-5):`
This performs the calculation of the minimum and maximum accelerated momentum for a proton for a shock near a protostar. Here, U is in km/s, T is in K, nh in cm^(-3), B in microGauss and Rsh in AU. This computes the maximum momentum through a number of constraint equations (see Padovani+2016, Gaches & Offner 2018). Returns the following dictionary:
```
{"pmin":pinj_MeVc*1E-3, 
"pmax":p_max, 
"q":q, 
"f0":f0, 
'emax':E_max, 
'emin':EfuncP(pinj_MeVc*1E-3), 
"pcr":PCR, 
'losstype':emaxtype}
```
Where pmin/emin and pmax/emax are the minimum(maximum) momenta/energy in units of GeV/c (GeV). `q` is the cosmic-ray power-law spectrum, `f0` is the normalization factor and `losstype` is an encoded number for what is constraining the cosmic ray acceleration. The encoding is 0:Shock age, 1:Energy losses, 2:Wave dampening, 3:Upstream escape, 4: Downstream escape.

The results of this function can be fed into the other class, described below.
### CRSpectrum
This function does a number of things, from initializing an internal power-law spectrum, to attenuating and diluting both the internal and a provided spectrum. Here are the primary functions
`def gen_spectrum(self, pmin, pmax, q, f0):`
This produces a power-law flux and number density spectrum following the distribution, f(p) = f_0 * p^(q) from pmin to pmax. The resulting flux output is in particles/s/cm^2/eV and the energy array in eV. It stores these internally.

`def add_range(self, lossFile):`
This function reads in a loss function (see above for that format) and stores an internal interpolated for R(E) and E(R) (inverse Range function).

`def attenuate_spec_v2(self, N):` and `def attenuate_provided_v2(self, espec, jspec, N, ERES = None):`
These attenuated the spectrum under the free-streaming approximation. The first uses the internally stored spectrum and the later uses provided arrays.

`def spatial_dilute_spec(self, r0, r1, which = 'attenuated', p = 1): `
Scales the internal spectrum by `j(E) ~ (r0/(r0+r1))^p`. Can decide to dilute the attenuated or the un-attenuated spectrum.

An example on how to use these functions is in the example_plots_gaches2018b.ipynb script. This notebook reproduces some of the plots from the Gaches & Offner (2018) paper.
