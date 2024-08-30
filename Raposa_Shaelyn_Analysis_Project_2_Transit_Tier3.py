#!/usr/bin/env python
# coding: utf-8

# # Shaelyn Raposa
# # Exoplanets - Analysis Project 2 (Transit Spectrum)

# In[1]:


import numpy as np
import emcee
import corner
from astropy.io import ascii
import matplotlib.colors as colors
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# assume: 
#  - a stellar radius of 0.78 Rsun
#  - a planetary 10-bar radius of 1.16 Rjup
#  - an H2 volume mixing ratio of 0.85
#  - an He volume mixing ratio of 0.15.


# ### Define a pressure grid that extends from the 10 bar radius to the top of the atmosphere.

# In[3]:


# use np.logspace, read in pressures in Pascals
press_grid = np.logspace(-4.0, 6.0, 100, endpoint=True)


# ### Read in water vapor and methane opacities (in m^2/molecule). Note these are given as *per molecule* of the absorbing species.

# In[4]:


dataset = ascii.read('opacities_transit.dat', data_start=2, delimiter = '|')
# wavelength, in um
wl = np.array(dataset['col1'][:])
# opacity, H2O
opac_H2O = np.array(dataset['col2'][:])
# opacity, CH4
opac_CH4 = np.array(dataset['col3'][:])


# ### Read in transit spectra.

# ### Tier 1 (trying this first)

# In[5]:


dataset_tier1 = ascii.read('transit_spectrum_ast510_f2020_tier1.dat', data_start=1, delimiter = '|')
# wavelength, in um, is the same as opacities file
# transit depth
trans_dep_t1 = np.array(dataset_tier1['col3'][:])
# uncertainty
trans_dep_t1_err = np.array(dataset_tier1['col4'][:])


# ### Tier 2/3 

# In[6]:


dataset2 = ascii.read('transit_spectrum_ast510_f2020_tier2_tier3.dat', data_start=1, delimiter = '|')
# wavelength, in um, is the same as opacities file
# transit depth
trans_dep_t2 = np.array(dataset2['col3'][:])
# uncertainty
trans_dep_t2_err = np.array(dataset2['col4'][:])


# ### Plot up the raw data.

# In[7]:


plt.plot(wl, trans_dep_t2, color = 'k')
plt.xlabel('Wavelength (microns)')
plt.ylabel('Transit Depth')
plt.title('Raw Data')
#plt.savefig('/Grad School/Exoplanets Fall 2020/Analysis Project 2/AP2_Raw_Data_Shae_Raposa.png', bbox_inches="tight",dpi=300)


# ### transit_depth.py

# In[8]:


# transit depth routine
#
# inputs:
#       Rp    -    10 bar planetary radius (in Jupiter radii)
#       Rs    -    stellar radius (in Solar radii)
#       z     -    grid of altitudes (in meters) with z[0]
#                  being the altitude of the top-most atmospheric
#                  level (i.e., lowest pressure gridpoint) and
#                  z[len(z)-1] being 0 meters and corresponding
#                  to the 10 bar planetary radius.
#       dtau  -    layer vertical optical depth matrix.  here,
#                  dtau[j-1,k] is the vertical optical depth across
#                  the jth layer at wavelength gridpoint k.  thus,
#                  dtau[0,0] is the optical depth at the first
#                  wavelength gridpoint across the atmospheric level
#                  that extends from z[0] to z[1].
#
def transit_depth(Rp,Rs,z,dtau):
    Nlev = len(z)        # number of levels
    Nlam = dtau.shape[1] # number of wavelengths
    Rj   = 6.9911e7      # jupiter radius (m)
    Rsun = 6.9550e8      # solar radius (m)

    # geometric path distribution
    Pb = np.zeros((Nlev,Nlev))
    for i in range(0,len(z)-1):
        b = Rp*Rj + (z[i]+z[i+1])/2.

        for j in range(0,z.shape[0]-1):
            r = Rp*Rj + (z[j]+z[j+1])/2.

            if r > b:
                Pb[i,j] = 2*r/(r**2 - b**2)**0.5

    # transit depth has value for each wavelength
    td   = np.zeros(Nlam)

    # loop over wavelength
    for k in range(0,Nlam):
        # transmission
        t     = np.zeros(Nlev-1)
        t[:]  = 1.

        for i in range(0,Nlev-2):
            tau = 0.

            for j in range(0,Nlev-2):
                tau = tau + Pb[i,j]*dtau[j,k]

            t[i] = np.exp(-tau)

        # integral over impact parameters
        for i in range(0,Nlev-2):
            td[k] = td[k] + 2*(1-t[i])*(Rp*Rj + 0.5*(z[i+1]+z[i]))*(z[i]-z[i+1])

        td[k] = td[k] + (Rp*Rj)**2
        td[k] = td[k]/(Rs*Rsun)**2

    return td


# ### Routines for Rayleigh Scattering cross section (m**2/molecules)

# In[9]:


# rayleigh scattering cross section (m**2/molecule)
#
# inputs:
#     lam    -    wavelength (in micrometers)
#     fH2    -    H2 volume mixing ratio
#     fHe    -    He volume mixing ratio

def rayleigh(lam,fH2,fHe):
    # convert wavelength to angstroms
    lam0 = lam*1.e4
    # compute rayleigh scattering cross section (note: lam0 in angstrom)
    sigmaH2 = ((8.140e-13)*(lam0**(-4.))*(1+(1.572e6)*(lam0**(-2.))+(1.981e12)*(lam0**(-4.))))*1e-4
    sigmaHe = ((5.484e-14)*(lam0**(-4.))*(1+(2.440e5)*(lam0**(-2.))))*1e-4
    return fH2*sigmaH2 + fHe*sigmaHe

# grey H2 cia cross section (m**2/molecule)
#
# inputs:
#
#     n     - total atmospheric number density (molecules/m**3)

def cia_H2(n):
    return n*(10**-6.)*(1/2.6867774e25)**2


# ## Forward model (ultimately outputs transit depths, using transit_depth function above).

# ### Test tier 1 case first - update: tested and now trying tier 2

# In[10]:


def ForwardModel(x, wl, trans_dep_t2, trans_dep_t2_err):  
    
    # define radius of planet (planetary 10-bar radius of 1.16 Rjup). convert to m 
    Rp = 1.16*(69.911e6)
    # define stellar radius (0.78 Rsun). convert to m
    Rs = 0.78*(696.34e6)
    
    # emcee fits for scale height, temperature, and gas concentration 
    H, T, lf_H2O, lf_CH4 = x
    f_H2O = 10**(lf_H2O)
    f_CH4 = 10**(lf_CH4)
    # Now, using pressure grid we can compute an altitude grid
    # Use hydrostatic equation, p(z)=p0*e^(-z/H)
    # Solving for z we get: z = H*ln(p0/pi), where p0 is our 10 bar pressure (1e6 Pascals) and pi is our pressure grid
    p0 = 1.0e6
    z = (H*np.log(p0/press_grid))*10.0**3.0
    
    # Now, we have pressure grid and corresponding altitude grid (press_grid, z)
    # can do height at top of layer - height at bottom of layer 
    del_z = np.zeros((len(z)))
    for i in range(0,len(z)):
        delta_z = z[i]-z[i-1]
        del_z[i] = np.abs(delta_z)
    # delta z is the same so I'll just set it equal to one value    
    del_z_val = del_z[2]
    
    # Now, have press_grid, z, and delz
    # we now want to calculate a number density profile using ideal gas law (p = n*k_b*T which gives us n=pi/k_b*T)
    # boltzmann constant, units (m^2*kg*s^-2*K^-1)
    k_b = 1.38064852e-23
    n_i = press_grid/(k_b*T)
    # compute average number density
    ni_avg = np.zeros((len(n_i)))
    for i in range(0,len(n_i)):
        ni_mid = 0.5*(n_i[i]+n_i[i-1])
        ni_avg[i] = ni_mid
    ni_avg = ni_avg[1:]
    
    # Now we have everything we need to compute optical depth across layer (pi, zi, deltaz, and ni)
    delta_tau = np.zeros((99,25))
    delta_tau_CIA = np.zeros((25))
    for i in range(0,len(ni_avg)):
        
        # First, do optical depth of water vapor (function of wavelength)
        delta_tau_H2O = ni_avg[i]*f_H2O*del_z_val*opac_H2O
        # now, add CH4 for tier 2
        delta_tau_CH4 = ni_avg[i]*f_CH4*del_z_val*opac_CH4
        # Now, similar calculations for optical depth across layer for Rayleigh Scattering 
        # going to be: fraction of atm that is H2 (85%, jupiter-like) * number density * layer thickness * Rayleigh scattering
        # cross-section for H2 (function of wavelength, use routine from notes doc) + same idea for helium contribution 
        # factoring common terms, and using rayleigh function we get:
        rs_cross = (rayleigh(wl,.85,.15))
        delta_tau_Ray = ni_avg[i]*del_z_val*rs_cross
        # CIA term (4th opacity source, collision induced absorption)
        #delta_tau_CIA[:] = 0.85*ni_avg[i]*del_z_val*cia_H2(ni_avg[i])
        delta_tau_CIA[:] = 0.85*ni_avg[i]*del_z_val*cia_H2(ni_avg[i])
        # add terms together 
        delta_tau_i = delta_tau_H2O + delta_tau_Ray + delta_tau_CIA + delta_tau_CH4
        delta_tau[i,:] = delta_tau_i

    # Now, we have everything we need to calculate transit depth using transit_depth function 
    #print(delta_tau[:,20])
    trans_depth = transit_depth(1.16,0.78,z,delta_tau)
    return trans_depth


# ### Define a log-likelihood function, lnlike, that takes as inputs: the free parameters (H, T, f_H2O, f_CH4), wl, transit depth, transit depth error.

# In[11]:


def lnlike(x, wl, trans_dep_t2, trans_dep_t2_err):
    # free parameters: scale height, temp, f_H2O
    H, T, f_H2O, f_CH4 = x
   
    # call forward model
    trans_depth_mod = ForwardModel(x, wl, trans_dep_t2, trans_dep_t2_err)
    
    return -0.5*(np.sum((trans_dep_t2-trans_depth_mod)**2.0/trans_dep_t2_err**2.0))


# ### Define a log prior function, that takes x as an input. For priors, limit the temperature to be between absolute zero and the temperature of a star (~3000 K), the scale height to be non-negative, and that the log-mixing ratio be less than 0 for water vapor(suggestion from class)

# In[12]:


def lnprior(x):
    H, T, f_H2O, f_CH4 = x
    
    if (T<0 or T>3000 or H<0 or f_H2O>0 or f_H2O<-10 or f_CH4>0 or f_CH4<-8):
        # unphysical
        return -np.inf
    else:
        return 0


# ### Define a log-probability function, lnprob, that implements Bayes theorem. 

# In[13]:


def lnprob(x, wl, trans_dep_t2, trans_dep_t2_err):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(x, wl, trans_dep_t2, trans_dep_t2_err)


# ### Define number of dimensions for fit, and number of walkers.

# In[14]:


# number of dimensions
ndim = 4
# number of walkers
nwalkers = 100


# ### Create backup/ save file.

# In[15]:


# initial position of walkers
x0 = [200, 1500, -4, -4]
pos = [x0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# create backup/ save file
fn = 'emcee.h5'
backend = emcee.backends.HDFBackend(fn)
backend.reset(nwalkers,ndim)


# ### Define the number of steps in the Markov chains, nstep, and number of burn-in steps

# In[16]:


nstep = 1000
nburn = 400


# ### Initialize the sampler and run emcee.

# In[17]:


# initialize sampler
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(wl, trans_dep_t2, trans_dep_t2_err),backend=backend)
# run mcmc for nstep steps
sampler.run_mcmc(pos, nstep,progress=True)


# In[18]:


# pull samples from chain/ discard burnin/thin by every other step/ flatten
reader = emcee.backends.HDFBackend(fn)
nthin = 2
samples = reader.get_chain(discard=nburn,thin=nthin,flat=True)
fig = corner.corner(samples,quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt = '.3f', color='xkcd:olive', labels=['H', 'T', 'log(fH2O)', 'log(fCH4)'])
#fig.savefig('/Grad School/Exoplanets Fall 2020/Analysis Project 2/Corner_Plot_Tier2_3_Shae_Raposa.png', bbox_inches="tight",dpi=300)


# ## Plot up residuals

# In[19]:


trans_dep_model = lnlike([200, 1500, -4, -4], wl, trans_dep_t2, trans_dep_t2_err)
trans_dep_vals = ForwardModel([200, 1500, -4, -4], wl, trans_dep_t2, trans_dep_t2_err)
plt.errorbar(wl, trans_dep_t2-trans_dep_vals, yerr=trans_dep_t2_err,fmt='-o')
plt.xlabel('Wavelength (microns)')
plt.ylabel('Residuals')
#plt.savefig('/Grad School/Exoplanets Fall 2020/Analysis Project 2/Residuals_Plot_Shae_Raposa.png', bbox_inches="tight",dpi=300)


# ### Note: Residuals are close to zero, which is good.

# # Calculate C/O ratio - Tier 3

# ### First, plot up a histogram of C/O values

# In[20]:


# unlog log(f_H2O) and log(f_CH4)
f_H2O = 10.0**(samples[:,2])
f_CH4 = 10.0**(samples[:,3])
# C/O
C_O = f_CH4/f_H2O


# In[21]:


plt.hist(C_O, bins=100, histtype = 'step')
plt.title('1D Marginal Distribution - C/O Ratio')
plt.ylabel('Probability Density')
plt.xlabel('C/O')
#plt.savefig('/Grad School/Exoplanets Fall 2020/Analysis Project 2/Histogram_Plot_Shae_Raposa.png', bbox_inches="tight",dpi=300)


# ### From the marginal distribution, derive the 16-, 50-, and 84- percentile values

# In[22]:


(n, bins, patches) = plt.hist(C_O, bins = 100, histtype = 'step')
CDF_CO = np.cumsum(n)/np.sum(n)
# midpoint for bins 
mid_bins = 0.5*(bins[1:]+bins[:-1])


# In[24]:


from scipy import interpolate
# determine C/O at +/- 1-sigma by interpolating 
cdf_interp = interpolate.interp1d(CDF_CO,mid_bins)
sigma1 = .682689492
print('Most-likely CO: ',cdf_interp(0.5))
print('+1-sigma: ',cdf_interp(sigma1 + (1-sigma1)/2) - cdf_interp(0.5))
print('-1-sigma: ',cdf_interp(0.5) - cdf_interp((1-sigma1)/2))


# In[ ]:




