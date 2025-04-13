#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# In[ ]:

# import necessary modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, PchipInterpolator
import math
from scipy.integrate import solve_ivp
import pandas as pd
from mpmath import hyp2f2
from numpy import log, exp

from power_spectra import find_S0, table, F_int
from background import rhodm, rhob, rho_tot, Ht, alpha, photon_mfp_full, cb2_full, kdsolve, a_rec, a_eq, a_background
from PMF_perturbations import pert_sharpcut_at_rec

#%%
########### INPUT PMF parameters #########################################
B0=0.56 #initial magnetic field strength in nG
kI=2.2*10**7 #Fourier wavenumber of initial coherence length scale of magnetic field in 1/Mpc, this will also roughly be the wavenumber with the peak enhancement in dark matter power spectrum
z_choose=0 #redshift at which power spectrum is needed.

[kd, B0_ev]=kdsolve(B0, kI, F_int) #finding evolution of kd and B0
print("magnetic field strength today is=",B0_ev(1)," nG")

#In[]
############## producing table of power spectrum #############
K = kI * 10**np.arange(-1.5, 2, 0.1) # table of k values of where to plot power spectrum
K = K[K < 438957709.293156] # cut short the array once values exceed the limit

Pk=np.zeros(K.shape[0]) #initializing power spectrum

j=0
for k in K:
    S0=find_S0(k,kI,B0)

    [a_dm_only,delta_dm_only,a_pmf,delta_dm_pmf,delta_b_pmf,theta_b_pmf,phi]=pert_sharpcut_at_rec(k,S0,kd,Ht,rhodm,rho_tot,photon_mfp_full,alpha,rhob,cb2_full,a_rec)
    
    a_choose=1/(1+z_choose)
    i_pmf=np.where(a_pmf>=a_choose)[0][0]
    i_dm=np.where(a_dm_only>=a_choose)[0][0]
    
    Pk[j]=delta_dm_only[i_dm]**2+delta_dm_pmf[i_pmf]**2#here delta_dm_only is dm perturbation without pmf in lcdm cosmology. The second one is delta_dm due to Pmf but without lcdm.
    
    print(k/kI)
    j=j+1

#%%
#plotting############################################################################################

%matplotlib widget
plt.figure(1)
minx2=10**0
# maxx2=10**4220000000.00000113
maxx2=10**5
plt.rcParams.update({'font.size': 22})
plt.loglog(K, Pk, color='k', linewidth=2.5,label=r"$B_I=$10 nG (1a)")


plt.loglog([kI,kI],[minx2,maxx2],'-k')
plt.text(kI*1.1,100,r"$\xi_{I1}^{-1}$")

plt.xlabel(r"$k$  (${\rm Mpc}^{-1}$)",labelpad=-3)
plt.ylabel(r"$k^3P(k)/[2\pi^2]$",labelpad=0)
plt.ylim([minx2,maxx2])
plt.xlim([K[0],K[-1]])
plt.legend(loc="upper right", ncol=2, fontsize='20')

plt.gcf().set_size_inches(10, 5,forward=True)
plt.gcf().subplots_adjust(left=0.11, right=0.99, bottom=0.16, top=0.97)
plt.minorticks_on()
plt.tick_params(length=8, width=1.5)
plt.tick_params(which='minor',length=5, width=1.5)

# %%
# plt.savefig("Pk.pdf", format="pdf", bbox_inches="tight")

# %%
