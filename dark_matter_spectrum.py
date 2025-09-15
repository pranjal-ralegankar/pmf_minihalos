# import necessary modules
import numpy as np

from power_spectra import find_S0, F_int  # Ensure these are correctly imported
from background import rhodm, rhob, rho_tot, Ht, alpha, photon_mfp_full, cb2_full, kdsolve, a_rec, a_eq, a_background
from PMF_perturbations import pert_sharpcut_at_rec


def Pk_fn(BI,kI,z_output):
    [kd, B_ev]=kdsolve(BI, kI, F_int) #finding evolution of kd and B
    print("magnetic field strength today is=",B_ev(1)," nG")

    ############## producing table of power spectrum #############
    K = kI * 10**np.arange(-1.5, 2, 0.1) # table of k values of where to plot power spectrum
    K = K[K < 220000000.00000113] # cut short the array once values exceed the limit

    Pk=np.zeros(K.shape[0]) #initializing power spectrum

    j=0
    for k in K:
        S0=find_S0(k,kI,BI)

        [a_dm_only,delta_dm_only,a_pmf,delta_dm_pmf,delta_b_pmf,theta_b_pmf,phi]=pert_sharpcut_at_rec(k,S0,kd,Ht,rhodm,rho_tot,photon_mfp_full,alpha,rhob,cb2_full,a_rec)
        
        a_output=1/(1+z_output)
        i_pmf=np.where(a_pmf>=a_output)[0][0]
        i_dm=np.where(a_dm_only>=a_output)[0][0]
        
        Pk[j]=delta_dm_only[i_dm]**2+delta_dm_pmf[i_pmf]**2#here delta_dm_only is dm perturbation without pmf in lcdm cosmology. The second one is delta_dm due to Pmf but without lcdm.
        
        print("computing P(k) for k/K_I=",k/kI)
        j=j+1
    
    return K, Pk

#%%