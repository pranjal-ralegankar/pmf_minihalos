#In[]:
# importing functions
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.interpolate import interp1d
import math
from scipy.integrate import solve_ivp
# In[ ]:
# Cosmological parameters and other CLASS parameters##############################################
common_settings = {# we need to set the output field to something although
                   # the really releveant outpout here will be set with 'k_output_values'
                   #'output':'mPk',
                   # value of k we want to polot in [1/Mpc]
                   #'k_output_values':k,
                   # LambdaCDM parameters
                   'h':0.67556,
                   'omega_b':0.022032,
                   'omega_cdm':0.12038,
                   'A_s':2.215e-9,
                   'n_s':0.9619,
                   'tau_reio':0.0925,
                   # Take fixed value for primordial Helium (instead of automatic BBN adjustment)
                   'YHe':0.246,
                   # other options and settings
                   'compute damping scale':'yes', # needed to output the time of damping scale crossing
                   'gauge':'newtonian'}
##############
#
# call CLASS
#
M = Class()
M.set(common_settings)
M.compute()

#background quantities
background = M.get_background()
H_table=background['H [1/Mpc]']
rhodm_table=background['(.)rho_cdm'] 
rhob_table=background['(.)rho_b']
rhor_table=background['(.)rho_g']+background['(.)rho_ur']
rhotot_table=background['(.)rho_tot']
a_background=1/(1+background['z'])
background_R=  3.0/4*(background['(.)rho_b'])/(background['(.)rho_g'])

a_at_rho_m_over_r = interp1d((rhob_table+rhodm_table)/(rhor_table),a_background)
a_eq = a_at_rho_m_over_r(1.)

rhodm = lambda a: np.where(a >= a_background[0], 
                           interp1d(a_background, rhodm_table, fill_value="extrapolate")(a), 
                           rhodm_table[0] * (a / a_background[0])**-3)

rhob = lambda a: np.where(a >= a_background[0], 
                          interp1d(a_background, rhob_table, fill_value="extrapolate")(a), 
                          rhob_table[0] * (a / a_background[0])**-3)

rhor = lambda a: np.where(a >= a_background[0], 
                          interp1d(a_background, rhor_table, fill_value="extrapolate")(a), 
                          rhor_table[0] * (a / a_background[0])**-4)

rho_tot = lambda a: np.where(a >= a_background[0], 
                             interp1d(a_background, rhotot_table, fill_value="extrapolate")(a), 
                             rhotot_table[0] * (a / a_background[0])**-4)

Ht = lambda a: np.where(a >= a_background[0], 
                        interp1d(a_background, H_table, fill_value="extrapolate")(a), 
                        H_table[0] * (a / a_background[0])**-2)

R = lambda a: np.where(a >= a_background[0], 
                       interp1d(a_background, background_R, fill_value="extrapolate")(a), 
                       background_R[0] * (a / a_background[0]))

quantities = M.get_current_derived_parameters(['z_rec'])
a_rec = 1/(1+quantities['z_rec'])

#thermodynamic quantities
ther=M.get_thermodynamics()
photon_mfp=1/ther["kappa' [Mpc^-1]"] #in comoving units coordinates
therm_a=1/(1+ther['z'])
cb2=ther['c_b^2']

def photon_mfp_full(a):
    if a>therm_a[-1]:
        ans=1/np.interp(1/a,1/therm_a,1/photon_mfp)
    else:
        ans=photon_mfp[-1]*(a/therm_a[-1])**2
    return ans

alpha=lambda a: 1/photon_mfp_full(a)/a/R(a)

def cb2_full(a):
    if a>therm_a[-1]:
        ans=1/np.interp(1/a,1/therm_a,1/cb2)
    else:
        ans=cb2[-1]*(therm_a[-1]/a)**1
    return ans

def dkgd(a,kgd):
    ans=-kgd**3/12*1/a**2/Ht(a)*photon_mfp_full(a)*(R(a)**2+16/15*(R(a)+1))/(1+R(a))**2
    return ans

astrt=10**-9
kgd=solve_ivp(dkgd,[astrt,1],[1/photon_mfp_full(astrt)*100,],method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)

l_gammaD=lambda a: 1/kgd.sol(a)
# a_BBN=fsolve(lambda a: M.baryon_temperature(1/a)-11604525006.1657,10**-13)[0]
# In[ ]:
#Solving kd############################################################################################
def kdsolve(B0,kI,F_int):
    #returns functions kd and B0 as a function of scale factor
    kmax=kI*100 #initial value of kd
    if kmax>1000:
        a_min=therm_a[-1]*(kmax*photon_mfp[-1])**-0.5
    else:
        a_at_k_over_mfp = interp1d(kmax*photon_mfp,therm_a)
        a_min = a_at_k_over_mfp(1.) #initial value of a from which I solve for kd

    Va02=2.2*10**-10*B0**2

    def dkd(a,kd):
        ans=-2*kd**3/3*F_int(kd/kI)*Va02/a**4/Ht(a)/(alpha(a)+Ht(a))
        return ans

    kd=solve_ivp(dkd,[a_min,1],[kmax*10,],method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)
    B0_ev=lambda a: B0*F_int(kd.sol(a)/kI)**0.5
    return kd, B0_ev

#In[]:
#[kd, B0_ev]=kdsolve(1,10**4,2)
#%matplotlib widget
#plt.loglog(kd.t,1/kd.y[0,:])