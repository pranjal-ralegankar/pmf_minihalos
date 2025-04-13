#functions that solve dark matter and baryon perturbations with and without PMF. 
# In[ ]:

from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

def pert_sharpcut_at_rec(k,S0,kd,Ht,rhodm,rho_tot,photon_mfp_full,alpha,rhob,cb2_full,a_rec):
    #Assuming logarithmic evolving delta_DM initially and zero phi
    #assume sharp cutoff of delta_b when theta_b feedbacks on magnetic fields
    norm=(2.215*10**-9*(k/0.002)**(0.9619-1))**0.5

    a_mfp=fsolve(lambda a: photon_mfp_full(a)*k-1,a_rec/10*(k*photon_mfp_full(a_rec/10))**-0.5)[0]
    a_hor=fsolve(lambda a: a*Ht(a)-k,(a_rec/10)**2*Ht(a_rec/10)/k)[0]
    
    def ddel_dm(a,delta,k): #DM only perturbation equations
        phi,dldm,thdm=delta[0],delta[1],delta[2]
        del_phi=-phi/a+(-k**2*phi-3./2*(a*Ht(a))**2*(rhodm(a)*dldm)/rho_tot(a))/(3*(a*Ht(a))**2)/a
        del_dldm=-(thdm/a**2/Ht(a) - 3*del_phi)
        del_thdm=-thdm/a+k**2/a**2/Ht(a)*phi

        return [del_phi,del_dldm,del_thdm]

    delta_dmi=-6.71*norm*np.log(0.5*200*a_hor/a_hor)
    theta_dmi=6.71*norm*200*a_hor*Ht(200*a_hor)
    deltasolve30=[0,delta_dmi,theta_dmi] #initial condition deep in subhorizon
    deltasolve3=solve_ivp(lambda a,y: ddel_dm(a,y,k),[200*a_hor,1],deltasolve30,method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
    delta_dm_only=deltasolve3.y[1,:]
    a_dm_only=deltasolve3.t[:]

    ###############My perturbation equations with pmf###############

    def ddel_baryon(a,delta,k): #DM and baryon perturbation equations
        phi,dldm,thdm,dlb,thb=delta[0],delta[1],delta[2],delta[3],delta[4]
        del_phi=-phi/a+(-k**2*phi-3./2*(a*Ht(a))**2*(rhob(a)*dlb+rhodm(a)*dldm)/rho_tot(a))/(3*(a*Ht(a))**2)/a
        del_dldm=-(thdm/a**2/Ht(a) - 3*del_phi)
        del_thdm=-thdm/a+k**2/a**2/Ht(a)*phi
        del_thb=-(1+alpha(a)/Ht(a))*thb/a+k**2/a**2/Ht(a)*phi+cb2_full(a)*k**2/a**2/Ht(a)*dlb+S0(kd.sol(a))/a**3/Ht(a)
        del_dlb=-(thb/a**2/Ht(a) - 3*del_phi)

        return [del_phi,del_dldm,del_thdm,del_dlb,del_thb]

    deltasolve0=[0,0,0,0,0] #initial condition deep in subhorizon
    deltasolve=solve_ivp(lambda a,y: ddel_baryon(a,y,k),[a_mfp,1],deltasolve0,method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely

    if k>kd.sol(10*a_rec):# for modes smaller than kd, trubulence at recombination will suppress delta_b
        i_turb=np.where(deltasolve.t>a_rec)[0][0] 
        #Solving DM only perturbations after baryon feedback, assume delta_b=0
        deltasolve20=[deltasolve.y[0,i_turb],deltasolve.y[1,i_turb],deltasolve.y[2,i_turb]]
        deltasolve2=solve_ivp(lambda a,y: ddel_dm(a,y,k),[deltasolve.t[i_turb],1],deltasolve20,method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely

        delta_dm_pmf=np.concatenate((deltasolve.y[1,:i_turb],deltasolve2.y[1,:]))
        a_pmf=np.concatenate((deltasolve.t[:i_turb],deltasolve2.t[:]))
        delta_b_pmf=np.concatenate((deltasolve.y[3,:i_turb],np.zeros(deltasolve2.t.shape[0])))
        theta_b_pmf=np.concatenate((deltasolve.y[4,:i_turb],np.zeros(deltasolve2.t.shape[0])))
        phi=np.concatenate((deltasolve.y[0,:i_turb],deltasolve2.y[0,:]))
    else:
        delta_dm_pmf=deltasolve.y[1,:]
        a_pmf=deltasolve.t
        delta_b_pmf=deltasolve.y[3,:]
        theta_b_pmf=deltasolve.y[4,:]
        phi=deltasolve.y[0,:]

    return([a_dm_only,delta_dm_only,a_pmf,delta_dm_pmf,delta_b_pmf,theta_b_pmf,phi])
