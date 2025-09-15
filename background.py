# =============================================================================
# Imports
# =============================================================================
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.interpolate import interp1d
import math
from scipy.integrate import solve_ivp
import pandas as pd
from scipy.optimize import fsolve

# =============================================================================
# Utility Functions
# =============================================================================

def table(f, t):
    """
    Evaluate function f at each value in array t and return as numpy array.
    """
    tablef = np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i] = f(t[i])
    return tablef

# =============================================================================
# Cosmological Parameters and CLASS Setup
# =============================================================================

common_settings = {
    # LambdaCDM parameters
    'h': 0.67556,
    'omega_b': 0.022032,
    'omega_cdm': 0.12038,
    'A_s': 2.215e-9,
    'n_s': 0.9619,
    'tau_reio': 0.0925,
    'YHe': 0.246,  # Primordial Helium
    # CLASS options
    'compute damping scale': 'yes', #needed to output photon diffusion scale
    'gauge': 'newtonian'
}
T0 = 2.34895e-10  # Present photon temperature in MeV

# Initialize CLASS
M = Class()
M.set(common_settings)
M.compute()

# =============================================================================
# Extract Background Quantities from CLASS
# =============================================================================

background = M.get_background()
H_table = background['H [1/Mpc]']
rhodm_table = background['(.)rho_cdm']
rhob_table = background['(.)rho_b']
rhor_table = background['(.)rho_g'] + background['(.)rho_ur']
rhode_table = background['(.)rho_lambda']
a_background = 1 / (1 + background['z'])
background_R = 3.0 / 4 * (background['(.)rho_b']) / (background['(.)rho_g'])

# =============================================================================
# Early Universe Degrees of Freedom
# =============================================================================

# Load degrees of freedom data from Excel
degrees_of_freedom_data = pd.read_excel('early universe degrees of freedom3.xlsx')
temperature_dof = degrees_of_freedom_data['K_bT_MeV']  # MeV
g_eff = degrees_of_freedom_data['Ge']
g_s = degrees_of_freedom_data['gs']

# Interpolating functions for g_eff and g_s as a function of temperature
logT = np.log(temperature_dof[::-1])  # Reverse for increasing order
g_eff_interp = lambda T: np.where(
    T > temperature_dof.iloc[0],
    g_eff.iloc[0],
    np.where(
        T < temperature_dof.iloc[-1],
        g_eff.iloc[-1],
        interp1d(logT, g_eff[::-1], kind='linear', fill_value="extrapolate")(np.log(T))
    )
)
g_s_interp = lambda T: np.where(
    T > temperature_dof.iloc[0],
    g_s.iloc[0],
    np.where(
        T < temperature_dof.iloc[-1],
        g_s.iloc[-1],
        interp1d(logT, g_s[::-1], kind='linear', fill_value="extrapolate")(np.log(T))
    )
)

# =============================================================================
# Temperature-Scale Factor Relations
# =============================================================================

Ttable = T0 * 10 ** np.arange(16, -0.1, -0.1)
a_T = lambda T: T0 / T * (g_s.iloc[-1] / g_s_interp(T)) ** (1 / 3)
atable = T0 / Ttable * (g_s.iloc[-1] / table(g_s_interp, Ttable)) ** (1 / 3)
T_a = lambda a: np.where(
    (a >= atable[0]) & (a <= atable[-1]),
    np.exp(interp1d(np.log(atable), np.log(Ttable), kind='linear', fill_value="extrapolate")(np.log(a))),
    T0 / a
)

# =============================================================================
# Matter-Radiation Equality
# =============================================================================

a_at_rho_m_over_r = interp1d((rhob_table + rhodm_table) / (rhor_table), a_background)
a_eq = a_at_rho_m_over_r(1.)

# =============================================================================
# Density Interpolating Functions
# =============================================================================

loga = np.log(a_background)
rhodm = lambda a: rhodm_table[0] * (a / a_background[0]) ** -3
rhob = lambda a: rhob_table[0] * (a / a_background[0]) ** -3

logrhor = np.log(rhor_table)
rho_ref = np.exp(interp1d(loga, logrhor, fill_value="extrapolate")(np.log(10 ** -7)))
def rhor(a):
    """
    Radiation density as a function of scale factor.
    Uses CLASS for a >= 1e-7, otherwise uses g_eff corrections.
    """
    if a >= 1e-7:
        ans = rhor_table[0] * (a / a_background[0]) ** -4
    else:
        Tref = T0 * (1e-7) ** -1  # MeV
        Tnow = T_a(a)
        ans = rho_ref / Tref ** 4 / g_eff_interp(Tref) * Tnow ** 4 * g_eff_interp(Tnow)
    return ans

rho_tot = lambda a: rhor(a) + rhodm(a) + rhob(a) + rhode_table[-1]
Ht = lambda a: rho_tot(a) ** 0.5  # Hubble rate in 1/Mpc

R = lambda a: background_R[0] * (a / a_background[0])

# =============================================================================
# Recombination and Thermodynamics
# =============================================================================

quantities = M.get_current_derived_parameters(['z_rec'])
a_rec = 1 / (1 + quantities['z_rec'])

ther = M.get_thermodynamics()
photon_mfp = 1 / ther["kappa' [Mpc^-1]"]  # Comoving units
therm_a = 1 / (1 + ther['z'])
cb2 = ther['c_b^2']

def photon_mfp_full(a):
    """
    Photon mean free path as a function of scale factor.
    """
    if a > 1 / 3 * 1e-4:
        ans = 1 / np.interp(1 / a, 1 / therm_a, 1 / photon_mfp) #near and after recombination I use interpolating table.
    else:
        ans=photon_mfp[-1]*(a/therm_a[-1])**2 #I do not including the effect due to changing degrees of freedom in photon mean free path for a<10**-7. It has negligible outcome because photon mean free path is not important for a<10**-7
    return ans

alpha = lambda a: 1 / photon_mfp_full(a) / a / R(a)


# For some reason, class is misbehaving and outputting negative values for sound speed near reionization. So I just ignore those values and extrapolate from last positive value.
valid_indices = cb2 > 0 
therm_a_2 = therm_a[valid_indices] #only include indices in interpolation where cb2 is positive
log_therma= np.log(therm_a_2[::-1]) #inverting because therm_a is in decreasing order
cb2_2 = cb2[valid_indices]
log_cb = np.log(cb2_2[::-1])
def cb2_full(a):
    """
    Baryon sound speed squared as a function of scale factor.
    """
    if a > 1 / 3 * 1e-4:
        ans = np.interp(np.log(a), log_therma, log_cb)
        ans = np.exp(ans)
    else:
        ans = cb2[-1] * (therm_a[-1] / a) ** 1
    return ans

# =============================================================================
# Magnetic Damping Scale Evolution
# =============================================================================

def dkgd(a, kgd):
    """
    Differential equation for photon diffusion scale.
    """
    ans = -kgd ** 3 / 12 * 1 / a ** 2 / Ht(a) * photon_mfp_full(a) * (R(a) ** 2 + 16 / 15 * (R(a) + 1)) / (1 + R(a)) ** 2
    return ans

astrt = 1e-9
kgd = solve_ivp(dkgd, [astrt, 1], [1 / photon_mfp_full(astrt) * 100], method='BDF', dense_output=True, atol=1e-6, rtol=1e-5)

def l_gammaD(a):
    """
    Photon diffusion length scale as a function of scale factor.
    """
    if a >= 1.1e-9:
        return 1 / kgd.sol(a)
    else:
        return 1 / kgd.sol(1.1e-9) * (a / (1.1 / 1e-9)) ** 1.5

# =============================================================================
# Neutrino Diffusion Length Scale
# =============================================================================

def g_target(T):
    """
    Number of fermions interacting with neutrinos at temperature T (MeV).
    """
    if T > 200:
        ans = (g_eff_interp(T) - 20) * 8 / 7
    elif T < 100:
        ans = (g_eff_interp(T) - 2) * 8 / 7
    else:
        ans = (g_eff_interp(T) - 2) * 8 / 7 * ((200 - T) / 100) ** 0.5 + (g_eff_interp(T) - 20) * 8 / 7 * ((T - 100) / 100) ** 0.5
    return ans

l_nuD=lambda a: 1.3*10**-5*(g_eff_interp(T_a(a))/10.75)**(1/12)*(10/g_target(T_a(a)))**(1/2)*(T_a(a))**-2.5 #assume T is in MeV

l_nu_mfp= lambda a: l_nuD(a)**2*a*Ht(a)
alpha_nu=lambda a: 7/4/g_eff_interp(T_a(a))/l_nu_mfp(a)/a #neutrino drag coefficient
a_solution = fsolve(lambda a: l_nu_mfp(a) - 7/4/g_eff_interp(T_a(a))/Ht(a)/a, 10**-7)
a_nu=a_solution[0] #scale factor when neutrino decouples from the plasma.


# =============================================================================
# Solving B and kD evolution from phase transition until the magnetic coherence scale enters photon mean free path
# =============================================================================

def BIkI_fromPT(T_PT):
    """
    Compute magnetic field and coherence scale evolution from phase transition temperature T_PT (MeV).
    Returns:
        B_K: function returning [B, K] at scale factor a
        a_I: scale factor when coherence scale enters photon diffusion scale
    """
    a_PT = T0 / T_PT * (g_s.iloc[-1] / g_s_interp(T_PT)) ** (1 / 3)
    k_PT = a_PT * Ht(a_PT) #comoving wavenumber at phase transition
    B_PT = 5.5 * (10 / g_eff_interp(T_PT)) ** (1 / 6) * 1e3  # nG comoving

    Va_PT = lambda a: (2.7e-4) * B_PT * g_eff_interp(T_a(a)) ** (1 / 6)
    kturb_PT = lambda a: (0.1 * Va_PT(a) / (a * Ht(a))) ** -1

    if k_PT < kturb_PT(a_PT):
        a_cross0 = fsolve(lambda a: kturb_PT(a) - k_PT, a_PT)[0]
    else:
        a_cross0 = a_PT

    k_hoskingI= kturb_PT(a_cross0) #initial coherence wavenumber when hosking conservation begins
    extrapolated_k=lambda a: k_hoskingI*(a/a_cross0)**(-4/9)*(g_eff_interp(T_a(a))/g_eff_interp((T_a(a_cross0))))**(4/27) #extrapolating from PT assuming Hosking conservation
    a_solution = fsolve(lambda a: extrapolated_k(a) - 1/l_nuD(a), a_PT)  # Start the search near a_PT
    a_cross=a_solution[0]

    def B_K(a,nB=2): #magnetic field and Fourier wavenumber of coherence length scale of magnetic field in 1/Mpc at a before photon diffusion regime
        if a<a_cross0:
            ansK=k_PT
            ansB=B_PT
        elif a<a_cross:
            ansK=extrapolated_k(a)
            ansB=B_PT*(ansK/k_PT)**(5/4) #extrapolating assuming Hosking conservation
        else:
            kI_nu=extrapolated_k(a_cross)#initial coherence wavenumber inside neutrino diffusion length scale when neutrino viscous damping begins
            BI_nu=B_PT*(kI_nu/k_PT)**(5/4)
            VaI2_nu=(2.7*10**-4)*BI_nu*g_s_interp(T_a(a_cross))**(2/3)/(g_eff_interp(T_a(a_cross))-21/4)**(1/2) #initial value of VaI2 inside neutrino diffusion length scale. I subtract neutrino degrees of freedom from baryon plasma
            kD_nu=lambda x: (VaI2_nu/x/(Ht(x)*alpha_nu(x))**0.5)**(-2/(nB+5))*kI_nu**((nB+3)/(nB+5)) #viscoud drag wavenumber due to neutrino freestreaming
            if a<a_nu:
                ansK=min(kD_nu(a),kI_nu) # kd_nu only becomes the coherence scale once it becomes smaller than kI_nu
                ansB=BI_nu*(ansK/kI_nu)**((nB+3)/2)
            else:
                kI_nu_after=min(kD_nu(a_nu),kI_nu)
                ansK=kI_nu_after*(a/a_nu)**(-4/9)*(g_eff_interp(T_a(a))/g_eff_interp(T_a(a_nu)))**(4/27) #extrapolating from PT assuming Hosking conservation
                ansB=BI_nu*(kI_nu_after/kI_nu)**((nB+3)/2)*(ansK/kI_nu_after)**(5/4)
        return [ansB,ansK]

    def debug_fsolve(a):
        bk_value = B_K(a, nB=2)[1]
        lgamma_value = 1 / l_gammaD(a)
        
        # Ensure both values are scalars
        if isinstance(bk_value, np.ndarray):
            bk_value = bk_value.item()  # Convert to scalar
        if isinstance(lgamma_value, np.ndarray):
            lgamma_value = lgamma_value.item()  # Convert to scalar
        
        # print(f"a: {a}, B_K: {bk_value}, 1/l_gammaD: {lgamma_value}")
        return bk_value - lgamma_value

    a_solution = fsolve(debug_fsolve, 10**-10)  # Start the search near 10^-7
    a_I = a_solution[0]  # Scale factor where coherence scale becomes equal to photon diffusion length scale. This sets the value of BI and kI used for solving perturbation equations.
    return [B_K, a_I]  # Returns a function B_K and scale factor a_I

# =============================================================================
# Magnetic field evolution inside the photon mean free path scale
# =============================================================================

def kdsolve(BI,kI,F_int):
    #returns functions kd and B as a function of scale factor. Here by BI I mean comoving magnetic field strength at initial time when coherence scale enters photon diffusion length scale.
    kmax=kI*100 #initial value of kd
    if kmax>1000:
        a_min=therm_a[-1]*(kmax*photon_mfp[-1])**-0.5
    else:
        a_at_k_over_mfp = interp1d(kmax*photon_mfp,therm_a)
        a_min = a_at_k_over_mfp(1.) #initial value of a from which I solve for kd

    Va02=2.2*10**-10*BI**2

    def dkd(a,kd):
        ans=-2*kd**3/3*F_int(kd/kI)*Va02/a**4/Ht(a)/(alpha(a)+Ht(a))
        return ans

    kd=solve_ivp(dkd,[a_min,1],[kmax*10,],method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)
    B_ev=lambda a: BI*F_int(kd.sol(a)/kI)**0.5
    return kd, B_ev

# =============================================================================
# Combined Evolution of Magnetic Field and Coherence Scale
# =============================================================================

def evolution_BK(T_PT, F_int):
    """
    Returns functions B_evolve(a) and k_evolve(a) for magnetic field and coherence scale evolution.
    Args:
        T_PT: phase transition temperature (MeV)
        F_int: spectral evolution function
    Returns:
        [B_evolve, k_evolve]: functions of scale factor a
    """
    [B_K, a_I] = BIkI_fromPT(T_PT)
    [BI, kI] = B_K(a_I)
    [kd, B_ev] = kdsolve(BI, kI, F_int)

    def B_evolve(a):
        if a < a_I:
            return B_K(a)[0]
        else:
            return B_ev(a)

    def k_evolve(a):
        if a < a_I:
            return B_K(a)[1]
        else:
            return min(kI, kd.sol(a)[0])

    return [B_evolve, k_evolve]

# =============================================================================
# End of File
# =============================================================================
