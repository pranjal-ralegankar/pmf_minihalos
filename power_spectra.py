# =============================================================================
# Imports
# =============================================================================
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import math
from scipy.integrate import dblquad, quad
import pandas as pd
from mpmath import hyp2f2
from numpy import log, exp, pi

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

Gamma = math.gamma

# =============================================================================
# Magnetic Field Power Spectrum Details
# =============================================================================

nB = 2
m = 4  # Determines the shape of power spectrum near peak

def tildeP_temp(xi):
    """
    dimensionless power spectrum function for magnetic field before they enter photon damping regime
    """
    ans = xi**nB/(1+xi**(m))**(1/m*(14/3+nB))
    return ans

# Calculate coherence wavenumber for the spectrum
xi_c_temp = quad(lambda x: x**2*tildeP_temp(x), 0, 100)[0] / quad(lambda x: x**1*tildeP_temp(x), 0, 100)[0]

def tildeP(xi, xd):
    """
    Power spectrum function with exponential cutoff.
    """
    ans = tildeP_temp(xi*xi_c_temp)*exp(-xi**2/xd**2)
    return ans

def xi_c(xD):
    """
    Returns coherence length scale as a function of xD.
    """
    ans = quad(lambda x: x**1*tildeP(x,xD), 0, 100)[0] / quad(lambda x: x**2*tildeP(x,xD), 0, 100)[0]
    return ans

Amp = quad(lambda x: x**2*tildeP(x,1000), 0, 100)[0]

def F(xd):
    """
    Dimensionless computation of B^2 from power spectrum.
    """
    ans = 1/Amp*quad(lambda x: x**2*tildeP(x,xd), 0, 100*min(1,xd))[0]
    return ans

# Interpolating F(xd) for efficiency
xdtable = 10**np.arange(-3,2.5,0.1)
Ftable = table(lambda x: F(x), xdtable)
Flog = PchipInterpolator(log(xdtable), log(Ftable))

def F_int(xd):
    """
    Interpolating function for F(xd).
    """
    if xd < xdtable[0]:
        ans = Ftable[0]*(xd/xdtable[0])**(nB+3)
    elif xd <= xdtable[-1]:
        ans = exp(Flog(log(xd)))
    else:
        ans = 1
    return ans

# =============================================================================
# Power Spectrum G Function
# =============================================================================

def Gnum(xi, xd):
    """
    Numerically compute G for the S0 power spectrum.
    """
    Gnum_integrand = (lambda y, t: tildeP(t, xd)*tildeP((xi**2+t**2-2*xi*t*y)**0.5, xd)/2/(xi**2+t**2-2*xi*t*y)*
        t**2*(2*t**2*(1-2*y**2+2*y**4)-4*xi*t*y**3+xi**2*(1+y**2)))
    Gnum_integrand2 = lambda y, t: Gnum_integrand(y, t*xd)
    ans = xd*dblquad(Gnum_integrand2, 0, np.inf, -1, 1)[0]
    return ans

# =============================================================================
# S0 Power Spectrum Calculation
# =============================================================================

def find_S0(k, kI, B0):
    """
    Find S0 as a function of kd given k, kI, and B0.
    Returns a function S0(kd).
    """
    xd_low_limit = 0.1*k/kI
    xd_up_limit = 20*max(k/kI,1)
    xdtable = 10**np.arange(log(xd_low_limit)/log(10), log(xd_up_limit)/log(10), 0.1)
    Gtable = table(lambda x: Gnum(k/kI, x), xdtable)
    Glog = PchipInterpolator(log(xdtable), log(Gtable))
    def G_int(xd):
        # Interpolating function for G
        if xd < xdtable[0]:
            ans = 0
        elif xd <= xdtable[-1]:
            ans = exp(Glog(log(xd)))
        else:
            ans = Gtable[-1]
        return ans

    Va02 = 2.2*10**-10*B0**2
    A = pi**2*Va02**2/4/kI**3*(1/Amp)**2
    PS0 = lambda xd: k**4*A*G_int(xd)
    S0 = lambda kd: (k**3/2/pi**2*PS0(kd/kI))**0.5
    return S0


