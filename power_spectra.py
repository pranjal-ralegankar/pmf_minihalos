# In[ ]:

# import necessary modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import math
from scipy.integrate import dblquad, quad
import pandas as pd
from mpmath import hyp2f2
from numpy import log, exp, pi

def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef

Gamma=math.gamma

###########################################################################################
# Details of magnetic field
nB=2
m=4 #determines the shape of power spectrum near peak. small m means slow transition and hence shallower peak. large m gives sharp peak.
def tildeP_temp(xi):
    ans=xi**nB/(1+xi**(m))**(1/m*(14/3+nB))
    return ans

xi_c_temp=quad(lambda x: x**2*tildeP_temp(x), 0, 100)[0]/quad(lambda x: x**1*tildeP_temp(x), 0, 100)[0] #finding the coherence wavenumber
def tildeP(xi,xd):
    ans=tildeP_temp(xi*xi_c_temp)*exp(-xi**2/xd**2)
    return ans

def xi_c(xD): #gives coherence length scale as a function of xd>>1, it is 1.
    ans=quad(lambda x: x**1*tildeP(x,xD), 0, 100)[0]/quad(lambda x: x**2*tildeP(x,xD), 0, 100)[0] #finding the coherence wavenumber
    return ans

Amp=quad(lambda x: x**2*tildeP(x,1000), 0, 100)[0]
def F(xd):
    ans=1/Amp*quad(lambda x: x**2*tildeP(x,xd), 0, 100*min(1,xd))[0]
    return ans

xdtable=10**np.arange(-3,2.5,0.1)
Ftable=table(lambda x: F(x),xdtable)
Flog=PchipInterpolator(log(xdtable),log(Ftable))
def F_int(xd):
    #an interpolating function for original F
    if xd<xdtable[0]:
        ans=Ftable[0]*(xd/xdtable[0])**(nB+3)
    elif xd<=xdtable[-1]:
        ans=exp(Flog(log(xd)))
    else:
        ans=1
    return ans

#In[]:
#Defining G: P_S0\propto k^4G(k/kI,kD/kI)

def Gnum(xi,xd):
    Gnum_integrand=(lambda y,t: tildeP(t,xd)*tildeP((xi**2+t**2-2*xi*t*y)**0.5,xd)/2/(xi**2+t**2-2*xi*t*y)*
    t**2*(2*t**2*(1-2*y**2+2*y**4)-4*xi*t*y**3+xi**2*(1+y**2)))
    Gnum_integrand2=lambda y,t: Gnum_integrand(y,t*xd)
    ans=xd*dblquad(Gnum_integrand2, 0, np.inf,-1,1)[0]
    return ans
# Finding S0 as a function of kd given a value of k, kI etc.
    
def find_S0(k,kI,B0):
    xd_low_limit=0.1*k/kI
    xd_up_limit=20*max(k/kI,1)
    xdtable=10**np.arange(log(xd_low_limit)/log(10),log(xd_up_limit)/log(10),0.1)
    Gtable=table(lambda x: Gnum(k/kI,x),xdtable)
    Glog=PchipInterpolator(log(xdtable),log(Gtable))
    def G_int(xd):
        #an interpolating function for original F
        if xd<xdtable[0]:
            ans=0
        elif xd<=xdtable[-1]:
            ans=exp(Glog(log(xd)))
        else:
            ans=Gtable[-1]
        return ans
    
    Va02=2.2*10**-10*B0**2
    A=pi**2*Va02**2/4/kI**3*(1/Amp)**2 #Amplitude of S0 power spectrum
    PS0=lambda xd: k**4*A*G_int(xd) #S0 power spectrum only as a function of kd/kI. 
    S0=lambda kd: (k**3/2/pi**2*PS0(kd/kI))**0.5
    return S0

    
#In[]:
###################testing python numerical G with table from mathematica################################
#xdtable=10*10**np.arange(-1,1.5,0.1)
#Gnum_table=table(lambda xd: Gnum(10,xd,nB),xdtable)
#Gan_table=table(lambda xd: G(10,xd),xdtable)
#%matplotlib widget
#plt.loglog(xdtable,Gnum_table)
#plt.loglog(xdtable,Gan_table)
# %%
