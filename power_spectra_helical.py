# In[ ]:

# import necessary modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import math
from scipy.integrate import dblquad, quad
from numpy import log, exp, pi
from scipy.integrate import solve_ivp
import pickle

def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef

Gamma=math.gamma

#nB=2 throughout
###########################################################################################
# Details of magnetic field
def Ek_Tina_fit(x):
    xd=0.3125
    d=3.35
    f=0.01236
    ans=1*x**4*exp(-1*x**3/xd**3+d*(x/xd)**0.5)+f*exp(-xd**4/x**4)*(x)**(-2)
    return ans
Pk_Tina=lambda x: Ek_Tina_fit(x)/x**2
kI_Tina=quad(lambda x: x**2*Pk_Tina(x), 0, 100)[0]/quad(lambda x: x**1*Pk_Tina(x), 0, 100)[0]
#For some reason the integral with Pk gives a shitty kI.

# xi_c_temp=quad(lambda x: x**2*tildeP_temp(x), 0, 100)[0]/quad(lambda x: x**1*tildeP_temp(x), 0, 100)[0] #finding the coherence wavenumber
def tildeP(xi,xd,b):
    ans=Pk_Tina(xi*kI_Tina)*exp(b*xi/xd-xi**2/xd**2)
    return ans
#xitable=10**np.arange(-2,2,0.1)
#plt.loglog(xitable,table(lambda x: tildeP(x,1000,0),xitable))

# Procedure to calculate b
# def db(t,b): #t is 1/xd because in solve_ivp time needs to increases. As I only know initial condition for xd->inf, it is better to replace xD with 1/t
#     ans=(2*quad(lambda x: x**3*tildeP(x*1/t,1/t,b), 0, 100)[0]/quad(lambda x: x**2*tildeP(x*1/t,1/t,b), 0, 100)[0]-b)/t
#     return ans
# b_sol=solve_ivp(db,[10**-2,100],[0,],method='BDF',dense_output=True,atol=1e-7,rtol=1e-7)
# def b(xd):
#     if xd>1/b_sol.t[0]:
#         ans=0
#     elif xd>1/b_sol.t[-1]:
#         ans=b_sol.sol(1/xd)
#     else:
#         C=b_sol.y[0,-1]**2-16*log(b_sol.t[-1])
#         ans=(16*log(1/xd)+C)**0.5
#     return ans
# b_table=[b_sol.t,b_sol.y[0,:]]
# with open('b_python.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(b_table, f)

with open('b_python.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    b_table = pickle.load(f)
blog=PchipInterpolator(log(b_table[:,0]),b_table[:,1])
def b(xd):
    if xd>1/b_table[0,0]:
        ans=0
    elif xd>1/b_table[-18,0]:
        ans=blog(-log(xd))
    else:
        C=b_table[-18,1]**2-16*log(b_table[-18,0])
        ans=(16*log(1/xd)+C)**0.5
    return ans


def xi_c(xD): #gives coherence length scale as a function of xd. for xd>>1, it is 1.
    ans=quad(lambda x: x**1*tildeP(x,xD,b(xD)), 0, 100)[0]/quad(lambda x: x**2*tildeP(x,xD,b(xD)), 0, 100)[0] #finding the coherence wavenumber
    return ans

#In[]
Amp=quad(lambda x: x**2*Pk_Tina(x*kI_Tina), 0, 100)[0]
def F(xd):
    btemp=b(xd)
    ans=xd**3/Amp*quad(lambda x: x**2*tildeP(x*xd,xd,btemp), 0, 100*min(1/xd,1))[0]
    return ans

xdtable=10**np.arange(-4,2,0.1)
Ftable=table(lambda x: F(x),xdtable)
Flog=PchipInterpolator(log(xdtable),log(Ftable))
def F_int(xd):
    #an interpolating function for original F
    if xd<xdtable[0]:
        btemp=b(xd)
        ans=pi**0.5/16*tildeP(10**-4,100,0)/(10**-4)**2/Amp*xd**5*exp(btemp**2/4)*btemp**2*(12+btemp**2)
    elif xd<=xdtable[-1]:
        ans=exp(Flog(log(xd)))
    else:
        ans=1
    return ans

#In[]:
#Defining G: P_S0\propto k^4G(k/kI,kD/kI)

def Gnum(xi,xd):
    btemp=b(xd)
    Gnum_integrand=(lambda y,t: tildeP(t,xd,btemp)*tildeP((xi**2+t**2-2*xi*t*y)**0.5,xd,btemp)/2/(xi**2+t**2-2*xi*t*y)*
    t**2*(2*t**2*(1-2*y**2+2*y**4)-4*xi*t*y**3+xi**2*(1+y**2)-2*(t+xi*y-2*t*y**2)*(xi**2+t**2-2*xi*t*y)**0.5))
    ans=dblquad(Gnum_integrand, 0, np.inf,-1,1)[0]
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
