# this program is going to estimate parameters from simulated datasets that originate from an OU process
# with noise added.  The parameters of the simulation and the length of the simulation are set through
# arguments
import langevin
import pandas as pd
import numpy as np
import argparse
import pymc3 as pm
import theano.tensor as tt
from scipy.stats import pearsonr
from scipy.optimize import root

class Ornstein_Uhlenbeck(pm.Continuous):
    """
    Ornstein-Uhlenbeck Process
    Parameters
    ----------
    B : tensor
        B > 0, B = exp(-(D/A)*delta_t)
    A : tensor
        A > 0, amplitude of fluctuation <x**2>=A
    delta_t: scalar
        delta_t > 0, time step
    """
    
    def __init__(self, A=None, B=None,
                 *args, **kwargs):
        super(Ornstein_Uhlenbeck, self).__init__(*args, **kwargs)
        self.A = A
        self.B = B
        self.mean = 0.
    
    def logp(self, x):
        A = self.A
        B = self.B
        
        x_im1 = x[:-1]
        x_i = x[1:]
        
        ou_like = pm.Normal.dist(mu=x_im1*B, tau=1.0/A/(1-B**2)).logp(x_i)
        return pm.Normal.dist(mu=0.0,tau=1.0/A).logp(x[0]) + tt.sum(ou_like)

# function to calculate A and B from the dataset
def OUanalytic2(data):
    N = data.size
    data1sq = data[0]**2
    dataNsq = data[-1]**2
    datasq = np.sum(data[1:-1]**2)
    datacorr = np.sum(data[0:-1]*data[1:])
    coef = [(N-1)*datasq,
       (2.0-N)*datacorr,
       -data1sq-(N+1)*datasq-dataNsq,
       N*datacorr]
    B=np.roots(coef)[-1]
    Q=(data1sq+dataNsq)/(1-B**2)
    Q=Q+datasq*(1+B**2)/(1-B**2)
    Q=Q-datacorr*2*B/(1-B**2)
    A = Q/N
    P2A = -N/A**2/2
    Btmp = B**2*(1+2*N)
    tmp = (1+Btmp)*(data1sq+dataNsq) + (2*Btmp + N + 1 -B**4*(N-1))*datasq - 2*B*(1+B**2+2*N)*datacorr
    P2B = -tmp/((1-B**2)**2*(data1sq+dataNsq + (1+B**2)*datasq - 2*B*datacorr))
    PAB = (N-1)*B/A/(1-B**2)
    dA = np.sqrt(-P2B/(P2A*P2B-PAB**2))
    dB = np.sqrt(-P2A/(P2A*P2B-PAB**2))
    return A,dA,B,dB

def OUresult2(data,deltat):
    A, dA, B ,dB = OUanalytic2(data)
    tau = -deltat/np.log(B)
    dtau = deltat*dB/B/np.log(B)**2
    return A,dA,tau,dtau

def OUcross(data1,data2,deltat):
    x1 = data1 + data2
    x2 = data1 - data2
    x1_A,x1_dA, x1_tau ,x1_dtau= OUresult2(x1,deltat)
    x2_A, x2_dA, x2_tau ,x2_dtau= OUresult2(x2,deltat)
    return (x1_A - x2_A)/x2_A, np.sqrt(x1_dA**2 + x1_A**2*x2_dA**2/x2_A**4)

def calc_fundstats(x):
    return x[0]**2+x[-1]**2,np.sum(x[1:-1]**2),np.sum(x[0:-1]*x[1:])

def b(D,A,delta_t):
    return np.exp(-D/A*delta_t)

def q(aep,ass,ac,b):
    return (aep + (1+b**2)*ass - 2*b*ac)/(1-b**2)

def dqdB(aep,ass,ac,b):
    return 2*(b*aep+2*b*ass-(1+b**2)*ac)/(1-b**2)**2

def d2qdB2(aep,ass,ac,b):
    return (3*b+1)/(1-b**2)**3*(aep+2*ass)-(4*b**3-2*b**2+6*b)/(1-b**2)**3*ac

def dBdA(b,D,A,delta_t):
    return b*D*delta_t/A**2

def dBdD(b,A,delta_t):
    return -b*delta_t/A

def d2BdA2(b,D,A,delta_t):
    return b*(D**2*delta_t**2/A**4-2*D*delta_t/A**3)

def d2BdD2(b,A,delta_t):
    return b*delta_t**2/A**2

def d2BdAdD(b,D,A,delta_t):
    return b*delta_t/A**2*(1-D**2*delta_t/A)

def d2qdD2(aep,ass,ac,b,A,delta_t):
    return d2qdB2(aep,ass,ac,b)*dBdD(b,A,delta_t)**2+dqdB(aep,ass,ac,b)*d2BdD2(b,A,delta_t)

def d2qdAdD(aep,ass,ac,b,D,A,delta_t):
    return d2qdB2(aep,ass,ac,b)*dBdA(b,D,A,delta_t)*dBdD(b,A,delta_t)+dqdB(aep,ass,ac,b)*d2BdAdD(b,D,A,delta_t)

def d2PdA2(N,aep,ass,ac,b,D,A,delta_t):
    return (N/2/A - 
            q(aep,ass,ac,b)/A**3 +
            (N-1)/(1-b**2)*(b*d2BdA2(b,D,A,delta_t) + dBdA(b,D,A,delta_t)**2*(1+2*b**2/(1-b**2))))
        
def d2PdAdD(N,aep,ass,ac,b,D,A,delta_t):
    return (dqdB(aep,ass,ac,b)*dBdD(b,A,delta_t)/2/A**2 -
            d2qdAdD(aep,ass,ac,b,D,A,delta_t)/2/A +
            (N-1)/(1-b**2)*(b*d2BdAdD(b,D,A,delta_t) + dBdA(b,D,A,delta_t)*dBdD(b,A,delta_t)*(1+2*b**2/(1-b**2))))

def d2PdD2(N,a1ep,a1ss,a1c,a2ep,a2ss,a2c,b1,b2,D,A1,A2,delta_t):
    return ((N-1)/(1-b1**2)*(b1*d2BdD2(b1,A1,delta_t) + dBdD(b1,A1,delta_t)**2*(1+2*b1**2/(1-b2**2)))+
           (N-1)/(1-b2**2)*(b2*d2BdD2(b2,A2,delta_t) + dBdD(b2,A2,delta_t)**2*(1+2*b2**2/(1-b2**2)))-
           d2qdD2(a1ep,a1ss,a1c,b1,A1,delta_t)/2/A1 -
           d2qdD2(a2ep,a2ss,a2c,b2,A2,delta_t)/2/A2)
           
def phi_deriv(x,a1ep,a1ss,a1c,a2ep,a2ss,a2c,delta_t,N):
    # x[0] = A1, x[1] = A2, x[2]=D
    A1 = x[0]
    A2 = x[1]
    D = x[2]
    b1 = b(D,A1,delta_t)
    b2 = b(D,A2,delta_t)
    Q1 = q(a1ep,a1ss,a1c,b1)
    Q2 = q(a2ep,a2ss,a2c,b2)
    dQ1 = dqdB(a1ep,a1ss,a1c,b1)
    dQ2 = dqdB(a2ep,a2ss,a2c,b2)
    y1 = -N*A1**2/2 + A1*Q1/2 + b1*D*delta_t*(A1*b1*(N-1)/(1-b1**2)-dQ1/2)
    y2 = -N*A2**2/2 + A2*Q2/2 + b2*D*delta_t*(A2*b2*(N-1)/(1-b2**2)-dQ2/2)
    y3 = (b1*(N-1)/(1-b1**2)-dQ1/A1/2)*b1/A1 + (b2*(N-1)/(1-b2**2)-dQ2/A2/2)*b2/A2
    return np.array([y1,y2,y3])

def correlated_ts(c,delta_t = 0.1,N=1000):
    # parameters for coupled oscillator
    K,D = 1.0,1.0
    data1 = langevin.time_series(A=1/K, D=D, delta_t=delta_t, N=N)
    data2 = langevin.time_series(A=1/(K+np.abs(c)), D=D, delta_t=delta_t, N=N)
    x1 = (data1 + data2)/2
    if c>0:
        x2 = (data1 - data2)/2
    else:
        x2 = (data2-data1)/2

    return x1,x2

#parameters
a_bound=5
M=400
N=1000

results = None
for rho in np.arange(0.25,0.3,0.1):
    for i in range(M):
        delta_t = 0.1
        coupling = 2*np.abs(rho)/(1-np.abs(rho))*np.sign(rho)
        x1,x2 = correlated_ts(coupling,N=N)
        prho = pearsonr(x1,x2)[0]
        print("OU cross correlation", OUcross(x1,x2,delta_t))
        print("pearson: ",prho)

        para = calc_fundstats(x1+x2) + calc_fundstats(x1-x2) +(delta_t,N)
        guessa1 = (x1+x2).std()**2
        guessa2 = (x1-x2).std()**2
        guessd = 0.5
        c_guess = (guessa1-guessa2)/guessa2
        print(guessa1,guessa2,guessd,c_guess/(2+c_guess))
        result = root(phi_deriv, [guessa1,guessa2,guessd],args=para)

        a1 = result.x[0]
        a2 = result.x[1]
        d = result.x[2]

        b1 = b(d,a1,delta_t)
        b2 = b(d,a2,delta_t)
        a1ep,a1ss,a1c = calc_fundstats(x1+x2)
        a2ep,a2ss,a2c = calc_fundstats(x1-x2)
        d2PdA2_1m = d2PdA2(N,a1ep,a1ss,a1c,b1,d,a1,delta_t)
        d2PdA2_2m = d2PdA2(N,a2ep,a2ss,a2c,b2,d,a2,delta_t)
        d2PdD2m = d2PdD2(N,a1ep,a1ss,a1c,a2ep,a2ss,a2c,b1,b2,d,a1,a2,delta_t)
        d2PdAdD_1m = d2PdAdD(N,a1ep,a1ss,a1c,b1,d,a1,delta_t)
        d2PdAdD_2m = d2PdAdD(N,a2ep,a2ss,a2c,b2,d,a2,delta_t)

        jacob = np.array([[d2PdA2_1m,0,d2PdAdD_1m],[0,d2PdA2_2m,d2PdAdD_2m],[d2PdAdD_1m,d2PdAdD_2m,d2PdD2m]])
        var = -np.linalg.inv(jacob)

        da1 = np.sqrt(var[0,0])
        da2 = np.sqrt(var[1,1])
        dd = np.sqrt(var[2,2])

        y1 = x1 + x2
        y2 = x1 - x2
        with pm.Model() as model:
            A1 = pm.Uniform('A1', lower=0, upper=a_bound)
            A2 = pm.Uniform('A2', lower=0, upper=a_bound)
            D = pm.Uniform('D',lower=0,upper=5)
            
            B1 = pm.Deterministic('B1',pm.math.exp(-delta_t * D / A1))
            B2 = pm.Deterministic('B2',pm.math.exp(-delta_t * D / A2))
                                
            path1 = Ornstein_Uhlenbeck('path1',A=A1, B=B1,shape=len(y1),observed=y1)
            path2 = Ornstein_Uhlenbeck('path2',A=A2, B=B2,shape=len(y2),observed=y2)
                                
            trace = pm.sample(10000,tune=2000)

        A1_trace = trace['A1']
        A2_trace = trace['A2']
        A1_mean = np.mean(A1_trace)
        A2_mean = np.mean(A2_trace)
        dA1 = np.std(A1_trace)
        dA2 = np.std(A2_trace)
        if A1_mean>A2_mean:
            C_trace = (A1_trace-A2_trace)/A2_trace
        else:
            C_trace = (A1_trace-A2_trace)/A1_trace

        C_mean = np.mean(C_trace)
        dC = np.std(C_trace)

        D_trace = trace['D']
        D_mean = np.mean(D_trace)
        dD = np.std(D_trace)

        print("predicted C: ",C_mean," +- ",dC)

        if results is None:
            results = [rho,prho,C_mean,dC, A1_mean,dA1,A2_mean,dA2,D_mean,dD,a1,da1,a2,da2,d,dd]
        else:
            results = np.vstack((results,[rho,prho,C_mean,dC, A1_mean,dA1,A2_mean,dA2,D_mean,dD,a1,da1,a2,da2,d,dd]))

column_names = ["rho","prho","C","dC","A1","dA1","A2","dA2","D","dD","a1","da1","a2","da2","d","dd"]
df=pd.DataFrame(results,columns=column_names)
print(df)
df.to_csv('correlations1k025.csv',index=False)
