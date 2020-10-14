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
a_bound=2
M=5
N=10000

results = None
for rho in np.arange(-0.9,1.0,0.1):
    for i in range(M):
        delta_t = 0.1
        coupling = 2*np.abs(rho)/(1-np.abs(rho))*np.sign(rho)
        x1,x2 = correlated_ts(coupling,N=N)
        prho = pearsonr(x1,x2)[0]
        print("OU cross correlation", OUcross(x1,x2,delta_t))
        print("pearson: ",prho)

        y1 = x1 + x2
        y2 = x1 - x2
        with pm.Model() as model:
            A1 = pm.Uniform('A1', lower=0, upper=a_bound)
            A2 = pm.Uniform('A2', lower=0, upper=a_bound)
            D = pm.Uniform('D',lower=0,upper=3)
            
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

        print("predicted C: ",C_mean," +- ",dC)

        if results is None:
            results = [rho,prho,C_mean,dC]
        else:
            results = np.vstack((results,[rho,prho,C_mean,dC]))

column_names = ["rho","prho","C","dC"]
df=pd.DataFrame(results,columns=column_names)
print(df)
df.to_csv('correlations10k.csv',index=False)
