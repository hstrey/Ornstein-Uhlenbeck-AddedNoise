# this program is going to estimate parameters from simulated datasets that originate from an OU process
# with noise added.  The parameters of the simulation and the length of the simulation are set through
# arguments
import langevin
import pandas as pd
import numpy as np
import argparse
import pymc3 as pm
import theano.tensor as tt

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--output', action='store', default="./test",
                        help='output file name')
    parser.add_argument('-a', '--amplitude', action='store', type=float, default=1.0,
                        help='amplitude of OU')
    parser.add_argument('-r', '--tau', action='store', type=float, default=1.0,
                        help='relaxation time tau of OU')
    parser.add_argument('-w', '--noise', action='store', type=float, default=1.0,
                        help='power of the noise')
    parser.add_argument('-n', '--length', action='store', type=int, default=2000,
                        help='length of simulation in timesteps')
    parser.add_argument('-s', '--samples', action='store', type=int, default=2000,
                        help='MCMC samples per run')
    parser.add_argument('-b', '--dtstart', action='store', type=float, default=0.01,
                        help='delta t range start')
    parser.add_argument('-e', '--dtend', action='store', type=float, default=2.0,
                        help='delta t range end')
    parser.add_argument('-l', '--dtlength', action='store', type=int, default=50,
                        help='delta t range number of points')
    parser.add_argument('-u', '--duplicates', action='store', type=int, default=1,
                        help='delta t range number of points')

    
    # get arguments
    arg = parser.parse_args()
    
    filename = arg.output
    a = arg.amplitude
    tau = arg.tau
    pn = arg.noise
    n = arg.length
    mc_samples = arg.samples
    dtstart = arg.dtstart
    dtend = arg.dtend
    dtlength = arg.dtlength
    dupl = arg.duplicates
    
    delta_t_list = np.linspace(dtstart,dtend,dtlength)

    result_array = None
    for delta_t in delta_t_list:
        print(delta_t)
        for i in range(dupl):
            data = langevin.time_series(A=a, D=a/tau, delta_t=delta_t, N=n)
            dataN = data + np.random.normal(loc=0.0, scale=np.sqrt(pn), size=n)
            with pm.Model() as model:
                B = pm.Beta('B', alpha=5.0,beta=1.0)
                A = pm.Uniform('A', lower=0, upper=5)
                sigma = pm.Uniform('sigma',lower=0,upper=5)
            
                path = Ornstein_Uhlenbeck('path',A=A, B=B,shape=len(dataN))
                dataObs = pm.Normal('dataObs',mu=path,sigma=sigma,observed=dataN)
                trace = pm.sample(mc_samples,cores=4)
        
            a_mean = trace['A'].mean()
            b_mean = trace['B'].mean()
            a_std = trace['A'].std()
            b_std = trace['B'].std()
            sigma_mean = trace['sigma'].mean()
            sigma_std = trace['sigma'].std()
            avgpath = np.mean(trace['path'],axis=0)
            stddiff = np.std(data-avgpath)
            stdpath = np.std(trace['path'],axis=0).mean()
        
            results = [delta_t,a_mean,a_std,b_mean,b_std,sigma_mean,sigma_std,stddiff,stdpath]
            if result_array is None:
                result_array = results
            else:
                result_array = np.vstack((result_array, results))
    
    column_names = ["delta_t","a_mean","a_std","b_mean","b_std","sigma_mean","sigma_std","stddiff","stdpath"]
    df=pd.DataFrame(result_array,columns=column_names)
    df.to_csv(filename+'.csv',index=False)

if __name__ == "__main__":
    main()