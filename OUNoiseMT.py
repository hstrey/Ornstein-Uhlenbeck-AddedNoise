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
    parser.add_argument('-o', '--output', action='store', default="./test",
                        help='output file name')
    parser.add_argument('-i', '--input', action='store', default="OUmt_sN05.npy",
                        help='input file name')
    parser.add_argument('-n', '--length', action='store', type=int, default=100,
                        help='how many records')
    parser.add_argument('-s', '--start', action='store', type=int, default=0,
                        help='starting record')

    
    # get arguments
    arg = parser.parse_args()
    
    outfilename = arg.output
    infilename = arg.input
    l = arg.length
    s = arg.start
    
    data = np.load(infilename)
    data = data[s:s+l]
    
    a_bound = 10
    result_df = pd.DataFrame(columns=['dt','A', 'dA','B','dB','s','ds'])
    for dataset in data:
        delta_t = dataset[0]
        ts = dataset[1:]
        print(delta_t)
        with pm.Model() as model:
            B = pm.Beta('B', alpha=1.0,beta=1.0)
            A = pm.Uniform('A', lower=0, upper=a_bound)
            sigma = pm.Uniform('sigma',lower=0,upper=5)

            path = Ornstein_Uhlenbeck('path',A=A, B=B,shape=len(ts))
            dataObs = pm.Normal('dataObs',mu=path,sigma=sigma,observed=ts)
            trace = pm.sample(2000,cores=4)
    
        a_mean = trace['A'].mean()
        b_mean = trace['B'].mean()
        a_std = trace['A'].std()
        b_std = trace['B'].std()
        sigma_mean = trace['sigma'].mean()
        sigma_std = trace['sigma'].std()
        result_dict = {'dt':delta_t,
                          'A':a_mean,
                          'dA':a_std,
                          'B':b_mean,
                          'dB':b_std,
                          's':sigma_mean,
                          'ds':sigma_std}
        print(result_dict)
        result_df = result_df.append(result_dict,ignore_index=True)
    print(result_df)                      
    result_df.to_csv(outfilename+'.csv',index=False)

if __name__ == "__main__":
    main()