import torch
import numpy as np
from torch.autograd import Variable
from joblib import Parallel, delayed
import multiprocessing

class RBM(object):
    
    def __init__(self, n_visible = 784, n_hidden = 500, W = None, v_bias = None, 
                 h_bias = None, batch_size = 0):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        
        if not W.any():
            initial_W = np.asarray(
                np.random.normal(loc = 0, scale = 1/n_visible,
                    size=(n_visible, n_hidden)
                    ),
                )
            W = initial_W
            
        if not v_bias.any():
            v_bias = np.zeros((1,n_visible))

        if not h_bias.any():
            h_bias = np.zeros((1,n_hidden))
            
        self.W = W
        self.v_bias = v_bias
        self.h_bias = h_bias


    def free_energy(self, v_sample, W, h_bias):
        num = len(v_sample)
        Wv = np.clip(np.matmul(v_sample,W) + h_bias,-80,80)
        hidden = np.log(1+np.exp(Wv)).sum(1)
        vbias = np.matmul(v_sample, self.v_bias.T).reshape(len(hidden))
        return -hidden.reshape(num)-vbias.reshape(num)

    
    def sample_h_given_v(self, v0_sample, W,h_bias):
        
        activation = np.matmul(v0_sample,W) + h_bias
        h1_mean = 1/(1+np.exp(-activation))
        h1_sample = np.random.binomial(1, p=h1_mean)

        return [h1_sample, h1_mean]

    
    def sample_v_given_h(self, h0_sample, W, h_bias):

        activation = np.matmul(h0_sample, W.T) + self.v_bias
        v1_mean = 1/(1+np.exp(-activation))
        v1_sample = np.random.binomial(1, p=v1_mean)
        return [v1_sample, v1_mean]

    
    def gibbs_hvh(self, h0_sample, W, h_bias):
        v1_sample, v1_mean = self.sample_v_given_h(h0_sample, W, h_bias)
        h1_sample, p_h1 = self.sample_h_given_v(v1_sample, W, h_bias)
        
        return [v1_sample, h1_sample, p_h1] 
    
    def gibbs_vhv(self, v0_sample, W, h_bias):
        h1_sample, h1_mean = self.sample_h_given_v(v0_sample, W, h_bias)
        v1_sample, p_v1 = self.sample_v_given_h(h1_sample, W, h_bias)
        
        return [h1_sample, v1_sample, p_v1]
        

    def rais(self, data, step = 1000, M = 100, parallel = False, seed = None):
        num_data = data.shape[0]
        result = 0
        if not parallel:
            p = []
            for i in range(M):
                logw = self.mcmc_r(data, step, num_data)
                p.append(logw)
            
            p = np.array(p)
            logmeanp = logmeanexp(p, axis = 0)
        else:
            num_cores = multiprocessing.cpu_count()

            p = Parallel(n_jobs=num_cores)(delayed(self.mcmc_r)(v = data, step = step, num_data = num_data, seed = seed) for i in range(M))
            
            p = np.array(p)
            
            logmeanp = logmeanexp(p, axis = 0)
            
        result = logmeanp.mean()
        
        return result
        
    def mcmc_r(self, v, step, num_data, seed = None):
        np.random.seed(seed)
        logZ0 = np.log((1+np.exp(self.v_bias))).sum() + np.log(1+np.exp(self.h_bias)).sum()        
        #h = self.sample_h_given_v(v, self.W, self.h_bias)
        logw = -self.free_energy(v,self.W,self.h_bias) - logZ0
        for k in range(step-1,-1,-1):
            a,v,c = self.gibbs_vhv(v, (k)*1.0/step*self.W, self.h_bias)
            logp_k = -self.free_energy(v, k*1.0/step*self.W, self.h_bias)
            logp_k1 = -self.free_energy(v, (k+1)*1.0/step*self.W, self.h_bias)  
            logw += logp_k - logp_k1
           
        return logw.reshape(num_data)
        
def rais(trained_model, data, step = 1000, M = 100, parallel = False):
	W = trained_model.W.data.numpy()
	v_bias = trained_model.v_bias.data.numpy()
	h_bias = trained_model.h_bias.data.numpy()
	n_visible, n_hidden = W.shape
	rbm = RBM(n_visible = n_visible, n_hidden = n_hidden, W = W, v_bias = v_bias, h_bias = h_bias)
	return rbm.rais(data, step = step, M = M, parallel = parallel)


def logmeanexp(x, axis=None):
    
    x = np.asmatrix(x)
    if not axis:
        n = len(x)
    else:
        n = x.shape[axis]
    
    x_max = x.max(axis)
    return (x_max + np.log(np.exp(x-x_max).sum(axis)) - np.log(n)).A
