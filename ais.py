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
        
        Wv = np.clip(np.matmul(v_sample,W) + h_bias,-80,80)
        hidden = np.log(1+np.exp(Wv)).sum(1)
        vbias = np.matmul(v_sample, self.v_bias.T).reshape(len(hidden))
        return -hidden-vbias

    
    def sample_h_given_v(self, v0_sample, W,h_bias):
        
        activation = np.clip(np.matmul(v0_sample,W) + h_bias,-80,80)
        h1_mean = 1/(1+np.exp(-activation))
        h1_sample = np.random.binomial(1, p=h1_mean)

        return [h1_sample, h1_mean]

    
    def sample_v_given_h(self, h0_sample, W, h_bias):

        activation = np.clip(np.matmul(h0_sample, W.T) + self.v_bias,-80,80)
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
        
    
    def ais(self, step = 100, M = 100, parallel = False, seed = None):

        logZ0 = np.log((1+np.exp(self.v_bias))).sum() + np.log(1+np.exp(self.h_bias)).sum()
        ratio = []
        if parallel:
            num_cores = multiprocessing.cpu_count()

            results = Parallel(n_jobs=num_cores)(delayed(self.mcmc)(step = step, seed = seed) for i in range(M))
            
           
            results = np.array(results).reshape(len(results), 1)
            logZ = logZ0 + logmeanexp(results, axis = 0)
        else:
            for i in range(M):
                ratio.append(self.mcmc(step, seed = seed))
                
            ratio = np.array(ratio).reshape(len(ratio),1)
            logZ = logZ0 + logmeanexp(ratio, axis = 0)
                
        return logZ
    def mcmc(self, step, seed):
        
        np.random.seed(seed)
        
        v = np.random.binomial(1, p=1/(1+np.exp(-self.v_bias)))
        
        logw = 0
        for k in range(step):
            logp_k = -self.free_energy(v, k*1.0/step*self.W, self.h_bias)
            logp_k1 = -self.free_energy(v, (k+1)*1.0/step*self.W, self.h_bias)
            logw += logp_k1 - logp_k
            

            v= self.gibbs_vhv(v, (k+1)*1.0/step*self.W, self.h_bias)[1]


        return logw

def logp_ais(trained_model, v_input, step = 1000, M = 100, parallel = False):
	W = trained_model.W.data.numpy()
	v_bias = trained_model.v_bias.data.numpy()
	h_bias = trained_model.h_bias.data.numpy()
	n_visible, n_hidden = W.shape
	rbm = RBM(n_visible = n_visible, n_hidden = n_hidden, W = W, v_bias = v_bias, h_bias = h_bias)
	return -np.mean(rbm.free_energy(v_input, W, h_bias))-rbm.ais(step = step, M = M, parallel = parallel)

def logp_var(trained_model, v_input, step = 1000, M = 100, parallel = False):
	W = trained_model.W.data.numpy()
	v_bias = trained_model.v_bias.data.numpy()
	h_bias = trained_model.h_bias.data.numpy()
	n_visible, n_hidden = W.shape
	rbm = RBM(n_visible = n_visible, n_hidden = n_hidden, W = W, v_bias = v_bias, h_bias = h_bias)
	return np.var(-rbm.free_energy(v_input, W, h_bias)-rbm.ais(step = step, M = M, parallel = parallel))

def get_rbm(trained_model):
	W = trained_model.W.data.numpy()
	v_bias = trained_model.v_bias.data.numpy()
	h_bias = trained_model.h_bias.data.numpy()
	n_visible, n_hidden = W.shape
	rbm = RBM(n_visible = n_visible, n_hidden = n_hidden, W = W, v_bias = v_bias, h_bias = h_bias)
	return rbm

def get_partition_function(trained_model, step = 1000, M = 100, parallel = False):
	W = trained_model.W.data.numpy()
	v_bias = trained_model.v_bias.data.numpy()
	h_bias = trained_model.h_bias.data.numpy()
	n_visible, n_hidden = W.shape
	rbm = RBM(n_visible = n_visible, n_hidden = n_hidden, W = W, v_bias = v_bias, h_bias = h_bias)
	return rbm.ais(step = step, M = M, parallel = parallel)

def logmeanexp(x, axis=None):
    
    x = np.asmatrix(x)
    if not axis:
        n = len(x)
    else:
        n = x.shape[axis]
    
    x_max = x.max(axis)
    return (x_max + np.log(np.exp(x-x_max).sum(axis)) - np.log(n)).A
