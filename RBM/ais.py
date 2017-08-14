import torch
import numpy as np
from torch.autograd import Variable
from joblib import Parallel, delayed
import multiprocessing


def logp(dbn, v, step = 1000, M_Z = 100, M_IS = 100, parallel = False):
    
    return np.mean(ulogprob(v, dbn, M = M_IS, parallel = parallel))-ais(dbn.rbm_layers[-1], step = step, M = M_Z, parallel = parallel)

def ais(rbm, step = 100, M = 100, parallel = False, seed = None):

    W = rbm.W.data.numpy().T
    v_bias = rbm.v_bias.data.numpy()
    h_bias = rbm.h_bias.data.numpy()
    
    logZ0 = np.log((1+np.exp(v_bias))).sum() + np.log(1+np.exp(h_bias)).sum()
    ratio = []
    if parallel:
        num_cores = multiprocessing.cpu_count()

        results = Parallel(n_jobs=num_cores)(delayed(mcmc)(step = step, seed = seed, W = W, h_bias = h_bias, v_bias = v_bias) for i in range(M))


        results = np.array(results).reshape(len(results), 1)
        logZ = logZ0 + logmeanexp(results, axis = 0)
    else:
        for i in range(M):
            ratio.append(mcmc(step, seed = seed,  W = W, h_bias = h_bias, v_bias = v_bias))

        ratio = np.array(ratio).reshape(len(ratio),1)
        logZ = logZ0 + logmeanexp(ratio, axis = 0)

    return logZ

def mcmc(step, seed, W, h_bias, v_bias):

    np.random.seed(seed)

    v = np.random.binomial(1, p=1/(1+np.exp(-v_bias))).reshape(1,-1)

    logw = 0
    for k in range(step):
        logp_k = -free_energy(v, k*1.0/step*W, h_bias, v_bias)
        logp_k1 = -free_energy(v, (k+1)*1.0/step*W, h_bias, v_bias)
        logw += logp_k1 - logp_k

        
        p_h, h = v_to_h(v, (k+1)*1.0/step*W, h_bias)
        p_v, v = h_to_v(h, (k+1)*1.0/step*W, v_bias)

    return logw


def v_to_h(v, W, h_bias):

    activation = np.clip(np.matmul(v,W) + h_bias,-80,80)
    p_h = 1/(1+np.exp(-activation))
    h = np.random.binomial(1, p=p_h)

    return p_h, h


def h_to_v(h, W, v_bias):

    activation = np.clip(np.matmul(h, W.T) + v_bias,-80,80)
    p_v = 1/(1+np.exp(-activation))
    v = np.random.binomial(1, p=p_v)
    return p_v, v

def free_energy(v, W, h_bias, v_bias):

    Wv = np.clip(np.matmul(v,W) + h_bias,-80,80)
    hidden = np.log(1+np.exp(Wv)).sum(1)
    vbias = np.matmul(v, v_bias.T).reshape(hidden.shape)
    return -hidden-vbias

def ulogprob(v, dbn, M = 1000, parallel = False):
    logw = np.zeros([M, len(v_input)])
    # samples = v_input
    if not parallel:
        for i in range(M):
            logw[i,:] += important_sampling(v, dbn)
    else:
        num_cores = multiprocessing.cpu_count()

        results = Parallel(n_jobs=num_cores)(delayed(important_sampling)(v = v, dbn = dbn) for i in range(M))
        logw += np.asarray(results)
           
    return logmeanexp(logw,0)

def important_sampling(v, dbn):
    
    logw = np.zeros(len(v))
    
    for l in range(dbn.n_layers-1):
        
        W = dbn.rbm_layers[l].W
        h_bias = dbn.rbm_layers[l].h_bias
        v_bias = dbn.rbm_layers[l].v_bias
        
        logw += -free_energy(v, W, h_bias, v_bias)
        v = v_to_h(v, W, h_bias, v_bias)[1]
        logw -= -free_energy_hidden(v, W.T, v_bias, h_bias)
        
    logw += -free_energy(v, dbn.rbm_layers[-1].W, dbn.rbm_layers[-1].h_bias, dbn.rbm_layers[-1].v_bias)
    return logw


def logmeanexp(x, axis=None):
    
    x = np.asmatrix(x)
    if not axis:
        n = len(x)
    else:
        n = x.shape[axis]
    
    x_max = x.max(axis)
    return (x_max + np.log(np.exp(x-x_max).sum(axis)) - np.log(n)).A