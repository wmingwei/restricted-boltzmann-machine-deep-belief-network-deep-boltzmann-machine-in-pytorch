import torch
import numpy as np
from torch.autograd import Variable
from joblib import Parallel, delayed
import multiprocessing


def logp(dbm, v, step = 1000, M_Z = 100, M_IS = 100, k = 100, parallel = False, seed = None, mean_logp = True):
    W = [w.data.numpy().T for w in dbm.W]
    bias = [b.data.numpy().T for b in dbm.bias]
    if mean_logp:
        return np.mean(ulogprob(v = v, W = W, bias = bias, M = M_IS, k = k, parallel = parallel, seed = seed))-ais(W = W, bias = bias, step = step, M = M_Z, parallel = parallel, seed = seed)
    else:
        return ulogprob(v=v, W = W, bias = bias, M = M_IS, k = k, parallel = parallel, seed = seed)-ais(W = W, bias = bias, step = step, M = M_Z, parallel = parallel, seed = seed)        

def ais(W, bias, step = 100, M = 100, parallel = False, seed = None):
    
    logZ0 = 0
    for i in bias:
        logZ0 += np.log((1+np.exp(i))).sum()
    ratio = []
    if parallel:
        num_cores = multiprocessing.cpu_count()

        results = Parallel(n_jobs=num_cores)(delayed(mcmc)(step = step, seed = seed, W = W, bias = bias) for i in range(M))


        results = np.array(results).reshape(len(results), 1)
        logZ = logZ0 + logmeanexp(results, axis = 0)
    else:
        for i in range(M):
            ratio.append(mcmc(step, seed = seed,  W = W, bias = bias))

        ratio = np.array(ratio).reshape(len(ratio),1)
        logZ = logZ0 + logmeanexp(ratio, axis = 0)

    return logZ

def mcmc(step, seed, W, bias):

    np.random.seed(seed)

    even_layer = [np.random.binomial(1, p=1/(1+np.exp(-bias[i]))).reshape(1,-1) for i in range(len(bias)) if i%2 ==0]

    logw = 0
    for k in range(step):
        logp_k = -free_energy(even_layer, [k*1.0/step*w for w in W], bias)
        logp_k1 = -free_energy(even_layer, [(k+1)*1.0/step*w for w in W], bias)
        logw += logp_k1 - logp_k

        
        p_odd_layer, odd_layer = even_to_odd(even_layer, [(k+1)*1.0/step*w for w in W], bias)
        p_even_layer, even_layer = odd_to_even(odd_layer, [(k+1)*1.0/step*w for w in W], bias)

    return logw
def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -80,80)))

def odd_to_even(odd_input, W, bias):
    even_p_output= []
    even_output= []
    for i in range(len(bias) - len(odd_input)):
        if i == 0:
            even_p_output.append(sigmoid(np.matmul(odd_input[i],W[2*i].T)+bias[2*i]))
        elif (len(bias) - len(odd_input) > len(odd_input)) and i == len(odd_input):
            even_p_output.append(sigmoid(np.matmul(odd_input[i-1],W[2*i-1]) + bias[2*i]))
        else:
            even_p_output.append(sigmoid(np.matmul(odd_input[i-1],W[2*i-1])+ bias[2*i] + np.matmul(odd_input[i],W[2*i].T)))
    for i in even_p_output:
        even_output.append(np.random.binomial(1, p=i))

    return even_p_output, even_output

def even_to_odd(even_input, W, bias):
    odd_p_output = [] 
    odd_output = []
    for i in range(len(bias) - len(even_input)):
        if (len(even_input) == len(bias) - len(even_input)) and i == len(even_input) - 1:
            odd_p_output.append(sigmoid(np.matmul(even_input[i],W[2*i])+bias[2*i+1]))
        else:
            odd_p_output.append(sigmoid(np.matmul(even_input[i],W[2*i]) + bias[2*i+1] + np.matmul(even_input[i+1], W[2*i+1].T)))

    for i in odd_p_output:
        odd_output.append(np.random.binomial(1, p=i))

    return odd_p_output, odd_output

def free_energy(even_layer, W, bias):
    
    bias_term = 0
    
    for i in range(len(even_layer)):
        bias_term += np.matmul(even_layer[i], bias[2*i].T)
        
    hidden_term = 0
    
    for i in range(len(bias)-len(even_layer)):
        wx_b = np.clip(np.matmul(even_layer[i], W[2*i]) + bias[2*i+1] + np.matmul(even_layer[i+1], W[2*i+1].T), -80, 80)
        hidden_term += np.log(1+np.exp(wx_b)).sum(1)
    bias_term = bias_term.reshape(hidden_term.shape)
    return -hidden_term - bias_term

def ulogprob(v, W, bias, M = 1000, k = 60, parallel = False, seed = None):
    logw = np.zeros([M, len(v)])
    
    if not parallel:
        for i in range(M):
            logw[i,:] += important_sampling(v, W, bias, k = k)
    else:
        num_cores = multiprocessing.cpu_count()

        results = Parallel(n_jobs=num_cores)(delayed(important_sampling)(v = v, W = W, bias = bias, k = k, seed = seed) for i in range(M))
        logw += np.asarray(results)
    return logmeanexp(logw,0)

def important_sampling(v, W, bias, k, seed = None):
    
    np.random.seed(seed)
    
    logw = np.zeros(len(v))
    even_layer = [v]
    for i in range(1,int(len(bias)/2)+1):
        even_layer.append(np.random.binomial(1,p = sigmoid(bias[2*i]).reshape(1,-1).repeat(len(v),axis = 0)))
    # [print(i.size()) for i in even_layer]
    for _ in range(k):
        p_odd_layer, odd_layer = even_to_odd(even_layer, W, bias)
        p_even_layer, even_layer = odd_to_even(odd_layer, W, bias)
        even_layer[0] = v
    
    logw += -free_energy(even_layer, W, bias)
    
    for i in range(1,len(even_layer)):
        logw -= np.log(np.abs(-even_layer[i]+1-p_even_layer[i])).sum(1)
        
    return logw


def logmeanexp(x, axis=None):
    
    x = np.asmatrix(x)
    if not axis:
        n = len(x)
    else:
        n = x.shape[axis]
    
    x_max = x.max(axis)
    return (x_max + np.log(np.exp(x-x_max).sum(axis)) - np.log(n)).A