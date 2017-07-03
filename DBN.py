import torch
import numpy
from torch.autograd import Variable
import RBM
from torch.utils import data as dtf

class DBN(object):
    def __init__(self, n_visible = 1600, n_hidden = [32,16], W = None, v_bias = None, h_bias=None,
                 trained = False):
        self.rbm_layers = []
        self.n_layers = len(n_hidden)
        
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_visible
            else:
                input_size = n_hidden[i-1]
            if trained:
                rbm = RBM.RBM(n_visible = input_size, 
                              n_hidden = n_hidden[i],
                              W = W[i],
                              h_bias = h_bias[i],
                              v_bias = v_bias[i])
            else:
                rbm = RBM.RBM(n_visible = input_size, 
                              n_hidden = n_hidden[i])
            self.rbm_layers.append(rbm)
    
    def greedy_train(self, lr = [1e-2, 1e-2], epoch = [100,100], batch_size = [50, 50], input_data = None, 
                     CD_k = 1, optimization_method = None, momentum = 0, gradient = False, L2_penalty = 0):
        
        for ith_rbm in range(self.n_layers):
            #print("training rbm %i" %ith_rbm)
            if ith_rbm:
                input_data = self.rbm_layers[ith_rbm-1].sample_h_given_v(Variable(input_data,requires_grad = False),
                                                                W = self.rbm_layers[ith_rbm-1].W,
                                                                h_bias = self.rbm_layers[ith_rbm-1].h_bias)[0].data
                #print("rbm %i data ready" %ith_rbm)
             
            self.rbm_layers[ith_rbm].train(lr = lr[ith_rbm], epoch = epoch[ith_rbm], batch_size = batch_size[ith_rbm], 
                                          input_data = input_data, CD_k = CD_k, optimization_method = optimization_method,
                                          momentum = momentum, gradient = gradient, L2_penalty = L2_penalty)
            
    def generate(self, iteration = 1):
        
        v_sample = self.rbm_layers[-1].generate(iteration = iteration)
        for i in range(self.n_layers-1):
            v_sample = self.rbm_layers[-2-i].sample_v_given_h(v_sample, self.rbm_layers[-2-i].W, self.rbm_layers[-2-i].h_bias)[0]
        return v_sample