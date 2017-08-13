import torch
import torch.autograd as autograd
import numpy as np
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from joblib import Parallel, delayed
import multiprocessing

def greedy_train(dbn, lr = [1e-3, 1e-4], epoch = [100, 100], batch_size = 50, input_data = None, weight_decay = [0,0], L1_penalty = [0,0], CD_k = 10, test_set = None):
    
    train_set = torch.utils.data.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    
    optimizers = [optim.Adam(dbn.rbm_layers[i].parameters(), lr = lr[i], weight_decay = weight_decay[i]) for i in range(dbn.n_layers)]
    
    for i in range(epoch[0]):
        
        for batch_idx, (data, target) in enumerate(train_loader):
            input_data = Variable(data)
            
            v_in, v_out = dbn(input_data, greedy = True, CD_k = CD_k)
            
            for i in range(dbn.n_layers):
                loss = dbn.rbm_layers[i].free_energy(v_in[i].detach()) - dbn.rbm_layers[i].free_energy(v_out[i].detach())
                loss.backward()
                
                optimizers[i].step()
            
                optimizers[i].zero_grad()
            
        if not type(test_set) == type(None):
            print("epoch %i: "%i, reconstruct_error(rbm, Variable(test_set)))
            

def generate(dbn, iteration = 1, prop_input = None):
    
    if not type(prop_input) == type(None):
        prop_v = Variable(torch.from_numpy(prop_input).type(torch.FloatTensor))
        for i in range(dbn.n_layers-1):
            prop_v = dbn.rbm_layers[i].v_to_h(prop_v)[0]
        prop = prop_v.data.mean()
    else:
        prop = 0.5
        
    v = torch.bernoulli(dbn.rbm_layers[-1].v_bias *0 + prop)
    for _ in range(iteration):
        
        p_h, h = dbn.rbm_layers[-1].v_to_h(v)
        
        p_v, v = dbn.rbm_layers[-1].h_to_v(h)
        
    for i in range(dbn.n_layers-1):
        p_v, v = dbn.rbm_layers[-2-i].h_to_v(v)
        
    return v