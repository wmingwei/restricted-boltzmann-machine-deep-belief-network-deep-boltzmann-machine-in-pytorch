import torch
import torch.autograd as autograd
import numpy as np
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from joblib import Parallel, delayed
import multiprocessing

def generate(rbm, iteration = 1, p = 0.5):
    
    v = torch.bernoulli(rbm.v_bias *0 + prop)
    for _ in range(iteration):
        
        p_h, h = rbm.v_to_h(v)
        
        p_v, v = rbm.h_to_v(h)
        
    return v


def reconstruct_error(rbm, v):
    p_h,h = rbm.v_to_h(v)
    p_v,recon_v = rbm.h_to_v(h)

    return torch.mean(torch.abs((v-recon_v))**2)


def train(rbm, lr = 1e-3, epoch = 100, batch_size = 50, input_data = None, weight_decay = 0, L1_penalty = 0, test_set = None, CD_k = 10):
    
    train_set = torch.utils.data.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    
    optimizer = optim.Adam(rbm.parameters(), lr = lr, weight_decay = weight_decay)
    
    for i in range(epoch):
        
        for batch_idx, (data, target) in enumerate(train_loader):
            input_data = Variable(data, CD_k = CD_k)
            
            v, v_ = rbm(input_data)
            
            loss = rbm.free_energy(v) - rbm.free_energy(v_.detach())
            
            loss.backward()
            
            optimizer.step()
            
            optimizer.zero_grad()
            
        if not type(test_set) == type(None):
            print("epoch %i: "%i, reconstruct_error(rbm, Variable(test_set)))
            