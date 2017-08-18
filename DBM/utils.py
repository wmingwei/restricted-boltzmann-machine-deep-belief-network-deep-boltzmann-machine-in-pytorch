import torch
import torch.autograd as autograd
import numpy as np
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from joblib import Parallel, delayed
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F

def greedy_train(dbm, lr = [1e-3, 1e-4], epoch = [100, 100], batch_size = 50, input_data = None, weight_decay = [0,0], L1_penalty = [0,0], CD_k = 10, test_set = None, initialize_v = False):
    
    train_set = torch.utils.data.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    
    # optimizers = [optim.Adam(dbn.rbm_layers[i].parameters(), lr = lr[i], weight_decay = weight_decay[i]) for i in range(dbn.n_layers)]
    
    for i in range(dbm.n_layers):
        print("Training the %ith layer"%i)
        optimizer = optim.Adam(dbm.rbm_layers[i].parameters(), lr = lr[i], weight_decay = weight_decay[i])
        if initialize_v:
            v = Variable(input_data)
            for ith in range(i):
                p_v, v = dbm.rbm_layers[ith].v_to_h(v)
            dbm.rbm_layers[i].v_bias.data.zero_()
            dbm.rbm_layers[i].v_bias.data.add_(torch.log(v.mean(0)/(1-v.mean(0))).data)
        for _ in range(epoch[i]):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = Variable(data)
                v, v_ = dbm(v_input = data, greedy = True, ith_layer = i, CD_k = CD_k)
                
                loss = dbm.rbm_layers[i].free_energy(v.detach()) - dbm.rbm_layers[i].free_energy(v_.detach()) + L1_penalty[0] * torch.sum(torch.abs(dbm.rbm_layers[i].W))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        if i > 0:
            dbm.bias[i].data = (dbm.rbm_layers[i-1].h_bias.data + dbm.rbm_layers[i].v_bias.data)/2
                
#     for _ in range(epoch[0]):
        
#         for batch_idx, (data, target) in enumerate(train_loader):
#             input_data = Variable(data)
            
#             v_in, v_out = dbn(input_data, CD_k = CD_k)
            
#             for i in range(dbn.n_layers):
#                 loss = dbn.rbm_layers[i].free_energy(v_in[i].detach()) - dbn.rbm_layers[i].free_energy(v_out[i].detach())
#                 loss.backward()
                
#                 optimizers[i].step()
            
#                 optimizers[i].zero_grad()

def joint_train(dbm, lr = 1e-3, epoch = 100, batch_size = 50, input_data = None, weight_decay = 0, k_positive=10, k_negative=10, alpha = [1e-1,1e-1,1]):
    u1 = nn.Parameter(torch.zeros(1))
    u2 = nn.Parameter(torch.zeros(1))
    optimizer = optim.Adam(dbm.parameters(), lr = lr, weight_decay = weight_decay)
    train_set = torch.utils.data.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    optimizer_u = optim.Adam([u1,u2], lr = lr/1000, weight_decay = weight_decay)
    for _ in range(epoch):
        print("training epoch %i with u1 = %.4f, u2 = %.4f"%_%u1%u2)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data)
            positive_phase, negative_phase= dbm(v_input = data, k_positive = k_positive, k_negative=k_negative, greedy = False)
            loss = energy(dbm = dbm, layer = positive_phase) - energy(dbm = dbm, layer = negative_phase)+alpha[0] * torch.norm(torch.norm(dbm.W[0],2,1)-u1)**2 + alpha[1]*torch.norm(torch.norm(dbm2.W[1],2,1)-u2)**2 + alpha[2] * (u1 - u2)**2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            optimizer_u.step()
            optimizer_u.zero_grad()
            
            
def energy(dbm, layer):
    E = -F.linear(dbm.bias[0],layer[0]).sum()
    for i in range(dbm.n_layers):
        E -= F.linear(layer[i], dbm.W[i]).view(-1)@(layer[i+1].view(-1)) + F.linear(dbm.bias[i+1],layer[i+1]).sum()
    
    return E

def generate(dbm, iteration = 1, n = 1):
    even_layer = []
    odd_layer = []
    for i in range(0, dbm.n_odd_layers):
        odd_layer.append(torch.bernoulli((dbm.bias[2*i+1]*0+0.5).view(1,-1).repeat(n, 1)))
    for _ in range(iteration):
        p_even_layer, even_layer = dbm.odd_to_even(odd_layer)
        p_odd_layer, odd_layer = dbm.even_to_odd(even_layer)
    
    return even_layer[0]
        
    