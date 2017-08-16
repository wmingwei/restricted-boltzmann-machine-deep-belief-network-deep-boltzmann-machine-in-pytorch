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

def greedy_train(dbn, lr = [1e-3, 1e-4], epoch = [100, 100], batch_size = 50, input_data = None, weight_decay = [0,0], L1_penalty = [0,0], CD_k = 10, test_set = None, initialize_v = False):
    
    train_set = torch.utils.data.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    
    # optimizers = [optim.Adam(dbn.rbm_layers[i].parameters(), lr = lr[i], weight_decay = weight_decay[i]) for i in range(dbn.n_layers)]
    
    for i in range(dbn.n_layers):
        print("Training the %ith layer"%i)
        optimizer = optim.Adam(dbn.rbm_layers[i].parameters(), lr = lr[i], weight_decay = weight_decay[i])
        if initialize_v:
            v = Variable(input_data)
            for ith in range(i):
                p_v, v = dbn.rbm_layers[ith].v_to_h(v)
            dbn.rbm_layers[i].v_bias.data.zero_()
            dbn.rbm_layers[i].v_bias.data.add_(torch.log(v.mean(0)/(1-v.mean(0))).data)
        for _ in range(epoch[i]):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = Variable(data)
                v, v_ = dbn(v_input = data, ith_layer = i, CD_k = CD_k)
                
                loss = dbn.rbm_layers[i].free_energy(v.detach()) - dbn.rbm_layers[i].free_energy(v_.detach()) + L1_penalty[0] * torch.sum(torch.abs(dbn.rbm_layers[i].W))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
#     for _ in range(epoch[0]):
        
#         for batch_idx, (data, target) in enumerate(train_loader):
#             input_data = Variable(data)
            
#             v_in, v_out = dbn(input_data, CD_k = CD_k)
            
#             for i in range(dbn.n_layers):
#                 loss = dbn.rbm_layers[i].free_energy(v_in[i].detach()) - dbn.rbm_layers[i].free_energy(v_out[i].detach())
#                 loss.backward()
                
#                 optimizers[i].step()
            
#                 optimizers[i].zero_grad()
            
        if not type(test_set) == type(None):
            print("epoch %i: "%i, reconstruct_error(rbm, Variable(test_set)))

            

def generative_fine_tune(dbn, lr = 1e-2, epoch = 100, batch_size = 50, input_data = None, CD_k = 1, optimization_method = "Adam", momentum = 0, weight_decay = 0, test_input = None):
    
    if optimization_method == "RMSprop":
        optimizer = optim.RMSprop(dbn.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    elif optimization_method == "SGD":
        optimizer = optim.SGD(dbn.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    elif optimization_method == "Adam":
        optimizer = optim.Adam(dbn.parameters(), lr = lr, weight_decay = weight_decay)   
    
    for i in dbn.parameters():
        i.mean().backward()
        
    train_set = torch.utils.data.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)

    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):

            sleep_wake(dbn = dbn, optimizer = optimizer, lr = lr, CD_k = CD_k, v = data, batch_size = batch_size)

        if not (type(test_input) == type(None)):

            print("fine tune", i, ais_dbn.logp_ais(self, test_input, step = 1000, M_Z = 20, M_IS = 100, parallel = True))

def sleep_wake(dbn, optimizer, lr = 1e-2, CD_k = 10, v = None, batch_size = 1):

    #get wake s tates
    wake_states = [Variable(v)]

    for ith_rbm in range(dbn.n_layers):
        if ith_rbm < dbn.n_layers-1:
            p_wake_state, wake_state = v_to_h(wake_states[ith_rbm], dbn.W_rec[ith_rbm], dbn.bias_rec[ith_rbm])
        else:
            p_wake_state, wake_state = v_to_h(wake_states[ith_rbm], dbn.W_mem, dbn.h_bias_mem)

        wake_states.append(wake_state.detach())

    #CD_k    
    sleep_top = wake_states[-1]

    for cd in range(CD_k):
        
        p_sleep_bottom, sleep_bottom = h_to_v(sleep_top, dbn.W_mem, dbn.v_bias_mem)
        p_sleep_top, sleep_top = v_to_h(sleep_bottom, dbn.W_mem, dbn.h_bias_mem)

    p_sleep_bottom, sleep_bottom = h_to_v(sleep_top, dbn.W_mem, dbn.v_bias_mem)

    #get sleep states
    sleep_states = [sleep_top.detach()]
    for ith_rbm in range(dbn.n_layers-1,-1,-1):
        if ith_rbm < dbn.n_layers-1:
            p_sleep_state, sleep_state = h_to_v(sleep_states[0], dbn.W_gen[ith_rbm], dbn.bias_gen[ith_rbm])
        else:
            p_sleep_state, sleep_state = h_to_v(sleep_top, dbn.W_mem, dbn.v_bias_mem)

        sleep_states = [sleep_state.detach()] + sleep_states
        
    optimizer.zero_grad()
    
    for ith_rbm in range(dbn.n_layers-1):

        #updata recgnition
        dbn.W_rec[ith_rbm].grad.data += (-(sleep_states[ith_rbm].t().mm(sleep_states[ith_rbm+1] - v_to_h(sleep_states[ith_rbm], dbn.W_rec[ith_rbm], dbn.bias_rec[ith_rbm])[1])).data/batch_size).t()
        

        dbn.bias_rec[ith_rbm].grad.data+=(-(sleep_states[ith_rbm+1] - v_to_h(sleep_states[ith_rbm], dbn.W_rec[ith_rbm], dbn.bias_rec[ith_rbm])[1]).sum(0).data/batch_size)
        
        # print(dbn.bias_rec[ith_rbm].grad.data.size())
        # print((-(sleep_states[ith_rbm+1] - v_to_h(sleep_states[ith_rbm], dbn.W_rec[ith_rbm], dbn.bias_rec[ith_rbm])[1]).sum(0).data/batch_size).size())
        #updata generation
        dbn.W_gen[ith_rbm].grad.data+=(-(wake_states[ith_rbm] - h_to_v(wake_states[ith_rbm+1], dbn.W_gen[ith_rbm], dbn.bias_gen[ith_rbm])[1]).t().mm(wake_states[ith_rbm+1]).data/batch_size).t()

        dbn.bias_gen[ith_rbm].grad.data+=(-(wake_states[ith_rbm] - h_to_v(wake_states[ith_rbm+1], dbn.W_gen[ith_rbm], dbn.bias_gen[ith_rbm])[1]).sum(0).data/batch_size)

    #updata memory

    dbn.W_mem.grad.data+=(-(wake_states[-2].t().mm(wake_states[-1]) - sleep_states[-2].t().mm(sleep_states[-1])).data/batch_size).t()

    dbn.v_bias_mem.grad.data+=(-(wake_states[-2] - sleep_states[-2]).sum(0).data/batch_size)

    dbn.h_bias_mem.grad.data+=(-(wake_states[-1] - sleep_states[-1]).sum(0).data/batch_size)

    optimizer.step()
    optimizer.zero_grad()
    
    return None
           

def v_to_h(v, W, h_bias):
    # p_h = F.sigmoid(v.mm(self.W.t()) + self.h_bias.repeat(v.size()[0],1))
    p_h = torch.sigmoid(F.linear(v,W,h_bias))
    h = torch.bernoulli(p_h)
    return p_h,h

def h_to_v(h, W, v_bias):
    p_v = torch.sigmoid(F.linear(h,W.t(),v_bias))
    v = torch.bernoulli(p_v)
    return p_v,v
    
def generate(dbn, iteration = 1, prop_input = None, annealed = False, n = 0):
    
    if not type(prop_input) == type(None):
        prop_v = Variable(torch.from_numpy(prop_input).type(torch.FloatTensor))
        for i in range(dbn.n_layers-1):
            prop_v = dbn.rbm_layers[i].v_to_h(prop_v)[0]
        prop = prop_v.data.mean()
    else:
        prop = 0.5
        
    h = torch.bernoulli((dbn.rbm_layers[-1].h_bias *0 + prop).view(1,-1).repeat(n, 0))
    p_v, v = dbn.rbm_layers[-1].h_to_v(h)
    
    if not annealed:
        for _ in range(iteration):

            p_h, h = dbn.rbm_layers[-1].v_to_h(v)

            p_v, v = dbn.rbm_layers[-1].h_to_v(h)
    else:
        for temp in np.linspace(3,0.6,25):
            for i in dbn.rbm_layers[-1].parameters():
                i.data *= 1.0/temp
                
            for _ in range(iteration):

                p_h, h = dbn.rbm_layers[-1].v_to_h(v)

                p_v, v = dbn.rbm_layers[-1].h_to_v(h)    

            for i in dbn.rbm_layers[-1].parameters():
                i.data *= temp
        
    for i in range(dbn.n_layers-1):
        p_v, v = dbn.rbm_layers[-2-i].h_to_v(v)
        
    return v