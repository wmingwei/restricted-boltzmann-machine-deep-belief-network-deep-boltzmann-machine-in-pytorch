import torch
import numpy
from torch.autograd import Variable
import RBM
from torch.utils import data as dtf

class DBN(object):
    def __init__(self, n_visible = 1600, n_hidden = [32,16], W = None, v_bias = None, h_bias=None,
                 batch_size = 50, trained = False):
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
                              v_bias = v_bias[i],
                              batch_size = batch_size)
            else:
                rbm = RBM.RBM(n_visible = input_size, 
                              n_hidden = n_hidden[i],
                              batch_size = batch_size)
            self.rbm_layers.append(rbm)
    
    def greedy_train(self, lr = [1e-2, 1e-2], epoch = [100,100], batch_size = [50, 50], input_data = None):
        
        train_set = dtf.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
        train_loader = dtf.DataLoader(train_set, batch_size = batch_size[0], shuffle=True)
        for ith_rbm in range(self.n_layers):
            print("training rbm %i" %ith_rbm)
            if ith_rbm:
                input_data = self.rbm_layers[ith_rbm-1].sample_h_given_v(Variable(input_data,requires_grad = False),
                                                                W = self.rbm_layers[ith_rbm-1].W,
                                                                h_bias = self.rbm_layers[ith_rbm-1].h_bias)[0].data
                train_set = dtf.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
                train_loader = dtf.DataLoader(train_set, batch_size = batch_size[ith_rbm], shuffle=True)
                print("rbm %i data ready" %ith_rbm)
            for i in range(epoch[ith_rbm]):
                cost = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    cost += self.rbm_layers[ith_rbm].get_cost_update(lr = lr[ith_rbm], v_input = Variable(data,requires_grad = False)).data
                print(cost)            