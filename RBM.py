import torch
import numpy
from torch.autograd import Variable
from torch.utils import data as dtf

class RBM(object):
    
    def __init__(self, n_visible = 784, n_hidden = 500, W = None, v_bias = None, 
                 h_bias = None, batch_size = 30):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        
        if not W:
            initial_W = numpy.asarray(
                numpy.random.normal(loc = 0, scale = 1/n_visible,
                    size=(n_visible, n_hidden)
                    ),
                )
            W = Variable(torch.from_numpy(initial_W).type(torch.FloatTensor), requires_grad = True)
            
        if not v_bias:
            v_bias = Variable(torch.zeros(1,n_visible).type(torch.FloatTensor), requires_grad=True)

        if not h_bias:
            h_bias = Variable(torch.zeros(1,n_hidden).type(torch.FloatTensor), requires_grad=True)
            
        self.W = W
        self.v_bias = v_bias
        self.h_bias = h_bias
        
    def free_energy(self, v_sample, W, h_bias):
        
        Wv = torch.clamp(v_sample.mm(W) + h_bias.repeat(v_sample.size()[0],1), min = -80, max = 80)
        hidden = torch.log(1+torch.exp(Wv)).sum(1)
        vbias = v_sample.mm(self.v_bias.transpose(0,1))        
        
        return -hidden-vbias
    
    def sample_h_given_v(self, v0_sample, W,h_bias):
        
        activation = v0_sample.mm(W) + h_bias.repeat(v0_sample.size()[0],1)
        h1_mean = torch.sigmoid(activation)
                
        h1_sample = torch.bernoulli(h1_mean)

        return [h1_sample, h1_mean]

    
    def sample_v_given_h(self, h0_sample, W, h_bias):

        activation = h0_sample.mm(W.transpose(0,1)) + self.v_bias.repeat(h0_sample.size()[0],1)
        v1_mean = torch.sigmoid(activation)
        v1_sample = torch.bernoulli(v1_mean)
        return [v1_sample, v1_mean]

    
    def gibbs_hvh(self, h0_sample, W, h_bias):
        v1_sample, v1_mean = self.sample_v_given_h(h0_sample, W, h_bias)
        h1_sample, p_h1 = self.sample_h_given_v(v1_sample, W, h_bias)
        
        return [v1_sample, h1_sample, p_h1] 
    
    def gibbs_vhv(self, v0_sample, W, h_bias):
        h1_sample, h1_mean = self.sample_h_given_v(v0_sample, W, h_bias)
        v1_sample, p_v1 = self.sample_v_given_h(h1_sample, W, h_bias)
        
        return [h1_sample, v1_sample, p_v1]
    
        
    
    def get_cost_update(self, lr = 1e-4, k=10, v_input = None):
        
        chain_v = v_input
        
        for i in range(k):
            chain_h, chain_v, chain_pv  = self.gibbs_vhv(chain_v, self.W, self.h_bias)

        loss = torch.mean(self.free_energy(v_input, self.W, self.h_bias)) - torch.mean(self.free_energy(chain_v.detach(), self.W, self.h_bias))


        loss.backward()
        
        if numpy.isnan(self.W.grad.data.numpy()).any():
            print('nan')
            
            
        if torch.sum(self.W.grad.data*self.W.grad.data) > 1000:
            self.W.grad.data = self.W.grad.data * 1000.0 / torch.sum(self.W.grad.data*self.W.grad.data)
            
        if torch.sum(self.v_bias.grad.data*self.v_bias.grad.data) > 1000:
            self.v_bias.grad.data = self.v_bias.grad.data * 1000.0 / torch.sum(self.v_bias.grad.data*self.v_bias.grad.data)
        if torch.sum(self.h_bias.grad.data*self.h_bias.grad.data) > 1000:
            self.h_bias.grad.data = self.h_bias.grad.data * 1000.0 / torch.sum(self.h_bias.grad.data*self.h_bias.grad.data)            
        
        
        self.W.data -= lr*self.W.grad.data
        self.v_bias.data -= lr*self.v_bias.grad.data
        self.h_bias.data -= lr*self.h_bias.grad.data

        self.W.grad.data.zero_()
        self.v_bias.grad.data.zero_()
        self.h_bias.grad.data.zero_()
        
        moniter_cost = self.reconstruct_cost(v_input)
        
        return moniter_cost
    
    def reconstruct_cost(self, v_input):
        reconstruct_v = self.gibbs_vhv(v_input, self.W, self.h_bias)[1]

        cost = torch.mean(torch.abs((v_input-reconstruct_v)))

        return cost
    
    def train(self, lr = 1e-2, epoch = 100, batch_size = 50, input_data = None):
        train_set = dtf.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
        train_loader = dtf.DataLoader(train_set, batch_size = batch_size, shuffle=True)
        
        for i in range(epoch):
            cost = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                cost += self.get_cost_update(lr = lr, v_input = Variable(data,requires_grad = False)).data
            print(cost)         
