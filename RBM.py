import torch
import numpy
from torch.autograd import Variable
from torch.utils import data as dtf
from torch import optim

class RBM(object):
    
    def __init__(self, n_visible = 784, n_hidden = 500, W = None, v_bias = None, 
                 h_bias = None):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        if not W:
            initial_W = numpy.asarray(
                #numpy.random.normal(loc = 0, scale = 1/n_visible,
                 #   size=(n_visible, n_hidden)
                 #   ),
                numpy.random.uniform(low = -1/numpy.sqrt(n_visible+n_hidden),
                                    high = 1/numpy.sqrt(n_visible+n_hidden),
                                    size=(n_visible, n_hidden))
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
    
    def generate(self, iteration = 1):
       
        v_samples = torch.bernoulli(self.v_bias *0 + 0.5)
        for i in range(iteration):
            h_samples, v_samples, chain_pv  = self.gibbs_vhv(v_samples, self.W, self.h_bias)
        
        return v_samples
            
    def get_cost_update(self, lr = 1e-2, k=10, v_input = None, optimizer = None, gradient = False, batch_size = 50):
        
        chain_v = v_input
        
        h_input, chain_v, chain_pv  = self.gibbs_vhv(chain_v, self.W, self.h_bias)
        
        for i in range(k):
            chain_h, chain_v, chain_pv  = self.gibbs_vhv(chain_v, self.W, self.h_bias)
        
        loss = torch.mean(self.free_energy(v_input, self.W, self.h_bias)) - torch.mean(self.free_energy(chain_v.detach(), self.W, self.h_bias))

        loss.backward()    
        if not gradient:
            self.W.grad.data.zero_()
            self.v_bias.grad.data.zero_()
            self.h_bias.grad.data.zero_()
            self.W.grad.data = -((v_input.transpose(0,1)).mm(h_input) - (chain_v.transpose(0,1)).mm(chain_h)).data
            self.v_bias.grad.data = -((v_input - chain_v).sum(0)).data
            self.h_bias.grad.data = -((h_input - chain_h).sum(0)).data
            
        if optimizer == None:
                   
        
        
            self.W.data -= lr*self.W.grad.data
            self.v_bias.data -= lr*self.v_bias.grad.data
            self.h_bias.data -= lr*self.h_bias.grad.data
            
            #print(torch.mean(v_input), torch.mean(self.W.grad.data))
            

            self.W.grad.data.zero_()
            self.v_bias.grad.data.zero_()
            self.h_bias.grad.data.zero_()
        else:
            optimizer.step()
            optimizer.zero_grad()
            
        moniter_cost = self.reconstruct_cost(v_input)
        
        return moniter_cost
    
    def reconstruct_cost(self, v_input):
        reconstruct_v = self.gibbs_vhv(v_input, self.W, self.h_bias)[1]

        cost = torch.mean(torch.abs((v_input-reconstruct_v)))

        return cost
    
    def train(self, lr = 1e-2, epoch = 100, batch_size = 50, input_data = None, optimization_method = None, CD_k = 1, momentum = 0, gradient = False, optimizer = None):
        train_set = dtf.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
        train_loader = dtf.DataLoader(train_set, batch_size = batch_size, shuffle=True)
        params = [self.W, self.v_bias, self.h_bias]
        
        if not optimizer:
            if optimization_method == "RMSprop":
                optimizer = optim.RMSprop(params, lr = lr)
            elif optimization_method == "SGD":
                optimizer = optim.SGD(params, lr = lr, momentum = momentum)
            else:
                optimizer = None
        else:
            optimizer = optimizer
            
        for i in range(epoch):
            cost = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                cost += self.get_cost_update(lr = lr, k = CD_k, v_input = Variable(data,requires_grad = False), optimizer = optimizer, gradient = gradient, batch_size = batch_size).data
            print(cost)
        return optimizer
