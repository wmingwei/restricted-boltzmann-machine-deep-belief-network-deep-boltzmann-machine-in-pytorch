import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

class RBM(nn.Module):
    def __init__(self,
                 n_visible=256,
                 n_hidden=64):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(n_hidden,n_visible).uniform_(-1.0/(n_visible+n_hidden), 1.0/(n_visible+n_hidden)))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
    
    def sample_from_p(self,p):
        return torch.bernoulli(p)
    
    def v_to_h(self,v):
        
        # p_h = F.sigmoid(v.mm(self.W.t()) + self.h_bias.repeat(v.size()[0],1))
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
    def h_to_v(self,h):
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
    def forward(self,v, CD_k = 10):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(CD_k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
    def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = torch.clamp(F.linear(v,self.W,self.h_bias),-80,80)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

class DBM(nn.Module):
    def __init__(self,
                 n_visible=256,
                 n_hidden=[100,64]):
        super(DBM, self).__init__()
        
        self.n_layers = len(n_hidden)
        self.rbm_layers = []
        self.n_odd_layers = int((self.n_layers+1)/2)
        self.n_even_layers = int((self.n_layers+2)/2)
        
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_visible
            else:
                input_size = n_hidden[i-1]
            rbm = RBM(n_visible = input_size, n_hidden = n_hidden[i])
            
            self.rbm_layers.append(rbm)
        
        self.W = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers)]
        self.bias = [nn.Parameter(self.rbm_layers[0].v_bias.data)]+[nn.Parameter(self.rbm_layers[i].h_bias.data) for i in range(self.n_layers)]
        
        for i in range(self.n_layers):
            self.register_parameter('W%i'%i, self.W[i])
        for i in range(self.n_layers+1):
            self.register_parameter('bias%i'%i, self.bias[i])
        
    def odd_to_even(self, odd_input = None):
        even_p_output= []
        even_output= []
        for i in range(self.n_even_layers):
            if i == 0:
                even_p_output.append(F.sigmoid(F.linear(odd_input[i],self.W[2*i].t(),self.bias[2*i])))
            elif (self.n_even_layers > self.n_odd_layers) and i == self.n_even_layers - 1:
                even_p_output.append(F.sigmoid(F.linear(odd_input[i-1],self.W[2*i-1],self.bias[2*i])))
            else:
                even_p_output.append(F.sigmoid(F.linear(odd_input[i-1],self.W[2*i-1],self.bias[2*i]) + F.linear(odd_input[i],self.W[2*i].t())))
        for i in even_p_output:
            even_output.append(torch.bernoulli(i))
            
        return even_p_output, even_output
    
    def even_to_odd(self, even_input = None):
        odd_p_output = [] 
        odd_output = []
        for i in range(self.n_odd_layers):
            if (self.n_even_layers == self.n_odd_layers) and i == self.n_odd_layers - 1:
                odd_p_output.append(F.sigmoid(F.linear(even_input[i],self.W[2*i],self.bias[2*i+1])))
            else:
                odd_p_output.append(F.sigmoid(F.linear(even_input[i],self.W[2*i],self.bias[2*i+1]) + F.linear(even_input[i+1],self.W[2*i+1].t())))
        
        for i in odd_p_output:
            odd_output.append(torch.bernoulli(i))
            
        return odd_p_output, odd_output
    
    def forward(self, v_input, k_positive = 10, k_negative=10, greedy = True, ith_layer = 0, CD_k = 10): #for greedy training
        if greedy:
            v = v_input
        
            for ith in range(ith_layer):
                p_v, v = self.rbm_layers[ith].v_to_h(v)

            v, v_ = self.rbm_layers[ith_layer](v, CD_k = CD_k)

            return v, v_
        
        v = v_input
        even_layer = [v]
        odd_layer = []
        for i in range(1, self.n_even_layers):
            even_layer.append(torch.bernoulli(self.bias[2*i]*0+0.5))
            
        for _ in range(k_positive):
            p_odd_layer, odd_layer = self.even_to_odd(even_layer)
            p_even_layer, even_layer = self.odd_to_even(odd_layer)
            even_layer[0] = v
            
        positive_phase_even = [i.detach().clone() for i in even_layer]
        positive_phase_odd =  [i.detach().clone() for i in odd_layer]
        for i, d in enumerate(positive_phase_odd):
            positive_phase_even.insert(2*i+1, positive_phase_odd[i])
        positive_phase = positive_phase_even
        for _ in range(k_negative):
            p_odd_layer, odd_layer = self.even_to_odd(even_layer)
            p_even_layer, even_layer = self.odd_to_even(odd_layer)

        negative_phase_even = [i.detach().clone() for i in even_layer]
        negative_phase_odd =  [i.detach().clone() for i in odd_layer]
        for i, d in enumerate(negative_phase_odd):
            negative_phase_even.insert(2*i+1, negative_phase_odd[i])
        negative_phase = negative_phase_even
        return positive_phase, negative_phase