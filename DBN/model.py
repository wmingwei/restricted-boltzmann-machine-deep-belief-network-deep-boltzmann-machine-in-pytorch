import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

class RBM(nn.Module):
    def __init__(self,
                 n_visible=256,
                 n_hidden=64,):
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

class DBN(nn.Module):
    def __init__(self,
                 n_visible=256,
                 n_hidden=[100,64],):
        super(DBN, self).__init__()
        
        self.n_layers = len(n_hidden)
        self.rbm_layers = []
        
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_visible
            else:
                input_size = n_hidden[i-1]
            rbm = RBM(n_visible = input_size, n_hidden = n_hidden[i])
            
            self.rbm_layers.append(rbm)
        
        self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layers-1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers-1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone()) for i in range(self.n_layers-1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in range(self.n_layers-1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)
        
        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])
        
        
    def forward(self, v_input, CD_k = 10): #for greedy training
        v = v_input
        
        v_in = []
        v_out = []
        for i_th in range(self.n_layers):
            v, v_ = self.rbm_layers[i_th](v, CD_k = CD_k)
            v_in.append(v.clone())
            v_out.append(v_.clone())
            v = self.rbm_layers[i_th].v_to_h(v)[1]

        return v_in, v_out