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
        self.CD_k = CD_k
    
    def sample_from_p(self,p):
        return torch.bernoulli(p)
    
    def v_to_h(self,v):
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