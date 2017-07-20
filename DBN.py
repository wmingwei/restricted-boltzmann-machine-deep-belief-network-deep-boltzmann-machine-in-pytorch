import torch
import numpy
from torch.autograd import Variable
import RBM
from torch.utils import data as dtf
from torch import optim
import ais_dbn

class DBN(object):
    def __init__(self, n_visible = 1600, n_hidden = [32,16], W = None, bias = None,
                 trained = False):
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
                              h_bias = bias[i+1],
                              v_bias = bias[i])
            else:
                rbm = RBM.RBM(n_visible = input_size, 
                              n_hidden = n_hidden[i])
                #if i != 0:
                 #   rbm.v_bias = self.rbm_layers[i-1].h_bias
            self.rbm_layers.append(rbm)
        
        self.W_rec = [Variable(torch.zeros(self.rbm_layers[i].W.size()).type(torch.FloatTensor), requires_grad=True) for i in range(self.n_layers-1)]
        self.W_gen = [Variable(torch.zeros(self.rbm_layers[i].W.size()).type(torch.FloatTensor), requires_grad=True) for i in range(self.n_layers-1)]
        self.bias_rec = [Variable(torch.zeros(self.rbm_layers[i].h_bias.size()).type(torch.FloatTensor), requires_grad=True) for i in range(self.n_layers-1)]
        self.bias_gen = [Variable(torch.zeros(self.rbm_layers[i].v_bias.size()).type(torch.FloatTensor), requires_grad=True) for i in range(self.n_layers-1)]
        self.W_mem = Variable(torch.zeros(self.rbm_layers[-1].W.size()).type(torch.FloatTensor), requires_grad=True)
        self.v_bias_mem = Variable(torch.zeros(self.rbm_layers[-1].v_bias.size()).type(torch.FloatTensor), requires_grad=True)
        self.h_bias_mem = Variable(torch.zeros(self.rbm_layers[-1].h_bias.size()).type(torch.FloatTensor), requires_grad=True)
    
        self.optimizer = None

        params_W = {"params":self.W_rec + self.W_gen + [self.W_mem]}
        params_bias = {"params":self.bias_rec + self.bias_gen + [self.v_bias_mem] + [self.h_bias_mem], "weight_decay":0}
        self.params = [params_W, params_bias]
        for para in self.params:
            for i in para["params"]:
                i.sum().backward()
    def greedy_train(self, lr = [1e-2, 1e-2], epoch = [100,100], batch_size = [50, 50], input_data = None, 
                     CD_k = 1, optimization_method = None, momentum = 0, gradient = True, weight_decay = 0, test_input = None, test_dbn = None, p_CD = [False, False], sparsity = 0, sparsity_rate = 0, mirror = False, new_optimizer = False):
        if not type(test_dbn) == type(None):
            dbn = self
        else:
            dbn = None
        for ith_rbm in range(self.n_layers):
            #print("training rbm %i" %ith_rbm)
            if mirror and ith_rbm>0:
                self.rbm_layers[ith_rbm].W.data = self.rbm_layers[ith_rbm-1].W.data.transpose(0,1)
                self.rbm_layers[ith_rbm].h_bias.data = self.rbm_layers[ith_rbm-1].v_bias.data
                self.rbm_layers[ith_rbm].v_bias.data = self.rbm_layers[ith_rbm-1].h_bias.data
                
                #print(ais_dbn.logp_ais(self, test_dbn,1000, 20, 100, True))
            if ith_rbm:
                #for i in range(int(5)):
                hidden_data = self.rbm_layers[ith_rbm-1].sample_h_given_v(Variable(input_data,requires_grad = False),
                                                                          W = self.rbm_layers[ith_rbm-1].W,
                                                                          h_bias = self.rbm_layers[ith_rbm-1].h_bias,
                                                                          v_bias = self.rbm_layers[ith_rbm-1].v_bias)[1].data
                self.rbm_layers[ith_rbm].train(lr = lr[ith_rbm], epoch = epoch[ith_rbm], batch_size = batch_size[ith_rbm], 
                                          input_data = hidden_data, CD_k = CD_k, optimization_method = optimization_method,
                                          momentum = momentum, weight_decay = weight_decay, gradient = gradient, p_CD = p_CD[ith_rbm], sparsity = sparsity, sparsity_rate = sparsity_rate, dbn = dbn, test_v = test_dbn,new_optimizer = new_optimizer)
                
                input_data = self.rbm_layers[ith_rbm-1].sample_h_given_v(Variable(input_data,requires_grad = False),
                                                                W = self.rbm_layers[ith_rbm-1].W,
                                                                h_bias = self.rbm_layers[ith_rbm-1].h_bias,
                                                                v_bias = self.rbm_layers[ith_rbm-1].v_bias)[0].data
                #print("rbm %i data ready" %ith_rbm)
            else: 
                self.rbm_layers[ith_rbm].train(lr = lr[ith_rbm], epoch = epoch[ith_rbm], batch_size = batch_size[ith_rbm], 
                                          input_data = input_data, CD_k = CD_k, optimization_method = optimization_method,
                                          momentum = momentum, weight_decay = weight_decay, gradient = gradient, p_CD = p_CD[ith_rbm], sparsity = sparsity, sparsity_rate = sparsity_rate, dbn = dbn, test_v = test_dbn,new_optimizer = new_optimizer)
            if not (type(test_input) == type(None)):
                
                print("layerwise", ith_rbm, ais_dbn.logp_ais(self, test_input, step = 1000, M_Z = 20, M_IS = 100, parallel = True))
                
        for ith_rbm in range(self.n_layers-1):
            self.W_rec[ith_rbm].data = self.rbm_layers[ith_rbm].W.data.clone()
            self.W_gen[ith_rbm].data = self.rbm_layers[ith_rbm].W.data.clone()
            self.bias_rec[ith_rbm].data = self.rbm_layers[ith_rbm].h_bias.data.clone()
            self.bias_gen[ith_rbm].data = self.rbm_layers[ith_rbm].v_bias.data.clone()
        self.W_mem.data = self.rbm_layers[-1].W.data.clone()
        self.v_bias_mem.data = self.rbm_layers[-1].v_bias.data.clone()
        self.h_bias_mem.data = self.rbm_layers[-1].h_bias.data.clone()
            
            #initialize gradient
        for para in self.params:
            for i in para["params"]:
                i.sum().backward()
            
    def generative_fine_tune(self, lr = 1e-2, epoch = 100, batch_size = 50, input_data = None, CD_k = 1, new_optimizer = True, optimization_method = None, momentum = 0, weight_decay = 0, test_input = None):
        if new_optimizer:
            if optimization_method == "RMSprop":
                self.optimizer = optim.RMSprop(self.params, lr = lr, momentum = momentum, weight_decay = weight_decay)
            elif optimization_method == "SGD":
                self.optimizer = optim.SGD(self.params, lr = lr, momentum = momentum, weight_decay = weight_decay)
            elif optimization_method == "Adam":
                self.optimizer = optim.Adam(self.params, lr = lr, weight_decay = weight_decay)                
        train_set = dtf.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
        train_loader = dtf.DataLoader(train_set, batch_size = batch_size, shuffle=True)
        
        for i in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                
                self.sleep_wake(lr = lr, CD_k = CD_k, v_input = data, batch_size = batch_size)
            
            if not (type(test_input) == type(None)):
                
                print("fine tune", i, ais_dbn.logp_ais(self, test_input, step = 1000, M_Z = 20, M_IS = 100, parallel = True))
            
    def sleep_wake(self, lr = 1e-2, CD_k = 10, v_input = None, batch_size = 1):
        
        #get wake states
        wake_states = [Variable(v_input,requires_grad = False)]
        
        for ith_rbm in range(self.n_layers):
            if ith_rbm < self.n_layers-1:
                wake_state = self.rbm_layers[ith_rbm].sample_h_given_v(wake_states[ith_rbm], self.W_rec[ith_rbm], self.bias_rec[ith_rbm], None)[0]
            else:
                wake_state = self.rbm_layers[-1].sample_h_given_v(wake_states[ith_rbm], self.W_mem, self.h_bias_mem, None)[0]

            wake_states.append(wake_state)

        #CD_k    
        sleep_top = wake_states[-1]

        for cd in range(CD_k):
            sleep_bottom, sleep_top, chain_ph  = self.rbm_layers[-1].gibbs_hvh(sleep_top, self.W_mem, self.h_bias_mem, self.v_bias_mem)

        sleep_bottom = self.rbm_layers[-1].sample_v_given_h(sleep_top, self.W_mem, self.h_bias_mem, self.v_bias_mem)[0]

        #get sleep states
        sleep_states = [sleep_top]
 
        for ith_rbm in range(self.n_layers-1,0,-1):
            if ith_rbm < self.n_layers-1:
                sleep_state = self.rbm_layers[ith_rbm].sample_v_given_h(sleep_states[0], self.W_gen[ith_rbm],
                                                                                        None, self.bias_gen[ith_rbm])[0]
            else:
                sleep_state = self.rbm_layers[-1].sample_v_given_h(sleep_top, self.W_mem, self.h_bias_mem, self.v_bias_mem)[0]

            sleep_states = [sleep_state] + sleep_states
            
        sleep_state = self.rbm_layers[0].sample_v_given_h(sleep_states[0], self.W_gen[0],
                                                                                        None, self.bias_gen[0])[1]
        sleep_states = [sleep_state] + sleep_states
        
        self.optimizer.zero_grad()
        
        for ith_rbm in range(self.n_layers-1):
            
            #updata recgnition
            self.W_rec[ith_rbm].grad.data =  -(sleep_states[ith_rbm].transpose(0,1).mm(sleep_states[ith_rbm+1] - self.rbm_layers[ith_rbm].sample_h_given_v(sleep_states[ith_rbm], self.W_rec[ith_rbm], self.bias_rec[ith_rbm], None)[1])).data/batch_size
            
            #print(ith_rbm, self.W_rec[ith_rbm].size(), self.W_rec[ith_rbm].grad.size())
            
            self.bias_rec[ith_rbm].grad.data = -(sleep_states[ith_rbm+1] - self.rbm_layers[ith_rbm].sample_h_given_v(sleep_states[ith_rbm], self.W_rec[ith_rbm], self.bias_rec[ith_rbm], None)[1]).sum(0).data/batch_size
            
            #print(ith_rbm, self.bias_rec[ith_rbm].size(), self.bias_rec[ith_rbm].grad.size())
            
            #updata generation
            self.W_gen[ith_rbm].grad.data = -(wake_states[ith_rbm] - self.rbm_layers[ith_rbm].sample_v_given_h(wake_states[ith_rbm+1], self.W_gen[ith_rbm], None, self.bias_gen[ith_rbm])[1]).transpose(0,1).mm(wake_states[ith_rbm+1]).data/batch_size
            
            #print(ith_rbm, self.W_gen[ith_rbm].size(), self.W_gen[ith_rbm].grad.size())
            
            self.bias_gen[ith_rbm].grad.data = -(wake_states[ith_rbm] - self.rbm_layers[ith_rbm].sample_v_given_h(wake_states[ith_rbm+1], self.W_gen[ith_rbm], None, self.bias_gen[ith_rbm])[1]).sum(0).data/batch_size
            
            #print(ith_rbm, self.bias_gen[ith_rbm].size(), self.bias_gen[ith_rbm].grad.size())
        
        #updata memory
        
        self.W_mem.grad.data = -(wake_states[-2].transpose(0,1).mm(wake_states[-1]) - sleep_states[-2].transpose(0,1).mm(sleep_states[-1])).data/batch_size
        
        #print(ith_rbm, self.W_mem.size(), self.W_mem.grad.size())
        
        self.v_bias_mem.grad.data = -(wake_states[-2] - sleep_states[-2]).sum(0).data/batch_size
        
        #print(ith_rbm, self.v_bias_mem.size(), self.v_bias_mem.grad.size())
        
        self.h_bias_mem.grad.data = -(wake_states[-1] - sleep_states[-1]).sum(0).data/batch_size
        
        #print(ith_rbm, self.h_bias_mem.size(), self.h_bias_mem.grad.size())
        
        self.optimizer.step()
        
        
        for ith_rbm in range(self.n_layers-1):
            
            self.rbm_layers[ith_rbm].W.data = self.W_gen[ith_rbm].data.clone()
            self.rbm_layers[ith_rbm].v_bias.data = self.bias_gen[ith_rbm].data.clone()

        self.rbm_layers[-1].W.data = self.W_mem.data.clone()
        self.rbm_layers[-1].v_bias.data = self.v_bias_mem.data.clone()
        self.rbm_layers[-1].h_bias.data = self.h_bias_mem.data.clone()
        
        return None
                
    def generate(self, iteration = 1, prop_input = None):
        if not type(prop_input) == type(None):
            prop_v = Variable(torch.from_numpy(prop_input).type(torch.FloatTensor))
            for i in range(self.n_layers-1):
                prop_v = self.rbm_layers[i].sample_h_given_v(prop_v, self.rbm_layers[i].W, self.rbm_layers[i].h_bias,
                                                             self.rbm_layers[i].v_bias)[0]
            prop = prop_v.data.mean()
        else:
            prop = 0.5
        v_sample = self.rbm_layers[-1].generate(iteration = iteration, prop = prop)
        for i in range(self.n_layers-1):
            v_sample = self.rbm_layers[-2-i].sample_v_given_h(v_sample, self.rbm_layers[-2-i].W, self.rbm_layers[-2-i].h_bias,self.rbm_layers[-2-i].v_bias)[0]
        return v_sample