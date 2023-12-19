from typing import Callable
import torch as pt
from ...model import *
from numpy import sqrt
from ...utils import nd_select, normalise_log_quantity, batched_select

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

class Markov_Switching(pt.nn.Module):
    def __init__(self, n_models:int, switching_diag: float, switching_diag_1: float, dyn = 'Boot'):
        super().__init__()
        self.dyn = dyn
        self.n_models = n_models
        tprobs = pt.ones(n_models) * ((1 - switching_diag - switching_diag_1)/(n_models - 2))
        tprobs[0] = switching_diag
        tprobs[1] = switching_diag_1
        self.switching_vec = pt.log(tprobs).to(device=device)
        self.dyn = dyn
        if dyn == 'Uni' or dyn == 'Deter':
            self.probs = pt.ones(n_models)/n_models
        else:
            self.probs = tprobs

    def init_state(self, batches, n_samples):
        if self.dyn == 'Deter':
            return pt.arange(self.n_models, device=device).tile((batches, n_samples//self.n_models)).unsqueeze(2)
        return pt.multinomial(pt.ones(self.n_models), batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=device)

    def forward(self, x_t_1, t):
        if self.dyn == 'Deter':
            return pt.arange(self.n_models, device=device).tile((x_t_1.size(0), x_t_1.size(1)//self.n_models)).unsqueeze(2) 
        shifts = pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(device).reshape([x_t_1.size(0),x_t_1.size(1)])
        new_models = pt.remainder(shifts + x_t_1[:, :, 1], self.n_models)
        return new_models.unsqueeze(2)
    
    def get_log_probs(self, x_t, x_t_1):
        shifts = (x_t[:,:,1] - x_t_1[:,:,1])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return self.switching_vec[shifts]


class Polya_Switching(pt.nn.Module):
    def __init__(self, n_models, dyn) -> None:
        super().__init__()
        self.dyn = dyn
        self.n_models = n_models
        self.ones_vec = pt.ones(n_models)
        
    def init_state(self, batches, n_samples):
        self.scatter_v = pt.zeros((batches, n_samples, self.n_models), device=device)
        i_models = pt.multinomial(self.ones_vec, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=device)
        return pt.concat((i_models, pt.ones((batches, n_samples, self.n_models), device=device)), dim=2)

    def forward(self, x_t_1, t):
        self.scatter_v.zero_()
        self.scatter_v.scatter_(2, x_t_1[:,:,1].unsqueeze(2).to(int), 1)
        c = x_t_1[:,:,2:] + self.scatter_v
        if self.dyn == 'Uni':
            return pt.concat((pt.multinomial(self.ones_vec,  x_t_1.size(0)*x_t_1.size(1), True).to(device).reshape([x_t_1.size(0),x_t_1.size(1), 1]), c), dim=2)
        return pt.concat((pt.multinomial(c.reshape(-1, self.n_models), 1, True).to(device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        probs = x_t[:, :, 2:]
        probs /= pt.sum(probs, dim=2, keepdim=True)
        s_probs = batched_select(probs, x_t_1[:, :, 1].to(int))
        return pt.log(s_probs)

class NN_Switching(pt.nn.Module):

    def __init__(self, n_models, recurrent_length):
        super().__init__()
        self.r_length = recurrent_length
        self.n_models = n_models
        self.forget = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid())
        self.self_forget = pt.nn.Sequential(pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Sigmoid())
        self.scale = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid())
        self.to_reccurrent = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Tanh())
        self.output_layer = pt.nn.Sequential(pt.nn.Linear(recurrent_length, n_models), pt.nn.Sigmoid())
        self.probs = pt.ones(n_models)/n_models

    def init_state(self, batches, n_samples):
        i_models = pt.multinomial(self.probs, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=device)
        if self.r_length > 0:
            return pt.concat((i_models, pt.zeros((batches, n_samples, self.r_length), device=device)), dim=2)
        else:
            return i_models

    def forward(self, x_t_1, t):
        old_model = x_t_1[:, :, 1].to(int).unsqueeze(2)
        one_hot = pt.zeros((old_model.size(0), old_model.size(1), self.n_models), device=device)
        one_hot = pt.scatter(one_hot, 2, old_model, 1)
        old_recurrent = x_t_1[:, :, 2:]
        c = old_recurrent * self.self_forget(old_recurrent)
        c *= self.forget(one_hot)
        c += self.scale(one_hot) * self.to_reccurrent(one_hot)
        return pt.concat((pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        models = x_t[:,:,1].to(int)
        probs = self.output_layer(x_t[:, :, 2:])
        probs = probs / pt.sum(probs, dim=2, keepdim=True)
        log_probs = batched_select(probs, models)
        return pt.log(log_probs)
    
class NN_Switching_LSTM(pt.nn.Module):

    def __init__(self, n_models, recurrent_length):
        super().__init__()
        self.r_length = recurrent_length
        self.n_models = n_models
        self.forget = pt.nn.Sequential(pt.nn.Linear(2*n_models, recurrent_length), pt.nn.Sigmoid())
        self.to_hidden = pt.nn.Sequential(pt.nn.Linear(2*n_models, recurrent_length), pt.nn.Tanh())
        self.self_gate = pt.nn.Sequential(pt.nn.Linear(2*n_models, recurrent_length), pt.nn.Sigmoid())
        self.output_layer = pt.nn.Sequential(pt.nn.Linear(2*n_models, n_models), pt.nn.Sigmoid())
        self.tanh = pt.nn.Sigmoid()
        self.probs = pt.ones(n_models)/n_models
        self.t_probs = pt.zeros(n_models, device=device)

    def init_state(self, batches, n_samples):
        i_models = pt.multinomial(self.probs, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=device)
        if self.r_length > 0:
            return pt.concat((i_models, pt.zeros((batches, n_samples, self.n_models + self.r_length), device=device)), dim=2)
        else:
            return i_models

    def forward(self, x_t_1, t):
        old_model = x_t_1[:, :, 1].to(int).unsqueeze(2)
        one_hot = pt.zeros((old_model.size(0), old_model.size(1), self.n_models), device=device)
        one_hot = pt.scatter(one_hot, 2, old_model, 1)
        old_probs = x_t_1[:, :, 2:2+self.n_models]
        old_recurrent = x_t_1[:, :, 2+self.n_models:]
        in_vector = pt.concat((one_hot, old_probs), dim=2)
        recurrent = old_recurrent * self.forget(in_vector)
        recurrent += self.self_gate(in_vector)*self.to_hidden(in_vector)
        new_probs = self.output_layer(in_vector) * self.tanh(recurrent)
        new_probs = new_probs / pt.sum(new_probs, dim=2, keepdim=True)
        self.t_probs = new_probs
        return pt.concat((pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), new_probs, recurrent), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        models = x_t[:,:,1].to(int)
        log_probs = batched_select(self.t_probs, models)
        return pt.log(log_probs)
    
class Simple_NN(pt.nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.net = pt.nn.Sequential(pt.nn.Linear(input, hidden), pt.nn.Tanh(), pt.nn.Linear(hidden, output))

    def forward(self, in_vec):
        return self.net(in_vec.unsqueeze(1)).squeeze()

class PF(Auxiliary_Feynman_Kac):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, a:list[int], b:list[int], var_s:float, switching_dyn:pt.nn.Module, dyn ='Boot', device:pt.device = device):
        super().__init__(device)
        self.n_models = len(a)
        self.a = pt.tensor(a, device = device)
        self.b = pt.tensor(b, device = device)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.var_factor = -1/(2*var_s)
        if dyn == 'Boot':
            self.PF_type = 'Bootstrap'
        else:
            self.PF_type ='Guided'

    def M_0_proposal(self, batches:int, n_samples: int):
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)
                                      
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=device)
        new_models = self.switching_dyn(x_t_1, t)
        index = new_models[:,:,0].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = ((scaling * x_t_1[:, :, 0]).unsqueeze(2) + bias.unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        locs = (scaling*pt.sqrt(pt.abs(x_t[:, :, 0])) + bias) 
        return self.var_factor * ((self.y[t] - locs)**2)


class RSDBPF(Auxiliary_Feynman_Kac):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, dyn='Boot', device:pt.device = device):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        if dyn == 'Boot':
            self.PF_type = 'Bootstrap'
        else:
            self.PF_type = 'Guided'

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=device) * self.sd_d
        new_models = self.switching_dyn(x_t_1, t)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=device)
        index = new_models[:, :, 0].to(int)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.dyn_models[m](x_t_1[:,:,0][mask])
        new_pos = (locs.unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.obs_models[m](x_t[:,:,0][mask])
        return self.var_factor * ((self.y[t] - locs)**2)


class RSDBPF_2(Auxiliary_Feynman_Kac):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, dyn='Boot', device:pt.device = device):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.PF_type = 'Guided'

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).tile((1, self.n_models)).unsqueeze(2)
        init_regimes = pt.repeat_interleave(pt.arange(self.n_models, device=device), n_samples).repeat((batches, 1))
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        init_regimes = pt.tile(init_regimes, (1, self.n_models, 1))
        for m in range(self.n_models):
            init_regimes[:, m*n_samples:(m+1)*n_samples, 0] = m
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        N = x_t_1.size(1)
        noise = self.x_dist.sample([x_t_1.size(0), N*self.n_models]).to(device=device) * self.sd_d
        new_models = self.switching_dyn(x_t_1, t)
        new_models = pt.tile(new_models, (1, self.n_models, 1))
        locs = pt.empty((x_t_1.size(0), N*self.n_models), device=device)
        for m in range(self.n_models):
            new_models[:, m*N:(m+1)*N, 0] = m
            locs[:, m*N:(m+1)*N] = self.dyn_models[m](x_t_1[:,:,0].unsqueeze(2))
        new_pos = (locs.unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1.tile((1,self.n_models,1)))

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.obs_models[m](x_t[:,:,0][mask])
        return self.var_factor * ((self.y[t] - locs)**2)
    

class DBPF(Auxiliary_Feynman_Kac):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, switching_diag:float, switching_diag_1:float, device:pt.device = device):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_model = Simple_NN(1, 8, 1)
        self.obs_model = Simple_NN(1, 8, 1)
        self.sd_d = pt.nn.Parameter(pt.rand(1, device=device)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1, device=device)*0.4 + 0.1)
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.PF_type = 'Bootstrap'

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).unsqueeze(2)
        return init_locs               
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=device) * self.sd_d
        locs = self.dyn_model(x_t_1)
        new_pos = locs.unsqueeze(2) + noise
        return new_pos
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        shifts = (x_t[:,:,1] - x_t_1[:,:,1])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return nd_select(self.switching_vec, shifts)

    def log_f_t(self, x_t, t: int):
        locs = self.obs_model(x_t)
        return self.var_factor * (self.y[t] - locs)**2
    
class RSDBPF_cheat(Auxiliary_Feynman_Kac):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, var_s:float, switching_dyn:pt.nn.Module, switching_diag:float, switching_diag_1:float, device:pt.device = device):
        super().__init__(device)
        self.a = pt.nn.Parameter(pt.rand(n_models, requires_grad=True)-0.5)
        self.b = pt.nn.Parameter(pt.rand(n_models, requires_grad=True)-0.5)
        self.c = pt.nn.Parameter(pt.rand(n_models, requires_grad=True)-0.5)
        self.d = pt.nn.Parameter(pt.rand(n_models, requires_grad=True)-0.5)
        self.sd_d = pt.nn.Parameter(pt.rand(1, device=device)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1, device=device)*0.4 + 0.1)
        self.n_models = n_models
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        
        self.switching_vec = pt.ones(self.n_models) * ((1 - switching_diag - switching_diag_1)/(self.n_models - 2))
        self.switching_vec[0] = switching_diag
        self.switching_vec[1] = switching_diag_1
        self.switching_vec = pt.log(self.switching_vec).to(device=device)
        self.PF_type = 'Guided'

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*self.sd_o**2)
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).unsqueeze(2)
        init_regimes = pt.multinomial(pt.ones(self.n_models), batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=device)
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=device)*self.sd_d
        index = self.switching_dyn(x_t_1, t).to(int)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.a[m]*(x_t_1[:,:,0][mask]) + self.b[m]
        new_pos = (locs.unsqueeze(2) + noise)
        test = new_pos.isnan()
        if pt.any(test):
            print('Error nan')
            raise SystemExit(0)
        return pt.cat((new_pos, index.unsqueeze(2)), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        shifts = (x_t[:,:,1] - x_t_1[:,:,1])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return nd_select(self.switching_vec, shifts)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=device)
        pos = pt.abs(x_t[:,:,0])
        #Needed for numerical stability of backward pass
        pos = pt.where(pos < 1e-5, pos + 1e-5, pos)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.c[m]*pt.sqrt(pos[mask]) + self.d[m]
        return self.var_factor * (self.y[t] - locs)**2
    
class Generates_0(Auxiliary_Feynman_Kac):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, device:pt.device = device):
        super().__init__(device)
        self.PF_type = 'Bootstrap'

    def M_0_proposal(self, batches:int, n_samples: int):
        return pt.zeros((batches, n_samples, 1), device=device)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        return pt.zeros_like((x_t_1))
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        None

    def log_f_t(self, x_t, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)
        
class LSTM(pt.nn.Module):

    def __init__(self, obs_dim, hid_dim, state_dim, n_layers) -> None:
        super().__init__()
        self.lstm = pt.nn.LSTM(obs_dim, hid_dim, n_layers, True, True, 0.0, False, state_dim, device)

    def forward(self, y_t):
            return self.lstm(y_t)[0]