from typing import Callable
import torch as pt
from ...model import *
from numpy import sqrt

class Markov_Switching(pt.nn.Module):
    def __init__(self, n_models:int, switching_diag: float, swithcing_diag_1: float):
        super().__init__()
        self.diag = switching_diag
        self.diag_1 = swithcing_diag_1
        self.n_models = n_models
        self.probs = pt.ones(n_models) * ((1 - self.diag - self.diag_1)/(n_models - 2))
        self.probs[0] = self.diag
        self.probs[1] = self.diag_1

    def initial_step(self, n_samples):
        return pt.multinomial(pt.ones(self.n_models), n_samples).unsqueeze(1)

    def forward(self, x_t_1, t):
        shifts = pt.multinomial(self.probs, x_t_1.size(0), True)
        new_models = pt.remainder(shifts + x_t_1[:, 1], self.n_models)
        return new_models.unsqueeze(1)
    
class Polya_Urn(pt.nn.Module):
    def __init__(self, n_models:int):
        super().__init__()
        self.n_models = n_models
        self.prob_vec = pt.ones(n_models)

    def initial_step(self, n_samples):
        init_regimes = pt.multinomial(pt.ones(self.n_models), n_samples, True).unsqueeze(1)
        init_counts = pt.ones((n_samples, self.n_models))
        self.add_matrix = pt.zeros((n_samples, self.n_models))
        return pt.concat((init_regimes, init_counts), dim=1)


    def forward(self, x_t_1, t):
        self.add_matrix.zero_()
        self.add_matrix.scatter_(1, x_t_1[:, 1].to(int).unsqueeze(1), 1)
        counts = x_t_1[:, -self.n_models:] + self.add_matrix
        regimes = pt.multinomial(counts, 1, True)
        return pt.concat((regimes, counts), dim =1)


class Regime_Switching_Linear_Gauss(Auxiliary_Feynman_Kac):
    def get_modules(self):
        return self.switching_dyn

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, a:list[int], b:list[int], var_s:float, switching_dyn:pt.nn.Module, device:pt.device = pt.device('cpu')):
        super().__init__(device)
        self.n_models = len(a)
        self.a = pt.Tensor(a, device = device)
        self.b = pt.Tensor(b, device = device)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)

    def M_0_proposal(self, n_samples: int):
        init_locs = self.init_x_dist.sample([n_samples]).unsqueeze(1)
        init_regimes = self.switching_dyn.initial_step(n_samples)
        return pt.cat((init_locs, init_regimes), dim = 1)                            
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0)])
        new_models = self.switching_dyn(x_t_1, t)
        index = new_models[:, 0].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = (scaling * x_t_1[:, 0] + bias + noise)
        return pt.cat((new_pos, new_models), dim = 1)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        pass

    def log_R_t(self, x_t, x_t_1, t: int):
        pass

    def log_f_t(self, x_t, t: int):
        pass

class Regime_Switching_Object(Simulated_Object):
    def __init__(self, model: Regime_Switching_Linear_Gauss, a, b, var_o, device = device):
        super().__init__(model, 20, 1, device)
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_o))
        self.a = pt.Tensor(a, device = device)
        self.b = pt.Tensor(b, device = device)

    def observation_generation(self):
        noise = self.y_dist.sample()
        index = self.x_t[:, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = (scaling * pt.sqrt(pt.abs(self.x_t[:, 0])) + bias + noise)
        return new_pos