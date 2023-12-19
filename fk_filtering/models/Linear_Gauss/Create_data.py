from typing import Callable
import torch as pt
from ...model import *

class Linear_Gauss_Bootstrap(Auxiliary_Feynman_Kac):
    def get_modules(self):
        pass

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, rho:pt.Tensor, covar_x:pt.Tensor, init_covar_x:pt.Tensor, device:pt.device = device):
        super().__init__(device)
        self.rho = rho
        self.x_dist = pt.distributions.MultivariateNormal(pt.zeros(covar_x.size(0)), covar_x)
        self.init_x_dist = pt.distributions.MultivariateNormal(pt.zeros(init_covar_x.size(0)), init_covar_x)

    def M_0_proposal(self, n_samples: int):
        return self.init_x_dist.sample([n_samples])
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0)])
        means = pt.einsum('ij, kj -> ki', self.rho, x_t_1)
        return means + noise
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        pass

    def log_R_t(self, x_t, x_t_1, t: int):
        pass

    def log_f_t(self, x_t, t: int):
        pass

class Linear_Gaussian_Object(Simulated_Object):
    def __init__(self, model: Linear_Gauss_Bootstrap, covar_y, x_to_y, device = device):
        super().__init__(model, 20, covar_y.size(0), device)
        self.y_dist = pt.distributions.MultivariateNormal(pt.zeros(covar_y.size(0)), covar_y)
        self.x_to_y = x_to_y

    def observation_generation(self):
        noise = self.y_dist.sample()
        mean = pt.einsum('ij, kj -> ki', self.x_to_y, self.x_t)
        return mean + noise
