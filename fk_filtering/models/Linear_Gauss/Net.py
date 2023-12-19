import torch as pt
from typing import Callable
from ...model import Auxiliary_Feynman_Kac
import torch.autograd.profiler as profiler

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

class LikelihoodNet(pt.nn.Module):
    def __init__(self, layer_info) -> None:
        super().__init__()
        layers = [pt.nn.Linear(layer_info[i], layer_info[i+1]) for i in range(len(layer_info) - 1)]
        Relu_list = [pt.nn.ReLU()]*(len(layer_info) -2)
        temp_list = [None]*(len(layer_info)*2 - 3)
        temp_list[::2] = layers
        temp_list[1:-1:2] = Relu_list
        self.stack = pt.nn.Sequential(*tuple(temp_list))

    def forward(self, x_t):
        return self.stack(x_t).squeeze(2)
    

class Gaussian_Test_Bootstrap(Auxiliary_Feynman_Kac):
    def get_modules(self):
        return self.obs_model

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, rho:pt.Tensor, covar_x:pt.Tensor, init_covar_x:pt.Tensor, obs_model):
        super().__init__()
        # Precalculate the Cholesky decomposition to avoid doing it at every timestep
        self.obs_model = obs_model
        self.rho = rho.to(device=device)
        self.x_dist = pt.distributions.MultivariateNormal(pt.zeros(covar_x.size(0)), covar_x)
        
        self.init_x_dist = pt.distributions.MultivariateNormal(pt.zeros(init_covar_x.size(0)), init_covar_x)

    def M_0_proposal(self, batches:int, n_samples: int):
        return self.init_x_dist.sample((batches, n_samples)).to(device=device)
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample((x_t_1.size(0), x_t_1.size(1))).to(device=device)
        means = pt.einsum('ij, bkj -> bki', self.rho, x_t_1)
        return means + noise
    
    def log_eta_t(self, x_t, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros((x_0.size(0), x_0.size(1)), device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)

    def log_f_t(self, x_t, t: int):
        expanded_y = self.y[t].unsqueeze(1).expand(-1, x_t.size(1), -1)
        out = self.obs_model(pt.concat((x_t, expanded_y), dim =2))
        if pt.isnan(out).any():
            if pt.isnan(x_t).any():
                print('x_t issue')
                raise SystemExit(0)
            if pt.isnan(expanded_y).any():
                print('y_t issue')
                raise SystemExit(0)
            print('NN issue')
            for p in self.obs_model.parameters():
                print(p)
            raise SystemExit(0)
        return out
    

class Gaussian_Det_Bootstrap(Auxiliary_Feynman_Kac):
    def get_modules(self):
        return self.obs_model

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, rho:pt.Tensor, covar_x:pt.Tensor, init_covar_x:pt.Tensor):
        super().__init__()
        # Precalculate the Cholesky decomposition to avoid doing it at every timestep
        self.rho = rho.to(device=device)
        self.x_dist = pt.distributions.MultivariateNormal(pt.zeros(covar_x.size(0)), covar_x)
        
        self.init_x_dist = pt.distributions.MultivariateNormal(pt.zeros(init_covar_x.size(0)), init_covar_x)
        self.PF_type ='Bootstrap'

    def M_0_proposal(self, batches:int, n_samples: int):
        return self.init_x_dist.sample((batches, n_samples)).to(device=device)
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample((x_t_1.size(0), x_t_1.size(1))).to(device=device)
        means = pt.einsum('ij, bkj -> bki', self.rho, x_t_1)
        return means + noise
    
    def log_eta_t(self, x_t, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros((x_0.size(0), x_0.size(1)), device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)

    def log_f_t(self, x_t, t: int):
        return -50*pt.cdist(0.9*x_t, self.y[t].unsqueeze(1)).squeeze()**2
    

class Gaussian_Test_Uniform_Bootstrap(Auxiliary_Feynman_Kac):
    def get_modules(self):
        return None

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, rho:pt.Tensor, covar_x:pt.Tensor, init_covar_x:pt.Tensor):
        super().__init__()
        self.rho = rho.to(device=device)
        self.x_dist = pt.distributions.MultivariateNormal(pt.zeros(covar_x.size(0)), covar_x)
        
        self.init_x_dist = pt.distributions.MultivariateNormal(pt.zeros(init_covar_x.size(0)), init_covar_x)

    def M_0_proposal(self, batches:int, n_samples: int):
        return self.init_x_dist.sample((batches, n_samples)).to(device=device)
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample((x_t_1.size(0), x_t_1.size(1))).to(device=device)
        means = pt.einsum('ij, bkj -> bki', self.rho, x_t_1)
        return means + noise
    
    def log_eta_t(self, x_t, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros((x_0.size(0), x_0.size(1)), device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)

    def log_f_t(self, x_t, t: int):  
        #print(-50*pt.cdist(self.y[t], x_t)**2)
        return -10*pt.cdist(0.5*x_t, self.y[t].unsqueeze(1)).squeeze()**2
    

class Gaussian_Test_Guided(Auxiliary_Feynman_Kac):
    def get_modules(self):
        return None

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, covar_x:pt.Tensor, init_covar_x:pt.Tensor):
        super().__init__()
        self.x_dist = pt.distributions.MultivariateNormal(pt.zeros(covar_x.size(0)), covar_x)
        self.sigma_a_b = pt.eye(1)*(1/(1 + (0.5**2) * 10))
        self.theta_hat_1 = (0.9*self.sigma_a_b).to(device=device)
        self.theta_hat_2 = (5*self.sigma_a_b).to(device=device)
        self.prop_dist = pt.distributions.MultivariateNormal(pt.zeros(1), self.sigma_a_b)
        self.init_x_dist = pt.distributions.MultivariateNormal(pt.zeros(init_covar_x.size(0)), init_covar_x)
        self.sigma_a_b = self.sigma_a_b.to(device=device)
        self.PF_type = 'Guided'

    def M_0_proposal(self, batches:int, n_samples: int):
        noise = self.init_x_dist.sample((batches, n_samples)).to(device=device)
        means = (self.y[0]*self.theta_hat_2).unsqueeze(1) + noise
        return means + noise
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample((x_t_1.size(0), x_t_1.size(1))).to(device=device)
        means = self.theta_hat_1*x_t_1 + (self.theta_hat_2*self.y[t]).unsqueeze(1)
        return means + noise
    
    def log_eta_t(self, x_t, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)

    def log_R_0(self, x_0, n_samples: int):
        prop_density = pt.cdist(x_0, (self.theta_hat_2*self.y[0]).unsqueeze(1)).squeeze()**2 + pt.log(self.sigma_a_b)
        dyn_density = pt.sum(x_0**2, dim=2)
        return prop_density - dyn_density

    def log_R_t(self, x_t, x_t_1, t: int):
        prop_density = pt.sum((self.theta_hat_1*x_t_1 + (self.theta_hat_2*self.y[t]).unsqueeze(1) - x_t)**2, dim=2) + pt.log(self.sigma_a_b)
        dyn_density = pt.sum((0.9*x_t_1 - x_t)**2, dim=2)
        return prop_density - dyn_density

    def log_f_t(self, x_t, t: int):  
        #print(-50*pt.cdist(self.y[t], x_t)**2)
        return -10*pt.cdist(0.5*x_t, self.y[t].unsqueeze(1)).squeeze()**2

class Jiaxi_test(Auxiliary_Feynman_Kac):
    class Parameter_Wrapper(pt.nn.Module):
        def __init__(self, D:int):
            super().__init__()
            self.theta1 = pt.nn.Parameter(pt.ones(1))
            self.theta2 = pt.nn.Parameter(pt.ones(1))
            self.M_0 = pt.nn.Linear(D, D, bias=False)
            self.M_t = pt.nn.Linear(2*D, D, bias=False)
            self.SD_0 = pt.nn.Parameter(pt.rand(D))
            self.SD_t = pt.nn.Parameter(pt.rand(D))
        
        def forward(self):
            pass

    def get_modules(self):
        return self.parameters

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, D:int, var_x:float, var_y:float):
        super().__init__()
        self.parameters = self.Parameter_Wrapper(D)
        self.D = D
        self.deterministic_prop_dist = pt.distributions.MultivariateNormal(pt.zeros(D), pt.eye(D))
        self.inv_x = pt.ones([1], device=device)*(1/var_x)
        self.inv_y = pt.ones([1], device=device)*(1/var_y)
        self.deterministic_prop_dist_0 = pt.distributions.MultivariateNormal(pt.zeros(D), pt.eye(D))
        self.PF_type = 'Guided'
        

    def M_0_proposal(self, batches:int, n_samples: int):
        noise = self.deterministic_prop_dist_0.sample((batches, n_samples)).to(device=device)
        means = self.parameters.M_0(self.y[0])
        return noise*pt.abs(self.parameters.SD_0.unsqueeze(1)) + means.unsqueeze(1)
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.deterministic_prop_dist.sample((x_t_1.size(0), x_t_1.size(1))).to(device=device)
        means = self.parameters.M_t(pt.concat((x_t_1, self.y[t].unsqueeze(1).expand(x_t_1.size())), dim = 2))
        return means + noise*pt.abs(self.parameters.SD_t.unsqueeze(1))
    
    def log_eta_t(self, x_t, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)

    def log_R_0(self, x_0, n_samples: int):
        prop_density = pt.cdist(x_0, self.parameters.M_0(self.y[0]).unsqueeze(1)).squeeze()**2 + pt.log(self.parameters.SD_0**2)
        dyn_density = pt.sum(x_0**2, dim=2)
        return prop_density - dyn_density

    def log_R_t(self, x_t, x_t_1, t: int):
        with profiler.record_function('prop'):
            prop_density = pt.sum((self.parameters.M_t(pt.concat((x_t_1, self.y[t].unsqueeze(1).expand(x_t_1.size())), dim = 2)) - x_t)**2, dim=2) + pt.log(self.parameters.SD_t**2)
        with profiler.record_function('dyn'):
            dyn_density = self.inv_x*pt.sum((self.parameters.theta1*x_t_1 - x_t)**2, dim=2) - pt.log(self.inv_x)
        with profiler.record_function('sub'):
            return prop_density - dyn_density

    def log_f_t(self, x_t, t: int):  
        with profiler.record_function('f_t'):
            return -self.inv_y*pt.sum((self.parameters.theta2*x_t - self.y[t].unsqueeze(1).expand(x_t.size()))**2, dim=2) + pt.log(self.inv_y)