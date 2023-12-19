import torch as pt
from typing import Callable
from ...model import Auxiliary_Feynman_Kac
import torch.autograd.profiler as profiler
import numpy as np
from time import sleep

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

class Position_Encoder(pt.nn.Module):
    def __init__(self, encoding_dim):
        super().__init__()
        self.NN = pt.nn.Sequential(
            pt.nn.Linear(4, 16),
            pt.nn.ReLU(True),
            pt.nn.Linear(16, 32),
            pt.nn.ReLU(True),
            pt.nn.Linear(32, 64),
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.2),
            pt.nn.Linear(64, encoding_dim),
        )
    
    def forward(self, x_t):
        features = pt.concat((x_t[:, :, :2], pt.sin(x_t[:,:,2:3]), pt.cos(x_t[:,:,2:3])), dim=2)
        return self.NN(features)

class Observation_Encoder(pt.nn.Module):
    def __init__(self, encoding_dim):
        super().__init__()
        self.NN = pt.nn.Sequential(
            pt.nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False), # 16*12*12
            pt.nn.ReLU(True),
            pt.nn.BatchNorm2d(16),
            pt.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 32*6*6
            pt.nn.ReLU(True),
            pt.nn.BatchNorm2d(32),
            pt.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 64*3*3
            pt.nn.ReLU(True),
            pt.nn.BatchNorm2d(64),
            pt.nn.Flatten(),
            pt.nn.Dropout2d(p=0.7),
            pt.nn.Linear(64*3*3, encoding_dim),
        )

    def forward(self, obs):
        true_obs = obs[:, 3:]/255
        true_obs = true_obs.reshape((obs.size(0), 24, 24, 3))
        true_obs = pt.permute(true_obs, (0, 3, 1, 2))
        return self.NN(true_obs)
    
class Observation_Decoder(pt.nn.Module):

    def __init__(self, encoding_dim):
        self.nn = pt.nn.Sequential(
            pt.nn.Linear(encoding_dim, 3 * 3 * 64),
            pt.nn.ReLU(True),
            pt.nn.Unflatten(-1, (64, 3, 3)),  # -1 means the last dim, (64, 3, 3)
            pt.nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False), # (32, 6,6)
            pt.nn.ReLU(True),
            pt.nn.BatchNorm2d(32),
            pt.nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False), # (16, 12,12)
            pt.nn.Dropout2d(p=0.7),
            pt.nn.ReLU(True),
            pt.nn.BatchNorm2d(16),
            pt.nn.ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False), # (3, 24, 24)
            pt.nn.Dropout2d(p=0.7),
            pt.nn.BatchNorm2d(3),
            pt.nn.Sigmoid()
        )
    
    def forward(self, encoded_obs):
        return self.nn(encoded_obs)
    
class likelihood_Net(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.NN = pt.nn.Sequential(
            pt.nn.Linear(1, 8),
            pt.nn.Tanh(),
            pt.nn.Linear(8, 1),
            pt.nn.Sigmoid()
        )
    
    def forward(self, cdist):
        return self.NN(cdist.unsqueeze(2)).squeeze()
    
class FCNN(pt.nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = pt.nn.Sequential(
            pt.nn.Linear(in_dim, hidden_dim),
            pt.nn.Tanh(),
            pt.nn.Linear(hidden_dim, hidden_dim),
            pt.nn.Tanh(),
            pt.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x.float())

class RealNVP(pt.nn.Module):

    def __init__(self, dim, hidden_dim = 4, base_net=FCNN):
        super().__init__()
        self.dim = dim
        self.dim_1 = self.dim - dim//2
        self.dim_2 = self.dim//2
        self.t1 = base_net(self.dim_1, self.dim_2, hidden_dim)
        self.s1 = base_net(self.dim_1, self.dim_2, hidden_dim)
        self.t2 = base_net(self.dim_2, self.dim_1, hidden_dim)
        self.s2 = base_net(self.dim_2, self.dim_1, hidden_dim)

    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                # pass
                pt.nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                # pass
                pt.nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                # pass
                pt.nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                # pass
                pt.nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        lower, upper = x[:, :, :self.dim_1], x[:, :, self.dim_1:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * pt.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * pt.exp(s2_transformed)
        z = pt.cat([lower, upper], dim=2)
        log_det = pt.sum(s1_transformed, dim=2) + \
                  pt.sum(s2_transformed, dim=2)
        return z ,log_det
    
    def inverse(self, z):
        lower, upper = z[:,:,:self.dim_1], z[:,:,self.dim_1:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * pt.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * pt.exp(-s1_transformed)
        x = pt.cat([lower, upper], dim=2)
        log_det = pt.sum(-s1_transformed, dim=2) + \
                  pt.sum(-s2_transformed, dim=2)
        return x, log_det


class Stacked_RealNVP(pt.nn.Module):
    def __init__(self, RealNVP_list) -> None:
        super().__init__()
        self.mods = pt.nn.ModuleList(RealNVP_list)

    def forward(self, x):
        Tx = x
        log_det = 1
        for mod in self.mods:
            Tx, log_det_t = mod.forward(Tx)
            log_det = log_det + log_det_t
        return Tx, log_det
    
    def inverse(self, z):
        Tz = z
        log_det = 1
        for mod in self.modules:
            Tz, log_det_t = mod.forward(Tz)
            log_det = log_det + log_det_t
        return Tz, log_det_t


class Maze_Model_Cos(Auxiliary_Feynman_Kac):
    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, encoding_size, maze_no, device:pt.device = device):
        super().__init__(device)
        self.pos_encoder = Position_Encoder(encoding_size)
        self.obs_encoder = Observation_Encoder(encoding_size)
        self.std = pt.nn.Parameter(pt.rand(1)*0.3)
        self.noise_gen = pt.distributions.Normal(0, 1)
        self.cos_sim = pt.nn.CosineSimilarity(dim = 2)
        if maze_no == 1:
            self.maze_size = (10, 5)
        elif maze_no == 2:
            self.maze_size = (15, 9)
        elif maze_no == 3:
            self.maze_size = (20, 13)
        else:
            raise ValueError('maze_no should be between 1 and 3')
        self.wall_size = 16.125
        self.PF_type = 'Bootstrap'
        
    def M_0_proposal(self, batches: int, n_samples: int):
        init_pos = pt.rand((batches, n_samples, 3), device = device)
        uniform_x = pt.ones(self.maze_size[0])
        uniform_y = pt.ones(self.maze_size[1])
        init_pos = pt.stack((pt.multinomial(uniform_x, (batches * n_samples), True).to(device=device).reshape(batches, n_samples), pt.multinomial(uniform_y, (batches * n_samples), True).to(device=device).reshape(batches, n_samples)), dim=2)
        init_pos = (init_pos + 0.5)*100
        init_pos = pt.concat((init_pos, pt.rand((batches, n_samples, 1), device=device) * pt.pi * 2 - pt.pi), dim=2)
        return init_pos
    
    @staticmethod
    def wrap_angle(angle:pt.Tensor):
        return ((angle - pt.pi) % (2*pt.pi)) - pt.pi

    def M_t_proposal(self, x_t_1, t: int):
        noise = self.noise_gen.sample((x_t_1.size())).to(device=device)
        noise = noise * self.std + 1
        noise = pt.clamp(noise, min=0.1).to(device=device)
        actions = (self.y[t][:, :3]).unsqueeze(1)/noise
        theta = x_t_1[:, :, 2]
        cos_t = pt.cos(theta)
        sin_t = pt.sin(theta)
        maze_frame_actions = pt.empty_like(x_t_1)
        maze_frame_actions[:, :, 0] = actions[:, :, 0] * cos_t - actions[:, :, 1] * sin_t
        maze_frame_actions[:, :, 1] = actions[:, :, 0] * sin_t + actions[:, :, 1] * cos_t
        maze_frame_actions[:, :, 2] = actions[:, :, 2]

        out = x_t_1 + maze_frame_actions
        # out = pt.concat((pt.clamp(out[:,:,0:1], self.wall_size, 100*self.maze_size[0] - self.wall_size), 
        # pt.clamp(out[:,:,1:2], self.wall_size, 100*self.maze_size[1] - self.wall_size), 
        # self.wrap_angle(out[:,:,2:3])), dim =2)
        out = pt.concat((out[:,:,0:1], 
        out[:,:,1:2], 
        self.wrap_angle(out[:,:,2:3])), dim =2)
        return out

    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        pass

    def log_R_t(self, x_t, x_t_1, t: int):
        pass

    def log_f_t(self, x_t, t: int):
        #encoded_obs = self.obs_encoder(self.y[t]/255)
        encoded_obs = self.obs_encoder(self.y[t])
        scaled_x_t = pt.concat(((x_t[:, :, 0:1]/ (self.maze_size[0]* 50)) -1, (x_t[:, :, 1:2]/ (self.maze_size[1]* 50)) -1, x_t[:, :, 2:3]), dim =2)
        encoded_pos = self.pos_encoder(scaled_x_t)
        #return pt.log(self.likelihood_NN(1 - self.cos_sim(encoded_pos, encoded_obs.unsqueeze(1)) + 1e-12))
        return -pt.log(1 - self.cos_sim(encoded_pos, encoded_obs.unsqueeze(1)) + 1e-12)
    

class Maze_Model_NVP(Auxiliary_Feynman_Kac):
    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, encoding_size, maze_no, device:pt.device = device):
        super().__init__(device)
        self.pos_encoder = Position_Encoder(encoding_size)
        self.obs_encoder = Observation_Encoder(encoding_size)
        self.RealNVP = Stacked_RealNVP([RealNVP(encoding_size, 4, FCNN) for _ in range(2)])
        self.std_p = pt.nn.Parameter(pt.rand(1)*40)
        self.std_a = pt.nn.Parameter(pt.rand(1))
        self.std = pt.nn.Parameter(pt.rand(1)*0.3)
        self.noise_gen = pt.distributions.Normal(0, 1)
        self.cos_sim = pt.nn.CosineSimilarity(dim = 2)
        if maze_no == 1:
            self.maze_size = (10, 5)
        elif maze_no == 2:
            self.maze_size = (15, 9)
        elif maze_no == 3:
            self.maze_size = (20, 13)
        else:
            raise ValueError('maze_no should be between 1 and 3')
        self.wall_size = 16.125
        self.PF_type = 'Bootstrap'
        
    def M_0_proposal(self, batches: int, n_samples: int):
        init_pos = pt.rand((batches, n_samples, 3), device = device)
        uniform_x = pt.ones(self.maze_size[0])
        uniform_y = pt.ones(self.maze_size[1])
        init_pos = pt.stack((pt.multinomial(uniform_x, (batches * n_samples), True).to(device=device).reshape(batches, n_samples), pt.multinomial(uniform_y, (batches * n_samples), True).to(device=device).reshape(batches, n_samples)), dim=2)
        init_pos = (init_pos + 0.5)*100
        init_pos = pt.concat((init_pos, pt.rand((batches, n_samples, 1), device=device) * pt.pi * 2 - pt.pi), dim=2)
        return init_pos
    
    @staticmethod
    def wrap_angle(angle:pt.Tensor):
        return ((angle - pt.pi) % (2*pt.pi)) - pt.pi

    def M_t_proposal(self, x_t_1, t: int):
        noise = self.noise_gen.sample((x_t_1.size())).to(device=device)
        noise = noise * self.std + 1
        noise = pt.clamp(noise, min=0.1).to(device=device)
        actions = (self.y[t][:, :3]).unsqueeze(1)/noise
        theta = x_t_1[:, :, 2]
        cos_t = pt.cos(theta)
        sin_t = pt.sin(theta)
        maze_frame_actions = pt.empty_like(x_t_1)
        maze_frame_actions[:, :, 0] = actions[:, :, 0] * cos_t - actions[:, :, 1] * sin_t
        maze_frame_actions[:, :, 1] = actions[:, :, 0] * sin_t + actions[:, :, 1] * cos_t
        maze_frame_actions[:, :, 2] = actions[:, :, 2]

        out = x_t_1 + maze_frame_actions
        # out = pt.concat((pt.clamp(out[:,:,0:1], self.wall_size, 100*self.maze_size[0] - self.wall_size), 
        # pt.clamp(out[:,:,1:2], self.wall_size, 100*self.maze_size[1] - self.wall_size), 
        # self.wrap_angle(out[:,:,2:3])), dim =2)
        out = pt.concat((out[:,:,0:1], 
        out[:,:,1:2], 
        self.wrap_angle(out[:,:,2:3])), dim =2)
        return out

    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        pass

    def log_R_t(self, x_t, x_t_1, t: int):
        pass

    def log_f_t(self, x_t, t: int):
        encoded_obs = self.obs_encoder(self.y[t])
        scaled_x_t = pt.concat(((x_t[:, :, 0:1]/ (self.maze_size[0]* 50)) -1, (x_t[:, :, 1:2]/ (self.maze_size[1]* 50)) -1, x_t[:, :, 2:3]), dim =2)
        encoded_pos = self.pos_encoder(scaled_x_t)
        encoded_dif = encoded_obs.unsqueeze(1) - encoded_pos
        gaussian_dif, scaling = self.RealNVP.forward(encoded_dif)
        return scaling - pt.sum(gaussian_dif**2)/2 - (encoded_obs.size(1)/2)*np.log(2*pt.pi)