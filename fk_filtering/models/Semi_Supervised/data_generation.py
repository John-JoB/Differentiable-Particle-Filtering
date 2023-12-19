import torch as pt
from typing import Callable
from ...model import Observation_Queue
import torch.autograd.profiler as profiler
from numpy import load, ndarray, histogram
from sys import getsizeof

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

def wrap_angle(angle:pt.Tensor):
    return ((angle - pt.pi) % (2*pt.pi)) - pt.pi

def add_noise_to_obs(measurements):
    new_o = pt.zeros([measurements.size(0), measurements.size(1),24, 24, 3], device = device)
    for i in range(measurements.size(0)):
        for j in range(measurements.size(1)):
            offsets = pt.randint(0, 8, (2,), device = device)
            new_o[i, j] = measurements[i, j, offsets[0]:offsets[0] + 24, offsets[1]:offsets[1] + 24, :]
    new_o += pt.normal(0.0, 20, new_o.size(), device=device)
    new_o = pt.clamp(new_o, 0, 255).to(pt.uint8)
    return new_o


def save_maze_data_help(data:ndarray) -> Observation_Queue:
    obs = pt.tensor(data['rgbd'], device=device).reshape((1000, 100, 32, 32, 4)).to(dtype=pt.float32)
    pos = pt.tensor(data['pose'], device=device).reshape((1000, 100, -1)).to(dtype=pt.float32)
    del data
    obs = obs[:,:,:,:,:3]
    pos[:,:,2] = pos[:,:,2]*pt.pi/180
    pos[:,:,2] = wrap_angle(pos[:,:,2])
    d_pos = pos[:, 1:, :] - pos[:, :-1, :]
    d_pos[:, :, 2] = wrap_angle(d_pos[:, : , 2])
    actions = pt.empty_like(pos)
    sin_angle = pt.sin(pos[:, :-1, 2])
    cos_angle = pt.cos(pos[:, :-1, 2])
    actions[:, 0, :] = 0
    actions[:, 1:, 0] = cos_angle * d_pos[:, :, 0] + sin_angle * d_pos[:, :, 1]
    actions[:, 1:, 1] = -sin_angle * d_pos[:, :, 0] + cos_angle*d_pos[:, :, 1]
    actions[:, 1:, 2] = d_pos[:, :, 2]
    action_noise = pt.normal(1, 0.1, size=actions.size(), device=device)
    actions *= action_noise
    obs = add_noise_to_obs(obs)
    obs = pt.flatten(obs, 2, -1)
    return Observation_Queue(xs=pos, ys=(actions, obs))

def save_maze_data(input_loc):
    data = load(f'{input_loc}train.npz')
    yield save_maze_data_help(data)
    data =  load(f'{input_loc}test.npz')
    yield save_maze_data_help(data)

