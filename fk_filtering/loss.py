import torch as pt
from torch._tensor import Tensor
from .results import Reporter
from .model import Observation_Queue
from typing import Any, Callable
from abc import ABCMeta
import time
 
class Loss(metaclass=ABCMeta):

    def silly_backward(self, *args, frequency=None, interval=None):
        res = self.forward(*args)
        if interval is None:
            interval = len(res)
        if frequency is None:
            frequency = 1
        res_i = pt.concat([t.reshape(1) for t in res[:interval]])
        l_i = pt.sum(res_i)
        #print(res_i)
        l_i.backward(retain_graph=True)
        if interval == len(res):
            return l_i
        res_c = res[interval:]
        for c, i in enumerate(res_c):
            #pt.cuda.synchronize()
            #s = time.time()
            self.t[0] = c+interval
            #i.register_hook(lambda grad: grad/len(vec))
            if self.t[0] == len(res) - 1:
                i.backward(retain_graph=False)
                i.detach_()
                break
            #if (c+pt.rand())%frequency==0:
            #print(c)
            i.backward(retain_graph=True)
            i.detach_()
            #pt.cuda.synchronize()
            #e = time.time()
            #print(e-s)
        r = pt.sum(pt.tensor(res))
        del res
        return r

    def __init__(self):
        super().__init__()
        self.t = [0]

    def per_point_loss(self, *args) -> pt.Tensor:
        pass

    def forward(self, *args) -> pt.Tensor:
        return pt.mean(self.per_point_loss(*args))

    def __call__(self, *args):
        return self.forward(*args)

class Supervised_L2_Loss(Loss):

    def __init__(self, function):
        super().__init__()
        self.function = function

    def per_point_loss(self, prediction:Reporter, truth:Observation_Queue) -> pt.Tensor:
        try:
            results = self.function(prediction.results)
        except AttributeError:
            results = prediction
        g_truth = self.function(truth.state)[:, :results.size(1), :]
        return pt.sum((results - g_truth)**2, dim=2)
    

class Maze_Supervised_Loss(Loss):
    def __init__(self, maze_no):
        super().__init__()
        if maze_no == 1:
            self.maze_size = (10, 5)
        elif maze_no == 2:
            self.maze_size = (15, 9)
        elif maze_no == 3:
            self.maze_size = (20, 13)
        else:
            raise ValueError('maze_no should be between 1 and 3')
    
    def per_point_loss(self, prediction:Reporter, truth:Observation_Queue) -> pt.Tensor:
        results = prediction.results
        g_truth = truth.state[:, :results.size(1), :]
        a_diff = pt.abs(results[:,:,2:3] - g_truth[:,:,2:3])
        diff = pt.concat(((results[:,:,0:1] - g_truth[:,:,0:1])/(self.maze_size[0]*50), 
                         (results[:,:,1:2] - g_truth[:,:,1:2])/(self.maze_size[0]*50), 
                         (pt.min(a_diff, 2*pt.pi - a_diff))/(2*pt.pi)), dim=2)
        return pt.sum((diff)**2, dim=2)


    
class Magnitude_Loss(Loss):
    def __init__(self, function, sign):
        super().__init__()
        self.function = function
        self.sign = sign

    def per_point_loss(self, reporter) -> pt.Tensor:
        return self.function(reporter.results) * self.sign
    
class AE_Loss(Loss):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def per_point_loss(self, *args) -> Tensor:
        try:
            obs = truth.observations[:,:,o_range:]
        except:
            obs = truth.observations[:,:,o_range[0]:o_range[1]]
        
        obs = obs.reshape(obs.size(0), obs.size(1), )

class Masked_Loss(Loss):
    def __init__(self, loss:Loss):
        super().__init__()
        self.loss = loss

    def per_point_loss(self, mask, *args) -> pt.Tensor:
        return (self.loss.per_point_loss(*args)*mask) * ((mask.size(0)*mask.size(1))/ pt.sum(mask).item())
    
class Compound_Loss(Loss):

    def __init__(self, loss_list):
        super().__init__()
        self.loss_list = loss_list

    def per_point_loss(self, arg_list) -> pt.Tensor:
        for i, (l, a) in enumerate(zip(self.loss_list, arg_list)):
            if i == 0:
                loss = l[1] * l[0].per_point_loss(*a)
            else:
                loss = loss + l[1] * l[0].per_point_loss(*a)
        return loss