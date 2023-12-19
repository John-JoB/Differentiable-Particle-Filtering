from .model import (
    Feynman_Kac,
    Simulated_Object,
)
from .utils import normalise_log_quantity
import numpy as np
from typing import Any, Callable, Union, Iterable
from copy import copy
from matplotlib import pyplot as plt
import torch as pt
from .resampling import Resampler
from torch import nn
import torch.autograd.profiler as profiler

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
"""

Main simulation functions for general non-differentiable particle filtering based on 
the algorithms presented in 'Choin, Papaspiliopoulos: An Introduction to Sequential 
Monte Carlo'

"""

 
class Differentiable_Particle_Filter(nn.Module):
    """
    Class defines a particle filter for a generic Feynman-Kac model, the Bootstrap,
    Guided and Auxiliary formulations should all use the same algorithm

    On initiation the initial particles are drawn from M_0, to advance the model a
    timestep call the forward function

    Parameters
    ----------
    model: Feynmanc_Kac
        The model to perform particle filtering on

    truth: Simulated_Object
        The object that generates/reports the observations

    n_particles: int
        The number of particles to simulate

    resampler: func(X, (N,) ndarray) -> X
        This class imposes no datatype for the state, but a sensible definition of
        particle filtering would want it to be an iterable of shape (N,),
        a lot of the code that interfaces with this class assumes that it is an ndarray
        of floats.

    ESS_threshold: float or int
        The ESS to resample below, set to 0 (or lower) to never resample and n_particles
        (or higher) to resample at every step
    
    """

    def __init__(
        self,
        model: Feynman_Kac,
        n_particles: int,
        resampler: Resampler,
        ESS_threshold: Union[int, float], 
        loss,
        threshold: int
        #state_scaling:float = 1., 
        #weight_scaling:float = 1.,
    ) -> None:
        super().__init__()
        self.resampler = resampler
        self.ESS_threshold = ESS_threshold
        self.n_particles = n_particles
        self.model = model
        self.threshold = threshold
        try:
            self.PF_type = self.model.PF_type
        except AttributeError:
            self.PF_type = 'Auxiliary'

    def __copy__(self):
        return Differentiable_Particle_Filter(
            copy(self.model),
            copy(self.truth),
            self.n_particles,
            self.resampler,
            self.ESS_threshold,
        )
    
    def initialise(self, truth:Simulated_Object) -> None:
        self.t = 0
        
        self.truth = truth
        self.model.set_observations(self.truth.get_observation, 0)
        self.x_t = self.model.M_0_proposal(self.truth.state.size(0), self.n_particles)
        self.x_t.requires_grad = True
        if self.PF_type == 'Bootstrap':
            self.log_weights = self.model.log_f_t(self.x_t, 0)
        elif self.PF_type == 'Guided':
            self.log_weights = self.model.log_G_0_guided(self.x_t, self.n_particles)
        else:
            self.log_weights = self.model.log_G_0(self.x_t, self.n_particles)
        self.log_normalised_weights = normalise_log_quantity(self.log_weights)
        self.order = pt.arange(self.n_particles, device=device)
        self.resampled = True
        self.log_weights_1 = pt.ones_like(self.log_weights)*(-np.log(self.n_particles))
    
    class grad_scaling(pt.autograd.Function):
        @staticmethod
        def forward(ctx, tensor, scale):
            ctx.save_for_backward(scale)
            return tensor

        @staticmethod
        def backward(ctx: Any, d_dx_t: Any) -> Any:
            scale = ctx.saved_tensors[0]
            #print(pt.max(d_dx_t))
            #print(pt.argmax(d_dx_t))
            d_dx_t = pt.clamp(d_dx_t, -1., 1.)
            return (scale*d_dx_t), None
        
    def silly_hook_creation(self, t, current, threshold, tensor):
        def silly_hook(grad, t, current, threshold, tensor):
            #if t == 10 or t == 11:
                #tensor[0] +=1
                #tensor[0].detach_()
                #print(id(tensor[0]))
            if current[0] - t < threshold:
                
                return pt.zeros_like(grad)
            return grad
            
        return lambda grad: silly_hook(grad, t, current, threshold, tensor)
    
    @staticmethod
    def grad_scale(grad):
        return grad*0


    def advance_one(self) -> None:
        """
        A function to perform the generic particle filtering loop (algorithm 10.3),
        advances the filter a single timestep.
        
        """

        self.t += 1
        if self.ESS_threshold < self.n_particles:
            mask = (1. / pt.sum(pt.exp(2*self.log_normalised_weights), dim=1)) < self.ESS_threshold
            resampled_x, resampled_w, _ = self.resampler(self.x_t[mask], self.log_normalised_weights[mask])
            self.x_t = self.x_t.clone()
            self.log_weights = self.log_normalised_weights.clone()
            self.x_t[mask] = resampled_x
            self.log_weights[mask] = resampled_w
            self.resampled = False
        else:
            self.x_t, self.log_weights, self.resampled_indices = self.resampler(self.x_t, self.log_normalised_weights)
            self.resampled = True
        self.x_t_1 = self.x_t.clone()
        self.model.set_observations(self.truth.get_observation, self.t)
        self.x_t = self.model.M_t_proposal(self.x_t_1, self.t)
        if self.PF_type == 'Bootstrap':
            self.log_weights += self.model.log_f_t(self.x_t, self.t)
        elif self.PF_type == 'Guided':
            self.log_weights += self.model.log_G_t_guided(self.x_t, self.x_t_1, self.t)
        else:
            self.log_weights += self.model.log_G_t(self.x_t, self.x_t_1, self.t)
        self.log_normalised_weights = normalise_log_quantity(self.log_weights)

    def forward(self, sim_object: Simulated_Object, iterations: int, statistics: Iterable):

        """
        Run the particle filter for a given number of time step
        collating a number of statistics

        Parameters
        ----------
        iterations: int
            The number of timesteps to run for

        statistics: Sequence of result.Reporter
            The statistics to note during run results are stored
            in these result.Reporter objects
        """
        self.initialise(sim_object)

        for stat in statistics:
            stat.initialise(self, iterations)

        for _ in range(iterations + 1):
            for stat in statistics:
                stat.evaluate(PF=self)
            if self.t == iterations:
                break
            self.advance_one()

        return statistics
    

    def display_particles(self, iterations: int, dimensions_to_show: Iterable, dims: Iterable[str], title:str):
        """
        Run the particle filter plotting particle locations in either one or two axes
        for each timestep. First plot after timestep one, each plot shows the current
        particles in orange, the previous particles in blue and, if availiable, the 
        true location of the observation generating object in red.

        Parameters
        ----------
        iterations: int
            The number of timesteps to run for

        dimensions_to_show: Iterable of int 
            Either length one or two, gives the dimensions of the particle state vector
            to plot. If length is one then all particles are plot at y=0.
        
        """
        if self.training:
            raise RuntimeError('Cannot plot particle filter in training mode please use eval mode')
        
        
        for i in range(iterations):
            x_last = self.x_t.clone()
            weights_last = self.log_normalised_weights.clone()
            self.advance_one()
            if len(self.x_t.shape) == 1:
                plt.scatter(x_last, np.zeros_like(self.x_t), marker="x")
                plt.scatter(self.x_t, np.zeros_like(self.x_t), marker="x")
                try:
                    print(self.truth.x_t)
                    print(self.model.y[self.t])
                    plt.scatter([self.truth.state[i+1]].detach(), [0], c="r")
                except AttributeError:
                    pass
                plt.show(block=True)
            elif len(dimensions_to_show) == 1:
                plt.scatter(
                    x_last[:, dimensions_to_show[0]].detach().to(device='cpu'),
                    np.zeros(len(self.x_t)),
                    marker="x",
                )
                plt.scatter(
                    self.x_t[:, dimensions_to_show[0]].detach().to(device='cpu'),
                    np.zeros(len(self.x_t)),
                    marker="x",
                    alpha=pt.exp(self.log_normalised_weights).detach().to(device='cpu')
                )
                try:
                    plt.scatter(self.truth.state[i+1, dimensions_to_show[0]].detach(), 0, c="r")
                except AttributeError:
                    pass
                plt.show(block=True)
            else:
                alpha = pt.exp(weights_last - pt.max(weights_last)).detach().to(device='cpu')
                plt.scatter(
                    x_last[0, :, dimensions_to_show[0]].detach().to(device='cpu'),
                    x_last[0, :, dimensions_to_show[1]].detach().to(device='cpu'),
                    marker="x", 
                    alpha=alpha
                )
                alpha = pt.exp(self.log_normalised_weights - pt.max(self.log_normalised_weights)).detach().to(device='cpu')
                plt.scatter(
                    self.x_t[0, :, dimensions_to_show[0]].detach().to(device='cpu'),
                    self.x_t[0, :, dimensions_to_show[1]].detach().to(device='cpu'),
                    marker="x",
                    alpha=alpha
                )
                plt.legend(['Current timestep particles', 'Previous timestep particles'])
                av = pt.sum(pt.exp(self.log_normalised_weights).unsqueeze(2)*self.x_t, dim=1).detach().cpu().numpy()
                
                try:
                    plt.scatter(
                        self.truth.state[0, i+1, dimensions_to_show[0]].detach().to(device='cpu'),
                        self.truth.state[0, i+1, dimensions_to_show[1]].detach().to(device='cpu'),
                        c="r",
                    )
                    plt.legend(['Previous timestep particles', 'Current timestep particles',  'Current timestep ground truth'])
                except AttributeError:
                    pass
                plt.scatter(av[0, dimensions_to_show[0]], av[0, dimensions_to_show[1]], c="g")
                plt.title(f'{title}: Timestep {i+1}')
                plt.xlabel(dims[0])
                plt.ylabel(dims[1])

                plt.show(block=True)
