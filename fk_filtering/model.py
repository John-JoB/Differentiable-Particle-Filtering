from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Callable, Iterable, Generator
import copy
from .utils import parallelise
import torch as pt
from joblib import cpu_count
import os
import shutil


cores = cpu_count()
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

class Feynman_Kac(pt.nn.Module, metaclass=ABCMeta):

    """
    Abstract base class for a Feynman-Kac model with all the functions required for
    particle filtering needing implementation

    To any model for particle filtering should sub-class 'Feynman_Kac'

    M_0_proposal and M_t_proposal should sample from their respective distributions

    G_0, G_t are pointwise density evaluation functions

    For generality and notational consistency, the observations are treated like model
    parameters and are set at each time-step rather than passed as function parameters
    For convienence I provide a separate update function for them

    I allow for the model to take any number of parameters, that should be passed as
    keyword arguments to __init__, and may be updated at anytime via the
    set_model_parameters method

    I override the copy functionality to create a copy but with new rng.

    Get modules should return the all the neural networks in the model as a ModuleList

    Parameters
    ----------
    **kwargs: any
        The model parameters to be passed to set_model_parameters

    """

    class reindexed_array:
        """
        Inner class to reindex an array, should be taken as immutable and read-only
        after creation
        Intended usage is to access observations at time t as y[t]

        Parameters
        ----------
        base_index: int
            The desired index of the first stored item.

        *args: any
            Arguments passed to np.array() for array creation

        **kwargs: any
            Arguments passed to np.array() for array creation
            
        """

        def __init__(self, base_index: int, ls):
            super().__init__()
            self.array = ls
            self.base_index = base_index
            self.device = device

        def __getitem__(self, index):
            return self.array[index - self.base_index]
        

    @abstractmethod
    def set_observations(self, get_observation: Callable, t: int):
        pass

    def __init__(self, device:pt.device=device) -> None:
        super().__init__()
        self.device = device
        self.rng = pt.Generator(device=self.device)
        

    def __copy__(self):
        """Copy the object but with new rng"""
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.set_rng()
        return out

    def set_rng(self, rng: pt.Generator = None) -> None:
        """If reproducable results are required then the random generator can be
        specified manually"""

        if rng is None:
            self.rng = pt.Generator(device=self.device)
            return
        self.rng = rng

    # Evaluate G_0
    @abstractmethod
    def log_G_0(self, x_0, n_samples):
        pass

    # Sample M_0
    @abstractmethod
    def M_0_proposal(self, n_samples: int):
        pass

    # Evaluate G_t
    @abstractmethod
    def log_G_t(self, x_t, x_t_1, t: int):
        pass

    # Sample M_t
    @abstractmethod
    def M_t_proposal(self, x_t_1, t: int):
        pass


class Auxiliary_Feynman_Kac(Feynman_Kac, metaclass=ABCMeta):

    """
    Base class for an auxiliary Feynman-Kac model

    Notes
    ------
    R_t are the Raydon-Nikodym derivatives M_t(x_t-1, dx_t) / P_t(x_t-1, dx_t) and
    should be computable for standard useage

    I provide the standard form for calculating the auxiliary weight functions G_t
    but they can be overridden with a direct calculation if desired
    either for performance or that it is possible to particle filter if the
    M_t(x_t-1, dx_t) / P_t(x_t-1, dx_t) are not computable/divergent but the
    f(x_t)M_t(x_t-1, dx_t) / P_t(x_t-1, dx_t) are. In in which case you should not
    define the ratios R. Although it will not be possible
    to recover the predictive distribution through the usual importance
    sampler in this case.

    """

    @abstractmethod
    def log_R_0(self, x_0, n_samples: int):
        pass

    @abstractmethod
    def log_R_t(self, x_t, x_t_1, t: int):
        pass

    @abstractmethod
    def log_f_t(self, x_t, t: int):
        pass

    @abstractmethod
    def log_eta_t(self, x_t, t: int):
        pass

    def log_G_0_guided(self, x_0, n_samples: int):
        return self.log_R_0(x_0, n_samples) + self.log_f_t(x_0, 0)

    def log_G_t_guided(self, x_t, x_t_1, t: int):
        return self.log_R_t(x_t, x_t_1, t) + self.log_f_t(x_t, t)

    def log_G_0(self, x_0, n_samples: int):
        return self.log_G_0_guided(x_0, n_samples) + self.log_eta_t(x_0, 0)

    def log_G_t(self, x_t, x_t_1, t: int):
        return (
            self.log_G_t_guided(x_t, x_t_1, t)
            + self.log_eta_t(x_t, t)
            - self.log_eta_t(x_t_1, t - 1)
        )


class State_Space_Object(metaclass=ABCMeta):
    """
    Base class for a generic state space object that can generate observations and
    update it's state

    The true state need not be availiable in which case this object can act like a
    queue of observations and return NaN whenever asked to evaluate the true state

    I keep a list that is assumed to contain observations at successive timesteps
    as well as an indexing variable to store the timestep of the first value in the
    list. Each time an observation is required, if it exists return it, if not
    advance the state and add observations sequentially to an array until the
    desired time is reached.

    I also provide copy functionality so that
    a copied state space object is a new state space object with the same RNG seeds
    so that running it again will produce consistent results

    Parameters
    -----------
    observation_history_length: int
        The number of observations to keep at any timestep

    observation_dimension: int
        The dimension of the observation vector

    Notes
    --------
    It is not recomended to interface with the class outside creation and the
    get_observation() method.

    The rng for transitioning states and for returning observations are separate
    because they can be done in different orders e.g. states 1:N can be generated
    without accessing any observations, but both states and observations are
    sequentially generated within their own series. Obviously do not try to access
    the rng generators outside of a child class. If you have time varying
    components to a subclass, the subclass must also override __copy__ and reset the
    relavent class state.

    """

    def __copy__(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.state_rng = pt.Generator(device=self.device)
        out.state_rng.manual_seed(out.state_seed)
        out.observation_rng = pt.Generator(device=self.device)
        out.observation_rng.manual_seed(out.observation_seed)
        out.observations = pt.empty_like(out.observations, device=self.device)
        out.time_index = 0
        out.object_time = 0
        out.first_object_set = False
        return out

    def __init__(self, observation_history_length: int, observation_dimension: int, device:pt.device = device):
        self.device = device
        self.state_rng = pt.Generator(device=self.device)
        self.state_seed = self.state_rng.initial_seed()
        self.observation_rng = pt.Generator(device=self.device)
        self.observation_seed = self.observation_rng.initial_seed()
        self.observation_history_length = observation_history_length
        self.observation_dimension = observation_dimension
        self.observations = pt.empty(
            [self.observation_history_length * 2, self.observation_dimension],
            device=self.device
        )
        self.first_object_set = False
        self.time_index = 0
        self.object_time = 0
        

    @abstractmethod
    def observation_generation(self):
        pass

    def forward(self):
        "Advance the state a timestep"
        self.object_time += 1

    def set_observation(self, t: int, value: pt.Tensor) -> None:
        """

        Update observation history with a new observation, if the observation is
        history is full copy the second half of the values stored to the first half of
        the array and start filling from the half way point

        Parameters
        ----------
            t: int
                Timestep of new observation

            value: ndarray
                Value of new observation

        """
        if self.time_index + self.observation_history_length * 2 <= t:
            self.observations[: self.observation_history_length] = self.observations[
                self.observation_history_length :
            ]
            self.time_index += self.observation_history_length
        self.observations[t - self.time_index, :] = value.squeeze()

    def get_observation(self, t):
        """
        Fetch observation at time t if it is not created then advance the object
        state and generate observations until time t
        """
        if t < 0:
            return pt.tensor([pt.nan]*self.observation_dimension, device=self.device)
        
        if t < self.time_index:
            raise ValueError(
                f"Trying to access observation at time {t}, "
                f"the earliest stored is at time {self.time_index}"
            )

        if t == 0 and not self.first_object_set:
            self.first_object_set = True
            self.set_observation(0, self.observation_generation())
            return self.observations[0]

        if t > self.object_time:
            for t_i in range(self.object_time + 1, t + 1):
                self.forward()
                self.set_observation(t_i, self.observation_generation())
        return self.observations[t - self.time_index]

    def true_function(self, function_of_interest: Callable):
        "Get a function of the current object state"
        try:
            return function_of_interest(self.x_t)
        except AttributeError:
            raise AttributeError(
                "This state space object does not have access to" "its true state"
            )


class Simulated_Object(State_Space_Object, metaclass=ABCMeta):

    """
    
    Base class for a simulated object. This object simulates a hidden Markov process,
    for an interpretable output the given model should always be the Bootstrap FK
    regardless of what algorithm is going to be used to filter.

    Parameters
    ----------
    In addition to the parameters of the State_Space_Object class this
    class also takes:

    model: Feynman_Kac
        The Feynman-Kac model to simulate from, in most cases this model should be
        Bootstrap

    """

    def __copy__(self):
        out = super().__copy__()
        out.model = copy.copy(out.model)
        out.model.set_rng(out.state_rng)
        out.x_t = out.model.M_0_proposal(1)
        return out

    def __init__(
        self,
        model: Feynman_Kac = None,
        observation_history_length: int = None,
        observation_dimension: int = None,
        device: pt.device = device
    ) -> None:
        super().__init__(observation_history_length, observation_dimension, device)
        self.model = model
        self.model.set_rng(self.state_rng)
        self.x_t = self.model.M_0_proposal(1)

    def forward(self):
        super().forward()
        self.x_t = self.model.M_t_proposal(self.x_t, self.object_time)


class Observation_Queue(State_Space_Object):
    """
    
    State space object act as a queue of observations and (optionally) state vectors
    Reimplements some methods in a simplified way to be more efficient for this
    special case.


    Parameters
    ----------
    xs: (T,s) ndarray or None, default: None
        An array containing the state of dimension s at every time in [0,T].
        If None and ys is not None then observations are not stored.
        Has no effect if ys is None.

    ys: (T, o) ndarray or None, default: None
        An array containing the observations of dimension s at every time in [0,T].
        If None then generate observations from the State_Space_Object
        converstion_object.

    conversion_object: State_Space_Object, default: None
        A state_space_object to have its observations and state (if availiable)
        memorised as a new Observation_Queue object.
        Must not be None if ys is None, using ys to load observations take priority
        otherwise.

    time_length: int or None, default: None
        The number of time steps of conversion_object to memorise
        Has no effect if conversion object is None.
        Must not be None if conversion_object is not None.

    Notes
    -------
    Calling set_observation from outside the class is already
    not recommended but will have even less predictable results if the object is
    an Observation_Queue

    """

    def __init__(self, xs: pt.Tensor = None, ys: pt.Tensor = None, conversion_object: State_Space_Object = None, time_length: int = None, device: pt.device = device):
        self.device = device
        self.object_time = 0
        if ys is not None:
            self.observations = ys
            if xs is not None:
                self.state = xs
            self.x_t = self.state[0]
            return

        

        try:
            state_dim = conversion_object.x_t.size(-1)
            self.state = pt.empty((state_dim[0], time_length + 1, state_dim), device=self.device)
            self.x_t = conversion_object.x_t
            state_availiable = True
        except AttributeError:
            state_availiable = False

        

        for t in range(time_length + 1):
            if t == 0:
                self.observations = pt.empty((conversion_object.get_observation(t).size(0), time_length + 1, conversion_object.observation_dimension))
            self.observations[t] = conversion_object.get_observation(t)
            if state_availiable:
                self.state[t] = conversion_object.x_t

    def __copy__(self):
        """
        Return a new Observation_Queue with the same
        observations and state set at time 0
        
        """
        try:
            out = Observation_Queue(xs=self.state, ys=self.observations, device=self.device)
        except AttributeError:
            out = Observation_Queue(ys=self.observations, device=self.device)
        return out

    def observation_generation(self):
        pass

    def forward(self):
        super().forward()
        try:
            self.x_t = self.state[:, self.object_time]
        except AttributeError:
            pass

    def get_observation(self, t):
        if t > self.object_time:
            self.forward()
        if isinstance(self.observations, tuple):
            return pt.concat([o[:, t, :] for o in self.observations], dim=1)
        else:
            return self.observations[:, t, :]
    

def create_simulated_files(path:str, T:int, make_object:Generator) -> None:

    if os.path.exists(path):
        print(f'Warning: This will overwrite the directory at path {path}')
        response = input('Input Y to confirm you want to do this:')
        if response != 'Y' and response != 'y':
            print('Halting')
            return
        try:
            shutil.rmtree(path)
        except:
            os.remove(path)
    
    os.mkdir(path)
    with pt.inference_mode():
        i=0
        for sim_object in make_object:
            if isinstance(sim_object, Observation_Queue): 
                batched_o = sim_object
            else:
                batched_o = Observation_Queue(conversion_object=make_object, time_length=T, device=device)

            if isinstance(batched_o.observations, tuple):
                for j in range(batched_o.observations[0].size(0)):
                    try:
                        pt.save(Observation_Queue(xs=batched_o.state[j, :T+1, :].clone(), ys=tuple(o[j, :T+1, :].clone() for o in batched_o.observations)), f'{path}{i}')
                    except:
                        pt.save(Observation_Queue(ys=tuple(o[j, :T+1, :].clone() for o in batched_o.observations)), f'{path}{i}')
                    i += 1
                continue
            
            for j in range(batched_o.observations.size(0)):
                try:
                    pt.save(Observation_Queue(xs=batched_o.state[j, :T+1, :].clone(), ys=batched_o.observations[j, :T+1, :].clone()), f'{path}{i}')
                except:
                    pt.save(Observation_Queue(ys=batched_o.observations[j, :T+1, :].clone()), f'{path}{i}')
                i += 1
    
    
    
class SimulatedDataset(pt.utils.data.Dataset):

    def __init__(self, path:str, lazy:bool=True) -> None:
        self.lazy = lazy
        if self.lazy:
            self.files = os.listdir(path)
            self.dir = path
            return
        
        files = os.listdir(path)
        dir = path
        self.data = [pt.load(f"{dir}/{file}") for file in files]
    
    def __len__(self):
        if self.lazy:
            return len(self.files)
        return len(self.data)
    
    def __getitem__(self, idx:int):
        if self.lazy:
            return pt.load(f"{self.dir}/{self.files[idx]}")
        return self.data[idx]
    
    def collate(self, batch:Iterable[Observation_Queue]):
        x_batch = pt.utils.data.default_collate([b.state for b in batch]).to(device=device)
        if isinstance(batch[0].observations, tuple):
            y_batch = tuple(pt.utils.data.default_collate([b.observations[i] for b in batch]).to(device=device) for i in range(len(batch[0].observations)))
        else:
            y_batch = pt.utils.data.default_collate([b.observations for b in batch]).to(device=device)
        return Observation_Queue(x_batch, y_batch)