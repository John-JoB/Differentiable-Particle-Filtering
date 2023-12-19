import warnings
import numpy as np
from typing import Union, Iterable, Callable
from joblib import cpu_count, Parallel, delayed
import torch as pt

cpu_cores = cpu_count()
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

def normalise_log_quantity(log_weights: pt.Tensor) -> pt.Tensor:
    """
    Take a vector in the log domain and return it normalised in the linear domain

    Parameters
    ----------
    log_weights: (B,N), ndarray
        Unormalised log weights

    Returns
    -------
    normalised_weights: (B,N), ndarray
        Normalised weights

    """
    return log_weights - pt.logsumexp(log_weights, dim=-1, keepdim=True)

def logsubexp(a:pt.Tensor, b:pt.Tensor):
    return a + pt.log(1. - pt.exp(b - a))

def nd_select(select_from, indecies):
    """
        Selecting an arbitarty shape tensor from a 1D tensor.

        Parameters
        -----------
    

    
    """
    index_shape = indecies.size()
    flat_indecies = indecies.flatten()
    flat_out = select_from[flat_indecies]
    return pt.reshape(flat_out, index_shape)

def batched_select(select_from, indecies):
    """
    Choose a 
    """
    flat_indecies = indecies.flatten()
    flat_indecies += pt.arange(flat_indecies.size(0), device = device)*select_from.size(-1)
    return select_from.flatten()[flat_indecies].reshape(indecies.size())



def bin(values, times, sample_points: Union[int, np.ndarray], log: bool = False):
    """Calculate bin averages given either the bin edges of total numbere of bins

    Parameters
    -----------
    values: (N,) ndarray
        An array of the values of the datapoints

    times: (N,) ndarray
        An array of the times at which the datapoints are taken,
        must be the same length as values

    sample_points: ndarray or int
        If sample_points is an ndarray it defines the bin edges
        If an int it defines the number of evenly spaced bins to cover the
        range of times
        Bins are left inclusive, right exclusive except for the final bin which
        is inclusive at both edges

    log: bool, optional, deafault = False
        If sample points is an integer, passing True will evenly space the bins on a
        log scale and False on a linear scale


    Returns
    -------
    bin_avgs: ndarrray
        The average of all points in a bin.
        Empty bins are not returned

    bin_centres:
        The mid-point of the bin, this is irrespective of the distribution of points in
        the bin, so a bin where the only point lies
        near an edge could distort the plot. A common case would be if the times are
        integers and the bin edges have integer spacing,
        in which case the bin_centres would be systematically too high for the data
        taken into account.

    """
    if isinstance(sample_points, int):
        if log:
            sample_points = np.logspace(
                times[0], times[-1], sample_points, dtype=np.float64
            )
        else:
            sample_points = np.linspace(
                times[0], times[-1], sample_points, dtype=np.float64
            )
    sample_indicies = np.searchsorted(times, sample_points)
    bin_centres = (sample_points[:-1] + sample_points[1:]) / 2
    bin_avgs = np.add.reduceat(values, sample_indicies)
    bin_avgs = bin_avgs[:-1]
    samples_per_bin = sample_indicies[1:] - sample_indicies[:-1]
    if sample_points[-1] == times[-1]:
        samples_per_bin[-1] += 1
        bin_avgs[-1] += values[-1]

    non_zeros = np.nonzero(samples_per_bin)
    return bin_avgs[non_zeros] / samples_per_bin[non_zeros], bin_centres[non_zeros]


def parallelise(functions: Iterable[Callable], cores=cpu_cores, backend='loky'):
    """
    Run a set of functions in parallel and return their outputs in a generator

    Parameters
    ----------
        functions: Iterable of functions None -> Object
            An iterable of functions to be evaluated in parallel, the functions should
            not take any input. If len(functions) is expected to be large it more
            efficient to make functions a generator than a list.

        cores: int
            Number of cpu cores to run process across, defaults to maximum availiable.
            If the number given is greater than the maximum then a warning is given,
            and the maximum is used.

    """
    if cpu_cores < cores:
        warnings.warn(
            "Warning, attempting to run particle filters on more cpus than"
            "are availiable, defaulting to max"
        )
        cores = cpu_cores
    return Parallel(n_jobs=cores, return_as="generator", prefer=backend)(
        delayed(f)() for f in functions
    )


def log_multi_gaussian_density(mean, data, covar=None, det_covar=None, inv_covar=None):
    """Calculates the density of multivariate gaussian at a set of samples where
    each sample may have a different mean but must have the same covariance.
    You can provide the covariance or it's determinant and inverse.

    Parameters
    ----------
    mean: (N, d) ndarray
        Mean of each sample, samples along axis 0.

    data: (N, d) ndarray
        Value of each sample

    covar: (d, d) ndarray, optional default: None 
        Covariance of each sample must not be None if either inv_covar or
        det_covar is not none

    det_covar: float, optional, default: None 
        The determinant of the covariance matrix

    inv_covar: (d, d) ndarray, optional, default: None 
        The inverse of the covariance matrix

    Returns
    -------------
    out: (N,) ndarray
        Density of each sample
    """
    if len(mean.shape) == 1:
        return -(np.log(2 * np.pi) + np.log(covar) + (1/covar)*((data-mean)**2))/2 
    if det_covar is None or inv_covar is None:
        det_covar = np.linalg.det(covar)
        inv_covar = np.linalg.inv(covar)
    dim = inv_covar.shape[0]
    prefactor = np.log(det_covar) + dim * np.log(2 * np.pi)
    disp = data - mean
    exponential_term = np.einsum(
        "ij, jk, ik -> i", disp, inv_covar, disp
    )
    return -(prefactor + exponential_term)/2
