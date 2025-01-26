import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple
# add the path to python_bindings to the system path
import sys
sys.path.append('/usr/local/lib/') # in this path the cmake is installing the python bindings library

from python_pcm_inversion import invert_shares

def invert_shares(x : ArrayLike, p : ArrayLike, sigma_p: float, sigma_x: ArrayLike, points : ArrayLike, weights : ArrayLike, data_shares : ArrayLike) -> ArrayLike:
    """
    inverts PCM market shares conditional on all other parameters of the model.

    Assume there are N products and M draws of heterogeneity.

    Parameters
    ----------
    x : ArrayLike size (N, D)
        Array of each product's characteristics. D is the number of dimensions of horizontak differentiation.
    p : ArrayLike size N
        Array of prices. It is important that the prices are sorted in increasing order. 
        If they are not sorted, then computing library will return runtime error.
    sigma_p : float
        Standard deviation of the price coefficient. Log of price elasicity 
        is assumed to be distributed normally with mean 0 and standard deviation sigma_p.
    sigma_x : ArrayLike size D
        Standard deviation of the characteristics coefficients.  Element i of this array is a scale of  
        the marginal utility of the horizontal characteristic i. Average marginal utility on top of vertical quality delta[i]
        is assumed to be zero.
    points : ArrayLike size num_points, each element an array of size M
         Array of points in the numerical integration grid. The grid is integrating the horizontal preferences in the popularion. 
        Each point is representing the shift of the marginal utilities. Id the point is [x0, x1, ..., xM-1], then 
        marginal utility of horizontal characteristic 0 for each product is x0*sigma_x[0], 
        marginal utility of characteristic 1 is x1*sigma_x[1] and so on.
    weights : ArrayLike size M
        Array of weights of the numerical integration grid. One is highly encouraged to have positive weights that sum to 1.
        It is not required, but using negative weights was found to produce negative unconditional market shares for some products, 
        and not summing weights to 1 is just a good practice.
    data_shares : ArrayLike size N
        Array of market shares of all real product (not including the outside option). 

    Returns an array of vertical qualities of size (N) such that if we substitute this solution into unconditional_share_without_jacobian 
    from pcm_market_share library, we get the predicted unconditional market shares equal to data_shares.
    unconditional_share_without_jacobian(d, x, p, sigma_p, sigma_x, points, weights) = data_shares with first argument d being the return of this function.
    """
    return invert_shares(x, p, sigma_p, sigma_x, points, weights, shares)