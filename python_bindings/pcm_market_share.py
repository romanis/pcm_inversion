import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple
# add the path to python_bindings to the system path
import sys
sys.path.append('/usr/local/lib/') # in this path the cmake is installing the python bindings library

import python_market_share as pms
import time
from  Tasmanian import TasmanianSparseGrid

# expose functions from python_market_share as normal python functions

def conditional_share_with_jacobian(delta : ArrayLike, p : ArrayLike, sigma_p: float, check_positive_shares = True) -> Tuple[ArrayLike, ArrayLike]:
    """
    Computes market share of vertical model conditional on draw of heterogeneity. In addition, also computes Jacobian matrix of 
    derivatives of market shares with respect to delta (vertical quality of each product).

    Assume there are N products.

    Parameters
    ----------
    delta : ArrayLike size N
        Array of each product's vertical qualities.
    p : ArrayLike size N
        Array of prices. It is important that the prices are sorted in increasing order. 
        If they are not sorted, then computing library will return runtime error.
    sigma_p : float
        Standard deviation of the price coefficient. Log of price elasicity 
        is assumed to be distributed normally with mean 0 and standard deviation sigma_p.
    check_positive_shares : bool
        whether to check if all the shares are positive conditional on deltas and prices inside the calculation function. 
        Skipping this check can speed up the calculation, but will cause runtime error if not all shares are indeed positive.

    Returns tuple where the first array of size N. ith element is the market shares of product i.
    The second array is size N, with each element itself an array of size N is the jacobian 
    of the market shares with respect to delta (vertical quality of each product). 
    Jacobian in its matrix form is symmetric and only has elements on 3 diagonals 
    (every product only competed with next more expensive and next less expensive product).


    """
    return pms.conditional_share_with_jacobian(delta, p, sigma_p, check_positive_shares)

def conditional_share_without_jacobian(delta : ArrayLike, p : ArrayLike, sigma_p: float, check_positive_shares = True) -> ArrayLike:
    """
    Computes market share of vertical model conditional on draw of heterogeneity. 

    Assume there are N products.

    Parameters
    ----------
    delta : ArrayLike size N
        Array of each product's vertical qualities.
    p : ArrayLike size N
        Array of prices. It is important that the prices are sorted in increasing order. 
        If they are not sorted, then computing library will return runtime error.
    sigma_p : float
        Standard deviation of the price coefficient. Log of price elasicity 
        is assumed to be distributed normally with mean 0 and standard deviation sigma_p.
    check_positive_shares : bool
        whether to check if all the shares are positive conditional on deltas and prices inside the calculation function. 
        Skipping this check can speed up the calculation, but will cause runtime error if not all shares are indeed positive.

    Returns array of size N market shares of every product.
    """
    return pms.conditional_share_without_jacobian(delta, p, sigma_p, check_positive_shares)

def unconditional_share_with_jacobian(
    delta : ArrayLike, x : ArrayLike, p : ArrayLike, sigma_p: float, sigma_x: ArrayLike, points: ArrayLike, weights: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
    """
    Computes Unconditional market share integrating conditional shares with the numerical integration grid and weights provided.
    . In addition, also computes Jacobian matrix of derivatives of market shares with respect to delta (average vertical quality of each product).

    Assume there are N products. Each product has M horizontal characteristics.

    Parameters
    ----------
    delta : ArrayLike size N
        Array of each product's average vertical qualities. Economically, these are the qualities that all people agree between them.
    x : ArrayLike size N, each element also array of size M
        Array of each product's horizontal qualities. Economically, these are the qualities of each product that people display varying preferences for.
    p : ArrayLike size N
        Array of prices. It is important that the prices are sorted in increasing order. 
        If they are not sorted, then computing library will return runtime error. There can be multiple products with same price.
        If there are products with the same price, then such products have to have some x characteristics that are different, otherwise 
        the model can fit any split of market shares between the products.
    sigma_p : float
        Standard deviation of the price coefficient. Log of price elasicity 
        is assumed to be distributed normally with mean 0 and standard deviation sigma_p.
    sigma_x : ArrayLike size M
        Standard deviation of the horizontal characteristics. Element i of this array is a scale of  
        the marginal utility of the horizontal characteristic i. Average marginal utility on top of vertical quality delta[i]
        is assumed to be zero.
    points : ArrayLike size num_points, each element an array of size M
        Array of points in the numerical integration grid. The grid is integrating the horizontal preferences in the popularion. 
        Each point is representing the shift of the marginal utilities. Id the point is [x0, x1, ..., xM-1], then 
        marginal utility of horizontal characteristic 0 for each product is x0*sigma_x[0], 
        marginal utility of characteristic 1 is x1*sigma_x[1] and so on.
    weights : ArrayLike size num_points
        Array of weights of the numerical integration grid. One is highly encouraged to have positive weights that sum to 1.
        It is not required, but using negative weights was found to produce negative unconditional market shares for some products, 
        and not summing weights to 1 is just a good practice.

    Returns tuple where the first array of size N. ith element is the market shares of product i. The second array is size N, 
    with each element itself an array of size N is the jacobian of the market shares with respect to delta (average vertical quality of each product).
    """
    return pms.unconditional_share_with_jacobian(delta, x, p, sigma_p, sigma_x, points, weights)

def unconditional_share_without_jacobian(
    delta : ArrayLike, x : ArrayLike, p : ArrayLike, sigma_p: float, sigma_x: ArrayLike, points: ArrayLike, weights: ArrayLike
    ) -> ArrayLike:
    """
    Computes Unconditional market share integrating conditional shares with the numerical integration grid and weights provided.
    

    Assume there are N products. Each product has M horizontal characteristics.

    Parameters
    ----------
    delta : ArrayLike size N
        Array of each product's average vertical qualities. Economically, these are the qualities that all people agree between them.
    x : ArrayLike size N, each element also array of size M
        Array of each product's horizontal qualities. Economically, these are the qualities of each product that people display varying preferences for.
    p : ArrayLike size N
        Array of prices. It is important that the prices are sorted in increasing order. 
        If they are not sorted, then computing library will return runtime error. There can be multiple products with same price.
        If there are products with the same price, then such products have to have some x characteristics that are different, otherwise 
        the model can fit any split of market shares between the products.
    sigma_p : float
        Standard deviation of the price coefficient. Log of price elasicity 
        is assumed to be distributed normally with mean 0 and standard deviation sigma_p.
    sigma_x : ArrayLike size M
        Standard deviation of the horizontal characteristics. Element i of this array is a scale of  
        the marginal utility of the horizontal characteristic i. Average marginal utility on top of vertical quality delta[i]
        is assumed to be zero.
    points : ArrayLike size num_points, each element an array of size M
        Array of points in the numerical integration grid. The grid is integrating the horizontal preferences in the popularion. 
        Each point is representing the shift of the marginal utilities. Id the point is [x0, x1, ..., xM-1], then 
        marginal utility of horizontal characteristic 0 for each product is x0*sigma_x[0], 
        marginal utility of characteristic 1 is x1*sigma_x[1] and so on.
    weights : ArrayLike size num_points
        Array of weights of the numerical integration grid. One is highly encouraged to have positive weights that sum to 1.
        It is not required, but using negative weights was found to produce negative unconditional

    Returns array of size N market shares of every product.
    """
    return pms.unconditional_share_without_jacobian(delta, x, p, sigma_p, sigma_x, points, weights)
