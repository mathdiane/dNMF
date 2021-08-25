# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 14:03:18 2021

@author: Amin
"""

import numpy as np
from scipy.optimize import nnls

@staticmethod
def histogram_match(a,b,nbins,type):
    """Takes as input two traces a (moving), b (reference) and outputs 
        normalized trace atransform that has a similar histogram as b
        Args:
            a (numpy.array): moving time series
            b (numpy.array): reference time series
            nbins (integer): number of bins to discretize both time series into
        Returns: 
            atransform (numpy.array): moved time series
            distance (double): histogram distance between atransform and b
    """
    
    #  discarding nans from time series
    a_nan_idx = ~np.isnan(a)
    b_nan_idx = ~np.isnan(b)
    a = a[a_nan_idx]
    b = b[b_nan_idx]
    
    Y = np.linspace(0,1,nbins)
    
    #  discretizing time series using quantiles
    abins = np.quantile(a,Y).T
    bbins = np.quantile(b,Y).T
    
    #  weighted linear regression of the matching quantiles
    if type == 'non-negative':
        beta = nnls(np.concatenate([abins,np.ones((abins.shape[0],1))],0),bbins)
    elif type == 'regular':
        beta = np.linalg.solve(np.concatenate([abins,np.ones((abins.shape[0],1))],0),bbins)
    
    # transformed time series with nan's put back in
    atransform = np.zeros(a_nan_idx.shape)*np.nan
    atransform[a_nan_idx] = a*beta[0] + beta[1]
    
    distance = np.nan
    
    return atransform,distance
