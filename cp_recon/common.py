import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import causal4.utils as c4u
from causal4.Causality import CausalityEstimator
from scipy.sparse.linalg import svds
from pathlib import Path, PosixPath
from typing import Union, Tuple

def estimate_std(rate:float, type:str='total',
                 dt:float=0.02, rate_I:float=None,
                 KN_ratio:float=0.01, EI_ratio:float=4.0,
                 JEE:float = 1.0,  JIE:float = 1.0,
                 JEI:float = -2.0, JII:float = -1.8):
    """ estiamte std of second order derivative of voltage
    Args:
        rate (float): firing rate of excitatory neurons, in unit of kHz
        dt (float): numerical time step, in unit of ms
        rate_I (float): firing rate of inhibitory neurons
        KN_ratio (float): ratio of noise to signal
        EI_ratio (float): ratio of excitatory to inhibitory neurons
        JEE, JIE, JEI, JII: synaptic weights
    Returns:
        std (float): standard deviation of second order derivative of voltage

    """
    if rate_I is None:
        rate_I = rate
    if type == 'total':
        W_E = JEE + JIE / EI_ratio
        W_I = JEI * EI_ratio + JII
        return np.sqrt(2*KN_ratio*dt*(
            EI_ratio * rate * W_E**2  + rate_I * W_I**2
            )/ (EI_ratio + 1.0))
    elif type == 'E':
        W_E = JEE
        W_I = JEI * EI_ratio
        return (1+1/EI_ratio)*np.sqrt(2*KN_ratio*dt*(
            EI_ratio * rate * W_E**2  + rate_I * W_I**2
            )/ (EI_ratio + 1.0))
    elif type == 'I':
        W_E = JIE / EI_ratio
        W_I = JII
        return (1+EI_ratio)*np.sqrt(2*KN_ratio*dt*(
            EI_ratio * rate * W_E**2  + rate_I * W_I**2
            )/ (EI_ratio + 1.0))
    else:
        raise ValueError('type should be total, E or I')

def square_windowed_mean(data, window_size=1000):
    """
    Calculate the square windowed mean of the data.
    """
    n = len(data)
    number_of_means = np.ceil(n / window_size).astype(int)
    if number_of_means*window_size > n:
        data = np.hstack([data, np.nan*np.ones(number_of_means*window_size - n)])
    result = np.nanmean(data.reshape(-1, window_size), axis=1)
    return result

def TCD(ts, data, window_size=1000, return_delta_mean=True):
    """
    Calculate the time course density (TCD) of the data.
    """
    assert len(ts) == len(data), "Length of time series and data must be the same"
    data_mean = square_windowed_mean(data, window_size)
    ts_mean = ts[::window_size][1:]
    if len(ts_mean) == len(data_mean)-1:
        ts_mean = np.concatenate((ts_mean, [ts_mean[-2]*2-ts_mean[-1]]))
    data_mean_delta = np.diff(data_mean)
    abnormal_pt_id = np.argmax(np.abs(data_mean_delta))
    cp_exist = (data_mean_delta[abnormal_pt_id]-data_mean_delta.mean()) > data_mean_delta.std() * 3
    if cp_exist:
        print('Change point exists')
        cp = ts_mean[abnormal_pt_id]
        print((np.abs(data_mean_delta).max()-data_mean_delta.mean())/ data_mean_delta.std())
    else:
        cp = np.nan
        print('No change point')
        print((np.abs(data_mean_delta).max()-data_mean_delta.mean())/ data_mean_delta.std())
    if return_delta_mean:
        return cp, ts_mean[:-1], data_mean_delta
    else:
        return cp

def square_windowed_Fstats(data, window_size=1000):
    """
    Calculate the square windowed mean of the data.
    """
    n = len(data)
    number_of_means = np.ceil(n / window_size).astype(int)
    if number_of_means*window_size > n:
        data = np.hstack([data, np.nan*np.ones(number_of_means*window_size - n)])
    result = np.nanmean(data.reshape(-1, window_size)**2, axis=1)
    return result[1:]/result[:-1]

import scipy.stats as stats 
def TCD_Ftest(ts, data, window_size=1000, p_thresh=1e-5, return_delta_mean=True, verbose=True):
    """
    Calculate the time course density (TCD) of the data.
    """
    assert len(ts) == len(data), "Length of time series and data must be the same"
    data_F = square_windowed_Fstats(data, window_size)
    P = stats.f(window_size, window_size).sf(data_F)
    ts_mean = ts[::window_size][1:]
    if len(ts_mean) == len(data_F)-1:
        ts_mean = np.concatenate((ts_mean, [ts_mean[-2]*2-ts_mean[-1]]))
    abnormal_pt_id = np.nonzero(P < p_thresh)[0]
    if len(abnormal_pt_id) > 0:
        cp = ts_mean[abnormal_pt_id]
        if verbose:
            print('Change point exists at ', cp)
            print('p-value: ', P[abnormal_pt_id])
    else:
        cp = np.nan
        if verbose:
            print('No change point')
    if return_delta_mean:
        return cp, ts_mean, data_F, P
    else:
        return cp