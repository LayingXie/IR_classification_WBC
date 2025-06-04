# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:10:01 2023

@author: mu24huj
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import namedtuple
import pandas as pd
from UtilsSpec import modified_z_scores, peak_fit, fit_degree, unifxaxis, interpfunc
from scipy.interpolate import splrep, splev
import seaborn as sns
import scipy.interpolate as interp
import matplotlib.colors as mcolors

def trim(x, trim_coef=0.0):
    """
    Function that cuts smallest and largest values in the list.

    :param x: input list
    :param trim_coef: portion of elements removed
    :return: output array
    """
    _x = np.asarray(x)
    x_abs = np.abs(_x)
    return _x[x_abs <= np.quantile(x_abs, 1 - trim_coef)]

def despike(spec, two_spec=False, despike_threshold=10, despike_order=2,
            win_smooth=9, win_bg=25, win=1, stop_callback=lambda: None):
    """
    Spectral data despiking.

    :param spec: SpData object, ndarray, or list with spectroscopic data corrupted by cosmic ray noise
    :param two_spec: indicate if in each point two spectra were measured
    :param despike_threshold: threshold for spike detection (in robust SD after enhancement)
    :param despike_order: order of discriminant for spike enhancement
    :param win_smooth: smoothing window for SD values used for spike detection (if two_spec: False)
    :param win_bg: window for baseline correction used in two spectra to find values that can
        replace despiked pixels (if two_spec: True)
    :param win: window around a spike to be corrected
    :param stop_callback: function that raises error when the execution should be stopped
    :return: SpData object, ndarray, or list with spike corrected spectral data
    """

    def despike_single(_spec):  # despiking of a single spectrum or for two spectra
        _two_spec = len(_spec) > 1  # check number of spectra
        if _two_spec:
            spectr = np.asarray(_spec)
            d_spec = np.diff(spectr, despike_order)  # calculate derivatives
            diff = np.abs(np.diff(d_spec, 1, 0)).max(axis=0)  # find differences between derivatives
        else:
            spectr = np.asarray(_spec[0])  # get a single spectrum
            d_spec = np.abs(np.diff(spectr, despike_order))  # calculate derivative of the spectrum
            # difference between derivative and smoothed derivative
            diff = d_spec - median_filter(d_spec, win_smooth, mode='nearest')

        robust_sd = np.std(trim(diff, 0.01))
        # find spikes by thresholding the difference
        spikes = np.array(diff) > despike_threshold * robust_sd
        if spikes.any():
            wx = np.hstack([np.repeat(False, (np.floor(despike_order / 2))), spikes,
                            np.repeat(False, (np.ceil(despike_order / 2)))])
            wx_range = np.flatnonzero(np.convolve(wx, [True] * (2 * win_bg + 1), mode='same'))
            wx_range = np.arange(wx_range.min(), wx_range.max() + 1)
            wx_win = np.convolve(wx, [True] * (2 * win + 1), mode='same')
            wx_win_tmp = wx_win[wx_range]
            if _two_spec:
                # two-spectra method
                tmp = spectr[:, wx_range]
                bg = np.apply_along_axis(snip, 1, tmp, 5)
                tmp = tmp - bg
                factor = np.apply_along_axis(lambda x: np.average(trim(x, 0.1)), 1, spectr)
                w_spec = np.argmax(np.abs(tmp[:, wx_win_tmp]), 0)
                for wsp in sorted(set(w_spec)):
                    wi = w_spec == wsp
                    patch = np.delete(tmp[:, wx_win_tmp][:, wi], wsp, 0).sum(0)
                    fac = np.delete(factor, wsp).sum()
                    patch = patch * factor[wsp] / (1 if fac == 0 else fac)
                    spectr[wsp, np.flatnonzero(wx_win)[wi]] = patch + bg[wsp, np.flatnonzero(wx_win_tmp)[wi]]
            else:
                # one-spectrum method
                spectr[wx_win] = median_filter(spectr[wx_range], win_smooth, mode='nearest')[wx_win_tmp]
        return spectr

    if isinstance(spec[0], (np.ndarray, list)):
        # for two-spectra method step size 2 in a loop and slice size 2 are used
        if two_spec:
            for i in range(0, len(spec), 2):
                stop_callback()
                w = slice(i, i + 2, 1)
                spec[w] = list(despike_single(spec[w]))
        else:
            for i, s in enumerate(spec):
                stop_callback()
                spec[i] = despike_single([s])
        return spec
    else:
        return despike_single([spec])


def snip(x, iterations=100, smoothing_window=9, lls=True, return_baseline=True):
    """
    Sensitive Nonlinear Iterative Peak (SNIP) clipping algorithm.

    :param x: SpData object, ndarray, or list with spectroscopic data
    :param iterations: number of iterations used in SNIP algorythm
    :param smoothing_window: smoothing window (default: 9); if 2 or less - smoothing is disabled
    :param lls: indicates if log-log-square_root (LLS) operator should be applied prior to baseline estimation
        (default: True)
    :param return_baseline: indicates if estimated baseline (default: True) or corrected spectra should be returned
    :return: SpData object, ndarray, or list (depending on input) with estimated baselines or corrected spectra
    """

    def snip_single(_x):  # SNIP baseline correction
        # BUG: in debug mode savgol_filter causes a ValueError error in python 3.10
        bg = savgol_filter(_x.copy(), smoothing_window, 2) if smoothing_window > 2 else np.asarray(_x.copy())

        bg[bg < 0] = 0
        if lls:  # log-log-square_root (LLS) operator
            bg = np.log(np.log(np.sqrt(np.maximum(bg, 0) + 1) + 1) + 1)

        for p in range(1, iterations + 1, 1):  # optimized snip loop (about 12 times faster)
            bg[p:-p] = np.minimum(bg[p:-p], (bg[p * 2:] + bg[:-p * 2]) / 2)

        if lls:  # back transformation of LLS operator
            bg = (np.exp(np.exp(bg) - 1) - 1) ** 2 - 1

        return bg if return_baseline else _x - bg  # return for a single spectrum

    # wrapper for different types of input data
    if isinstance(x[0], (np.ndarray, list)):
        return np.asarray([snip_single(xi) for xi in x])
    else:
        return snip_single(x)


def interpfunc(x_old, x_new, data):
    """
    Parameters
    ----------
    data : numpy array
        Input data.

    Returns
    -------
    newdata : numpy array
        Interpolated data.

    """
    if data.shape[0] != 1:
        newdata = np.zeros([data.shape[0], len(x_new)])
        for i in range(data.shape[0]):
            TMPout = interp.interp1d(x_old, data[i,:], axis=0)
            newdata[i,:] = TMPout(x_new)
    else:
        TMPout = interp.interp1d(x_old, data, axis=0)
        newdata = TMPout(x_new)    
    return newdata


def normal_spectra(spectra, norm_method='vector'):
    if norm_method == 'vector':
        norm_spec = spectra / np.sqrt((spectra ** 2).sum(1))[:, None]
    elif norm_method == 'l1':
        norm_spec = spectra / np.abs(spectra).sum(1)[:, None]
    elif norm_method == 'max':
        norm_spec = spectra / spectra.max(1)[:, None]
    else:
        raise ValueError("Invalid normalization method...")
    return norm_spec


def plot_mean_sd(wn, spectra, titlename):
    plt.plot(wn, np.mean(spectra, axis=0), label='Mean')
    plt.fill_between(wn, np.mean(spectra, axis=0) - np.std(spectra, axis=0), 
                     np.mean(spectra, axis=0) + np.std(spectra, axis=0), 
                     alpha = 0.2, label='SD')
    plt.xlabel("Wavenumber / cm $^{-1}$", fontsize=15)
    plt.ylabel('Intensity / arb. u.', fontsize=15)
    plt.title(titlename, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.show()
    
def plot_group_mean(wn, spectra, group, gap=0):
    means = []
    sds = []
    for g in np.unique(group):
        ix = np.argwhere(group==g)[:, 0]
        means.append(np.mean(spectra[ix,:], 0))
        sds.append(np.std(spectra[ix, :], 0))
    means = np.row_stack(means)
    sds = np.row_stack(sds)
    sns.set(rc = {'figure.figsize':(12,8)})
    for i in range(len(np.unique(group))):
        plt.fill_between(wn, i*gap+means[i, :] - sds[i, :], i*gap+means[i, :] + sds[i, :], alpha=0.2)
    
    for i in range(len(np.unique(group))):
        sns.lineplot(x=wn, y=i*gap+means[i, :], label=np.unique(group)[i])
    plt.legend(loc="upper right",fontsize=10)
    plt.ylabel('Raman Intensity / arb. u.',fontsize=20)
    plt.xlabel("Wavenumber / cm $^{-1}$",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    return means
    
def plot_spectra(wn, spectra):    
    cols = list(mcolors.TABLEAU_COLORS)
    for i in range(spectra.shape[0]):
        sns.lineplot(x=wn, y=spectra[i, :], color=cols[i%len(cols)])
    plt.ylabel('Raman Intensity / arb. u.',fontsize=20)
    plt.xlabel("Wavenumber / cm $^{-1}$",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()             