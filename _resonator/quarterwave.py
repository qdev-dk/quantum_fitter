# -*- coding: utf-8 -*-
""" 
Fit functions and procedures to fit data from hanger-style resonators

@author: T2_2
"""
import cmath

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import eig, inv
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from copy import deepcopy

from scipy.stats import chisquare
from scipy.constants import hbar
from scipy.constants import pi

# Elipse fitting functions taken from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
def fitEllipse(x,y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1 = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2 = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1 = np.sqrt(abs(up/down1))
    res2 = np.sqrt(abs(up/down2))
    return np.array([res1, res2])

# Find FWHM
def fwhm(f,S21):
    mag = np.abs(S21)
    mag_half_max = (mag.min() + np.mean([mag[[0,-1]]]))/2
    f0_idx = np.argmin(mag)

    half_max_left = np.abs((mag[0:f0_idx]-mag_half_max)).argmin()
    half_max_right = np.abs((mag[f0_idx::]-mag_half_max)).argmin()
    return abs(f[half_max_right+f0_idx]-f[half_max_left])

# Full function for S21 response of the resonator.
def S21funct(f,f_0,Qi,Qe_mag,Qe_theta,A,alpha,phi1,phi2):
    Qi *= 1e3
    Qe_mag *= 1e3
    Qe = Qe_mag*np.exp(-1j*Qe_theta)
    Qc = 1/(np.real(1/Qe))
    Q = 1/(1/Qc + 1/Qi)
    S = A*(1 + alpha*(f-f_0)/f_0)*(1 - (Q/Qe)/(1 + 2*1j*Q*(f-f_0)/f_0))*np.exp(1j*(phi1*f + phi2))
    s = np.hstack([np.real(S),np.imag(S)]).T
    return s

def run_feedline(f, S21, sgw = None, linecomp = None, guess=()):
    '''
    Robust fitting function for a hanger-style qaurter wavelength resonator coupled to a feedline
    f (numpy array):            Frequency values
    S21 (complex numpy array):  Complext S21 measurement
    sgw (integer, None):        Filter-window for savgol filter (2nd order)
    linecomp (integer, None):   Number of endpoint samples to average to determine the background
    '''
    if sgw:
        rr = savgol_filter(np.real(S21), sgw, 2)
        ri = savgol_filter(np.imag(S21), sgw, 2)
        S21 = np.vectorize(complex)(rr, ri)

    # Unwrap phase and subtract linear frequency dependence from endpoints
    linFit = [0,0]
    if linecomp:
        phase = np.unwrap(np.angle(S21))
        fitwin = int(linecomp*len(f))
        linFit = np.polyfit([np.mean(f[:fitwin]), np.mean(f[-fitwin:])],
                            [np.mean(phase[:fitwin]), np.mean(phase[-fitwin:])], 1)
    S21 = S21*np.exp(-1j*linFit[0]*f)

    # Format complex data to real/imag for fitting function
    S21_R_I = (np.hstack([np.real(S21),np.imag(S21)])).T

    # Find guess parameters
    f0 = f[np.argmin(np.abs(S21))]

    # Normalize data
    A = np.mean(np.abs(S21)[[0,-1]])
    S21_norm = S21/A

    if guess is None:
        # Fit inverse data in complex plane to an ellipse and get average diameter
        a = fitEllipse(np.real(1/S21_norm), np.imag(1/S21_norm))
        axes = ellipse_axis_length(a)
        D = np.max(axes[[0, 1]])
        Qi = 2*(f0/fwhm(f, S21_norm))
        guess = [Qi, Qi/(2*D)]
    guess = [f0, *guess, 0, A, 0, 0, np.angle(np.mean(S21_norm))]

    f0, Qi, Qe_mag, _, A, _ = guess
    param_bounds=(      [0.9*f0,    0,        0,          -np.pi,     0,      -np.inf, -np.inf, -np.inf],
                        [1.1*f0,    20*Qi,   20*Qe_mag,  np.pi,     2*A,  np.inf,  np.inf,  np.inf])

    popt, pcov = curve_fit(S21funct, f, S21_R_I, p0=guess, bounds=param_bounds)
    popt[6] += linFit[0]
    guess[6] += linFit[0]

    l = len(f)
    S21_guess = S21funct(f, *guess)
    S21_guess = S21_guess[0:l] + 1j*S21_guess[l::]

    if popt is not None:
        S21_opt = S21funct(f, *popt)
        S21_opt = S21_opt[0:l] + 1j*S21_opt[l::]
        angle = np.exp(-1j*(f*popt[6] + popt[7]))
        f0 = popt[0]
    else:
        f0 = guess[0]
        S21_opt = S21_guess
        angle = np.exp(-1j*(f*guess[6] + guess[7]))

    #S21_R_I_opt = (np.hstack([np.real(S21_opt),np.imag(S21_opt)])).T

    #chi2, p_value = chisq(popt)

    S21 = S21*np.exp(1j*linFit[0]*f)
    return (guess, popt, pcov, S21, S21funct)

def run_reflection():
    """ 
    Robust fitting function for a reflection-style qaurter wavelength resonator coupled to an input.
    f (numpy array):            Frequency values
    S21 (complex numpy array):  Complext S21 measurement
    sgw (integer, None):        Filter-window for savgol filter (2nd order)
    linecomp (integer, None):   Number of endpoint samples to average to determine the background
    """
    pass