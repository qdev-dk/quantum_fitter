""" 
Functions to load data, shape, fit and plot
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import qcodes as qc

from scipy.constants import pi, hbar

from qdev_wrappers import show_num
from qcodes.dataset.data_set import load_by_id

from qcodes.dataset.sqlite.database import connect
from qcodes.dataset.data_export import reshape_2D_data

from ..fitting.quarterwave import run_feedline

def load_polardata(id, axesname=('frequency_set', 'power_set'), dataname=('S21_magnitude', 'S21_phase'), path=None):
    """ Load polar data from a VNA trace and return ([axis], S21, mid)
    - path can point either to a db-file or a dataset folder (old dataset)
    """
    backend = 'DB'
    if path is None:
        dataset = load_by_id(id)
        data = dataset.get_parameter_data()
    elif path.endswith('.db'):
        if not os.path.isfile(path): 
            raise ValueError(f" The given path ({path}) does not exist.")
        dataset = load_by_id(id, conn = connect(path, version=0))
        data = dataset.get_parameter_data()
    else:
        backend = 'FILE'
        if not os.path.isdir(path): 
            raise ValueError(f" The given path ({path}) does not point to an existing folder.")
        data = show_num(id, samplefolder=path, do_plots=False)[0][0]

    values = []; axes = []
    if backend == 'DB':
        for mask in dataname:
            if len(axesname) > 1:
                x,y,z = reshape_2D_data(x = data[mask][axesname[0]],
                                           y = data[mask][axesname[1]], 
                                           z = data[mask][mask])
                axes = (x,y)
                values.append(z)                           
            else:
                axes = data[mask][axesname[0]]
                values.append(data[mask][mask])
    else:
        for mask in dataname:
            for key in data.arrays.keys():
                if mask in key:
                    values.append(data.arrays[key].ndarray)
                    break
            else:
                raise ValueError(f" Can not extract '{mask}' from this dataset. Check the data or the data argument" )

        for mask in axesname:
            for key in data.arrays.keys():
                if mask in key:
                    setpoints = data.arrays[key].ndarray
                    if len(setpoints.shape) > 1:
                        setpoints = setpoints[0]
                    axes.append(setpoints)
        
    S21 = values[0] * np.exp(1j*values[1])   
    if not list(filter(lambda x: x > 1, S21.shape)):
        S21 = np.asarray(S21)

    if len(axes) == 0:
        raise ValueError('We should be able to extract at least one parameter from the data. Check the data or the axes argument' )

    return (axes, S21, id)

def plot_quality(axes, S21, id=None, sgw = None, linecomp = 0.05, fitter = run_feedline, fit_as_guess=False,
                 convert_power = False, label='', num = None, ylim = None, **kwargs):
    if len(axes) < 2:
        raise ValueError('Data should be 2-dimensional')
    f = axes[0]*1E-6
    power = axes[1]

    results = []
    guess = None
    for i, Pi in enumerate(power):
        trace = S21[i,:]
        try:
            _, popt, pcov, _, _ = fitter(f, trace, sgw=sgw, linecomp=linecomp, guess=guess, **kwargs)
            if fit_as_guess: guess = popt[1:3]
            results.append((Pi, popt, pcov))
        except Exception as err:
            # print(f'Skipped power {Pi}')
            print(err)

    data = []
    for pi, popt, pcov in results:
        #f0, Qi, Qe, Qe_theta, A, alpha, phi1, phi2 = popt
        pcov = np.diag(pcov)
        # pi, f0, f0err, int, int_err, ext, ext_err
        data.append((pi, popt[0], pcov[0], popt[1], pcov[1], popt[2], pcov[2]))
    data = np.asarray(data)

    if convert_power:
        power = 0.001*10**((data[:,0])/10)
        # average photon number
        data[:,0] = (2*power * data[:,5]**2) / (hbar/np.real(1/data[:,5]) * (2*pi*data[:,0]*1e6)**2)    
    
    # plotting
    if not num:
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                                     figsize = kwargs.pop('figsize', (4,4)))
        ax3 = ax1.twinx()

        ax1.arrow(0.10, 0.08, -0.06, 0, head_width=0.05, head_length=0.02, 
          edgecolor = 'tab:red', facecolor = 'tab:red', 
          transform = ax1.transAxes)

        ax3.arrow(1-0.10, 0.08, 0.06, 0, head_width=0.05, head_length=0.02, 
                edgecolor = 'tab:blue', facecolor = 'tab:blue', 
                transform = ax3.transAxes)
    else:   
        ax1, ax3, ax = plt.figure(num = num).axes

    lh1 = ax1.errorbar(data[:,0], data[:,3], yerr = data[:,4], label = label+r'$Q_{int}$', linestyle='none',  marker = 'o', color = 'tab:red')
    lh3 = ax3.errorbar(data[:,0], data[:,5], yerr = data[:,6], label = label+r'$Q_{ext}$', linestyle='none', marker = 'x', color = 'tab:blue') 
    lh2 = ax2.errorbar(data[:,0], data[:,1]*1E-3, yerr = data[:,2]*1E-3, label = label+r'$f_0$', linestyle='none', marker = 'x', color = 'tab:orange') 

    # plt.legend(lh1+lh2, [h.get_label() for h in lh1+lh2])  
    ax1.set_ylabel('$Q_{int}$ [$10^3$]')
    ax3.set_ylabel('$Q_{ext}$ [$10^3$]')
    ax2.set_ylabel('$f_0$ (GHz)')
    
    if convert_power:
        ax2.set_xlabel('average intraresonator photon number $<n_{ph}>$')
        ax2.set_xscale("log", nonposx = 'clip')
    else:
        ax2.set_xlabel('$P$ (dBm)')

    if ylim:
        ax1.set_ylim(0, ylim[0])
        ax3.set_ylim(0, ylim[1])

        favg = np.mean(data[:,1])*1E-3
        ax2.set_ylim(favg-1E-3, favg+1E-3)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.title(f"Data from {id}")
    plt.tight_layout()
    plt.show()    

    return data, (ax1, ax2, ax3)

def plot_polarfit(f, S21, id=None, power = None, sgw = None, linecomp = 0.05, 
                  guess=None, fitter = run_feedline, newfig = False, clear=True, **kwargs):
    f = f*1E-6
    p_guess, popt, pcov, S21f, model = fitter(f, S21, sgw=sgw, linecomp=linecomp, guess=guess, **kwargs)
    
    l = len(f)
    S21_guess = model(f,*p_guess)
    S21_guess = S21_guess[0:l] + 1j*S21_guess[l::]
    if popt is not None: 
        S21_opt = model(f,*popt)
        S21_opt = S21_opt[0:l] + 1j*S21_opt[l::]
        angle = np.exp(-1j*(f*popt[6] + popt[7]))
        f0 = popt[0]
    else:
        f0 = p_guess[0]
        S21_opt = S21_guess
        angle = np.exp(-1j*(f*p_guess[6] + p_guess[7]))
        
    if newfig:    
        fig = plt.figure(figsize=kwargs.get('figsize', (10,3.4)))
    else:
        fig = plt.figure(10)
        if clear: fig.clf()
        fig.set_size_inches(kwargs.get('figsize', (10,3.4)))
        
    txt = f'#{id}: ' if id else ''
    txt += r'$f_0$ = {:.0f} MHz'.format(f0)
    txt += r', $Q_{{int}}$ = {:.0f}'.format(popt[1]*1E3)
    txt += r', $Q_{{ext}}$ = {:.0f}'.format(popt[2]*1E3)
    if power:
        txt += r', $P_{VNA}$ =' +str(power) +'dBm'
    plt.suptitle(txt, fontsize=11)
 
#    fgs = fig.add_gridspec(nrows=1, ncols=4, left=0.05, right=0.48, wspace=0.05)
    
    # First panel
    ax1 = plt.subplot(141)
    # plt.scatter(S21.real, S21.imag, color='tab:blue')
    if sgw or linecomp: 
        plt.scatter(S21f.real, S21f.imag, s=2, color='k')
    
    plt.plot(S21_opt.real,S21_opt.imag,'r')
    # plt.plot(S21_guess.real,S21_guess.imag,'k')

    plt.title('Raw S21 complex-plane', fontsize=10)
    plt.xlabel(r'S21.real')
    plt.ylabel(r'S21.imag')
    ax1.grid()
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))    
    
    # Second panel
    ax2 = plt.subplot(142)
    # plt.scatter((S21*angle).real,(S21*angle).imag, color='tab:blue')
    if sgw or linecomp: 
        plt.scatter((S21f*angle).real, (S21f*angle).imag, s=2, color='k')
    plt.plot((S21_opt*angle).real, (S21_opt*angle).imag,'r')
    # plt.plot((S21_guess*angle).real,(S21_guess*angle).imag,'k')
    
    plt.title('Cable delay-free S21', fontsize=10)
    plt.xlabel(r'S21.real')
    plt.ylabel(r'S21.imag')
    ax2.grid()
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))    

    # Third panel
    ax3 = plt.subplot(143)
    # S21_l = 20*np.log10(np.abs(S21))  
    S21_lf = 20*np.log10(np.abs(S21f))
    S21_opt_l = 20*np.log10(np.abs(S21_opt)) 
    # S21_guess_l = 20*np.log10(np.abs(S21_guess))  

    # plt.scatter(f - f0, S21_l, color='tab:blue')
    if sgw or linecomp: 
        plt.scatter(f - f0, S21_lf, s=2, color='k')
    plt.plot(f - f0, S21_opt_l,'r')
    # plt.plot(f - f0, S21_guess_l ,'k')

    plt.title('S21 Magnitude', fontsize=10)
    plt.xlabel(r'Frequency - $f_0$ (MHz)')
    plt.ylabel('dB')

    # Fourth panel
    plt.subplot(144)
    # plt.scatter(f - f0, np.angle(S21*angle), color='tab:blue')
    if sgw or linecomp: 
        plt.scatter(f - f0, np.angle(S21f*angle), s=2, color='k')
    plt.plot(f - f0, np.angle(S21_opt*angle),'r')
    # plt.plot(f - f0, np.angle(S21_guess*angle) ,'k')

    plt.title('S21 phase', fontsize=10)
    plt.xlabel(r'Frequency - $f_0$ (MHz)')
    plt.ylabel('angle (rad)')

    plt.show()
    
    defs = [('left', 0.08), ('right', 0.99),
            ('bottom', 0.15) , ('top', 0.85),
            ('wspace', 0.45)]
    
    params = {k: kwargs.get(k, d) for k, d in defs}
    plt.subplots_adjust(**params)

    return popt, pcov, S21f, model 