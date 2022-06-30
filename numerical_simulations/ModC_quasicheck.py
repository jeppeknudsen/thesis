# %%

from constants import *
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.signal import find_peaks, square
from numpy.core.defchararray import array
import time
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
import os






# %% Model C
# Cleaned script

# Script solving the differential equations for model C as described in Liv's thesis adding an external oscillator in the form of modifying the parameter Kappa
#
# %% Imports



# Import constants fro mconstant file


# %% Timetaking
start_time = time.time()


# %% ModC function

@njit
def square_wave_func(t, omega):
    T = 2*np.pi / omega
    if t % T < 0.5 * T:
        square_val = 0
    else:
        square_val = 1

    return square_val


@njit
def ModC(t, x, amp, omega, waveshape, param_var):

    T = 2*np.pi / omega

    # CHOOSE WAVEFORM: SINE, SQUARE or NONE
    if waveshape == 1:
        sine_val = (np.sin(omega * t) + 1) / 2

        wave_factor = 1 - amp * sine_val

    elif waveshape == 2:

        square_val = square_wave_func(t, omega)

        wave_factor = 1 - amp * square_val

    else:
        wave_factor = 1

    # CHOOSE KAPPA, LAMBDA or NO VARIATION
    if param_var == 1:
        pdot = alpha - (kappa * wave_factor) * \
            x[2] * x[0] / (x[0] + lam_param)  # p53

    elif param_var == 2:
        pdot = alpha - kappa * x[2] * x[0] / \
            (x[0] + lam_param * wave_factor)  # p53

    else:
        pdot = alpha - kappa * x[2] * x[0] / (x[0] + lam_param)

    mmdot = xsi*x[0]*x[0] - delta*x[1]  # Mdm2

    mdot = epsilon*x[1] - eta*x[2]  # Mdm2

    xdot = np.array([pdot, mmdot, mdot])

    return xdot

# %% Runge-Kutta 4 solver. Takes initial conditions, start and end time, time increment size and parameters.
# Initial condiitons size corresponds to number of equaitons in model.
@njit
def RK4(x0, t0, tf, dt, amp, omega, waveshape, param_var):

    t = np.arange(t0, tf, dt)
    nt = t.size

    nx = x0.size
    x = np.zeros((nx, nt))

    x[:, 0] = x0

    for k in range(nt - 1):
        k1 = dt * ModC(t[k], x[:, k], amp, omega, waveshape, param_var)
        k2 = dt * ModC(t[k], x[:, k] + k1/2, amp, omega, waveshape, param_var)
        k3 = dt * ModC(t[k], x[:, k] + k2/2, amp, omega, waveshape, param_var)
        k4 = dt * ModC(t[k], x[:, k] + k3, amp, omega, waveshape, param_var)

        dx = (k1 + 2*k2 + 2*k3 + k4)/6

        x[:, k+1] = x[:, k] + dx

    return x, t


# %%

# height trim determines percentage of max height accepted, length_trim determines percentage of max period found in first run
def peak_identifier(x, t, skip_perc=0.8, height_trim=0.8, length_trim=0.3):
    # FOR VERY DETAILED PEAK IDENTIFICATION WITH MULTI-PEAK-PERIODS: height_trim = 0.95, length_trim = 0.8)

    tf = t[-1]
    dt = t[1] - t[0]

    # Index to skip until stabilized
    ind_steady = int(np.ceil(skip_perc * tf/dt))
    x_steady = np.array(x[0, ind_steady:])  # Trimmed p53 level
    t_steady = np.array(t[ind_steady:])

    height_threshold = x_steady.max() * height_trim

    peaks_init, _ = find_peaks(x_steady, height=height_threshold)

    max_peaks_init = peaks_init[peaks_init > height_threshold]

    T_peaks_init = np.array([])

    # Find temporal distance between peaks in order to find max distance
    for j in range(len(max_peaks_init) - 1):
        peak_init_dist = (max_peaks_init[j+1] - max_peaks_init[j])
        T_peaks_init = np.append(T_peaks_init, peak_init_dist)

    if T_peaks_init.size == 0:  # If no peaks found return empty list
        T_peaks = np.array([])
        peak_indices = np.array([])

    else:
        length_threshold = length_trim * T_peaks_init.max()

        peak_indices, _ = find_peaks(
            x_steady, height=height_threshold, distance=length_threshold)  # ,

        T_peaks = np.array([])  # List of times between peaks

        for j in range(len(peak_indices) - 1):
            peak_dist = (peak_indices[j+1] - peak_indices[j])*dt
            T_peaks = np.append(T_peaks, peak_dist)

    return T_peaks, peak_indices+ind_steady


# %% Function that identifies characteristics of identified peaks

def period_determiner(T_peaks):

    T_period = 0

    n_min = 4  # Minimum required number of peaks

    n_deci = 2  # Number of decimals to compare

    T_peaks_round = T_peaks  # np.around(T_peaks,decimals=n_deci)

    T_len = len(T_peaks)

    if T_len < n_min:
        T_period = 0
    else:
        # DOUBLE CONDITIONAL DEMANDS CONSISTENT PERIOD
        if (T_peaks_round[0] - T_peaks_round[1]) < 10**-n_deci and (T_peaks_round[1] - T_peaks_round[2]) < 10**-n_deci:
            T_period = T_peaks[0]

        elif T_peaks_round[0] - T_peaks_round[2] < 10**-n_deci and T_peaks_round[1] - T_peaks_round[3] < 10**-n_deci:
            T_period = T_peaks[0] + T_peaks[1]

        # elif T_peaks_round[0] - T_peaks_round[3] < 10**-n_deci:
        #     T_period = T_peaks[0] + T_peaks[1] + T_peaks[2]

        else:
            T_period = 0

    return T_period

# %% Function that finds Tp-TN relationship (entrainment) for various periods of external oscillator


def Tp_vs_TN(T_start, T_end, T_steps, t0, tf, dt, x0):
    T_input = np.linspace(T_start, T_end, T_steps)
    omega_array = 2 * np.pi / T_input
    T_output = np.zeros_like(omega_array)
    T_output_var = np.zeros_like(omega_array)
    T_peaks_arr = []

    p_arr = np.zeros((T_steps, int(tf/dt)))
    m_m_arr = np.zeros_like(p_arr)
    m_arr = np.zeros_like(p_arr)
    peak_indices_arr = []

    for index, omega_input in enumerate(tqdm(omega_array)):

        x, t = RK4(x0, t0, tf, dt, A_osc, omega_input, waveshape, param_var)

        T_peaks, peak_indices = peak_identifier(x, t)

        T_peaks_arr.append(T_peaks)
        peak_indices_arr.append(peak_indices)

        p_arr[index, :] = x[0, :]
        m_m_arr[index, :] = x[1, :]
        m_arr[index, :] = x[2, :]

        # period_determiner(T_peaks) #T_peaks.mean()# NB Mean is used, should probably be changed?
        T_output[index] = T_peaks.mean()
        T_output_var[index] = T_peaks.var()

    return T_input, T_output, T_output_var, T_peaks_arr, p_arr, m_m_arr, m_arr, peak_indices_arr

# %%


def var_IC_poincare(p0_start, p0_end, m_m0_start, m_m0_end, m0_start, m0_end,  n_IC, T_chosen, t0, tf, dt):

    omega_input = 2 * np.pi / T_chosen

    p0_arr = np.linspace(p0_start, p0_end, n_IC)
    m_m0_arr = np.linspace(m_m0_start, m_m0_end, n_IC)
    m0_arr = np.linspace(m0_start, m0_end, n_IC)

    x0_arr = np.stack((p0_arr, m_m0_arr, m0_arr), axis=-1)

    p_arr = np.zeros((n_IC, int(tf/dt)))
    m_m_arr = np.zeros_like(p_arr)
    m_arr = np.zeros_like(p_arr)
    # peak_indices_arr = []

    for index, x0_ind in enumerate(tqdm(x0_arr)):

        x, t = RK4(x0_ind, t0, tf, dt, A_osc,
                   omega_input, waveshape, param_var)

        # T_peaks, peak_indices = peak_identifier(x, t)

        # T_peaks_arr.append(T_peaks)
        # peak_indices_arr.append(peak_indices)

        p_arr[index, :] = x[0, :]
        m_m_arr[index, :] = x[1, :]
        m_arr[index, :] = x[2, :]

        # T_output[index] = T_peaks.mean()# period_determiner(T_peaks) #T_peaks.mean()# NB Mean is used, should probably be changed?
        # T_output_var[index]= T_peaks.var()

    return p_arr, m_m_arr, m_arr


# %% Poincare section

def poincare_section(x, y, z, x_intersect, y_range, z_range):
    '''
    Finds intersections with chosen Poincare section (2d) 
    for given phase space trajectory (3d)
    y_range and z_range indicate limits of poincare section and
    x_intersect gives the location of section in x-direction.

    '''
    # x_poincare = []

    y_poincare = np.array([])
    z_poincare = np.array([])
    y_min = y_range[0]
    y_max = y_range[1]
    z_min = z_range[0]
    z_max = z_range[1]

    x_len = len(x)

    mask_y_max = y <= y_max
    mask_y_min = y >= y_min
    mask_z_max = z <= z_max
    mask_z_min = z >= z_min

    # Make a collected mask to filter out clear outliers
    mask_comp = mask_y_max * mask_y_min * mask_z_max * mask_z_min

    print(mask_comp)
    for i in range(x_len-1):
        if mask_comp[i]:
            if x[i+1] < x_intersect and x[i] > x_intersect:
                y_intersect = (y[i+1] + y[i])/2
                z_intersect = (z[i+1] + z[i])/2

                y_poincare = np.append(y_poincare, y_intersect)
                z_poincare = np.append(z_poincare, z_intersect)

            elif x[i+1] > x_intersect and x[i] < x_intersect:
                y_intersect = (y[i+1] + y[i])/2
                z_intersect = (z[i+1] + z[i])/2

                y_poincare = np.append(y_poincare, y_intersect)
                z_poincare = np.append(z_poincare, z_intersect)

    return y_poincare, z_poincare

# %% Varying kappa


def kappa_sin(t, amp, T):

    omega = 2 * np.pi/T

    sine_val = (np.sin(omega * t) + 1) / 2
    wave_factor = 1 - amp * sine_val

    return kappa*wave_factor


# %% INFO
def run_info(waveshape, param_var):
    if waveshape == 1:
        wavename = 'SINE WAVE'
    elif waveshape == 2:
        wavename = 'SQUARE WAVE'
    else:
        wavename = 'NONE'

    if param_var == 1:
        param_name = 'KAPPA'
    elif param_var == 2:
        param_name = 'LAMBDA'
    else:
        param_name = 'NONE'

    print(f"""

    A_osc is {A_osc} and T_kappa is {T_kappa}

    Waveform is: {wavename}

    Varied paramater is: {param_name}

    """)


# %%

dt = 0.001
tf = 1200*5
t0 = 0
x0 = np.array([0, 0, 0])

T_kappa = 5
A_osc = 0.3
waveshape = 1
param_var = 1
omega_kappa = 1

# T_start = 0.1
# T_end = 11
# T_steps = 101  # 101 for nice periods
# T_list = np.linspace(T_start, T_end, T_steps)
# T_chosen = T_list[38]

# %% Calculate 'natural' period
x_natural, t_natural = RK4(x0, t0, tf, dt, A_osc, omega_kappa, 0, 0)
T_list, peak_ind = peak_identifier(x_natural, t_natural)
T_natural = T_list[0]


# %% CHOISES
A_osc = 0.086
T_factor = 0.7
T_chosen = T_natural * T_factor
omega_kappa =2*np.pi/ T_chosen



# %%
x0 = np.array([1.5, 0.01, 1])


x, t = RK4(x0, t0, tf, dt, A_osc, omega_kappa, 1, 1)

sin_kappa = kappa_sin(t, A_osc, T_chosen)


# %%
label_size = 15
title_size = 15
linewidth = 0.3

A_string = f'{A_osc*1000:.0f}'.zfill(4)
T_string = f'{T_factor*100:.0f}'.zfill(4)

save_graph = False

trim_pct = 0.8

trim_idx = np.int32(tf*trim_pct/dt)
print(f'Trim idx is {trim_idx}')

fig, ax = plt.subplots(figsize=(6,6),dpi=250)
ax.plot(sin_kappa[trim_idx:], x[0][trim_idx:], lw=linewidth)

ax.set_ylabel(f'p53 [a.u.]', fontsize=label_size)
ax.set_xlabel(f'$\kappa_{{osc}}$', fontsize=label_size)
ax.set_title(f'$\kappa_{{osc}}$ vs p53 for $T_{{osc}}={T_factor}T_{{natural}}$ and $A_{{osc}}={A_osc}$,\nfrom $t={int(np.round(tf*trim_pct))}$ to $t={int(np.round(tf))}$', fontsize=title_size)

fig.tight_layout()

if save_graph:
    graph_dir = r'GRAPHDIR'
    os.chdir(graph_dir)
    fig.savefig(f'quasi_check_A_' + A_string + '_T_' + T_string + '.png', facecolor='w',dpi=250,bbox_inches = "tight")


# %%
