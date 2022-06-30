# %% Model C
#
# Script solving the differential equations for model C as described in Liv's thesis adding an external oscillator in the form of modifying the parameter Kappa
#
# Extended to identify peaks of oscillating levels of p53

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
from scipy.signal import find_peaks, square
import os

graph_dir = r'GRAPHPATH'
script_dir = r'SCRIPTPATH'

# %% Timetaking
start_time = time.time()


# %% Parameters

alpha = 10
beta = 0
xsi = 0.01
delta = 0.25
epsilon = 10
eta = 1
gamma = 100
jota = 1
kappa = 10
lam_param = 0.12
mu = 25
nu = 0.4
tau_1 = 1.6
tau_2 = 1.1

# %% Function that takes input of variable values and parameteters, and outputs the differential change acoording to the model.


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



# %%

x0 = np.array([0,0,0])
t0 = 0
tf = 1200
dt = 0.001
paramvar = 1
waveshape = 1

# amp_start = 0.1
# amp_end = 0.3
# amp_int = 0.1

# amp_list = np.arange(amp_start, amp_end+amp_int, amp_int)
amp_list = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

x_natural, t_natural = RK4(x0, t0, tf, dt, 0, 1, 0, 0)
T_list_peaks, peak_ind = peak_identifier(x_natural, t_natural)
T_natural = T_list_peaks[0]


# %% Solve for x and t

A_chosen = 0.15
T_factor = 0.7

T_chosen = T_factor * T_natural
omega_chosen = 2* np.pi / T_chosen

x_chosen, t = RK4(x0, t0, tf, dt, A_chosen, omega_chosen, 1, 1)




# %% Plotting transcient period

save_fig = True
plt.rcParams.update({'font.size': 12})


# Phase plot

trim_perc = 0.8
trim_ind = np.int32(len(t)*trim_perc)


fig1 = plt.figure(dpi=250,figsize=(6,6))
ax1 = fig1.add_subplot(111, projection='3d')

ax1.plot(x_chosen[0,trim_ind:], x_chosen[1,trim_ind:], x_chosen[2,trim_ind:], c='red', lw=2,label=f'Et label')



ax1.set_xlabel(r'$p53$ [a.u.]')
ax1.set_ylabel(r'$m_{mRNA}$ [a.u.]')
ax1.set_zlabel(r'$m$ [a.u.]')
# ax1.set_xlim(0,5)
# ax1.set_ylim(0.07,0.38)
# ax1.set_zlim(0.8,2.75)
# ax1.legend(loc='best')
fig1.tight_layout()

ax1.set_title(f'Phase plot for $A={A_chosen:.2f}$ and $T = {T_factor:.2f} T_{{natural}}$,\nfrom $t={int(trim_perc*tf)}$ to $t={int(tf)}$') #


A_string = str(int(A_chosen*1000)).zfill(4)

T_ex_string = str(int(T_factor*100)).zfill(3)
T_string = T_ex_string[0] + '_' + T_ex_string[1:]

if save_fig:
    os.chdir(graph_dir)
    fig1.savefig('modC_phase_T_' + T_string + '_A_' +  A_string +'.png',dpi=250, facecolor='w', bbox_inches='tight')
    # fig1.clf()
    # plt.close(fig1)
    # gc.collect()
# %%

# %%
