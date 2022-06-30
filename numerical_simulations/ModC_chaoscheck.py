# %% Model C 
# Cleaned script

# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
from scipy.signal import find_peaks, square
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import os

# Import constants fro mconstant file
from constants import *

# %% Timetaking
start_time = time.time()

plt.rcParams.update({'font.size': 10})



# %% ModC function

# @njit # FORSØG MED SQUAREWAVE
# def square_wave_func(t, omega): # Square wave function that starts at 0 and goes to 1 after half a period
#     T = 2*np.pi / omega
#     x = np.ones_like(t)

#     mask = t % T <= T/2 # Mask that is true for all first halfs of period

#     x[mask] = 0

#     return x
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

    ############### CHOOSE WAVEFORM: SINE, SQUARE or NONE
    if waveshape == 1:
        sine_val = ( np.sin(omega * t) + 1) / 2

        wave_factor = 1 - amp * sine_val

    elif waveshape == 2:

        square_val = square_wave_func(t,omega)

        wave_factor = 1 - amp * square_val

    else:
        wave_factor = 1

    ######## CHOOSE KAPPA, LAMBDA or NO VARIATION
    if param_var == 1:
        pdot = alpha - (kappa * wave_factor ) * x[2] * x[0] / (x[0] + lam_param) # p53

    elif param_var == 2:
        pdot = alpha - kappa * x[2] * x[0] / (x[0] + lam_param * wave_factor ) # p53
    
    else:
        pdot = alpha - kappa * x[2] * x[0] / (x[0] + lam_param)

    mmdot = xsi*x[0]*x[0] - delta*x[1] # Mdm2

    mdot = epsilon*x[1] - eta*x[2] # Mdm2

    xdot = np.array( [ pdot, mmdot, mdot ] )

    return xdot

# %% Runge-Kutta 4 solver. Takes initial conditions, start and end time, time increment size and parameters.
# Initial condiitons size corresponds to number of equaitons in model.
@njit
def RK4(x0, t0, tf, dt, amp, omega, waveshape, param_var):

    t = np.arange(t0,tf,dt)
    nt = t.size

    nx = x0.size
    x = np.zeros((nx,nt))

    x[:,0] = x0

    for k in range(nt -1):
        k1 = dt * ModC(t[k], x[:,k], amp, omega, waveshape, param_var)
        k2 = dt * ModC(t[k], x[:,k] + k1/2, amp, omega, waveshape, param_var)
        k3 = dt * ModC(t[k], x[:,k] + k2/2, amp, omega, waveshape, param_var)
        k4 = dt * ModC(t[k], x[:,k] + k3, amp, omega, waveshape, param_var)

        dx = (k1 + 2*k2 + 2*k3 + k4)/6

        x[:,k+1] = x[:,k] + dx
    
    return x, t


# %% 

def peak_identifier(x, t, skip_perc=0.8, height_trim = 0.8, length_trim = 0.3): # height trim determines percentage of max height accepted, length_trim determines percentage of max period found in first run
# FOR VERY DETAILED PEAK IDENTIFICATION WITH MULTI-PEAK-PERIODS: height_trim = 0.95, length_trim = 0.8)

    tf = t[-1]
    dt = t[1] - t[0]

    ind_steady = int(np.ceil(skip_perc * tf/dt)) # Index to skip until stabilized
    x_steady = np.array(x[0,ind_steady:]) # Trimmed p53 level
    t_steady = np.array(t[ind_steady:])


    height_threshold = x_steady.max()* height_trim

    peaks_init, _ = find_peaks(x_steady, height=height_threshold)

    max_peaks_init = peaks_init[peaks_init>height_threshold]

    T_peaks_init = np.array([]) #

    for j in range(len(max_peaks_init) - 1): # Find temporal distance between peaks in order to find max distance
        peak_init_dist = (max_peaks_init[j+1] - max_peaks_init[j])
        T_peaks_init = np.append(T_peaks_init, peak_init_dist)

    if T_peaks_init.size == 0: # If no peaks found return empty list
        T_peaks = np.array([])
        peak_indices = np.array([])

    else: 
        length_threshold = length_trim * T_peaks_init.max()

        peak_indices, _ = find_peaks(x_steady,height = height_threshold, distance=length_threshold) # , 


        T_peaks = np.array([]) # List of times between peaks

        for j in range(len(peak_indices) - 1):
            peak_dist = (peak_indices[j+1] - peak_indices[j])*dt
            T_peaks = np.append(T_peaks, peak_dist)
        
    return T_peaks, peak_indices+ind_steady


# %% Function that identifies characteristics of identified peaks

def period_determiner(T_peaks):

    T_period = 0

    n_min = 4 # Minimum required number of peaks

    n_deci = 2 # Number of decimals to compare

    T_peaks_round = T_peaks #np.around(T_peaks,decimals=n_deci)

    T_len = len(T_peaks)

    if T_len < n_min:
        T_period = 0
    else:
        if (T_peaks_round[0] - T_peaks_round[1]) < 10**-n_deci and (T_peaks_round[1] - T_peaks_round[2]) < 10**-n_deci :# DOUBLE CONDITIONAL DEMANDS CONSISTENT PERIOD
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

    p_arr = np.zeros((T_steps,int(tf/dt)))
    m_m_arr = np.zeros_like(p_arr)
    m_arr = np.zeros_like(p_arr)
    peak_indices_arr = []

    for index, omega_input in enumerate(tqdm(omega_array)):

        x, t = RK4(x0, t0, tf, dt, A_osc, omega_input, waveshape, param_var)

        T_peaks, peak_indices = peak_identifier(x, t)

        T_peaks_arr.append(T_peaks)
        peak_indices_arr.append(peak_indices)

        p_arr[index, :] = x[0,:]
        m_m_arr[index, :] = x[1,:]
        m_arr[index, :] = x[2,:]

        T_output[index] = T_peaks.mean()# period_determiner(T_peaks) #T_peaks.mean()# NB Mean is used, should probably be changed?
        T_output_var[index]= T_peaks.var()


    return T_input, T_output, T_output_var, T_peaks_arr, p_arr, m_m_arr, m_arr, peak_indices_arr

# %%

def var_IC_poincare(p0, m_m0, m0,  n_IC, T_chosen, t0, tf, dt):
    
    omega_input = 2 * np.pi / T_chosen
    
    p0_arr = p0
    m_m0_arr = m_m0
    m0_arr = m0

    x0_arr = np.stack((p0_arr, m_m0_arr, m0_arr) ,axis=-1)

    p_arr = np.zeros((len(m_m0),int(tf/dt)))
    m_m_arr = np.zeros_like(p_arr)
    m_arr = np.zeros_like(p_arr)
    # peak_indices_arr = []

    for index, x0_ind in enumerate(tqdm(x0_arr)):

        x, t = RK4(x0_ind, t0, tf, dt, A_osc, omega_input, waveshape, param_var)

        # T_peaks, peak_indices = peak_identifier(x, t)

        # T_peaks_arr.append(T_peaks)
        # peak_indices_arr.append(peak_indices)

        p_arr[index, :] = x[0,:]
        m_m_arr[index, :] = x[1,:]
        m_arr[index, :] = x[2,:]

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
    
    mask_comp = mask_y_max * mask_y_min * mask_z_max * mask_z_min # Make a collected mask to filter out clear outliers

    print(mask_comp)
    for i in range(x_len-1):
        if mask_comp[i]:
            if x[i+1] < x_intersect and x[i] > x_intersect:
                y_intersect = (y[i+1] + y[i])/2
                z_intersect = (z[i+1] + z[i])/2
                
                y_poincare = np.append(y_poincare,y_intersect)
                z_poincare = np.append(z_poincare,z_intersect)

            elif x[i+1] > x_intersect and x[i] < x_intersect:
                y_intersect = (y[i+1] + y[i])/2
                z_intersect = (z[i+1] + z[i])/2
                
                y_poincare = np.append(y_poincare,y_intersect)
                z_poincare = np.append(z_poincare,z_intersect)


    return y_poincare, z_poincare

# %% Varying kappa

def kappa_sin(t, amp, T):
    
    omega = 2 * np.pi/T

    sine_val = ( np.sin(omega * t) + 1) / 2
    wave_factor = 1 - amp * sine_val

    return kappa*wave_factor




# %% 

dt = 0.001
tf = 1200 
t0 = 0
x0 = np.array([0,0,0])



x0_start = 0
x0_end = 2
n_x0 = 10
T_start = 0.1
T_end = 11
T_steps = 101 # 101 for nice periods

# %% Calculate 'natural' period
A_osc = 0.47#0.086 # Amplitude of oscillations relative to kappa_0
T_factor = 2#0.7



omega_kappa = 1
x_natural, t_natural = RK4(x0, t0, tf, dt, A_osc, omega_kappa, 0, 0)
T_list, peak_ind = peak_identifier(x_natural,t_natural)
T_natural = T_list[0]







T_kappa =  T_natural * T_factor# Period of external oscillator in hours

omega_kappa = 2*np.pi / T_kappa

waveshape = 1 # Waveshape of varied parameter. 0 is no varying, 1 is sine waev, 2 is square wave.

param_var = 1 # 1 IS KAPPA, 2 IS LAMBDA AND 0 IS NONE


# %% 
p0 = 1.5
m0 = 1

m_m0_start = 0.1
m_m0_inc = 0.0001
n_m_m0 = 10
m_m0_end = m_m0_start+n_m_m0*m_m0_inc*2
m_m0_list = np.arange(m_m0_start, m_m0_end, m_m0_inc)[:n_m_m0]
print(m_m0_list)

# %%

p53_arr = []
for i, m_m0 in enumerate(tqdm(m_m0_list)):
    x, t = RK4(np.array([p0, m_m0, m0]), t0, tf, dt, A_osc, omega_kappa, 1, 1)
    p53_arr.append(x[0])


# %%
max_factor = 1.5
skip_perc = 0.975
zoom_perc = 0.2
save_fig = False


import matplotlib as mpl
plt.rcParams.update({'font.size': 16})

cmap = mpl.cm.get_cmap('jet')

x_max = int(np.round(zoom_perc*tf))
x_min = 0

fig1, ax1 = plt.subplots(figsize=(12,5), dpi=250)

for j, p53_trace in enumerate(p53_arr):
    ax1.plot(t, p53_trace,zorder=n_m_m0 - j, c=cmap(j/n_m_m0), lw=0.9)
ax1.set_xlim(x_min,x_max)

ax1.set_title(f'{n_m_m0} trajectories for $A_{{osc}}={A_osc}$, $T_{{osc}}={T_factor}T_{{natural}}$\nInitial $m_m$ varied by 0.1%')
ax1.set_xlabel('Time [h]')
ax1.set_ylabel('p53 [a.u.]')

fig1.tight_layout()

graph_dir = r'GRAPHDIR'
os.chdir(graph_dir)
if save_fig:
    fig1.savefig(f'chaos_trajectories.png', facecolor='w',dpi=250,bbox_inches = "tight")

# fig7, ax7 = plt.subplots(figsize=(10,4), dpi=250)

# #### Plotting state 1
# ax7.plot(t,x_arr_state1[0], label=f'State 1', color=colors[0], zorder=2, lw=2)
# ax7.plot(t[peak_indices_state1],x_arr_state1[0,peak_indices_state1], 'o', label=f'Identified peaks for state 1', color='orange', markersize=8, markeredgecolor='k', alpha = 1, zorder=10)

# #### Plotting state 2
# ax7.plot(t,x_arr_state2[0], label=f'State2', color=colors[1], zorder=2, lw=2)
# ax7.plot(t[peak_indices_state2],x_arr_state2[0,peak_indices_state2], 'X', label=f'Identified peaks for state 2', color='lawngreen', markersize=8, markeredgecolor='k', alpha = 1, zorder=10)


# ax7.set_xlim(x_min,x_max)
# ax7.set_ylim(-0.1,5)

# ax7_1 = ax7.twinx()

# ax7_1.plot(t,x_kappa_chosen, color='darkgreen', label='Kappa', alpha=0.6, zorder=1, ls='--')
# ax7_1.set_ylim(0,kappa*max_factor)
# ax7_1.set_ylabel('kappa [a.u.]')
# ax7_1.set_yticks([6,8,10, 12])

# ax7.set_xlabel('Time [h]')
# ax7.set_ylabel('p53 [a.u.]')
# ax7.set_title(f'Multiple stabilities for $A_{{osc}}={A_osc:.3f}$ and $T_{{osc}}={T_factor:.1f}T_{{natural}}$')

# lines, labels = ax7.get_legend_handles_labels()
# lines1, labels1 = ax7_1.get_legend_handles_labels()
# ax7.legend(lines + lines1, labels + labels1, loc='upper right', ncol=3, framealpha=1)

# ax7.grid(which='both', ls='--')
# # ax.legend(loc='best')

# fig7.tight_layout()

# # fig7.savefig('m0_var_trajectories_Tind38', dpi=500)

# graph_dir = r'GRAPHDIR'
# os.chdir(graph_dir)
# if save_traces:
#     fig7.savefig(f'multi_stab_trajectory_2_edited.png', facecolor='w',dpi=250,bbox_inches = "tight")
# # %%








# %%

# %%

# %%
