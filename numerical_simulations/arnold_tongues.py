# %%


from configparser import Interpolation
from itertools import filterfalse
from operator import length_hint
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
import matplotlib as mpl

dname = r'DIRECTORYPATH'
os.chdir(dname)

# %% Model C
# Cleaned script

# Script solving the differential equations for model C as described in Liv's thesis adding an external oscillator in the form of modifying the parameter Kappa
#
# %% Imports

from constants import *


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

def file_reader(T, A):
    A_string = f'{A*100:.0f}'.zfill(3)
    T_string = f'{T*100:.0f}'.zfill(3)
    
    file_name = f'T_'+T_string+f'_A_'+A_string+'.npz' 

    with np.load(file_name) as data_file:
        
        x = data_file['x']

    return x


def data_creater(T_list, amp_list):
    data_dir = r'DATAPATH'
    os.chdir(data_dir)


    file_list = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]



    for Amp in tqdm(amp_list):

        A_string = f'{Amp*100:.0f}'.zfill(3)

        for T in T_list:
            T_string = f'{T*100:.0f}'.zfill(3)

            if f'T_'+T_string+f'_A_'+A_string+'.npz' in file_list:
                """ IF FILE IS ALREADY REGISTERED CONTINUE, MEANS NO OVERWRITING"""
                continue
            else:#
                omega_kappa = 2*np.pi/(T*T_natural)
                x, t = RK4(x0, t0, tf, dt, Amp, omega_kappa, 1, 1)
                np.savez(f'T_'+T_string+f'_A_'+A_string+'.npz', x=x[:,0:-1:10], Amp = Amp, T=T)





# %%
## Best are height=0.5  and length=0.3
height_trim=0.5
length_trim=0.3
print(f'height trim = {height_trim}')
print(f'length trim = {length_trim}')
def peak_identifier(x, t, skip_perc=0.7, height_trim=height_trim, length_trim=length_trim): #skip_perc=0.8, height_trim=0.8, length_trim=0.3
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
            T_period = None

    return T_period

# %% Function that finds Tp-TN relationship (entrainment) for various periods of external oscillator
def Tp_vs_Tn2(T_list, Amp, t):

    T_peaks_arr = []
    peak_indices_arr = []
    T_output = np.zeros_like(T_list)
    T_output_var = np.zeros_like(T_list)


    for index, T in enumerate(tqdm(T_list)):

        x = file_reader(T,Amp)

        T_peaks, peak_indices = peak_identifier(x, t)

        T_peaks_arr.append(T_peaks)
        peak_indices_arr.append(peak_indices)

        T_output[index] = T_peaks.mean()#period_determiner(T_peaks) #T_peaks.mean()# NB Mean is used, should probably be changed?
        T_output_var[index]= T_peaks.var()


    return T_output, T_output_var, T_peaks_arr, peak_indices_arr




# def Tp_vs_TN(T_start, T_end, T_steps, t0, tf, dt, x0):
#     T_input = np.linspace(T_start, T_end, T_steps)
#     omega_array = 2 * np.pi / T_input
#     T_output = np.zeros_like(omega_array)
#     T_output_var = np.zeros_like(omega_array)
#     T_peaks_arr = []

#     p_arr = np.zeros((T_steps,int(tf/dt)))
#     m_m_arr = np.zeros_like(p_arr)
#     m_arr = np.zeros_like(p_arr)
#     peak_indices_arr = []

#     for index, omega_input in enumerate(tqdm(omega_array)):

#         x, t = RK4(x0, t0, tf, dt, A_osc, omega_input, waveshape, param_var)

#         T_peaks, peak_indices = peak_identifier(x, t)

#         T_peaks_arr.append(T_peaks)
#         peak_indices_arr.append(peak_indices)

#         p_arr[index, :] = x[0,:]
#         m_m_arr[index, :] = x[1,:]
#         m_arr[index, :] = x[2,:]

#         T_output[index] = T_peaks.mean()# period_determiner(T_peaks) #T_peaks.mean()# NB Mean is used, should probably be changed?
#         T_output_var[index]= T_peaks.var()


#     return T_input, T_output, T_output_var, T_peaks_arr, p_arr, m_m_arr, m_arr, peak_indices_arr



# %%

x0 = np.array([0,0,0])
t0 = 0
tf = 600
dt = 0.001
paramvar = 1
waveshape = 1



amp_start = 0.01
amp_end = 0.41
amp_int = 0.01

# amp_list = np.arange(amp_start, amp_end+amp_int, amp_int)
amp_list = np.arange(amp_start,amp_end,amp_int)

# amp_chosen = 0.1

x_natural, t = RK4(x0, t0, tf, dt, 0, 1, 0, 0)
T_list_peaks, peak_ind = peak_identifier(x_natural, t)
T_natural = T_list_peaks[0]


# T_list = np.array([0.7, 0.8, 1.65, 2.05])
T_start = 0.02
T_end = 2.5
dT = 0.02

T_list = np.arange(T_start, T_end, dT)
T_chosen_list = T_list * T_natural

data_creater(T_list, amp_list)



# %% Read data

# %%
data_dir = r'DATADIR'
os.chdir(data_dir)

# x_T = []

# for T in T_list:
#     x_A = []
#     for A in amp_list:
#         x = file_reader(T, A)
#         x_A.append(x)
#     x_T.append(x_A)

# %%
t_peaks_arr = []
y_t_peaks_arr = []
for T in T_chosen_list:
    t_peaks = np.arange(1/4*T,tf,T)
    t_peaks_arr.append(t_peaks)
    y_t_peaks_arr.append(np.zeros_like(t_peaks))



t = t[::10]

T_output_list = []
T_output_var_list = []


ratios_list = np.array([ 1/4, 1/3, 1/2, 2/3, 3/4, 1, 4/3, 3/2, 5/3, 2])
ratios_str_list = np.array(['1/4', '1/3', '1/2', '2/3','3/4', '1', '4/3', '3/2', '5/3', '2'])

# for j, amp_chosen in enumerate(amp_list):

#     T_output, T_output_var, T_peaks_arr, peak_indices_arr = Tp_vs_Tn2(T_list, amp_chosen, t)
#     T_mod_list = []
#     for ii in range(len(T_output)):
#         if T_output_var[ii]>0.01:
#             T_mod_list.append(np.nan)
#         else:
#             T_mod_list.append(T_output[ii])
#     T_output_list.append(T_mod_list)
#     T_output_var_list.append(T_output_var)

for j, amp_chosen in enumerate(amp_list):

    T_output, T_output_var, T_peaks_arr, peak_indices_arr = Tp_vs_Tn2(T_list, amp_chosen, t)
    T_mod_list = []
    for ii in range(len(T_output)):
        for ratio in ratios_list:
            if np.isclose(T_chosen_list[ii]/T_output[ii],ratio,atol=0.01):
                check_ratio = True
                break
            else:
                check_ratio=False
        if T_output_var[ii]<1000 and check_ratio:
            T_mod_list.append(T_output[ii])
        else:
            T_mod_list.append(np.nan)
    T_output_list.append(T_mod_list)
    T_output_var_list.append(T_output_var)





def x_uncoupled(T_list, T_natural):
    return T_natural/T_list

x_uncoupl = x_uncoupled(T_chosen_list, T_natural)




# %%
plt.rcParams.update({'font.size': 12})
labelsize=17

cmap = mpl.cm.get_cmap('tab10')#mpl.colors.ListedColormap(['white', 'red'])
bounds= np.array([ -0.02, 1/4, 1/3, 1/2, 2/3, 3/4, 1, 4/3, 3/2, 5/3, 2])+0.01
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig2, ax2 = plt.subplots(figsize=(8,6),dpi=250)
im2 = ax2.imshow(1/(T_output_list/T_chosen_list), aspect='auto', extent=[0,T_end,amp_start,amp_end], origin='lower', interpolation='none', cmap=cmap, norm=norm)
ax2.set_xticks(ratios_list)
plt.xticks(rotation=90)
ax2.set_xticklabels(ratios_str_list)
cbar = fig2.colorbar(im2, ax=ax2, ticks=ratios_list) # , cmap=cmap, norm=norm, ticks=bounds
cbar.set_ticklabels(ratios_str_list)
cbar.set_label(f'$T_{{osc}}/T_{{measured}}$', fontsize=labelsize)

ax2.set_title(f'Arnold tongues for $T_{{osc}}=0$ to $T_{{osc}}=2.5 T_{{natural}}$', fontsize=labelsize+2)

ax2.set_ylabel(f'$A_{{osc}}$', fontsize=labelsize)
ax2.set_xlabel(f'$T_{{osc}}/T_{{natural}}$', fontsize=labelsize)

fig2.tight_layout()


save_arnold = False

if save_arnold:
    dir_name = r'DATADIR'

    os.chdir(dir_name)

    fig2.savefig(f'arnold_tongues.png',dpi=250, facecolor=('w'), bbox_inches='tight')

# cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
# cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
# cbar.set_label(f'$T_{{est}}/T_{{N}}$', fontsize=labelsize)

# %%



