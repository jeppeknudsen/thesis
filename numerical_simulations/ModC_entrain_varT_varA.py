# %%


from itertools import filterfalse
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

dname = r'SCRIPTDIR'
os.chdir(dname)

# print(f'The directory is {dname}')

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
tf = 600
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


T_list = np.array([0.7, 0.8, 1.64, 2.08])
T_chosen_list = T_list * T_natural

def data_creater(T_list, amp_list):
    data_dir = r'DATADIR'
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

data_creater(T_list, amp_list)


# %% Read data

def file_reader(T, A):
    A_string = f'{A*100:.0f}'.zfill(3)
    T_string = f'{T*100:.0f}'.zfill(3)
    
    file_name = f'T_'+T_string+f'_A_'+A_string+'.npz' 

    with np.load(file_name) as data_file:
        
        x = data_file['x']

    return x

# %%
data_dir = r'DATADIR'
os.chdir(data_dir)


x_T = []

for T in T_list:
    x_A = []
    for A in amp_list:
        x = file_reader(T, A)
        x_A.append(x)
    x_T.append(x_A)

# %%
t_peaks_arr = []
y_t_peaks_arr = []
for T in T_chosen_list:
    t_peaks = np.arange(1/4*T,tf,T)
    t_peaks_arr.append(t_peaks)
    y_t_peaks_arr.append(np.zeros_like(t_peaks))


# %%
trim_start = 0.9
trim_end = 1
idx_start = np.int32(trim_start * tf / 0.01)
idx_end = np.int32(trim_end * tf / 0.01)

t = np.arange(0,tf,0.01)
plt.rcParams.update({'font.size': 14})

fig1, ax1 = plt.subplots(len(amp_list),len(x_T), figsize=(8,10),sharex=True,dpi=250)

for i, x_A in enumerate(x_T):
    for j, x_data in enumerate(x_A):
        ax1[j,i].plot(t[idx_start:idx_end],x_data[0,idx_start:idx_end])
        ax1[j,i].set_xlim(t[idx_start],t[idx_end-1]+0.01)
        ax1[j,i].plot(t_peaks_arr[i],y_t_peaks_arr[i]+0.2,marker='|',c='orange',ms=9,linestyle='None')

for ii, amp in enumerate(amp_list):
    ax1[ii,0].set_ylabel(f'$A = {amp}$')

for jj, T_label in enumerate(T_list):
    ax1[0,jj].set_title(f'$T = {T_label} T_{{natural}}$')
    ax1[-1,jj].set_xlabel('Time [h]')
fig1.tight_layout()

# %%
graph_dir = r'GRAPHDIR'
save_graph = False
if save_graph:
    os.chdir(graph_dir)
    fig1.savefig('entrain_overview.png',facecolor='w',dpi=250)


# %%
A_idx_chosen = 4
T_idx_chosen = 0

T_chosen = T_chosen_list[T_idx_chosen]
A_chosen = amp_list[A_idx_chosen]
y_sin = A_chosen*(np.sin(2*np.pi/T_chosen * t) + 1) / 2

plt.rcParams.update({'font.size': 15})


fig2,ax2 = plt.subplots(1,1,figsize=(6,4),dpi=250, sharex=True)
ax2.plot(t,x_T[T_idx_chosen][A_idx_chosen][0],c='red', label=f'p53', lw=3)
ax2.plot(t_natural, x_natural[0],'orange',alpha=0.55, label='Unperturbed system')
ax2.set_xlim(540,600)
ax2.legend(loc='best')
ax2.set_title(f'Steady state\n$T_{{osc}}={T_list[T_idx_chosen]} T_{{natural}}$, $A_{{osc}}={amp_list[A_idx_chosen]}$')
ax2.set_xlabel('Time [h]')
ax2.set_ylabel('p53 [a.u.]')

ax2.grid('on', alpha=0.6)
# ax2.legend(loc='upper right')



ax2_1 = ax2.twinx()
ax2_1.plot(t,kappa*(1-y_sin), c='green', label=f'$\kappa$',alpha=0.9)

lines, labels = ax2.get_legend_handles_labels()
lines1, labels1 = ax2_1.get_legend_handles_labels()
ax2_1.legend(lines + lines1, labels + labels1, loc='upper right')
ax2_1.set_ylim(kappa*A_chosen*2,np.max(kappa*(1-y_sin))*2.5)
ax2_1.set_ylabel('$\kappa$ [a.u.]')
ax2_1.set_yticks([5,10])










fig2.tight_layout()
save_fig = False
if save_fig:
    os.chdir(graph_dir)
    fig2.savefig('single_entrain_steady.png',dpi=250, facecolor='w')






# %%
