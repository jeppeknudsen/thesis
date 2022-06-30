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

graph_dir = r'GRAPHDIR'
script_dir = r'SCRIPTDIR'

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

A_osc = 0.0 # Amplitude of oscillations relative to kappa_0. Must be <=1. To be varied later
T_kappa = 5

omega = (2*np.pi) / T_kappa# Period factor, to be varied later
# A_osc = 0.4 and omega = 2 for testing
# Interesting for omega=6 and A=0.2 AND A=0.5


print(f' A_osc is {A_osc}, omega is {omega} and T_kappa is {T_kappa}')

#pdot = alpha - (kappa - kappa * A_osc * square(t * 2 * np.pi / T_kappa)) # np.sin( omega * t  ) )*x[2] * x[0] / (x[0] + lam_param) # p53 SQUARE



@njit
def ModC(t, x, amp, omega):

    pdot = alpha - (kappa - kappa * amp * np.sin( omega * t  ) )*x[2] * x[0] / (x[0] + lam_param) # p53

    mmdot = xsi*x[0]*x[0] - delta*x[1] # Mdm2

    mdot = epsilon*x[1] - eta*x[2] # Mdm2

    xdot = np.array( [ pdot, mmdot, mdot ] )

    return xdot

# %% Runge-Kutta 4 solver. Takes initial conditions, start and end time, time increment size and parameters.
# Initial condiitons size correspond to number of equaitons in model.
@njit
def RK4(x0, t0, tf, dt, amp, omega):

    t = np.arange(t0,tf,dt)
    nt = t.size

    nx = x0.size
    x = np.zeros((nx,nt))

    x[:,0] = x0

    for k in range(nt -1):
        k1 = dt * ModC(t[k], x[:,k], amp, omega)
        k2 = dt * ModC(t[k], x[:,k] + k1/2, amp, omega)
        k3 = dt * ModC(t[k], x[:,k] + k2/2, amp, omega)
        k4 = dt * ModC(t[k], x[:,k] + k3, amp, omega)

        dx = (k1 + 2*k2 + 2*k3 + k4)/6

        x[:,k+1] = x[:,k] + dx
    
    return x, t

# f = lambda t, x : ModC(x, params)

# %% Initial conditions and time adjustments

x0 = np.array([0,0,0])

t0 = 0
tf = 1200
dt=0.001 # Step size


# %% Solve for x and t
x, t = RK4(x0, t0, tf, dt, A_osc, omega)


# %% Cut off initial points to investigate steady state and avoid first high peak

skip_perc = 0.25 # 
ind_steady = int(np.ceil(skip_perc * tf/dt)) # Index to skip until
x_steady = np.array(x[0,ind_steady:]) # Trimmed p53 level
t_steady = np.array(t[ind_steady:])

# %% Peakfinder - Homemade possibility if time permits?
# First I find peaks that are at a percentage of the max point of the trimmed data, typically 80%
# I then find the temporal distance between these peaks. I thenrefind peaks of with the first condition AND demand that the peaks are spaced at least 60% of the max distance of peaks from the previous peak find

height_trim = 0.8 # Factor of trim relative to height
length_trim = 0.6 # Factor of trim in length, i.e. time.

height_threshold = x_steady.max()* height_trim

peaks_init, _ = find_peaks(x_steady, height=height_threshold)

max_peaks_init = peaks_init[peaks_init>height_threshold]

T_peaks_init = np.array([]) #

for j in range(len(max_peaks_init) - 1): # Find temporal distance between peaks in order to find max distance
    peak_init_dist = (max_peaks_init[j+1] - max_peaks_init[j])
    T_peaks_init = np.append(T_peaks_init, peak_init_dist)

length_threshold = length_trim * T_peaks_init.max()

# height_threshold = np.mean(x_steady[peaks_init])*0.99 # Factor of 0.99 to accomodate slight difference in peaks


peaks, _ = find_peaks(x_steady,height = height_threshold, distance=length_threshold) # , 


# %% Time between peaks

T_peaks = np.array([])

for j in range(len(peaks) - 1):
    peak_dist = (peaks[j+1] - peaks[j])*dt
    T_peaks = np.append(T_peaks, peak_dist)

print(f'''
The average period is measured to be {T_peaks.mean()}\n
The std is found to be {T_peaks.var()}

''')

# %% Kappa plot

def kappa_osc(t, A_kappa, omega_kappa):
    return (kappa - kappa * A_kappa * np.sin( omega_kappa * t ) )

kappa_var = np.array(kappa_osc(t, A_osc, omega))

# %% Plotting transcient period

save_fig = False


kappa_scale = 1 # Adjust kappa level to fit into graph

end_time = time.time()
print(f'Elapsed time is {end_time - start_time} seconds')

plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots(figsize=(6,4), dpi=250)
ax.plot(t,x[0,:], label='p53 level', lw=3, c='red') # Plotting p53
ax.plot(t,x[1,:], label='m_RNA level', lw=3, c='green') # Plotting p53
ax.plot(t,x[2,:], label='Mdm2 level', lw=3, c='blue') # Plotting p53



# ax.set_ylim(0, x.max()*1.2)
ax.set_xlim(t0,30)
ax.set_title(f'Transient state')

# ax.set_title(f'Oscillating kappa with amplitude of {A_osc} and period of {T_kappa:.2f} h')
ax.grid('on', alpha=0.6)
ax.set_xlabel('Time [h]')
ax.set_ylabel('p53, m_RNA, Mdm2 [a.u.]')
ax.legend(loc='upper right')

fig.tight_layout()
if save_fig:
    os.chdir(graph_dir)
    fig.savefig('modC_plain.png',dpi=250, facecolor='w')



# %% Plotting stable state

kappa_scale = 1 # Adjust kappa level to fit into graph

end_time = time.time()
print(f'Elapsed time is {end_time - start_time} seconds')

# plt.rcParams.update({'font.size': 18})

fig2, ax2 = plt.subplots(figsize=(6,4), dpi=250)
# ax3 = ax2.twinx()

# ax2.plot(t,x[0,:], 'r', label='p53 level', lw=3) # Plotting p53
ax2.plot(t,x[0,:], label='p53 level', lw=3, c='red') # Plotting p53
ax2.plot(t,x[1,:], label='m_RNA level', lw=3, c='green') # Plotting p53
ax2.plot(t,x[2,:], label='Mdm2 level', lw=3, c='blue') # Plotting p53

# ax3.plot(t, kappa_scale*kappa_var, 'b', label=f'kappa level', alpha=0.4, lw=2) # Plotting kappa oscillation

ax2.plot(t_steady[peaks], x_steady[peaks], 'X', ms=10, color = 'gold', markeredgecolor='k', label='Identified peaks')

# ax2.axvline(x=t[ind_steady], ls='dashed', color='k', label='Cut off point')



# lines, labels = ax2.get_legend_handles_labels()
# lines1, labels1 = ax3.get_legend_handles_labels()
# ax3.legend(lines + lines1, labels + labels1, loc='upper right')
ax2.legend(loc='upper right')
# ax2.set_ylim(0, x.max()*1.2)
ax2.set_xlim(tf-630,tf-600)
# ax3.set_ylim(0,kappa_var.max())
ax2.set_title(f'Steady state')
ax2.grid('on', alpha=0.6)
ax2.set_xlabel('Time [h]')
ax2.set_ylabel('p53, m_RNA, Mdm2 [a.u.]')
# ax3.set_ylabel('kappa [a.u.]')

# ax2.legend(loc='best')

# ax3.set_ylim(0,kappa_var.max())
fig2.tight_layout()
if save_fig:
    fig2.savefig('modC_plain_period',dpi=250, facecolor='w')
# ax.plot(t,x[1,:], 'b')
# ax.plot(t,x[2,:], 'black')
# ax.set_ylim([0,6]);

# ax[1].plot(x[0,:],x[1,:])
# %%
