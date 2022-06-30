# %% Model C 
# Cleaned script

# Script solving the differential equations for model C as described in Liv's thesis adding an external oscillator in the form of modifying the parameter Kappa
#
# %% Imports

# %matplotlib ipympl
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
from numpy.core.defchararray import array
from scipy.signal import find_peaks, square
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import os

# Import constants fro mconstant file

## Import constants by adding path to folder
sys.path.append(r'CONSTANTSPATH')
from constants import *


# %% Timetaking
start_time = time.time()

font_point = 16


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


# %%
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

    # print(mask_comp)
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


def find_min_pc_intersect_data(x):
    

    p_poincare_half = 1.5 # NBNBNB LOOKING AT FIXED p53 level
    # p_poincare_half = (np.max(x[0,trim_index:])-np.min(x[0,trim_index:]))/2+np.min(x[0,trim_index:])

    m_m_poincare_min = np.min(x[1,:])
    m_m_poincare_max = np.max(x[1,:])
    m_m_poincare_half = (m_m_poincare_max-m_m_poincare_min)/2+m_m_poincare_min



    m_poincare_half = (np.max(x[2,:])-np.min(x[2,:]))/2+np.min(x[2,:])
    m_poincare_min = np.min(x[2,:])
    m_poincare_max = np.max(x[2,:])

    m_m_range = np.array([m_m_poincare_min, m_m_poincare_half])
    m_range = np.array([0,m_poincare_max])

    y_poincare, z_poincare = poincare_section(x[0,:], x[1,:], x[2,:], p_poincare_half, m_m_range, m_range)


    uniq_dec = 5 # ROUNDING DECIMAL - SENSITIVITY OF DETECTION

    y_pc_round = np.around(y_poincare, decimals=uniq_dec)
    z_pc_round = np.around(z_poincare, decimals=uniq_dec)

    y_pc_uniq, y_pc_uniq_ind, y_pc_uniq_count = np.unique(y_pc_round, return_index=True, return_counts=True)
    z_pc_uniq, z_pc_uniq_ind, z_pc_uniq_count = np.unique(z_pc_round, return_index=True, return_counts=True)

    if len(y_poincare) == len(y_pc_uniq):
        # print(f'NO PERIODICITY DETECTED for {x0}')
        return None, None


    ind_intersect_min = y_pc_uniq_ind[0]

    return y_pc_round[ind_intersect_min], z_pc_round[ind_intersect_min]

# %% Initials
t0 = 0
tf = 1200
dt = 0.001

trim_perc = 0.9 # Where data as been cut
trim_index = int(np.ceil( trim_perc * tf/dt ))

t = np.arange(t0, tf, dt)[trim_index:]



A_start = 0.01
A_end = 0.5
n_A = np.int((A_end - A_start) *100) +1


p0_start = 1.5  
p0_end = 1.5
m_m0_start = 0.01
m_m0_end = 0.1
m0_start = 1
m0_end = 1

n_IC = 50
IC_stepsize = 0.001

m_m0_list = np.arange(m_m0_start, m_m0_end, IC_stepsize)


# data_dir_list = ['fix_T_0_70_varA_varx0', 'fix_T_0_80_varA_varx0','fix_T_0_90_varA_varx0','fix_T_0_95_varA_varx0', 'fix_T_1_00_varA_varx0','fix_T_1_05_varA_varx0']
# T_data_list = [0.70, 0.80, 0.90, 0.95, 1, 1.05]



data_dir_list = ['fix_T_0_70_varA_varx0'] # , 'fix_T_0_80_varA_varx0','fix_T_0_90_varA_varx0','fix_T_0_95_varA_varx0', 'fix_T_1_00_varA_varx0','fix_T_1_05_varA_varx0','fix_T_1_10_varA_varx0','fix_T_1_25_varA_varx0','fix_T_1_40_varA_varx0','fix_T_1_50_varA_varx0','fix_T_1_65_varA_varx0'
T_data_list = [0.70]#, 0.80, 0.90, 0.95, 1, 1.05, 1.10, 1.25, 1.40, 1.50, 1.65



# %% DATA IMPORT

def total_variance_varA_varIC(data_dir):
    """
    Function that takes data directory as input and outputs variance for a chosen Poincare section.
    """


    data_dir_str = r'DATADIR\\' + data_dir
    

    os.chdir(data_dir_str) # Change directory to data folder

    # Make list of available files (TAKES ALL FILES): 
    file_list = [f for f in os.listdir(data_dir_str) if os.path.isfile(os.path.join(data_dir_str, f))]

    ## FIRST INITIALISE EMPTY LISTS AND SO ON
    amp_list = []

    ampcount = 0
    # file_list = file_list[-250:] # FORSØGER MED 10

    var_list = []
    var_tot_fixamp = 0

    pc_min_y = []
    pc_min_z = []

    # SNIPPET FOR GOING THROUGH X-data
    for ii, file_name in enumerate(tqdm(file_list)):
        # Load each file and carry out calculations within this section


        with np.load(file_name) as data_file:
            # FIRST LOAD DATA
            x = data_file['x']
            Amp = data_file['Amp']
            x0 = data_file['x0']

            if Amp not in amp_list:
                
                if ii != 0:
                    var_list.append(np.var(pc_min_y)+np.var(pc_min_z))

                amp_list.append(Amp)
                
                ampcount += 1
                var_tot_fixamp = 0
                pc_min_y = []
                pc_min_z = []
                
            
            # Finding range for poincare section
            m_m_poincare_min = np.min(x[1,:])
            m_m_poincare_max = np.max(x[1,:])
            m_m_poincare_half = (m_m_poincare_max-m_m_poincare_min)/2+m_m_poincare_min

            m_poincare_half = (np.max(x[2,:])-np.min(x[2,:]))/2+np.min(x[2,:])
            m_poincare_min = np.min(x[2,:])
            m_poincare_max = np.max(x[2,:])

            m_m_range = np.array([m_m_poincare_min, m_m_poincare_half])
            m_range = np.array([0,m_poincare_max])


            y_pc, z_pc = poincare_section(x[0], x[1], x[2], p0_start, m_m_range, m_range)

            y_pc_var = np.var(y_pc)#
            z_pc_var = np.var(z_pc)

            pc_min_y.append(np.max(y_pc))
            pc_min_z.append(np.max(z_pc))

            var_tot_fixamp = var_tot_fixamp + y_pc_var + z_pc_var


            if ii == len(file_list)-1:
                var_list.append(np.var(pc_min_y)+np.var(pc_min_z))
    

    return amp_list, var_list

# amp_list, var_list = total_variance_varA_varIC(data_dir_list[0])

############################################
T_outtake = 0.4
############################################

T_ex_string = str(int(T_outtake*100)).zfill(3)
T_string = T_ex_string[0] + '_' + T_ex_string[1:]

# data_dir = r'DATADIR'

data_dir_name = 'fix_T_' + T_string + '_varA_varx0'

amp_list, var_list = total_variance_varA_varIC(data_dir_name)
amp_list = np.stack(amp_list,axis=0)


# amp_list = np.arange(0,0.301,0.001)

# %%


# %% RESET PATH

file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)
# %%

plt.rcParams.update({'font.size': 16})

# Qualitatively observed dual stable states
# x_dual_state_arise = [0.169, 0.245]
# x_dual_state_dissap = [0.204, 0.267]



fig, ax = plt.subplots(figsize=(10,4),dpi=500)
fig.suptitle('Variance between different stable states')



ax.plot(amp_list,var_list, color='red', label='Variance for 100 different ICs')
# ax.vlines(x_dual_state_arise, 0, np.max(var_list), colors = 'k', ls='--', lw=2, label= 'Beginning of dual state', zorder=3)
# ax.vlines(x_dual_state_dissap, 0, np.max(var_list), colors = 'b', ls='--', lw=2, label='End of dual state', zorder=4 )

# ax.legend(loc='best')
# ax.set_ylim(-0.1*np.max(var_list),1.7*np.max(var_list))
# ax.set_xlim(0.15,0.3)
ax.set_xlabel('Amplitude')
ax.set_ylabel('Variance [a.u.]')
ax.legend(loc='best', framealpha = 1)
fig.tight_layout()

# fig.savefig('Variance_T_07_100IC', dpi=500)



# fig.text(0.5, 0.04, 'Amplitude', ha='center', fontsize=font_point)
# fig.text(0.04, 0.5, 'Variance [a.u.]', va='center', rotation='vertical', fontsize=font_point)
# fig.tight_layout(rect=[0.05, 0.05, 0.92, 0.95])

# fig.savefig('PC_int_variance_8Ts',dpi=500)

# %%

# def GetSpacedElements(array, numElems = 4):
#     """
#     Function that returns number of equidistant indexes of list
#     """
#     out = np.array(np.round(np.linspace(0, len(array)-1, numElems))).astype(int)
#     return out



# # p0_start = 1.5
# # p0_end = 2
# m_m0_start = 0.01
# m_m0_end = 0.1
# m0_start = 1
# m0_end = 1

# n_IC = 50
# IC_stepsize = 0.001

# m_m0_list = np.arange(m_m0_start, m_m0_end, IC_stepsize)



# def data_outtake(T_ex, Amp_ex, IC_list, n_IC):
#     """
#     Function that gives a list of filenames for a chosen T and amplitude.
#     IC_list is the possible ICs to choose from, and n_IC is the number of chosen data sets. These are taken from the IC_list with equidistant values.
#     """
    
#     IC_idx = GetSpacedElements(IC_list, n_IC)
#     IC_chosen_list = IC_list[IC_idx]

#     Amp_ex_string = str(int(Amp_ex*1000)).zfill(4)

#     T_ex_string = str(int(T_ex*100)).zfill(3)
#     T_string = T_ex_string[0] + '_' + T_ex_string[1:]
#     data_ex_dir =     r'DATAPATH'

    
#     data_ex_list = []
#     for IC_val in IC_chosen_list:
#         IC_val_str = str(int((IC_val*1000))).zfill(4)
#         data_filename = 'T_'+ T_string + '_A_' + Amp_ex_string + '_' + IC_val_str + '.npz'

#         data_ex_list.append(data_filename)


#     return data_ex_list, data_ex_dir

#     # # Change directory to corresponding amplitude
#     # data_dir_str = r'DATAPATH\\' + data_dir
    

#     # os.chdir(data_dir_str) # Change directory to data folder


# # %%
# Amp_outtake = 0.158
# T_outtake = 0.7
# n_IC_outtake = 100




# data_ex_list, data_ex_dir = data_outtake(T_outtake, Amp_outtake, m_m0_list, n_IC_outtake)

# idx = np.where(np.round(T_data_list,3)==T_outtake)[0][0]



# font_point = 16



# fig2, ax2 = plt.subplots()
# fig2.suptitle(f'Variance for T={T_outtake}', fontsize=font_point)


# ax2.plot(amp_list,var_list, label=f'T = {T_data_list[idx]} T_natural', color='red')
# ax2.axvline(x=Amp_outtake, color='blue', ls='--', label=f'Investigated amplitude, A={Amp_outtake}')
# # ax.set_title(f'Variance for external period T={T_data_list[ii]}')
# ax2.legend(loc='best')
# ax2.set_ylim(-0.05*np.max(var_list),1.3*np.max(var_list))
# ax2.set_xlim(np.min(amp_list),np.max(amp_list))
# # %%




# os.chdir(data_ex_dir)

# x_data_ex = []
# x0_data_ex = []
# for ii, file_name in enumerate(data_ex_list):
#     # Load each file and carry out calculations within this section


#     with np.load(file_name) as data_file:
#         # FIRST LOAD DATA
#         x_data_ex.append(data_file['x'])
#         x0_data_ex.append(data_file['x0'])

# # %%

# im_dir = r'IMPATH'

# import gc

# def plot_ex_traj(x_data_arr, x0_data_arr, amp, save_fig=False):

#     # Phase plot
#     color= plt.cm.hsv(np.linspace(0,1,len(x_data_arr)))

#     fig1 = plt.figure(dpi=250,figsize=(6,5))
#     ax1 = fig1.add_subplot(111, projection='3d')

#     for jj, x_arr in enumerate(x_data_arr):
#         ax1.plot(x_arr[0,:], x_arr[1,:], x_arr[2,:], c=color[jj], lw=1,label=f'x0 = {x0_data_arr[jj][1]}', zorder=len(x_data_arr)-jj)


    
#     ax1.set_xlabel('p53 [a.u.]')
#     ax1.set_ylabel('m_mRNA [a.u.]')
#     ax1.set_zlabel('m [a.u.]')
#     ax1.set_xlim(0,5)
#     ax1.set_ylim(0.07,0.38)
#     ax1.set_zlim(0.8,2.75)
#     # ax1.legend(loc='best')
#     fig1.tight_layout()


#     fig1.suptitle(f'Phase portrait for {len(x_data_arr)} different ICs\nA={amp:.3f}') #
    
#     if save_fig:
#         os.chdir(im_dir)
#         fig1.savefig('phaseplot_amp_' + str(int(amp*1000)).zfill(4)) 
    
    
    
#     fig1.clf()
#     plt.close(fig1)
#     gc.collect()



# for Amp in tqdm(amp_list):
#     if Amp<0.0:
#         data_ex_list, data_ex_dir = data_outtake(T_outtake, Amp, m_m0_list, n_IC_outtake)

#         os.chdir(data_ex_dir)

#         x_data_ex = []
#         x0_data_ex = []
#         for ii, file_name in enumerate(data_ex_list):
#             # Load each file and carry out calculations within this section


#             with np.load(file_name) as data_file:
#                 # FIRST LOAD DATA
#                 x_data_ex.append(data_file['x'])
#                 x0_data_ex.append(data_file['x0'])


#         plot_ex_traj(x_data_ex, x0_data_ex, Amp, save_fig=False)
    
