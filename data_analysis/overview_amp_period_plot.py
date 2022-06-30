# %% Introductory analysis, data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from scipy.signal import find_peaks, square
from scipy.fftpack import fft, ifft, fftfreq


# %% Polyomial correction
5
def poly_correct(y_data, ndof = 3):
    """Function that takes a"""
    x_poly = np.arange(len(y_data))
    poly_coeffs = np.polyfit(x_poly, y_data, deg = ndof)
    poly_values = np.polyval(poly_coeffs,x_poly)
    data_corrected_poly = y_data - poly_values
    data_corrected = data_corrected_poly# - np.min(data_corrected_poly)
    return data_corrected



# %% Making list of csv-data files in data direcotry

data_dir = r'DATAPATH'

os.chdir(data_dir)

data_files = glob.glob('*.csv')

# %% File focus



# df = pd.DataFrame(data_run)

# df.describe
# %%

def tweak_peak_finder(y_data):
    peaks_idx = find_peaks(y_data, prominence=np.mean(y_data)/3, distance = 2*6)
    return peaks_idx[0]


def nutlin_data(file_name, y_data):
    if file_name == 'MP_2020_12_04_measurements_YFP.csv':
        t_start = 98
        t_end = 461
        t_period = 11
        concentration = 2

    elif file_name == 'MP_2021_04_17_measurements_YFP.csv':
        t_start = 98
        t_end = 527
        t_period =  11
        concentration = 0.5

    elif file_name == 'MP_2021_08_11_measurements_YFP.csv':
        t_start = 93
        t_end = 522
        t_period = 11 
        concentration = 1

    elif file_name == 'MP_2021_08_16_measurements_YFP.csv':
        t_start = 93
        t_end = 527
        t_period = 11 
        concentration = 0.25

    elif file_name == 'MP_2021_08_22_measurements_YFP.csv':
        t_start = 93
        t_end = 359
        t_period = 8 
        concentration = 0.5

    elif file_name == 'MP_2021_09_12_measurements_YFP.csv':
        t_start = 94
        t_end = 391
        t_period = 9
        concentration = 0.5
    
    elif file_name == 'MP_Freq_40_05_measurements_YFP.csv':
        t_start = 125
        t_end = 257
        t_period = 4
        concentration = 0.5

    elif file_name == 'MP_Freq_70_05_measurements_YFP.csv':
        t_start = 99
        t_end = 414
        t_period = 7
        concentration = 0.5
    
    else:
        t_start = 1
        t_end = 2
        t_period = 9
        concentration = 0

    x_nutlin = np.arange(len(y_data))/6
    x_square = np.arange(t_end - t_start)/6
    y_square = (square(2*np.pi*x_square / (t_period))+1)/2 * concentration
    y_nutlin = np.pad(y_square,[t_start, len(y_data)-t_end], mode = 'constant')

    t_end_full = t_end + int(6*t_period/2)

    return x_nutlin, y_nutlin, concentration, t_start, t_end_full, t_period



data_idx = 7 # Choose which file to analyse
run_idx = 3

data_run = np.loadtxt(data_files[data_idx],delimiter=',')

print(
    f"""
    The chosen file is {data_files[data_idx]} and run index is {run_idx}
    
    It contains a number of {len(data_run)} experiments
    """
)


# %% Average and std

data_num, data_len = data_run.shape

data_corrected_arr = np.zeros_like(data_run)

for i, data in enumerate(data_run):
    data_corrected_arr[i,:] = poly_correct(data, ndof=3)

data_run_mean = np.mean(data_corrected_arr, axis = 0)
data_run_std = np.std(data_corrected_arr, axis = 0)

x_nutlin, y_nutlin, concentration, t_start_idx, t_end_idx, nutlin_period = nutlin_data(data_files[data_idx], data_run[0])

# %% Plotting
colors = ['red', 'blue', 'green']
plt.rcParams.update({'font.size': 12})


periods_arr = [4, 7, 8, 9, 11, 11, 11, 11]
concentrations_arr = [0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 1, 2]

experiments_number = [207, 159, 105, 112, 106, 102, 97, 23]



fig, ax = plt.subplots(figsize=(6,3), dpi=150)
ax.scatter(periods_arr, concentrations_arr, zorder=10, label = 'Measurement points', color= 'gold', s=100, edgecolors='k')
ax.set_xlabel('Nutlin period [h]')
ax.set_ylabel(f'Nutlin concentration [$\mu M$]')

ax.vlines(5.5, -1, 5, ls='dotted', color='k', label=f'$T_{{natural}}$ (Estimate)')
# ax.vlines(5.5*3/2, -1, 5, ls='dotted', color='green',alpha=0.7)
ax.vlines(5.5*2, -1, 5, ls='dotted', color='blue',alpha=0.7, label=f'$2 T_{{natural}}$')

ax.legend(loc='upper left', framealpha=1)

ax.set_ylim(0,2.1)
ax.set_xlim(3,12)

ax.set_title('Overview of experiments')

fig.tight_layout()


save_graph = False

if save_graph:
    graph_dir = r'DATAPATH'
    os.chdir(graph_dir)
    fig.savefig('measurements_overview.png', dpi=150, facecolor='w')

# %%
