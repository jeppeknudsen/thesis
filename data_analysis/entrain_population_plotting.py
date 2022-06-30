# %% Introductory analysis, data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from scipy.signal import find_peaks, square
from scipy.fftpack import fft, ifft, fftfreq
from tqdm import tqdm


# %% Polyomial correction

def poly_correct(y_data, ndof = 3):
    """Function that takes a"""
    x_poly = np.arange(len(y_data))
    poly_coeffs = np.polyfit(x_poly, y_data, deg = ndof)
    poly_values = np.polyval(poly_coeffs,x_poly)
    data_corrected_poly = y_data - poly_values
    data_corrected = data_corrected_poly# - np.min(data_corrected_poly)
    return data_corrected



# %% Making list of csv-data files in data direcotry

data_dir = r'DATADIR'

os.chdir(data_dir)

data_files = glob.glob('*.csv')

# %% File focus



# df = pd.DataFrame(data_run)

# df.describe
# %%


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

save_graphs=True

data_run = np.loadtxt(data_files[data_idx],delimiter=',')

print(
    f"""
    The chosen file is {data_files[data_idx]} and run index is {run_idx}
    
    It contains a number of {len(data_run)} experiments
    """
)


# %% Average and std

run_number = 1

data_num, data_len = data_run.shape

data_corrected_arr = np.zeros_like(data_run)

for i, data in enumerate(data_run):
    data_corrected_arr[i,:] = poly_correct(data, ndof=3)

data_run_mean = np.mean(data_corrected_arr, axis=0)
data_run_std = np.std(data_corrected_arr, axis=0)

data_run = data_corrected_arr[run_number]

x_nutlin, y_nutlin, concentration, t_start_idx, t_end_idx, nutlin_period = nutlin_data(data_files[data_idx], data_run)

# %% Plotting
colors = ['red', 'blue', 'green']
plt.rcParams.update({'font.size': 8})



# %% Oeak finding



peak_idx_arr = []
peak_props_arr = []
for n_data_run in range(data_num):

    # Peak adjustments
    peak_height = None
    peak_threshold = None
    peak_distance = 15 # 15 er 2.5 h
    peak_prominence_factor = 1/5
    if np.max(data_corrected_arr[n_data_run])>2000:
        peak_prominence = 2000*peak_prominence_factor
    else:
        peak_prominence = peak_prominence_factor *(np.max(data_corrected_arr[n_data_run])-np.min(data_corrected_arr[n_data_run])) # Del med 5 virker godt
    peak_width = None #



    peak_idx, peak_props = find_peaks(data_corrected_arr[n_data_run], height=peak_height, threshold=peak_threshold, distance=peak_distance, prominence=peak_prominence, width=peak_width)
    peak_idx_arr.append(peak_idx)
    peak_props_arr.append(peak_props)




# %%

def plot_peak_traces(n_start_graph=0, n_graphs = 5, save_graphs=False):


    colors = ['red', 'blue', 'green']
    plt.rcParams.update({'font.size': 8})

    label_size = 12
    x = np.arange(0,data_len)/6
    t_max = data_len/6

    # n_graphs = 5
    # n_start_graph = 20

    h_ratios= np.ones(n_graphs+1)*2
    h_ratios[-1]=1

    fig, ax = plt.subplots(n_graphs+1,1,figsize=(10,7),sharex = True, gridspec_kw={'height_ratios': h_ratios}, dpi = 150)

    for i in range(n_start_graph,n_start_graph+n_graphs):
        ax[i-n_start_graph].plot(x[0:t_start_idx+1], data_corrected_arr[i][0:t_start_idx+1], color=colors[0], label='Mean')

        ax[i-n_start_graph].plot(x[t_start_idx:t_end_idx+1], data_corrected_arr[i][t_start_idx:t_end_idx+1], color=colors[1], label='Mean')

        ax[i-n_start_graph].plot(x[t_end_idx:], data_corrected_arr[i][t_end_idx:], color=colors[2], label='Mean')

        ax[i-n_start_graph].plot(x[peak_idx_arr[i]],data_corrected_arr[i][peak_idx_arr[i]], marker= 'X', ls='None', color='orange', mec = 'black', mew=0.5, ms=7)

        ax[i-n_start_graph].axvline(x[t_start_idx], color = 'k', ls='dotted', alpha = 0.99, label='Nutlin on/off')
        ax[i-n_start_graph].axvline(x[t_end_idx], color = 'k', ls='dotted', alpha = 0.99)
        
        ax[i-n_start_graph].annotate(f'Data run number {i+1}', xy=(0.83, 0.85), xycoords='axes fraction')

    ax[n_graphs].step(x_nutlin, y_nutlin, 'seagreen', label=f'Nutlin period {nutlin_period} h')
    ax[n_graphs].set_ylabel(r'Nutlin [$\mu$M]', fontsize=label_size)
    ax[n_graphs].set_ylim(-0.1*concentration, 1.15*concentration)
    ax[n_graphs].set_xlabel('Time [h]', fontsize=14)
    ax[n_graphs].legend(loc='upper right')

    ax[n_graphs].set_xlim(0,t_max)

    fig.text(0.02, 0.5, 'p53 concentration [a.u.]', position=(0.01,0.4), rotation='vertical', fontsize=14)
    fig.suptitle(f'Traces and identified peaks for $C={concentration} \mu M$ and $T={nutlin_period} h$, data runs {n_start_graph+1}-{n_start_graph+n_graphs+1}', fontsize=16)

    fig.tight_layout(rect=[0.01, 0.01, 1, 1])#rect=[0.01, 0.01, 1, 0.90]

    if save_graphs:
        dir_name = r'GRAPHDIR\\'+f'{data_files[data_idx][3:-21]}_individual_peak_graphs'

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        os.chdir(dir_name)


        fig.savefig(f'Peak_ident_{data_files[data_idx][3:-21]}_runs_'+f'{n_start_graph}'.zfill(3)+f'_'+f'{n_start_graph+n_graphs}'.zfill(3), dpi=250, facecolor='w')
    
    plt.close()

# %%

n_graphs = 5

n_start_list = np.arange(0, data_num, n_graphs)

save_loop_graphs = True

for n_start in tqdm(n_start_list):
    if n_start == n_start_list[-1]:
        plot_peak_traces(n_start_graph=n_start, n_graphs=data_num-n_start, save_graphs=save_loop_graphs)
        # print(f'n_start_graph={n_start}, n_graphs={data_num-n_start+1}')
    
    else:
        plot_peak_traces(n_start_graph=n_start, n_graphs=n_graphs, save_graphs=save_loop_graphs)



# %%
