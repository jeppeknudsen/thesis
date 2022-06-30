## peak_temporal_diff_entrainment

# %% Introductory analysis, data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from scipy.signal import find_peaks, square
from scipy import interpolate
from tqdm import tqdm
import matplotlib as mpl


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

data_dir = r'DATAPATH'

os.chdir(data_dir)

data_files = glob.glob('*.csv')

# %%


def nutlin_data(file_name, y_data):
    if file_name == 'MP_2020_12_04_measurements_YFP.csv':
        t_start = 98
        t_end = 461
        t_period = 11
        concentration = 2
        entrain_est = 1

    elif file_name == 'MP_2021_04_17_measurements_YFP.csv':
        t_start = 98
        t_end = 527
        t_period =  11
        concentration = 0.5
        entrain_est = 1

    elif file_name == 'MP_2021_08_11_measurements_YFP.csv':
        t_start = 93
        t_end = 522
        t_period = 11 
        concentration = 1
        entrain_est = 1

    elif file_name == 'MP_2021_08_16_measurements_YFP.csv':
        t_start = 93
        t_end = 527
        t_period = 11 
        concentration = 0.25
        entrain_est = 1

    elif file_name == 'MP_2021_08_22_measurements_YFP.csv':
        t_start = 93
        t_end = 359
        t_period = 8 
        concentration = 0.5
        entrain_est = 1

    elif file_name == 'MP_2021_09_12_measurements_YFP.csv':
        t_start = 94
        t_end = 391
        t_period = 9
        concentration = 0.5
        entrain_est = 1
    
    elif file_name == 'MP_Freq_40_05_measurements_YFP.csv':
        t_start = 125
        t_end = 257
        t_period = 4
        concentration = 0.5
        entrain_est = 1

    elif file_name == 'MP_Freq_70_05_measurements_YFP.csv':
        t_start = 99
        t_end = 414
        t_period = 7
        concentration = 0.5
        entrain_est = 1


    else:
        t_start = 1
        t_end = 2
        t_period = 9
        concentration = 0
        entrain_est = 1

    x_nutlin = np.arange(len(y_data))/6
    x_square = np.arange(t_end - t_start)/6
    y_square = (square(2*np.pi*x_square / (t_period))+1)/2 * concentration
    y_nutlin = np.pad(y_square,[t_start, len(y_data)-t_end], mode = 'constant')

    t_end_full = t_end + int(6*t_period/2)

    return x_nutlin, y_nutlin, concentration, t_start, t_end_full, t_period, entrain_est



data_idx = 3# Choose which file to analyse
run_idx = 3
# %%
def flatten(t):
    return np.array([item for sublist in t for item in sublist])

data_idx_list = range(8)
save_graphs=False




peak_lag_collected_arr_1 = []
peak_lag_collected_arr_2 = []
peak_lag_collected_arr_3 = []


for data_idx in data_idx_list:
    data_run = np.loadtxt(data_files[data_idx],delimiter=',')


    run_number = 1

    data_num, data_len = data_run.shape

    x = np.arange(0,data_len)/6

    data_corrected_arr = np.zeros_like(data_run)

    for i, data in enumerate(data_run):
        data_corrected_arr[i,:] = poly_correct(data, ndof=3)

    data_run_mean = np.mean(data_corrected_arr, axis=0)
    data_run_std = np.std(data_corrected_arr, axis=0)

    data_run = data_corrected_arr[run_number]

    x_nutlin, y_nutlin, concentration, t_start_idx, t_end_idx, nutlin_period, entrain_est = nutlin_data(data_files[data_idx], data_run)




    def get_ave_values(xvalues, yvalues, n = 5):
        signal_length = len(xvalues)
        if signal_length % n == 0:
            padding_length = 0
        else:
            padding_length = n - signal_length//n % n
        xarr = np.array(xvalues)
        yarr = np.array(yvalues)
        xarr.resize(signal_length//n, n)
        yarr.resize(signal_length//n, n)
        xarr_reshaped = xarr.reshape((-1,n))
        yarr_reshaped = yarr.reshape((-1,n))
        x_ave = xarr_reshaped[:,0]
        y_ave = np.nanmean(yarr_reshaped, axis=1)
        return x_ave, y_ave



    n_average = 3

    peak_idx_arr = []
    peak_props_arr = []
    periods_arr = []

    peak_av_arr = []
    x_av_arr = []
    y_av_arr = []
    periods_av_arr = []

    for n_data_run in range(data_num):

        x_av, y_av = get_ave_values(x, data_corrected_arr[n_data_run], n=n_average)
        x_av_arr.append(x_av)
        y_av_arr.append(y_av)

        # Peak adjustments
        peak_height = None
        peak_threshold = None
        peak_distance = 15 # 15 er 2.5 h
        high_prominence_list = [1, 2]
        if data_idx in high_prominence_list:
            peak_prominence_factor = 1/4
        else:
            peak_prominence_factor = 1/6
        if np.max(data_corrected_arr[n_data_run])>6000 and data_idx in high_prominence_list:
            peak_prominence = 6000*peak_prominence_factor
        elif np.max(data_corrected_arr[n_data_run])>2000 and data_idx not in high_prominence_list:
            peak_prominence = 2000*peak_prominence_factor
        else:
            peak_prominence = peak_prominence_factor *(np.max(data_corrected_arr[n_data_run])-np.min(data_corrected_arr[n_data_run])) # Del med 5 virker godt
        peak_width = None #



        peak_idx, peak_props = find_peaks(data_corrected_arr[n_data_run], height=peak_height, threshold=peak_threshold, distance=peak_distance, prominence=peak_prominence, width=peak_width)
        peak_idx_arr.append(peak_idx)
        peak_props_arr.append(peak_props)


        peak_av_idx, peak_props_av = find_peaks(y_av, height=peak_height, threshold=peak_threshold, distance=peak_distance/n_average, prominence=peak_prominence, width=peak_width)
        peak_av_arr.append(peak_av_idx)


        periods_temp = np.diff(peak_idx)

        periods_arr.append(np.append(periods_temp,periods_temp[-1])/(6*nutlin_period))

        periods_av_temp = np.diff(peak_av_idx)
        periods_av_arr.append(np.append(periods_av_temp,periods_av_temp[-1])/(6*nutlin_period/n_average))





    peak_lag_arr_p1 = []
    peak_lag_arr_p2 = []
    peak_lag_arr_p3 = []

    for peaks_run in peak_idx_arr:
        peak_lag_temp_1 = []
        peak_lag_temp_2 = []
        peak_lag_temp_3 = []
        for peak_idx in peaks_run:
            # Phase 1
            if peak_idx < t_start_idx:
                peak_lag_temp_1.append(peak_idx%int(nutlin_period*6))
            # Phase 2
            elif peak_idx >= t_start_idx and peak_idx < t_end_idx:
                peak_lag_temp_2.append((peak_idx-t_start_idx)%int(nutlin_period*6))
            # Phase 3
            elif peak_idx >= t_end_idx:
                peak_lag_temp_3.append((peak_idx-t_end_idx)%int(nutlin_period*6))
            else:
                print('ADVARSEL')
        peak_lag_arr_p1.append(peak_lag_temp_1)
        peak_lag_arr_p2.append(peak_lag_temp_2)
        peak_lag_arr_p3.append(peak_lag_temp_3)
    

    peak_lag_1 = flatten(peak_lag_arr_p1)/6
    peak_lag_2 = flatten(peak_lag_arr_p2)/6
    peak_lag_3 = flatten(peak_lag_arr_p3)/6

    peak_lag_collected_arr_1.append(peak_lag_1)
    peak_lag_collected_arr_2.append(peak_lag_2)
    peak_lag_collected_arr_3.append(peak_lag_3)


#%%
save_graphs = True


plt.rcParams.update({'font.size': 18})
concentration_list = np.array([2, 0.5, 1, 0.25, 1, 1, 1, 1])
period_list = np.array([11, 11, 11, 11, 8, 9, 4, 7])

fix_T_idx_list = [0, 1, 2, 3]
fix_C_idx_list = [4, 5, 6, 7]

fix_T_colors = ['lime', 'limegreen', 'green', 'darkgreen']
fix_C_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

y_vals_fix_T = []
x_vals_fix_T = []
for i in fix_T_idx_list:
    bin_list = np.arange(0,period_list[i]+0.5, 0.5)
    y_vals, bin_edges = np.histogram(peak_lag_collected_arr_2[i], bins=bin_list)
    y_vals_fix_T.append(y_vals)

    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    x_vals_fix_T.append(bin_centers)


fig1, ax1 = plt.subplots(2,2,figsize=(12,10), sharex=True, dpi=250)


ax1[0, 0].hist(peak_lag_collected_arr_2[fix_T_idx_list[3]], bins=np.arange(0,period_list[fix_T_idx_list[3]], 0.5), label=f'$C_N = {concentration_list[fix_T_idx_list[3]]} \mu M$', color = fix_T_colors[0])
ax1[0, 0].set_ylim(-0.01,np.max(y_vals_fix_T[3]*1.4))
ax1[0, 0].legend(loc='best')
ax1[0, 0].set_xlim(0,11)
# ax1[0, 0].set_ylabel(f'Counts pr $0.5$ h', fontsize=22)


ax1[0, 1].hist(peak_lag_collected_arr_2[fix_T_idx_list[1]], bins=np.arange(0,period_list[fix_T_idx_list[1]], 0.5), label=f'$C_N = {concentration_list[fix_T_idx_list[1]]} \mu M$', color = fix_T_colors[1])
ax1[0, 1].set_ylim(-0.01,np.max(y_vals_fix_T[1]*1.4))
ax1[0, 1].legend(loc='best')
ax1[0, 1].set_xlim(0,11)

ax1[1, 0].hist(peak_lag_collected_arr_2[fix_T_idx_list[2]], bins=np.arange(0,period_list[fix_T_idx_list[2]], 0.5), label=f'$C_N = {concentration_list[fix_T_idx_list[2]]} \mu M$', color = fix_T_colors[2])
ax1[1, 0].set_ylim(-0.01,np.max(y_vals_fix_T[2]*1.4))
ax1[1, 0].legend(loc='best')
ax1[1, 0].set_xlim(0,11)


ax1[1,1].hist(peak_lag_collected_arr_2[fix_T_idx_list[0]], bins=np.arange(0,period_list[fix_T_idx_list[0]], 0.5), label=f'$C_N = {concentration_list[fix_T_idx_list[0]]} \mu M$', color = fix_T_colors[3])
ax1[1,1].set_ylim(-0.01,np.max(y_vals_fix_T[0]*1.4))
ax1[1,1].legend(loc='best')
ax1[1,1].set_xlim(0,11)



fig1.suptitle(f'Time between registered peak and nearest Nutlin initiation, $T_N = 11 h$', fontsize = 24, y=0.98)
fig1.text(0.0, 0.5, f'Counts pr $0.5$ h', va='center', rotation='vertical', fontsize=22)
fig1.text(0.5, 0.0, 'Time [h]', ha='center', fontsize=22)


fig1.tight_layout()

if save_graphs:
    dir_name = r'GRAPHPATH'

    os.chdir(dir_name)
    fig1.savefig(f'peak_locations_fix_T.png',dpi=250, facecolor=('w'), bbox_inches='tight')




y_vals_fix_C = []
x_vals_fix_C = []
for i in fix_C_idx_list:
    bin_list = np.arange(0,period_list[i]+0.5, 0.5)
    y_vals, bin_edges = np.histogram(peak_lag_collected_arr_2[i], bins=bin_list)
    y_vals_fix_C.append(y_vals)

    bin_centers = (bin_edges[1:] + bin_edges[:-1])/(2)# *period_list[i]
    x_vals_fix_C.append(bin_centers)


fig2, ax2 = plt.subplots(2,2,figsize=(12,10), dpi=250)


ax2[0, 0].hist(peak_lag_collected_arr_2[fix_C_idx_list[2]], bins=np.arange(0,period_list[fix_C_idx_list[2]], 0.5), label=f'$T_N = {period_list[fix_C_idx_list[2]]} h$', color = fix_C_colors[0])
ax2[0, 0].set_ylim(-0.01,np.max(y_vals_fix_C[2]*1.3))
ax2[0, 0].legend(loc='best')
ax2[0, 0].set_xlim(0,period_list[fix_C_idx_list[2]])
# ax2[0].set_ylabel(f'Counts pr $0.5$ h', fontsize=22)


ax2[0, 1].hist(peak_lag_collected_arr_2[fix_C_idx_list[3]], bins=np.arange(0,period_list[fix_C_idx_list[3]], 0.5), label=f'$T_N = {period_list[fix_C_idx_list[3]]} h$', color = fix_C_colors[1])
ax2[0, 1].set_ylim(-0.01,np.max(y_vals_fix_C[3]*1.3))
ax2[0, 1].legend(loc='best')
ax2[0, 1].set_xlim(0,period_list[fix_C_idx_list[3]])
ax2[0, 1].set_xticks([0,3.5,7])



ax2[1,0].hist(peak_lag_collected_arr_2[fix_C_idx_list[0]], bins=np.arange(0,period_list[fix_C_idx_list[0]], 0.5), label=f'$T_N = {period_list[fix_C_idx_list[0]]} h$', color = fix_C_colors[2])
ax2[1,0].set_ylim(-0.01,np.max(y_vals_fix_C[0]*1.3))
ax2[1,0].legend(loc='best')
ax2[1,0].set_xticks([0,4,8])
ax2[1,0].set_xlim(0,period_list[fix_C_idx_list[0]])

ax2[1,1].hist(peak_lag_collected_arr_2[fix_C_idx_list[1]], bins=np.arange(0,period_list[fix_C_idx_list[1]], 0.5), label=f'$T_N = {period_list[fix_C_idx_list[1]]} h$', color = fix_C_colors[3])
ax2[1,1].set_ylim(-0.01,np.max(y_vals_fix_C[1]*1.3))
ax2[1,1].legend(loc='best')
ax2[1,1].set_xticks([0,4.5,9])
ax2[1,1].set_xlim(0,period_list[fix_C_idx_list[1]])

fig2.suptitle(f'Time between registered peak and nearest Nutlin initiation, $C_N = 0.5 \mu M$', fontsize = 24, y=0.98)
fig2.text(0.0, 0.5, f'Counts pr $0.5$ h', va='center', rotation='vertical', fontsize=22)
fig2.text(0.5, 0.0, 'Time [h]', ha='center', fontsize=22)

fig2.tight_layout()

if save_graphs:
    dir_name = r'GRAPHPATH'

    os.chdir(dir_name)
    fig2.savefig(f'peak_locations_fix_C.png',dpi=250, facecolor=('w'), bbox_inches='tight')


# %%

#%%
plt.rcParams.update({'font.size': 18})
concentration_list = np.array([2, 0.5, 1, 0.25, 1, 1, 1, 1])
period_list = np.array([11, 11, 11, 11, 8, 9, 4, 7])

fix_T_idx_list = [0, 1, 2, 3]
fix_C_idx_list = [4, 5, 6, 7]

fix_T_colors = ['lime', 'limegreen', 'green', 'black']
fix_C_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

y_vals_fix_T = []
x_vals_fix_T = []
for i in fix_T_idx_list:
    bin_list = np.arange(0,period_list[i]+0.5, 0.5)
    y_vals, bin_edges = np.histogram(peak_lag_collected_arr_2[i], bins=bin_list, density=True)
    y_vals_fix_T.append(y_vals)

    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    x_vals_fix_T.append(bin_centers)


fig1, ax1 = plt.subplots(1,4,figsize=(16,4), sharex=True)


ax1[0].plot(x_vals_fix_T[3],y_vals_fix_T[3], label=f'$T_N = {concentration_list[fix_T_idx_list[3]]}$', color = fix_T_colors[0])
ax1[0].set_ylim(-0.01,np.max(y_vals_fix_T[3]*1.3))
ax1[0].legend(loc='best')
ax1[0].set_xlim(0,11)

ax1[1].plot(x_vals_fix_T[1],y_vals_fix_T[1], label=f'$T_N = {concentration_list[fix_T_idx_list[1]]}$', color = fix_T_colors[1])
ax1[1].set_ylim(-0.01,np.max(y_vals_fix_T[1]*1.3))
ax1[1].legend(loc='best')

ax1[2].plot(x_vals_fix_T[2],y_vals_fix_T[2], label=f'$T_N = {concentration_list[fix_T_idx_list[2]]}$', color = fix_T_colors[2])
ax1[2].set_ylim(-0.01,np.max(y_vals_fix_T[2]*1.3))
ax1[2].legend(loc='best')

ax1[3].plot(x_vals_fix_T[0],y_vals_fix_T[0], label=f'$T_N = {concentration_list[fix_T_idx_list[0]]}$', color = fix_T_colors[3])
ax1[3].set_ylim(-0.01,np.max(y_vals_fix_T[0]*1.3))
ax1[3].legend(loc='best')







y_vals_fix_C = []
x_vals_fix_C = []
for i in fix_C_idx_list:
    bin_list = np.arange(0,period_list[i]+0.5, 0.5)
    y_vals, bin_edges = np.histogram(peak_lag_collected_arr_2[i], bins=bin_list, density=True)
    y_vals_fix_C.append(y_vals)

    bin_centers = (bin_edges[1:] + bin_edges[:-1])/(2)# *period_list[i]
    x_vals_fix_C.append(bin_centers)


fig4, ax4 = plt.subplots(1,4,figsize=(16,4))


ax4[0].plot(x_vals_fix_C[2],y_vals_fix_C[2], label=f'$T_N = {concentration_list[fix_C_idx_list[2]]}$', color = fix_C_colors[0], lw=3)
ax4[0].set_ylim(-0.01,np.max(y_vals_fix_C[2]*1.3))
ax4[0].legend(loc='best')
ax4[0].set_xlim(0,period_list[fix_C_idx_list[2]])

ax4[1].plot(x_vals_fix_C[3],y_vals_fix_C[3], label=f'$T_N = {concentration_list[fix_C_idx_list[3]]}$', color = fix_C_colors[1], lw=3)
ax4[1].set_ylim(-0.01,np.max(y_vals_fix_C[3]*1.3))
ax4[1].legend(loc='best')

ax4[2].plot(x_vals_fix_C[0],y_vals_fix_C[0], label=f'$T_N = {concentration_list[fix_C_idx_list[0]]}$', color = fix_C_colors[2], lw=3)
ax4[2].set_ylim(-0.01,np.max(y_vals_fix_C[0]*1.3))
ax4[2].legend(loc='best')

ax4[3].plot(x_vals_fix_C[1],y_vals_fix_C[1], label=f'$T_N = {concentration_list[fix_C_idx_list[1]]}$', color = fix_C_colors[3], lw=3)
ax4[3].set_ylim(-0.01,np.max(y_vals_fix_C[1]*1.3))
ax4[3].legend(loc='best')


fig4.tight_layout()




# %%






