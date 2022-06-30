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



data_idx = 7# Choose which file to analyse
run_idx = 3

save_graphs=False

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

x = np.arange(0,data_len)/6

data_corrected_arr = np.zeros_like(data_run)

for i, data in enumerate(data_run):
    data_corrected_arr[i,:] = poly_correct(data, ndof=3)

data_run_mean = np.mean(data_corrected_arr, axis=0)
data_run_std = np.std(data_corrected_arr, axis=0)

data_run = data_corrected_arr[run_number]

x_nutlin, y_nutlin, concentration, t_start_idx, t_end_idx, nutlin_period, entrain_est = nutlin_data(data_files[data_idx], data_run)

# %% Plotting
colors = ['red', 'blue', 'green']
plt.rcParams.update({'font.size': 8})



# %% Oeak finding



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



# f = interpolate.interp1d(x, y)

# %% Plot index
plot_idx = 0

# %% Unsmoothened
plot_individual = False
t_max = data_len/6
if plot_individual:
    colors = ['red', 'blue', 'green']
    plt.rcParams.update({'font.size': 8})

    label_size = 8


    fig3, ax3 = plt.subplots(3,1,sharex = True, gridspec_kw={'height_ratios': [6, 4, 2]}, dpi = 250)


    ax3[0].plot(x[0:t_start_idx+1], data_corrected_arr[plot_idx][0:t_start_idx+1], color=colors[0])

    ax3[0].plot(x[t_start_idx:t_end_idx+1], data_corrected_arr[plot_idx][t_start_idx:t_end_idx+1], color=colors[1])

    ax3[0].plot(x[t_end_idx:], data_corrected_arr[plot_idx][t_end_idx:], color=colors[2])

    ax3[0].plot(x[peak_idx_arr[plot_idx]],data_corrected_arr[plot_idx][peak_idx_arr[plot_idx]], marker= 'X', ls='None', color='orange', mec = 'black', mew=0.5, ms=7)


    # ax3[0].plot(x[peak_idx],data_run_mean[peak_idx], marker= 'X', ls='None', color='orange', mec = 'black', mew=0.5, ms=7)



    ax3[0].axvline(x[t_start_idx], color = 'k', ls='dotted', alpha = 0.99, label='Nutlin on/off', )
    ax3[0].axvline(x[t_end_idx], color = 'k', ls='dotted', alpha = 0.99)



    ax3[0].set_ylabel('p53 level [au]', fontsize=label_size)
    ax3[0].set_xlim(0,t_max)
    ax3[0].set_ylim(1.1*np.min(data_corrected_arr[plot_idx]), 1.2*np.max(data_corrected_arr[plot_idx]))
    ax3[0].set_title(f'Identified peaks and periods for single run ({plot_idx+1}), datafile {data_files[data_idx][3:-21]}\np53 adjusted to 3rd deg poly', fontsize=label_size+2)

    ax3[1].step(peak_idx_arr[plot_idx]/6,periods_arr[plot_idx], where='post', c='maroon', label='T_est/T_nutlin')
    ax3[1].legend(loc='upper right')
    ax3[1].set_ylabel('Period ratio', fontsize=label_size)

    ax3[1].axvline(x[t_start_idx], color = 'k', ls='dotted', alpha = 0.99, label='Nutlin on/off')
    ax3[1].axvline(x[t_end_idx], color = 'k', ls='dotted', alpha = 0.99)


    ax3[2].plot(x_nutlin, y_nutlin, 'seagreen', label=f'Nutlin period {nutlin_period} h', ds='steps')
    ax3[2].set_ylabel(r'Nutlin [$\mu$M]', fontsize=label_size)
    ax3[2].set_ylim(-0.1*concentration, 1.15*concentration)
    ax3[2].set_xlabel('Time [h]', fontsize=label_size)
    ax3[2].legend(loc='upper right')

    # if save_individual:
    # fig, ax = plt.subplots(figsize=(8,2),dpi=150) 
    # ax.step(peak_idx_arr[plot_idx][0:-1],periods_arr[plot_idx], where='post')



# %% Smoothened
save_graphs= False
plot_all = False
if plot_all:
    for plot_idx in tqdm(range(data_num)):
        colors = ['red', 'blue', 'green']
        plt.rcParams.update({'font.size': 12})

        label_size = 12

        t_max = data_len/6

        fig2, ax2 = plt.subplots(3,1, figsize=(8,5), sharex = True, gridspec_kw={'height_ratios': [6, 4, 2]}, dpi = 250)


        ax2[0].plot(x[0:t_start_idx+1], data_corrected_arr[plot_idx][0:t_start_idx+1], color=colors[0], alpha = 0.5)

        ax2[0].plot(x[t_start_idx:t_end_idx+1], data_corrected_arr[plot_idx][t_start_idx:t_end_idx+1], color=colors[1], alpha = 0.5)

        ax2[0].plot(x[t_end_idx:], data_corrected_arr[plot_idx][t_end_idx:], color=colors[2], alpha = 0.5)

        # ax2[0].plot(x[peak_idx_arr[plot_idx]],data_corrected_arr[plot_idx][peak_idx_arr[plot_idx]], marker= 'X', ls='None', color='orange', mec = 'black', mew=0.5, ms=7)

        ax2[0].plot(x_av_arr[plot_idx], y_av_arr[plot_idx], c='black', alpha=1, lw=0.8, label=f'p53 averaged over {n_average} points')

        ax2[0].plot(x_av_arr[plot_idx][peak_av_arr[plot_idx]],y_av_arr[plot_idx][peak_av_arr[plot_idx]], marker= 'X', ls='None', color='orange', mec = 'black', mew=0.5, ms=7)
        ax2[0].legend(loc='upper right')


        # ax2[0].plot(x[peak_idx],data_run_mean[peak_idx], marker= 'X', ls='None', color='orange', mec = 'black', mew=0.5, ms=7)



        ax2[0].axvline(x[t_start_idx], color = 'k', ls='dotted', alpha = 0.99, label='Nutlin on/off')
        ax2[0].axvline(x[t_end_idx], color = 'k', ls='dotted', alpha = 0.99)



        ax2[0].set_ylabel('p53 level [au]', fontsize=label_size)
        ax2[0].set_xlim(0,t_max)
        ax2[0].set_ylim(1.1*np.min(data_corrected_arr[plot_idx]), 1.3*np.max(data_corrected_arr[plot_idx]))
        ax2[0].set_title(f'Identified peaks and periods for $C={concentration} \mu M$ and $T={nutlin_period} h$ (run {plot_idx+1})', fontsize=label_size+2)

        ax2[1].step(peak_av_arr[plot_idx]*n_average/6,periods_av_arr[plot_idx], where='post', c='maroon', label=f'$T_{{est}}/T_{{nutlin}}$')
        ax2[1].legend(loc='upper right')
        ax2[1].set_ylabel('Period ratio', fontsize=label_size)

        ax2[1].axvline(x[t_start_idx], color = 'k', ls='dotted', alpha = 0.99, label='Nutlin on/off')
        ax2[1].axvline(x[t_end_idx], color = 'k', ls='dotted', alpha = 0.99)


        ax2[2].plot(x_nutlin, y_nutlin, 'seagreen', label=f'Nutlin period {nutlin_period} h', ds='steps')
        ax2[2].set_ylabel(r'Nutlin [$\mu$M]', fontsize=label_size)
        ax2[2].set_ylim(-0.1*concentration, 1.15*concentration)
        ax2[2].set_xlabel('Time [h]', fontsize=label_size)
        ax2[2].legend(loc='upper right')
        
        fig2.tight_layout()

        if save_graphs:
            dir_name = r'GRAPHPATH\\'+f'{data_files[data_idx][3:-21]}_period_determ_individ'

            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

            os.chdir(dir_name)


            fig2.savefig(f'Period_determ_{data_files[data_idx][3:-21]}_run_'+f'{plot_idx+1}'.zfill(3), dpi=250, facecolor='w')

        plt.close()


## fig2.tight_layout()
## graph_dir = r'C:\Users\jeppe\Documents\Fysik\Thesis\data_collab_work\Entrainment_data_2021\experiments_csv\Graphs'
## os.chdir(graph_dir)
## fig2.savefig('stage_corrected_new_p53_avg_'+data_files[data_idx][3:-21]+'.png', dpi=150, facecolor='w')


# %% 

ensemble_plot = False

if ensemble_plot:
    x_av_interpol_arr = []
    y_av_interpol_arr = []
    for plot_idx in range(data_num):

        f = interpolate.interp1d(peak_av_arr[plot_idx]*n_average/6, periods_av_arr[plot_idx], kind='previous')

        dt = 0.1
        x_min = np.min(peak_av_arr[plot_idx]*n_average/6)
        x_max = np.max(peak_av_arr[plot_idx]*n_average/6)

        delta_x_min = x_min/dt
        delta_x_max = (data_len/6 - x_max)/dt


        xnew = np.arange(x_min, x_max, dt)

        ynew = f(xnew)

        ynew_padded = np.pad(ynew, (int(delta_x_min),int(delta_x_max)), 'constant', constant_values=(np.nan, np.nan))

        x_av_interpol_arr.append(xnew)
        y_av_interpol_arr.append(ynew_padded)




# %%
if ensemble_plot:


    y_period_avg_focus = []
    for y_period in y_av_interpol_arr:
        t_start_new = int((t_start_idx/6*10)+nutlin_period*10/2)
        t_end_new = int((t_end_idx/6*10)- nutlin_period*10/2)
        # y_period_avg_focus.append(np.mean(y_period[t_start_idx:t_end_idx])) # Bare mean
        y_period_no_nan = np.nan_to_num(y_period, nan=entrain_est) # nan-values are replaced with estimated entrain so they do not add to distance
        y_period_avg_focus.append(np.mean(np.abs(np.array(y_period_no_nan[t_start_new:t_end_new])-entrain_est))) # Absolut afstand
        
    y_period_sorting = np.argsort(y_period_avg_focus)


if ensemble_plot:

    plt.rcParams.update({'font.size': 12})

    labelsize=18
    # Making colormap
    cmap = mpl.cm.get_cmap('rainbow')
    bounds=np.arange(0.3,2.5,0.2)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig4, ax4 = plt.subplots(figsize=(6,9),dpi=250)
    # np.array(y_av_interpol_arr)[y_period_sorting]
    im4 = ax4.imshow(np.array(y_av_interpol_arr)[y_period_sorting], aspect='auto', extent=[0,t_max,data_num,0],cmap=cmap, norm=norm, interpolation='none') # vmax=2.3,vmin=0.3
    cbar = fig4.colorbar(im4, ax=ax4, cmap=cmap, norm=norm, ticks=bounds)
    cbar.set_label(f'$T_{{est}}/T_{{N}}$', fontsize=labelsize)
    ax4.set_title(f'$T_{{est}}/T_{{N}}$ for $T_{{N}}={nutlin_period} h$, $C_N = {concentration} \mu M$', fontsize=labelsize)

    ax4.axvline(x[t_start_idx], color = 'k', ls='dashed', alpha = 0.99, label='Nutlin on/off', lw = 3)
    ax4.axvline(x[t_end_idx], color = 'k', ls='dashed', alpha = 0.99, lw = 3)

    ax4.set_ylabel(f'Experiment number', fontsize=labelsize)
    ax4.set_xlabel(f'Time [h]', fontsize=labelsize)

    save_graphs = True

    if save_graphs:
        dir_name = r'GRAPHPATH'

        os.chdir(dir_name)
        A_string = str(int(concentration*100)).zfill(3)
        fig4.savefig(f'period_est_by_period_nut_T_{int(nutlin_period)}_C_' + A_string +'.png',dpi=250, facecolor=('w'))


# %%

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


#%%

def flatten(t):
    return np.array([item for sublist in t for item in sublist])

plt.rcParams.update({'font.size': 14})

bin_list = np.arange(0,nutlin_period, 0.5)


peak_lag_1 = flatten(peak_lag_arr_p1)/6
peak_lag_2 = flatten(peak_lag_arr_p2)/6
peak_lag_3 = flatten(peak_lag_arr_p3)/6

colors = ['red', 'blue', 'green']
fig, ax = plt.subplots(1,3,figsize=(12,4), dpi=250, sharex=True)
ax[0].hist(peak_lag_1, bins=bin_list,color=colors[0], density=False, label='Before')
ax[0].set_xlim(0, nutlin_period)
ax[0].set_ylabel('Number of peaks pr 0.5 h', fontsize=14)
ax[0].legend(loc='best')

ax[1].hist(peak_lag_2, bins=bin_list,color=colors[1], density=False, label='During')
ax[1].set_xlabel('Time [h]', fontsize=14)
ax[1].legend(loc='best')


ax[2].hist(peak_lag_3, bins=bin_list,color=colors[2], density=False, label='After')
ax[2].legend(loc='best')


fig.suptitle(f'Time between registered peak and nearest nutlin initiation,\n$C={concentration} \mu M$ and $T={nutlin_period} h$', fontsize=16)

fig.tight_layout()

save_graphs = False

if save_graphs:
    dir_name = r'GRAPHPATH'

    os.chdir(dir_name)

    fig.savefig(f'p53_peak_time_after_nutlin_initiation_{data_files[data_idx][3:-21]}.png',dpi=250, facecolor=('w'))



# %%








































# def get_ave_values(xvalues, yvalues, n = 5):
#     signal_length = len(xvalues)
#     if signal_length % n == 0:
#         padding_length = 0
#     else:
#         padding_length = n - signal_length//n % n
#     xarr = np.array(xvalues)
#     yarr = np.array(yvalues)
#     xarr.resize(signal_length//n, n)
#     yarr.resize(signal_length//n, n)
#     xarr_reshaped = xarr.reshape((-1,n))
#     yarr_reshaped = yarr.reshape((-1,n))
#     x_ave = xarr_reshaped[:,0]
#     y_ave = np.nanmean(yarr_reshaped, axis=1)
#     return x_ave, y_ave

# n_average = 4

# x_av, y_av = get_ave_values(x, data_run_mean, n=n_average)


# peak_height = None
# peak_threshold = None
# peak_distance = 3 # 15 er 2.5 h
# peak_prominence = 100
# peak_width = 1 #


# peak_idx_av, peak_props_av = find_peaks(y_av, height=peak_height, threshold=peak_threshold, distance=peak_distance, prominence=peak_prominence, width=peak_width)



# fig1, ax1 = plt.subplots(figsize=(10,6), dpi=100)
# ax1.plot(x, data_run_mean, color='red', label='Mean, all runs', alpha=0.7)
# ax1.plot(x_av, y_av, color='blue', label=f'Mean data averaged over {n_average} points ({n_average/6:.2f} h)')

# ax1.plot(x_av[peak_idx_av],y_av[peak_idx_av], marker= 'X', ls='None', color='orange', mec = 'black', mew=0.5, ms=7)
# ax1.legend(loc='best')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('p53 [a.u]')
# ax1.set_xlim(0,np.max(x))
# ax1.set_title('Mean data plain and time-averaged')