import pickle
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date

# ICP:
# 1/ any_error_flag == True, jeżeli: kraniektomia albo procent artefaktów w slicie > 80% albo średnie ICP w slicie < 0 albo procent wartości nieprawidłowych (-99999) w slicie > 10%
# 2/ cały signal nan, jeżeli: slice zawiera tylko artefakty (klasa 5) albo jest artefaktem w formie "szpili" (bardzo wysokie punktowe ICP)
#
# ABP:
# 1/ any_error_flag == True, jeżeli: kraniektomia albo procent artefaktów w slicie > 80% albo średnie ICP w slicie < 0 albo procent wartości nieprawidłowych (-99999) w slicie > 10%
#
# PRx, HR, PSI:
# 1/ any_error_flag == True, jeżeli kraniektomia albo procent artefaktów w slicie > 80% albo średnie ICP w slicie < 0 albo procent wartości nieprawidłowych (-99999) w slicie > 10%
#
# PRx:
# 1/ signal == None, jeżeli slice jest za krótki, żeby policzyć PRx
# 2/ cały signal nan, jeżeli w żadnym oknie do PRx w obrębie slice'a nie da się policzyć PRx (tzn. długość slice'a jest wystarczająca, ale każde okno zawiera nany)
#
# PSI:
# 1/ time_start nan, fs == 0, signal pusty (pusta tablica), jeżeli: slice zawiera tylko artefakty (klasa 5) albo slice jest za krótki, żeby policzyć PSI
#
# Dodatkowo: jeśli kiedykolwiek fs == 0: pomija się slice


def get_full_data(slices):
    # max_time = 7 * 24 * 60 * 60
    valid_t, valid_signal = [], []
    for ids, slice in enumerate(slices):
        if not slice['any_error_flag']:
            t0 = slice['time_start']
            fs = slice['fs']
            signal = slice['signal']
            # Dopisano konwersję na listę w przypadku, gdy slice był obiektem np.Float64 (jednoelementowy slice nie był listą)
            if isinstance(signal, np.float64):
                signal = [signal]
            # Dopisano warunek sprawdzający, czy przypadkiem nie zachodzi dzielenie przez 0 (tak było w próbce A)
            if signal is not None and len(signal) > 0 and not np.all(np.isnan(signal)) and fs != 0:
                n = len(signal)
                tline = np.linspace(t0, t0 + n * 1 / fs, n)
                valid_t.extend(tline)
                valid_signal.extend(signal)

    valid_t = np.asarray(valid_t)
    valid_signal = np.asarray(valid_signal)
    return valid_t, valid_signal


def binary_search_nearest(sorted_list, target):
    left, right = 0, len(sorted_list) - 1

    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] == target:
            return (sorted_list[mid], mid)
        elif sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    if right < 0:
        return (sorted_list[0], 0)
    elif left >= len(sorted_list):
        return (sorted_list[-1], -1)
    else:
        if abs(target - sorted_list[left]) < abs(target - sorted_list[right]):
            return (sorted_list[left], left)
        else:
            return (sorted_list[right], right)


def find_nearest_values(times, min_times, params):
    min_times_array = np.array(min_times)

    nearest_time_values = []
    nearest_time_values_params = []
    for i in range(len(times)):
        nearest_time_values.append([])
        nearest_time_values_params.append([])

    for time in min_times_array:
        for idx, sublist in enumerate(times):
            nearest_value, nearest_value_index = binary_search_nearest(
                sublist, time)
            nearest_time_values[idx].append(nearest_value)
            nearest_time_values_params[idx].append(
                params[idx][nearest_value_index])

    return nearest_time_values, nearest_time_values_params


def introduce_data_gaps(times, params):
    for i in range(len(times)):
        for j in range(1, len(times[i])):
            if times[i][j] - times[i][j-1] > 60:
                try:
                    times[i].insert(j, times[i][j-1] + 30)
                except:
                    times[i] = np.insert(times[i], j, times[i][j-1] + 30)
                try:
                    params[i].insert(j, np.nan)
                except:
                    params[i] = np.insert(params[i], j, np.nan)

    return times, params


def plot_params(times, params, x_axis_seconds_to_hours=False):
    fig, axes = plt.subplots(6, 1, sharex=True)
    titles = ['ICP', 'ABP', 'AMP', 'HR', 'PRx', 'PSI']
    plt.subplots_adjust(hspace=0.5)  # Odstępy na tytuły między wykresami

    for i in range(len(times)):
        if x_axis_seconds_to_hours:
            times[i] = [x / 60 / 60 for x in times[i]]
        axes[i].plot(times[i], params[i])
        axes[i].set_title(titles[i])

    return (fig, axes)


def sample_window(times, params, size_hours, threshold_pct):
    size_seconds = int(size_hours * 60 * 60)

    # Przygotowanie list o odpowiedniej długości
    param_avgs = []
    param_avgs_window_indexes = []
    stop_flags = []
    for i in range(len(times)):
        param_avgs.append([])
        param_avgs_window_indexes.append([])
        stop_flags.append([])

    # Główna pętla
    n = 1
    index_limit = 0
    while True:
        current_time_limit = n * size_seconds  # Prawa granica okna
        previous_time_limit = (n-1) * size_seconds  # Lewa granica okna

        for i in range(len(times)):
            j = 1
            while times[i][j] - times[i][j-1] == 0 or times[i][j] == np.NaN or times[i][j-1] == np.NaN:
                j += 1

            max_count_of_samples_in_window = int(
                size_seconds / (times[i][j] - times[i][j-1]))
            index_limit = binary_search_nearest(
                times[i], current_time_limit)[1]
            previous_index_limit = binary_search_nearest(
                times[i], previous_time_limit)[1]

            # Sprawdzanie, czy nie została przekroczona długość listy
            if previous_index_limit != -1:
                # Wycinek listy parametrów ograniczony odpowiednimi granicami
                params_index_limit_cutout = params[i][previous_index_limit: index_limit + 1]
                params_index_limit_cutout = np.array(params_index_limit_cutout)[
                    ~np.isnan(params_index_limit_cutout)].tolist()
                # Sprawdzanie przekroczenia warunku minimalnego wypełnienia okna
                if len(params_index_limit_cutout) >= round(max_count_of_samples_in_window * (threshold_pct / 100)):
                    avg = np.average(params_index_limit_cutout)
                    param_avgs[i].append(avg)
                else:
                    param_avgs[i].append(np.nan)

                param_avgs_window_indexes[i].append(n)
            else:
                stop_flags[i] = [1]

        if stop_flags.count([1]) == len(times):
            break
        n += 1

    return (param_avgs, param_avgs_window_indexes)


def sample_window_return_fill_value(times, params, size_hours, threshold_pct):
    size_seconds = int(size_hours * 60 * 60)

    # Przygotowanie list o odpowiedniej długości
    param_avgs = []
    param_avgs_window_indexes = []
    fill_values = [[], [], [], [], [], []]
    stop_flags = []
    for i in range(len(times)):
        param_avgs.append([])
        param_avgs_window_indexes.append([])
        stop_flags.append([])

    # Główna pętla
    n = 1
    index_limit = 0
    while True:
        current_time_limit = n * size_seconds  # Prawa granica okna
        previous_time_limit = (n-1) * size_seconds  # Lewa granica okna

        for i in range(len(times)):
            j = 1
            while times[i][j] - times[i][j-1] == 0 or times[i][j] == np.NaN or times[i][j-1] == np.NaN:
                j += 1

            max_count_of_samples_in_window = int(
                size_seconds / (times[i][j] - times[i][j-1]))
            index_limit = binary_search_nearest(
                times[i], current_time_limit)[1]
            previous_index_limit = binary_search_nearest(
                times[i], previous_time_limit)[1]

            # Sprawdzanie, czy nie została przekroczona długość listy
            if previous_index_limit != -1:
                # Wycinek listy parametrów ograniczony odpowiednimi granicami
                params_index_limit_cutout = params[i][previous_index_limit: index_limit + 1]
                params_index_limit_cutout = np.array(params_index_limit_cutout)[
                    ~np.isnan(params_index_limit_cutout)].tolist()

                fill_value = len(params_index_limit_cutout) / \
                    max_count_of_samples_in_window

                # Sprawdzanie przekroczenia warunku minimalnego wypełnienia okna
                if len(params_index_limit_cutout) >= round(max_count_of_samples_in_window * (threshold_pct / 100)):
                    avg = np.average(params_index_limit_cutout)
                    param_avgs[i].append(avg)
                else:
                    param_avgs[i].append(np.nan)

                param_avgs_window_indexes[i].append(n)
                fill_values[i].append(fill_value)
            else:
                stop_flags[i] = [1]

        if stop_flags.count([1]) == len(times):
            break
        n += 1

    return (param_avgs, param_avgs_window_indexes, fill_values)


def append_histogram_lists(icp, abp, amp, hr, prx, psi, params):
    histogram_lists = [icp, abp, amp, hr, prx, psi]
    for id, list in enumerate(params):
        if len(list) > 7:
            list = list[:7]
        histogram_lists[id] = histogram_lists[id] + list
    return histogram_lists[0], histogram_lists[1], histogram_lists[2], histogram_lists[3], histogram_lists[4], histogram_lists[5]


# Parametry służące włączanie funkcji analitycznych skryptu
PCT_FILL_ANALYSIS = False
DRAW_PARAMETERS_HISTOGRAMS = False # okno 24h / 8h do zmiany niżej
PARAMETER_VALUES_TO_EXCEL = False
PLOT_GRAPHS = False

if PCT_FILL_ANALYSIS:
    pct_fill_df = pd.DataFrame(columns=['id', '8h', '24h'])

if PARAMETER_VALUES_TO_EXCEL:
    icp = []
    abp = []
    amp = []
    hr = []
    prx = []
    psi = []


exclude_df = pd.read_csv('exclusions_final_v2.csv')
troublemakers = []

with tqdm(total=len(os.listdir(rf'{os.getcwd()}\data\MeanICP')), desc="Processing", unit="file") as pbar:
    for filename in os.listdir(rf'{os.getcwd()}\data\MeanICP'):
        id = filename[:filename.find("_")]
        if id.lower() not in [exclude_id.lower() for exclude_id in exclude_df['Patient'].tolist()]:
            files = [f'MeanICP\{filename}', f'MeanABP\{id}_meanABP.pkl', f'AMPpp\{id}_AMPpp.pkl',
                     f'HR\{id}_HR.pkl', f'PRx\{id}_PRx.pkl', f'PSI\{id}_PSI.pkl']
            times = []
            params = []
            for file in files:
                p = Path(f'{os.getcwd()}\data\{file}')
                with open(p, 'rb') as f:
                    data = pickle.load(f)
                try:
                    t_param, param = get_full_data(data)
                    times.append(t_param)
                    params.append(param)
                except Exception as e:
                    print(e)

            try:
                # Wyznaczanie takiego rejestru czasu, który ma minimum wpisów (stanowi ogranicznik dla pozostałych)
                for i in range(len(times)):
                    if i == 0:
                        min_times_length = len(times[i])
                        min_times_index = i
                    elif min_times_length > len(times[i]):
                        min_times_length = len(times[i])
                        min_times_index = i

                min_times = times.pop(min_times_index)
                min_times_param = params.pop(min_times_index)

                synced_times, synced_params = find_nearest_values(times, min_times, params)
                synced_times.insert(min_times_index, min_times)
                synced_params.insert(min_times_index, min_times_param)
                times.insert(min_times_index, min_times)
                params.insert(min_times_index, min_times_param)

                window_params_8h, window_indexes_8h, window_fill_pct_8h = sample_window_return_fill_value(
                    synced_times, synced_params, 8, 50)
                window_params_24h, window_indexes_24h, window_fill_pct_24h = sample_window_return_fill_value(
                    synced_times, synced_params, 24, 50)


                if PARAMETER_VALUES_TO_EXCEL:
                    # order: [icp, abp, amp, hr, prx, psi]
                    patient_data_8h = {'ICP': window_params_8h[0],
                                    'ABP': window_params_8h[1],
                                    'AMP': window_params_8h[2],
                                    'HR': window_params_8h[3],
                                    'PRx': window_params_8h[4],
                                    'PSI': window_params_8h[5]}
                    
                    patient_data_24h = {'ICP': window_params_24h[0],
                                    'ABP': window_params_24h[1],
                                    'AMP': window_params_24h[2],
                                    'HR': window_params_24h[3],
                                    'PRx': window_params_24h[4],
                                    'PSI': window_params_24h[5]}

                    patient_df_8h = pd.DataFrame(patient_data_8h)
                    patient_df_24h = pd.DataFrame(patient_data_24h)

                    try:
                        with pd.ExcelWriter('parameters_values_8h.xlsx', mode='a') as writer:
                            patient_df_8h.to_excel(writer, sheet_name=f'{id}')
                    except:
                        with pd.ExcelWriter('parameters_values_8h.xlsx', mode='w') as writer:
                            patient_df_8h.to_excel(writer, sheet_name=f'{id}')

                    try:
                        with pd.ExcelWriter('parameters_values_24h.xlsx', mode='a') as writer:
                            patient_df_8h.to_excel(writer, sheet_name=f'{id}')
                    except:
                        with pd.ExcelWriter('parameters_values_24h.xlsx', mode='w') as writer:
                            patient_df_8h.to_excel(writer, sheet_name=f'{id}')

                if PLOT_GRAPHS:
                    today_date = str(date.today())
                    output_directory = os.path.join(os.getcwd(), 'outputs', 'plots', today_date)
                    os.makedirs(output_directory, exist_ok=True)

                    window_indexes_8h, window_params_8h = introduce_data_gaps(window_indexes_8h, window_params_8h)
                    window_indexes_24h, window_params_24h = introduce_data_gaps(window_indexes_24h, window_params_24h)

                    windowed_plot_8h = plot_params(window_indexes_8h, window_params_8h)[1]
                    plt.savefig(os.path.join(output_directory, f'{id}_8H_50%.png'))
                    plt.close()
                    
                    windowed_plot_24h = plot_params(window_indexes_24h, window_params_24h)[1]
                    plt.savefig(os.path.join(output_directory, f'{id}_24H_50%.png'))
                    plt.close()

                if PCT_FILL_ANALYSIS:
                    pct_fill_df.loc[len(pct_fill_df)] = [id, window_fill_pct_8h, window_fill_pct_24h]
                
                if DRAW_PARAMETERS_HISTOGRAMS:
                    icp, abp, amp, hr, prx, psi = append_histogram_lists(icp, abp, amp, hr, prx, psi, window_params_8h)
                    # icp, abp, amp, hr, prx, psi = append_histogram_lists(icp, abp, amp, hr, prx, psi, window_params_24h)

            except Exception as e:
                troublemakers.append((id, e))
                if PCT_FILL_ANALYSIS:
                    pct_fill_df.loc[len(pct_fill_df)] = [id, (e, params), times]

        pbar.update(1)

if len(troublemakers) > 0:
    with open('troublemakers.txt', 'w') as file:
        for item in troublemakers:
            file.write(f'{str(item)}\n')

if PCT_FILL_ANALYSIS:
    pct_fill_df.to_csv('pct_fill_data.csv', index=False)

if DRAW_PARAMETERS_HISTOGRAMS:
    histogram_lists = [icp, abp, amp, hr, prx, psi]
    for i in range(len(histogram_lists)):
        data = histogram_lists[i]
        plt.hist(data)
        plt.savefig(f'{i}_histogram.png')
        plt.close()

    histogram_list_max_len = max(len(icp), len(abp), len(amp), len(hr), len(prx), len(psi))
    for item in [icp, abp, amp, hr, prx, psi]:
        while len(item) < histogram_list_max_len:
            item.append(np.NaN)

    data_dict = {'ICP': icp,
                 'ABP': abp,
                 'AMP': amp,
                 'HR': hr,
                 'PRx': prx,
                 'PSI': psi}
    histogram_data_df = pd.DataFrame(data_dict)
    histogram_data_df.to_csv('histogram_data.csv', index=False)

    # - Wykresy w edytowalnej formie, albo w większym formacie
