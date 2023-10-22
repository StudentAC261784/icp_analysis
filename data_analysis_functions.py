import os

import matplotlib.pyplot as plt
import numpy as np

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

def create_and_return_directory(base_dir, *subdirs):
    directory = os.path.join(base_dir, *subdirs)
    os.makedirs(directory, exist_ok=True)
    return directory