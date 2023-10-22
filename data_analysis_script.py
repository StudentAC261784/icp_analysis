import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date

import data_analysis_functions as dafun

# Parametry służące włączaniu funkcji analitycznych skryptu
PCT_FILL_ANALYSIS = False # Generowanie .csv z danymi dotyczącymi % wypełnienia kolejnych okien 8h i 24h per parametr per pacjent
DRAW_PARAMETERS_HISTOGRAMS = False # Generowanie histogramów rozkładów parametrów
PARAMETER_VALUES_TO_EXCEL = False # Tworzenie .xlsx z wartościami parametrów per pacjent w oknach 8h i 24h
PLOT_GRAPHS = False # Generowanie wykresów parametrów z podziałem na oknach 8h i 24h
SPLIT_GRAPHS_BASED_ON_GOSE_SCORE = False # Ma znaczenie jedynie jeśli PLOT_GRAPHS == True - rozdział wykresów w zależności od wyniku leczenia
DRAW_BOX_PLOTS = True # Generowanie wykresów pudełkowych

signals_names = ['icp', 'abp', 'amp', 'hr', 'prx', 'psi']

if PCT_FILL_ANALYSIS:
    pct_fill_df = pd.DataFrame(columns=['id', '8h', '24h'])

if PARAMETER_VALUES_TO_EXCEL:
    icp_8h = []
    abp_8h = []
    amp_8h = []
    hr_8h = []
    prx_8h = []
    psi_8h = []

    icp_24h = []
    abp_24h = []
    amp_24h = []
    hr_24h = []
    prx_24h = []
    psi_24h = []

if PLOT_GRAPHS:
    today_date = str(date.today())
    base_dir = os.path.join(os.getcwd(), 'outputs', 'plots', today_date)

    if SPLIT_GRAPHS_BASED_ON_GOSE_SCORE:
        # Tworzenie odpowiednich folderów na wykresy w zależności od okna, wyniku leczenia
        output_directory_good_8h = dafun.create_and_return_directory(base_dir, 'good', '8h')
        output_directory_bad_8h = dafun.create_and_return_directory(base_dir, 'bad', '8h')
        output_directory_none_8h = dafun.create_and_return_directory(base_dir, 'none', '8h')

        output_directory_good_24h = dafun.create_and_return_directory(base_dir, 'good', '24h')
        output_directory_bad_24h = dafun.create_and_return_directory(base_dir, 'bad', '24h')
        output_directory_none_24h = dafun.create_and_return_directory(base_dir, 'none', '24h')
    else:
        output_directory_8h = dafun.create_and_return_directory(base_dir, '8h')
        output_directory_24h = dafun.create_and_return_directory(base_dir, '24h')

if DRAW_BOX_PLOTS:
    box_data_good_8h, box_data_good_24h, box_data_bad_8h, box_data_bad_24h = [
        [[] for _ in range(6)] for _ in range(4)
    ]

exclude_df = pd.read_csv('exclusions_final_v2.csv')
troublemakers = []

with tqdm(total=len(os.listdir(rf'{os.getcwd()}\data\MeanICP')), desc="Processing", unit="file") as pbar:
    if SPLIT_GRAPHS_BASED_ON_GOSE_SCORE or DRAW_BOX_PLOTS:
        gose_outcome_df = pd.read_csv('outcomes_processed.csv')
        patient_id_gose_outcome_dict = dict(
            zip([x.lower() for x in gose_outcome_df['Patient'].values.tolist()], gose_outcome_df['Gose3m'].fillna('').values.tolist())
            )

    x = 0
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
                    t_param, param = dafun.get_full_data(data)
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

                # Wyszukiwanie w każdym wektorze czasu elementów, które są najbardziej zbliżone do kolejnych elementów najkrótszego wektora
                synced_times, synced_params = dafun.find_nearest_values(times, min_times, params)
                # Dopisywanie do list również wyjęte wcześniej listy
                synced_times.insert(min_times_index, min_times)
                synced_params.insert(min_times_index, min_times_param)

                times.insert(min_times_index, min_times)
                params.insert(min_times_index, min_times_param)
                
                window_params_8h, window_indexes_8h, window_fill_pct_8h = dafun.sample_window_return_fill_value(
                    synced_times, synced_params, 8, 50)
                window_params_24h, window_indexes_24h, window_fill_pct_24h = dafun.sample_window_return_fill_value(
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

                    # Poszerzanie/Generowanie plików .xlsx dla okien 8h i 24h
                    try:
                        with pd.ExcelWriter('parameters_values_8h.xlsx', mode='a') as writer:
                            patient_df_8h.to_excel(writer, sheet_name=f'{id}')
                    except:
                        with pd.ExcelWriter('parameters_values_8h.xlsx', mode='w') as writer:
                            patient_df_8h.to_excel(writer, sheet_name=f'{id}')

                    try:
                        with pd.ExcelWriter('parameters_values_24h.xlsx', mode='a') as writer:
                            patient_df_24h.to_excel(writer, sheet_name=f'{id}')
                    except:
                        with pd.ExcelWriter('parameters_values_24h.xlsx', mode='w') as writer:
                            patient_df_24h.to_excel(writer, sheet_name=f'{id}')

                if PLOT_GRAPHS:
                    # Dodawanie wartości null do danych, żeby na wykresie nie łączyć punktów między którymi wartość nie została policzona
                    window_indexes_with_gaps_8h, window_params_with_gaps_8h = dafun.introduce_data_gaps(window_indexes_8h, window_params_8h)
                    window_indexes_with_gaps_24h, window_params_with_gaps_24h = dafun.introduce_data_gaps(window_indexes_24h, window_params_24h)

                    if SPLIT_GRAPHS_BASED_ON_GOSE_SCORE:
                        patient_gose_score = patient_id_gose_outcome_dict[id.lower()]
                        
                        if patient_gose_score == '':
                            output_dir_8h = output_directory_none_8h
                            output_dir_24h = output_directory_none_24h
                        elif patient_gose_score < 5:
                            output_dir_8h = output_directory_bad_8h
                            output_dir_24h = output_directory_bad_24h
                        else:
                            output_dir_8h = output_directory_good_8h
                            output_dir_24h = output_directory_good_24h

                        windowed_plot_8h = dafun.plot_params(window_indexes_with_gaps_8h, window_params_with_gaps_8h)[1]
                        plt.savefig(os.path.join(output_dir_8h, f'{id}_8H_50%.png'))
                        plt.close()

                        windowed_plot_24h = dafun.plot_params(window_indexes_with_gaps_24h, window_params_with_gaps_24h)[1]
                        plt.savefig(os.path.join(output_dir_24h, f'{id}_24H_50%.png'))
                        plt.close()

                    else:
                        windowed_plot_8h = dafun.plot_params(window_indexes_with_gaps_8h, window_params_with_gaps_8h)[1]
                        plt.savefig(os.path.join(output_directory_8h, f'{id}_8H_50%.png'))
                        plt.close()
                        
                        windowed_plot_24h = dafun.plot_params(window_indexes_with_gaps_24h, window_params_with_gaps_24h)[1]
                        plt.savefig(os.path.join(output_directory_24h, f'{id}_24H_50%.png'))
                        plt.close()

                if DRAW_BOX_PLOTS:
                    patient_gose_score = patient_id_gose_outcome_dict[id.lower()]

                    # Okno 8h
                    for i in range(len(window_params_8h)):
                        if patient_gose_score != '':
                            # Dzielenie pacjentów na grupy w zależności od wyniku leczenia
                            target_list = box_data_bad_8h if patient_gose_score < 5 else box_data_good_8h
                            for j in range(len(window_params_8h[i])):
                                try:
                                    target_list[i][j].append(window_params_8h[i][j])
                                except:
                                    target_list[i].append([window_params_8h[i][j]])

                    # Okno 24h
                    for i in range(len(window_params_24h)):
                        if patient_gose_score != '':
                            # Dzielenie pacjentów na grupy w zależności od wyniku leczenia
                            target_list = box_data_bad_24h if patient_gose_score < 5 else box_data_good_24h
                            for j in range(len(window_params_24h[i])):
                                try:
                                    target_list[i][j].append(window_params_24h[i][j])
                                except:
                                    target_list[i].append([window_params_24h[i][j]])

                            
                if PCT_FILL_ANALYSIS:
                    pct_fill_df.loc[len(pct_fill_df)] = [id, window_fill_pct_8h, window_fill_pct_24h]
                
                if DRAW_PARAMETERS_HISTOGRAMS:
                    icp_8h, abp_8h, amp_8h, hr_8h, prx_8h, psi_8h = dafun.append_histogram_lists(
                        icp_8h, abp_8h, amp_8h, hr_8h, prx_8h, psi_8h, window_params_8h
                        )
                    icp_24h, abp_24h, amp_24h, hr_24h, prx_24h, psi_24h = dafun.append_histogram_lists(
                        icp_24h, abp_24h, amp_24h, hr_24h, prx_24h, psi_24h, window_params_24h
                        )

            except Exception as e:
                troublemakers.append((id, e))
                if PCT_FILL_ANALYSIS:
                    pct_fill_df.loc[len(pct_fill_df)] = [id, (e, params), times]

        pbar.update(1)
        # x += 1
        # if x == 51:
        #     break

if len(troublemakers) > 0:
    with open('troublemakers.txt', 'w') as file:
        for item in troublemakers:
            file.write(f'{str(item)}\n')

if PCT_FILL_ANALYSIS:
    pct_fill_df.to_csv('pct_fill_data.csv', index=False)

if DRAW_PARAMETERS_HISTOGRAMS:
    histogram_lists_8h = [icp_8h, abp_8h, amp_8h, hr_8h, prx_8h, psi_8h]
    histogram_lists_24h = [icp_24h, abp_24h, amp_24h, hr_24h, prx_24h, psi_24h]

    for signal_name, data_8h, data_24h in zip(signals_names, histogram_lists_8h, histogram_lists_24h):
        plt.hist(data_8h)
        plt.savefig(f'{signal_name}_histogram_8h.png')
        plt.close()
        plt.hist(data_24h)
        plt.savefig(f'{signal_name}_histogram_24h.png')
        plt.close()

    # Sprowadzanie list do jednej długości
    histogram_list_max_len_8h = max(map(len, histogram_lists_8h))
    for item in histogram_lists_8h:
        item += [np.NaN] * (histogram_list_max_len_8h - len(item))

    histogram_list_max_len_24h = max(map(len, histogram_lists_24h))
    for item in histogram_lists_24h:
        item += [np.NaN] * (histogram_list_max_len_24h - len(item))

    # Tworzenie DataFrames, eksport do .csv
    data_dict_8h = {signal_name: data_8h for signal_name, data_8h in zip(signals_names, histogram_lists_8h)}
    histogram_data_df_8h = pd.DataFrame(data_dict_8h)
    histogram_data_df_8h.to_csv('histogram_data_8h.csv', index=False)

    data_dict_24h = {signal_name: data_24h for signal_name, data_24h in zip(signals_names, histogram_lists_24h)}
    histogram_data_df_24h = pd.DataFrame(data_dict_24h)
    histogram_data_df_24h.to_csv('histogram_data_24h.csv', index=False)

if DRAW_BOX_PLOTS:
    box_file_names = ['_boxplot_bad_8h.png', '_boxplot_good_8h.png', '_boxplot_bad_24h.png', '_boxplot_good_24h.png']

    for i in range(len(signals_names)):
        for index, box_data in enumerate([box_data_bad_8h, box_data_good_8h, box_data_bad_24h, box_data_good_24h]):
            max_length = max(len(sublist) for sublist in box_data[i])
            # Sprowadzanie list do tej samej długości
            same_length_data = [sublist + [None] * (max_length - len(sublist)) for sublist in box_data[i]]
            # Tworzenie DataFrame
            df = pd.DataFrame(same_length_data)
            df = df.T
            boxplot = df.boxplot()
            # plt.savefig(f'outputs/boxplots/{signals[i]}_boxplot_bad_8h.png', bbox_inches='tight', dpi=100)
            plt.savefig(rf'outputs/boxplots/{signals_names[i]}{box_file_names[index]}')
            plt.close()

print('done')