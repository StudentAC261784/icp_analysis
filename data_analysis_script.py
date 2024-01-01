import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

import data_analysis_functions as dafun

def create_synced_signal_files(signals_info):
    troublemakers = []

    for synced_directory in signals_info['synced_directories']:
        dafun.create_and_return_directory(synced_directory)

    with tqdm(total=len(os.listdir(signals_info['directories'][0])), desc="Syncing signals", unit="patient") as pbar:
        for filename in os.listdir(signals_info['directories'][0]):
            patient_id = filename[:filename.find("_")]

            files = [
                f'{directory}\{patient_id}_{signals_info["files_suffixes"][directory_index]}.pkl'
                for directory_index, directory in enumerate(signals_info['directories'])
            ]
            times = []
            params = []

            for file in files:
                p = Path(file)
                with open(p, 'rb') as f:
                    data = pickle.load(f)
                try:
                    t_param, param = dafun.get_full_data(data)
                    times.append(t_param)
                    params.append(param)
                except Exception as e:
                    print(e)

            try:
                # Find the time array with the minimum entries
                min_times_index = np.argmin([len(t) for t in times])
                min_times = times.pop(min_times_index)
                min_times_param = params.pop(min_times_index)

                # Find nearest values for each time array
                synced_times, synced_params = dafun.find_nearest_values(times, min_times, params)
                # Insert the minimum time array back
                synced_times.insert(min_times_index, min_times)
                synced_params.insert(min_times_index, min_times_param)

                for i in range(len(synced_times)):
                    signal_dict = {'time': synced_times[i], 'signal': synced_params[i]}
                    output_file_path = f'{signals_info["synced_directories"][i]}\{patient_id}_synced_{signals_info["files_suffixes"][i]}.pkl'

                    with open(output_file_path, 'wb') as handle:
                        pickle.dump(signal_dict, handle)
            except Exception as e:
                print(e)
                troublemakers.append((patient_id, e))

            pbar.update(1)

    if troublemakers:
        print('WARNING: Some patients have caused errors. Check troublemakers_data_sync.txt for more details.')
        with open('troublemakers_data_sync.txt', 'w') as file:
            for item in troublemakers:
                file.write(f'{str(item)}\n')

def sample_windows(script_functionality, signals_info, windows_info, general_exclude_file=None):#box_plots_limits, additional_exclusions):
    troublemakers = []

    if general_exclude_file != None:
        general_exclude_df = pd.read_csv(general_exclude_file)
        general_exclude_set = set(general_exclude_df['Patient'].str.lower())
    else:
        general_exclude_set = set()

    if script_functionality['split_drawn_graphs_by_outcome']:
        # Uogólnić poza GOSE, na Good/Bad outcome
        gose_outcome_df = pd.read_csv('outcomes_processed.csv')
        patient_id_gose_outcome_dict = dict(zip([x.lower() for x in gose_outcome_df['Patient'].values.tolist()], gose_outcome_df['Gose3m'].fillna('').values.tolist()))

    for window_index, window in enumerate(windows_info):
        if script_functionality['export_window_pct_fill_data']:
            # DataFrame na dane o % wypełnienia kolejnych okien
            pct_fill_df = pd.DataFrame(columns=['id'] + signals_info['names'])

        if os.path.exists(rf'outputs/parameters_values_per_patient_{window["size"]}h.xlsx'):
            os.remove(rf'outputs/parameters_values_per_patient_{window["size"]}h.xlsx')
            
        with tqdm(total=len(os.listdir(signals_info['synced_directories'][0])), desc=f"Sampling window {window['size']}h. {window_index + 1}/{len(windows_info)}", unit="patient") as pbar:
            for filename in os.listdir(signals_info['synced_directories'][0]):
                patient_id = filename[:filename.find("_")]

                if patient_id.lower() not in general_exclude_set:
                    files = [f'{directory}\{patient_id}_synced_{signals_info["files_suffixes"][directory_index]}.pkl' for directory_index, directory in enumerate(signals_info['synced_directories'])]
                    
                    synced_times, synced_params = zip(*[(data['time'], data['signal']) for file in files for data in [pickle.load(open(file, 'rb'))]])

                    window_params, window_indexes, window_fill_pct = dafun.sample_window(synced_times, synced_params, window['size'], window['fill_threshold'], True)
                    try:
                        if script_functionality['export_window_pct_fill_data']:
                            row = [list(elem) for elem in window_fill_pct]
                            row.insert(0, patient_id)
                            pct_fill_df.loc[len(pct_fill_df)] = row

                        # Eksportowanie wartości parametrów per pacjent do pliku .xlsx
                        patient_data = {signals_info['names'][i]: window_params[i] for i in range(len(window_params))}
                        patient_df = pd.DataFrame(patient_data)
                        # Poszerzanie/Generowanie plików .xlsx
                        try:
                            with pd.ExcelWriter(f'outputs/parameters_values_per_patient_{window["size"]}h.xlsx', engine='openpyxl', mode='a') as writer:
                                patient_df.to_excel(writer, sheet_name=f'{patient_id}')
                        except FileNotFoundError:
                            with pd.ExcelWriter(f'outputs/parameters_values_per_patient_{window["size"]}h.xlsx', mode='w') as writer:
                                patient_df.to_excel(writer, sheet_name=f'{patient_id}')

                        # Rysowanie wykresów parametrów per pacjent
                        if script_functionality['draw_parameter_graphs_per_patient']:
                            output_directories = dafun.create_graphs_directories(script_functionality, window['size'])

                            # Dodawanie wartości null do danych, żeby na wykresie nie łączyć punktów między którymi wartość nie została policzona
                            window_indexes_with_gaps, window_params_with_gaps = dafun.introduce_data_gaps(window_indexes, window_params)

                            # Rozdzielanie wykresów względem wyniku leczenia
                            if script_functionality['split_drawn_graphs_by_outcome']:
                                patient_gose_score = patient_id_gose_outcome_dict.get(patient_id.lower(), '')

                                if patient_gose_score == '':
                                    output_dir = output_directories[3]
                                elif patient_gose_score < 5:
                                    output_dir = output_directories[2]
                                else:
                                    output_dir = output_directories[1]
                                output_dirs = [os.path.join(output_dir, f'{patient_id}_{window["size"]}h_{window["fill_threshold"]}%.{format}')
                                            for format in script_functionality["parameter_graphs_file_formats"]]
                                dafun.plot_params(window_indexes_with_gaps, window_params_with_gaps, signals_info['names'], signals_info['units'], window['size'], output_dirs)
                            # Albo tylko po rozmiarze okna
                            else:
                                output_dirs = [os.path.join(output_directories[0], f'{patient_id}_{window["size"]}h_{window["fill_threshold"]}%.{format}') 
                                            for format in script_functionality["parameter_graphs_file_formats"]]
                                dafun.plot_params(window_indexes_with_gaps, window_params_with_gaps, signals_info['names'], signals_info['units'], window['size'], output_dirs)

                    except Exception as e:
                        print(e)
                        troublemakers.append((patient_id, e))

                pbar.update(1)
        
        if script_functionality['export_window_pct_fill_data']:
            pct_fill_df.to_csv(f'outputs/pct_fill_data_{window["size"]}h.csv', index=False)

        # Kopiowanie pliku również do folderu data, aby nie zaburzać stałości miejsca, z którego czerpane są dane
        shutil.copyfile(f'outputs/parameters_values_per_patient_{window["size"]}h.xlsx',
                        f'data/synced/parameters_values_per_patient_{window["size"]}h.xlsx')

        if len(troublemakers) > 0:
            print(f'WARNING: Some patients have caused errors. Check troublemakers_sampling_{window["size"]}h.txt for more details.')
            with open(f'troublemakers_sampling_{window["size"]}h.txt', 'w') as file:
                for item in troublemakers:
                    file.write(f'{str(item)}\n')

def perform_analysis(script_functionality, signals_info, box_plots_limits, additional_exclusions=None): 
    troublemakers = []

    if script_functionality['draw_box_plots']:
        # Uogólnić poza GOSE, na Good/Bad outcome
        gose_outcome_df = pd.read_csv('outcomes_processed.csv')
        patient_id_gose_outcome_dict = dict(zip([x.lower() for x in gose_outcome_df['Patient'].values.tolist()], gose_outcome_df['Gose3m'].fillna('').values.tolist()))

    for window_index, window in enumerate(windows_info):
        good_count, poor_count = [0, 0]

        if script_functionality['draw_parameter_histograms']:
            # np.arrays z danymi do histogramów
            histogram_params_arrays = [np.array([]) for _ in range(len(signals_info['names']))]

        if script_functionality['draw_box_plots']:
            # Listy z wartościami parametrów podzielonymi na outcome do boxplotów
            box_data_good, box_data_poor = [[[] for _ in range(len(signals_info['names']))] for _ in range(2)]

        if additional_exclusions != None:
            # Listy zawierające informacje o wykluczeniach
            additional_exclude_df = pd.read_csv(additional_exclusions[window_index]['window_size_specific'])
            additional_exclude_list = [exclude_id.lower() for exclude_id in additional_exclude_df['Patient'].tolist()] 

            specific_window_indexes_exclude_df = pd.read_csv(additional_exclusions[window_index]['window_index_specific'], delimiter=';')
            specific_window_indexes_exclude_df['Patient'] = specific_window_indexes_exclude_df['Patient'].str.lower()    
            specific_window_indexes_exclude_dict = dict(zip(specific_window_indexes_exclude_df['Patient'], specific_window_indexes_exclude_df['Window_indexes_to_exclude']))
        else:
            additional_exclude_list = []
            specific_window_indexes_exclude_dict = {}

        source_excel = pd.ExcelFile(f'data/synced/parameters_values_per_patient_{window["size"]}h.xlsx')
        sheet_names = source_excel.sheet_names

        with tqdm(total=len(sheet_names), desc=f"Analyzing data for window {window['size']}h.' {window_index + 1}/{len(windows_info)}", unit="patient") as pbar:
            for patient_id in sheet_names:
                try:
                    if patient_id.lower() not in additional_exclude_list:
                        patient_gose_score = patient_id_gose_outcome_dict.get(patient_id.lower(), '')
                        if patient_gose_score != '':
                            if patient_gose_score < 5:
                                poor_count += 1
                            else:
                                good_count += 1
                        current_sheet_data = pd.read_excel(source_excel, patient_id)

                        num_columns = current_sheet_data.shape[1]
                        window_params = [current_sheet_data.iloc[:, i].tolist() for i in range(1, num_columns)]

                        # Zamienianie wartości odrzucanych okien o konkretnych indeksach na np.NaN
                        if patient_id.lower() in specific_window_indexes_exclude_dict:
                            try:
                                exclude_indexes = specific_window_indexes_exclude_dict[patient_id.lower()].split(',')
                            except:
                                exclude_indexes = [specific_window_indexes_exclude_dict[patient_id.lower()]]
                            exclude_indexes = [int(item) for item in exclude_indexes]
                            for exclude_index in exclude_indexes:
                                for i in range(len(window_params)):
                                    window_params[i][exclude_index] = np.NaN

                        # Przygotowywanie danych pod wykresy pudełkowe
                        if script_functionality['draw_box_plots']:
                            patient_gose_score = patient_id_gose_outcome_dict.get(patient_id.lower(), '')
                            for i in range(len(window_params)):
                                if patient_gose_score != '':
                                    # Dzielenie pacjentów na grupy w zależności od wyniku leczenia
                                    target_list = box_data_poor if patient_gose_score < 5 else box_data_good
                                    for j in range(len(window_params[i])):
                                        try:
                                            target_list[i][j].append(window_params[i][j])
                                        except:
                                            target_list[i].append([window_params[i][j]])

                        # Przygotowanie danych pod histogramy
                        if script_functionality['draw_parameter_histograms']:
                            histogram_params_arrays = dafun.append_histogram_arrays(histogram_params_arrays, window_params)
                    pbar.update(1)
                except Exception as e:
                    print(e)
                    troublemakers.append((patient_id, e))

            source_excel.close()

            if script_functionality['draw_parameter_histograms']:
                for signal_name, signal_unit, data in zip(signals_info['names'], signals_info['units'], histogram_params_arrays):
                    plt.hist(data)
                    plt.title(f'Histogram rozkładu wartości parametru {signal_name} w oknie {window["size"]}h')
                    plt.xlabel(f'{signal_name} {signal_unit}')
                    plt.ylabel('Częstość')

                    for format in script_functionality['parameter_histograms_file_formats']:
                        plt.savefig(f'outputs/histograms/{signal_name}_histogram_{window["size"]}h.{format}')
                    plt.close()

                # Sprowadzanie list do jednej długości
                histogram_list_max_len = max(map(len, histogram_params_arrays))
                histogram_params_arrays = [np.pad(arr, (0, histogram_list_max_len - len(arr)), constant_values=np.nan)
                                                for arr in histogram_params_arrays]

                # Tworzenie DataFrames, eksport do .csv
                data_dict = {signal_name: data for signal_name, data in zip(signals_info["names"], histogram_params_arrays)}
                histogram_data_df = pd.DataFrame(data_dict)
                histogram_data_df.to_csv(f'outputs/histograms/histogram_data_{window["size"]}h.csv', index=False)

            if script_functionality['draw_box_plots']:
                for i in range(len(signals_info['names'])):
                    for index, box_data in enumerate([box_data_poor, box_data_good]):
                        max_length = max([len(sublist) for sublist in box_data[i]])

                        # Próg wypełnienia danymi
                        percentage_threshold = script_functionality['box_plots_threshold_value']
                        min_length = max_length * percentage_threshold // 100
                        # Filtrowanie elementów, które nie przekraczają progu wypełnienia danymi
                        data = [sublist for sublist in box_data[i] if len(sublist) >= min_length]

                        # Sprowadzanie list do tej samej długości
                        same_length_data = [sublist + [None] * (max_length - len(sublist)) for sublist in data]
                        # Tworzenie DataFrame
                        df = pd.DataFrame(same_length_data)
                        df = df.T

                        if script_functionality['export_parameter_values_per_window']:
                            df.to_csv(rf'data/parameter_values_data/{signals_info["names"][i]}_{"poor" if index == 0 else "good"}_{window["size"]}h.csv', index=False)
                        
                        if script_functionality['limit_box_plots_x_axis']:
                            tick_range = range(box_plots_limits[window_index]['range_indexes'][0], box_plots_limits[window_index]['range_indexes'][1])
                            tick_length = len(tick_range)
                        else:
                            tick_range = range(0, len(df))
                            tick_length = len(tick_range)

                        match index:
                            case 0:
                                if script_functionality['limit_box_plots_x_axis']:
                                    df = df.iloc[:, box_plots_limits[window_index]['range_indexes'][0]:box_plots_limits[window_index]['range_indexes'][1]]
                                positions = np.arange(1, df.shape[1] + 1) - 0.2
                                fig, ax = plt.subplots(figsize=(10, 8))
                                bp0 = ax.boxplot([df[col].dropna() for col in df.columns], positions=positions, boxprops=dict(color='red', facecolor='red', alpha=0.5),
                                            medianprops=dict(color='red'), flierprops=dict(markeredgecolor='red'), patch_artist=True, widths=0.3)
                            case 1:
                                if script_functionality['limit_box_plots_x_axis']:
                                    df = df.iloc[:, box_plots_limits[window_index]['range_indexes'][0]:box_plots_limits[window_index]['range_indexes'][1]]
                                positions = np.arange(1, df.shape[1] + 1) + 0.2
                                bp1 = ax.boxplot([df[col].dropna() for col in df.columns], positions=positions, boxprops=dict(color='green', facecolor='green', alpha=0.5),
                                            medianprops=dict(color='green'), flierprops=dict(markeredgecolor='green'), patch_artist=True, widths=0.3)

                                if tick_length <= 10:
                                    ax.set_xticks(np.arange(1, df.shape[1] + 1))
                                    ax.set_xticklabels([i for i in tick_range])
                                elif 10 < tick_length < 16:
                                    ax.set_xticks(np.arange(1, df.shape[1] + 1))
                                    ax.set_xticklabels([i if i % 2 == 0 else '' for i in tick_range])
                                else:
                                    ax.set_xticks(np.arange(1, df.shape[1] + 1))
                                    ax.set_xticklabels([i if i % 3 == 0 else '' for i in tick_range])
                                ax.legend([bp0['boxes'][0], bp1['boxes'][0]], ['Zły wynik leczenia', 'Dobry wynik leczenia'],
                                        loc='upper right')

                                plt.xlabel(f'Indeks okna {window["size"]}h')
                                plt.ylabel(f'{signals_info["names"][i]} {signals_info["units"][i]}')
                                plt.title(f'Średnie wartości parametru {signals_info["names"][i]} w oknach {window["size"]}h')
                                for format in script_functionality['box_plots_file_formats']:
                                    file_path = rf'outputs/boxplots/{signals_info["names"][i]}_boxplot_{window["size"]}h.{format}'
                                    plt.savefig(file_path)
                                plt.close()
        print('done')
        print(poor_count, good_count)

        if len(troublemakers) > 0:
            print(f'WARNING: Some patients have caused errors. Check troublemakers_analysis_{window["size"]}h.txt for more details.')
            with open(f'troublemakers_analysis_{window["size"]}h.txt', 'w') as file:
                for item in troublemakers:
                    file.write(f'{str(item)}\n')


if __name__ == '__main__':

    signals_info = {'names' : ['ICP', 'ABP', 'AMP', 'HR', 'PRx', 'PSI'],
                'units' : ['[mm Hg]', '[mm Hg]', '[mm Hg]', '[BPM]', '[-]', '[-]'],
                'files_suffixes' : ['meanICP', 'meanABP', 'AMPpp', 'HR', 'PRx', 'PSI'],
                'directories' : [rf'{os.getcwd()}\data\MeanICP', rf'{os.getcwd()}\data\MeanABP', rf'{os.getcwd()}\data\AMPpp',
                                        rf'{os.getcwd()}\data\HR', rf'{os.getcwd()}\data\PRx', rf'{os.getcwd()}\data\PSI'],
                'synced_directories' : [rf'{os.getcwd()}\data\synced\MeanICP', rf'{os.getcwd()}\data\synced\MeanABP', rf'{os.getcwd()}\data\synced\AMPpp',
                                                rf'{os.getcwd()}\data\synced\HR', rf'{os.getcwd()}\data\synced\PRx', rf'{os.getcwd()}\data\synced\PSI']}

    script_functionality = {
        'export_window_pct_fill_data' : True, # Generowanie .csv z danymi dotyczącymi % wypełnienia okien per parametr per pacjent
        'draw_parameter_histograms' : True, # Generowanie histogramów rozkładów parametrów
        'parameter_histograms_file_formats' : ['png', 'svg', 'pdf'], # Ma znaczenie jeśli draw_parameter_histograms == True - formaty pliku graficznego, na przykład 'svg', 'png'
        'draw_parameter_graphs_per_patient' : True, # Generowanie wykresów parametrów per pacjent z podziałem na okna
        'parameter_graphs_file_formats' : ['png', 'svg', 'pdf'], # Ma znaczenie jeśli draw_parameter_graphs_per_patient == True - formaty pliku graficznego, na przykład 'svg', 'png'
        'split_drawn_graphs_by_outcome' : True, # Ma znaczenie jeśli draw_parameter_graphs_per_patient == True - podział wykresów w zależności od wyniku leczenia
        'draw_box_plots' : True, # Generowanie wykresów pudełkowych
        'limit_box_plots_x_axis' : True, # Ograniczenie boxplotów do przedziału osi X
        'box_plots_threshold_value' : 10, # Ma znaczenie jeśli draw_box_plots == True - próg (w %) ilości danych per indeks okna na wykresach pudełkowych. 0 -> brak progu
        'box_plots_file_formats' : ['png', 'svg', 'pdf'], # Ma znaczenie jeśli draw_box_plots == True - formaty pliku graficznego, na przykład 'svg', 'png'
        'export_parameter_values_per_window' : True # Ma znaczenie, jeśli draw_box_plots == True - eksport wartości parametrów per okno do plików .csv wykorzystywanych w skrypcie statystycznym
    }

    windows_info = [{'size' : 8, 'fill_threshold' : 50}, {'size' : 24, 'fill_threshold' : 50}]

    box_plots_limits = [{'range_indexes' : [3, 24]}, {'range_indexes' : [1, 8]}]

    additional_exclusions = [{'window_size_specific' : 'additional_patient_exclusions_8h.csv', 
                              'window_index_specific' : 'specific_window_indexes_exclusions_8h.csv'},
                            {'window_size_specific' : 'additional_patient_exclusions_24h.csv',
                             'window_index_specific' : 'specific_window_indexes_exclusions_24h.csv'}]

    # create_synced_signal_files(signals_info)
    # sample_windows(script_functionality, signals_info, windows_info, 'general_patient_exclusions.csv')
    perform_analysis(script_functionality, signals_info, box_plots_limits, additional_exclusions)