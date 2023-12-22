from scipy.stats import shapiro, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import pandas as pd

def calculate_median_differences(output_file_name, window_ranges_dict, window_sizes_to_analyze):
    excel_writer = pd.ExcelWriter(f'{os.getcwd()}\outputs\statistics\{output_file_name}', engine='xlsxwriter')
    done_files = []
    for filename in os.listdir(rf'{os.getcwd()}\data\parameter_values_data'):
        if filename not in done_files:
            filename_bad = filename.replace('good', 'poor')
            done_files.append(filename)
            done_files.append(filename_bad)
            filepath = f'{os.getcwd()}\data\parameter_values_data\{filename}'
            filepath_bad = f'{os.getcwd()}\data\parameter_values_data\{filename_bad}'

            param, outcome, window_size = filename.replace('.csv', '').split('_')
            if window_size in window_sizes_to_analyze:
                df_good = pd.read_csv(filepath)
                df_bad = pd.read_csv(filepath_bad)
                window_range = window_ranges_dict[window_size]
                df_good = df_good.iloc[:, window_range] # tylko okna o odpowiednich indeksach
                df_bad = df_bad.iloc[:, window_range]
                
                median_df = pd.concat([df_bad.median(), df_good.median()], axis=1)
                median_df.columns = ['median_bad', 'median_good']
                median_df.insert(0, 'window_index', window_range)
                median_df.insert(3, 'median_bad - median_good', median_df['median_bad'] - median_df['median_good'])

                median_df.to_excel(excel_writer, sheet_name=f'{param}_{window_size}', index=False)

    excel_writer.save()

def perform_mann_whitney_test(output_file_name, window_ranges_dict, window_sizes_to_analyze):
    done_files = []
    excel_writer = pd.ExcelWriter(f'{os.getcwd()}\outputs\statistics\{output_file_name}', engine='xlsxwriter')
    for filename in os.listdir(rf'{os.getcwd()}\data\parameter_values_data'):
        if filename not in done_files:
            filename_poor = filename.replace('good', 'poor')
            done_files.append(filename)
            done_files.append(filename_poor)
            filepath = f'{os.getcwd()}\data\parameter_values_data\{filename}'
            filepath_poor = f'{os.getcwd()}\data\parameter_values_data\{filename_poor}'

            param, outcome, window_size = filename.replace('.csv', '').split('_')
            if window_size in window_sizes_to_analyze:
                df_good = pd.read_csv(filepath)
                df_poor = pd.read_csv(filepath_poor)
                window_range = window_ranges_dict[window_size]
                df_good = df_good.iloc[:, window_range].T # tylko okna o odpowiednich indeksach
                df_poor = df_poor.iloc[:, window_range].T 

                results = []

                for (row_poor, row_good) in zip(df_poor.iterrows(), df_good.iterrows()):
                    _, row_poor = row_poor
                    _, row_good = row_good

                    stat, p_value = mannwhitneyu(row_poor.dropna(), row_good.dropna())
                    results.append(p_value)

                results_df = pd.DataFrame(results, columns=['mann_whitney_p_value'])
                results_df.insert(0, 'window_index', window_range)
                results_df.to_excel(excel_writer, sheet_name=f'{param}_{window_size}', index=False)
    excel_writer.save()

def perform_shapiro_test(output_file_name, window_ranges_dict, window_sizes_to_analyze):
    excel_writer = pd.ExcelWriter(f'{os.getcwd()}\outputs\statistics\{output_file_name}', engine='xlsxwriter')
    for filename in os.listdir(rf'{os.getcwd()}\data\parameter_values_data'):
        filepath = f'{os.getcwd()}\data\parameter_values_data\{filename}'
        param, outcome, window_size = filename.replace('.csv', '').split('_')

        if window_size in window_sizes_to_analyze:
            df = pd.read_csv(filepath)

            window_range = window_ranges_dict[window_size]
            df = df.iloc[:, window_range].T

            p_value_df = pd.DataFrame(df.apply(lambda row: shapiro(row.dropna())[1], axis=1), columns=['p_value'])
            p_value_df.insert(0, 'window_index', window_range)

            p_value_df.to_excel(excel_writer, sheet_name=filename.replace('.csv', ''), index=False)

    excel_writer.save()

def perfmorm_two_way_anova_test(output_file_name, window_ranges_dict, window_sizes_to_analyze):
    done_files = []
    excel_writer = pd.ExcelWriter(f'{os.getcwd()}\outputs\statistics\{output_file_name}', engine='xlsxwriter')
    for filename in os.listdir(rf'{os.getcwd()}\data\parameter_values_data'):
        if filename not in done_files:
            filename_poor = filename.replace('good', 'poor')
            done_files.append(filename)
            done_files.append(filename_poor)
            filepath = f'{os.getcwd()}\data\parameter_values_data\{filename}'
            filepath_poor = f'{os.getcwd()}\data\parameter_values_data\{filename_poor}'

            param, outcome, window_size = filename.replace('.csv', '').split('_')
            if window_size in window_sizes_to_analyze:
                df_good = pd.read_csv(filepath)
                df_poor = pd.read_csv(filepath_poor)
                window_range = window_ranges_dict[window_size]
                df_good = df_good.iloc[:, window_range] # tylko okna o odpowiednich indeksach
                df_poor = df_poor.iloc[:, window_range]

                temp_df = pd.DataFrame()
                for i in range(df_good.shape[1]):
                    temp_df_good = pd.DataFrame({f'value': df_good.iloc[:, i], 'hours': (i+1)*int(window_size[:-1]), 'outcome': 'good'})
                    temp_df_poor = pd.DataFrame({f'value': df_poor.iloc[:, i], 'hours': (i+1)*int(window_size[:-1]), 'outcome': 'poor'})

                    temp_df = pd.concat([temp_df, temp_df_good, temp_df_poor], axis=0)
                    
                temp_df.dropna(inplace=True)

                model = ols(f'value ~ C(outcome) + C(hours) + C(outcome):C(hours)', data = temp_df).fit()
                result = sm.stats.anova_lm(model, typ=2)
                results_df = pd.DataFrame({'f': result.iloc[0:3, 2], 'p_value': result.iloc[0:3,3]})
                results_df.insert(0, 'variable', ['outcome', 'time_passed', 'outcome*time_passed'])
                results_df.to_excel(excel_writer, sheet_name=f'{param}_{window_size}', index=False)
    excel_writer.save()


if __name__ == '__main__':
    # Słownik limitów indeksów okien
    window_ranges_dict = {
        "8h" : range(3, 24),
        "24h" : range(1, 8)
    }

    perform_shapiro_test('shapiro_results.xlsx', window_ranges_dict, ["8h", "24h"])
    # calculate_median_differences('median_results.xlsx', window_ranges_dict, ["8h", "24h"])
    perform_mann_whitney_test('mann_whitney_results.xlsx', window_ranges_dict, ["24h"])
    perfmorm_two_way_anova_test('anova_results.xlsx', window_ranges_dict, ["24h"])