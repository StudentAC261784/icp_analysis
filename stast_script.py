from scipy.stats import shapiro
import os
import pandas as pd

excel_writer = pd.ExcelWriter(f'{os.getcwd()}\outputs\shapiro_results.xlsx', engine='xlsxwriter')

for filename in os.listdir(rf'{os.getcwd()}\data\parameter_values_data'):
    filepath = f'{os.getcwd()}\data\parameter_values_data\{filename}'
    filename = filename.replace('.csv', '')
    param, outcome, window = filename.split('_')
    df = pd.read_csv(filepath)
    if window == "8h":
        window_range = range(3, 24)
        df = df.iloc[:, window_range].T # okna o indeksach od 3 do 23
    else:
        window_range = range(1, 8)
        df = df.iloc[:, window_range].T # okna o indeksach od 1 do 7
    
    p_value_df = pd.DataFrame(df.apply(lambda row: shapiro(row.dropna())[1], axis=1), columns=['shapiro_p_value'])
    p_value_df.insert(0, 'window_index', window_range)

    p_value_df.to_excel(excel_writer, sheet_name=filename, index=False)

excel_writer.save()
