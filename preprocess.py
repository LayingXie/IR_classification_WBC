#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:41:07 2024

@author: laying
"""
import pandas as pd
import numpy as np
import re
import os
from PreprocSpec import snip, normal_spectra
from scipy.signal import savgol_filter



def process_region(df_transposed, lower_bound, upper_bound, iterations, smoothing_window, apply_savgol, window_length, polynomial):
    numeric_values = pd.to_numeric(df_transposed.iloc[0, :], errors='coerce')
    columns_to_remove = numeric_values[(numeric_values > upper_bound) | (numeric_values < lower_bound)].index
    df_filtered = df_transposed.drop(columns=columns_to_remove)
    
    absorbance = df_filtered.iloc[1:, :].apply(pd.to_numeric, errors='coerce')
    wavenumber = df_filtered.iloc[0, :]

    absorbance = absorbance - np.min(absorbance)
    spec_baseline = snip(np.array(absorbance), iterations, smoothing_window, return_baseline=False)
    
    if apply_savgol:
        spec_smooth = savgol_filter(spec_baseline.copy(), window_length, polynomial)
    else:
        spec_smooth = spec_baseline.copy()
    
    return wavenumber, spec_smooth

def process_data(df, region1=(1000, 1700), region2=(2820, 3000), iterations1=70, iterations2=50, smoothing_window=2, window_length=7, polynomial=2, apply_savgol=True, norm_method ='vector'):
    df_transposed = df.T
    
    lower_bound1, upper_bound1 = region1
    lower_bound2, upper_bound2 = region2
    
    wavenumber_1, spec_smooth_1 = process_region(df_transposed, lower_bound1, upper_bound1, iterations1, smoothing_window, apply_savgol, window_length, polynomial)
    wavenumber_2, spec_smooth_2 = process_region(df_transposed, lower_bound2, upper_bound2, iterations2, smoothing_window, apply_savgol, window_length, polynomial)

    wavenumber_1 = wavenumber_1.to_numpy().reshape(-1, 1)
    wavenumber_2 = wavenumber_2.to_numpy().reshape(-1, 1)
    wavenumber = np.vstack((wavenumber_2, wavenumber_1))
    
    merged_data = np.concatenate((spec_smooth_2, spec_smooth_1), axis=1)
    spec_norm = normal_spectra(merged_data, norm_method = norm_method)
    
    merged_data2 = np.concatenate((wavenumber, spec_norm.T), axis=1)
    return merged_data2

def pool_data(data):
    mat_reshaped = np.reshape(data.values[:, 1:], (data.shape[0], 64, 64))
    pooled_mat = np.mean(
        mat_reshaped.reshape(mat_reshaped.shape[0], 64 // 4, 4, 64 // 4, 4), axis=(2, 4)
    )
    return pd.DataFrame(pooled_mat.reshape(pooled_mat.shape[0], -1))

def generate_patient_file_name(file_name):
    match = re.match(r'(\d+)', file_name)
    if match:
        prefix_number = match.group(1)
        return f"Patient#{prefix_number}"
    else:
        return None
    
    
    
def process_files(folder_path, metadata_path, pool_data, generate_patient_file_name, 
                   merge_on_metadata_col, merge_additional_cols, cells):
    all_data = pd.DataFrame()
    first_row = None
    file_names = []

    for filename in os.listdir(folder_path):
        if filename.startswith('.') or not filename.endswith('.csv'):
            continue  

        file_path = os.path.join(folder_path, filename)

        try:
            data = pd.read_csv(file_path, header=None, dtype=float, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"Error reading file {file_path} with UTF-8 encoding. Trying alternative encodings.")
            try:
                data = pd.read_csv(file_path, header=None, dtype=float, encoding='utf-16-le')
            except UnicodeDecodeError:
                print(f"Error reading file {file_path} with UTF-16-LE encoding. Trying Latin-1 encoding.")
                data = pd.read_csv(file_path, header=None, dtype=float, encoding='ISO-8859-1')

        if data.empty:
            print(f"Warning: Empty file {file_path}. Skipping.")
            continue

        pooled_data = pool_data(data)

        if first_row is None:
            first_row = data.iloc[:, :1]

        all_data = pd.concat([all_data, pooled_data], axis=1)
        file_names.extend([filename] * (pooled_data.shape[1]))

    if first_row is not None:
        all_data.insert(0, 'Wavenumber', first_row)

    all_data.loc[len(all_data)] = [None] + file_names
    all_data.fillna("filename", inplace=True)

    df = all_data.T.reset_index(drop=False)
    df.columns = df.iloc[0]

    metadata = pd.read_excel(metadata_path)
    result_df = pd.merge(df[['filename']], metadata[[merge_on_metadata_col,merge_additional_cols]], 
                         left_on='filename', right_on=merge_on_metadata_col, how='left')

    y = result_df.iloc[1:, 2]
    y.index = range(len(y))
    
    file_names = result_df.iloc[1:, 1]
    patient = [generate_patient_file_name(file_name) for file_name in file_names]
    patient = pd.DataFrame(patient)

    dict_df = {"group": y, "patient": patient.iloc[:, 0]}

    spectra = all_data.iloc[:-1, :].T
    spectra.columns = spectra.iloc[0]
    spectra = spectra.iloc[1:].reset_index(drop=True)
    spectra = pd.DataFrame(spectra, dtype=object)

    final_df = pd.concat([pd.DataFrame(dict_df), spectra], axis=1)
    wavenumber = final_df.iloc[:, 2:].columns
    
    cell_type_value = cells
    final_df.insert(2, 'cells', cell_type_value)

    return final_df, wavenumber


def process_and_combine_files(folders, metadata_path, pool_data, generate_patient_file_name):
    combined_df = pd.DataFrame()
    for folder, merge_on_metadata_col, merge_additional_cols, cells in folders:
        final_df, wavenumber = process_files(folder, metadata_path, pool_data, generate_patient_file_name, 
                                             merge_on_metadata_col, merge_additional_cols, cells)
        combined_df = pd.concat([combined_df, final_df], axis=0, ignore_index=True)
    return combined_df, wavenumber
    
    