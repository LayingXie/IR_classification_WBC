#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:59:07 2024

@author: laying
"""

import pandas as pd
import numpy as np
from brokenaxes import brokenaxes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_scores(df, n_components=8, group_colors=None, label_mapping=None, scale_data=True, wavenumber=None, show_legend=True, plot_combinations=None, show_labels=True, wspace=0.1, hspace=0.1, n=3):
    """
    Plot PCA scores scatter of patients.
    
    Parameters:
        df (DataFrame): The input DataFrame containing spectral data and group labels.
        n_components (int): Number of principal components to plot. Default is 8.
        group_colors (dict): Custom colormap for patient groups. Keys are group labels, values are colormap.
        label_mapping (dict): Mapping of group labels to their corresponding names.
        scale_data (bool): Whether to scale the data. Default is True.
        plot_combinations (list of tuple): List of tuples specifying the PC combinations to plot. Default is None.
        show_labels (bool): Whether to show the labels on the scatter plot. Default is True.
    """
    if scale_data:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.iloc[:, 2:])
    else:
        scaled_data = df.iloc[:, 2:]
    
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled_data)
    
    if group_colors is None:
        group_colors = {0: plt.cm.Blues, 2: plt.cm.Greens, 1: plt.cm.Reds}  # Default colormap
        
    if label_mapping is None:
        label_mapping = {0: 'Malignant solid tumors', 1: 'Lymphoma', 2: 'Leukemia'}  # Default label mapping

    custom_cmaps = {}
    for group, color_map in group_colors.items():
        unique_patients = df[df['group'] == group]['patient'].unique()
        custom_cmaps[group] = {patient: color_map(i / len(unique_patients)) for i, patient in enumerate(unique_patients)}

    if plot_combinations is None:
        fig, axs = plt.subplots(n_components, n_components, figsize=(8, 8))

        for i in range(n_components):
            for j in range(n_components):
                ax = axs[i, j]
                if i != j:
                    for group, cmap in custom_cmaps.items():
                        indices = np.where(df['group'] == group)[0]
                        ax.scatter(scores[indices, i], scores[indices, j], c=[cmap[patient] for patient in df.iloc[indices]['patient']], alpha=0.5, label=label_mapping[group])
                else:
                    ax.text(0.2, 0.2, f'PC{i + 1}', fontsize=12, ha='center', va='center')
                if i == n_components - 1:
                    ax.set_xlabel(f'PC{j + 1}')
                if j == 0:
                    ax.set_ylabel(f'PC{i + 1}')

        if show_legend:
            legend_handles = []
            for group, cmap in custom_cmaps.items():
                unique_patients = df[df['group'] == group]['patient'].unique()
                for patient in unique_patients:
                    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=cmap[patient], label=f'{label_mapping[group]}: {patient}'))

            plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=2)
        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.show()
    else:
        
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))

        for ax, (i, j) in zip(axs.flat, plot_combinations):
            for group, cmap in custom_cmaps.items():
                indices = np.where(df['group'] == group)[0]
                sc = ax.scatter(scores[indices, i], scores[indices, j], c=[cmap[patient] for patient in df.iloc[indices]['patient']], alpha=0.5, label=label_mapping[group])
            
            ax.set_xlabel(f'PC{i + 1}')
            ax.set_ylabel(f'PC{j + 1}')
            #ax.set_title(f'PC{i + 1} vs PC{j + 1}')
            if show_labels:
                seen_labels = set()
                for idx in range(scores.shape[0]):
                   # ax.annotate(df.iloc[idx]['patient'], (scores[idx, i], scores[idx, j]), fontsize=8, alpha=0.75)
                    patient_label = df.iloc[idx]['patient']
                    label_number = patient_label.split('#')[-1]  # 提取 # 后面的数字部分

                    if label_number  not in seen_labels:
                        ax.annotate(label_number , (scores[idx, 0], scores[idx, 1]), fontsize=8, alpha=0.75)
                        seen_labels.add(label_number)


#        if show_legend:
#            legend_handles = []
#            for group, cmap in custom_cmaps.items():
#                unique_patients = df[df['group'] == group]['patient'].unique()
#                for patient in unique_patients:
#                    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=cmap[patient], label=f'{label_mapping[group]}: {patient}'))
            
#            fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.2, 0.5), fontsize=12)
            
        if show_legend:
            legend_handles = []
    
            # 使用预定义的组颜色
            for group, cmap in group_colors.items():
                color = cmap(0.6)  # 使用颜色映射中间的颜色
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markersize=5, 
                               markerfacecolor=color, 
                               label=f'{label_mapping[group]}')
                )
    
            # 添加透明度说明
            
            axs[0, 0].legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0, 1), fontsize=6)


        
        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.show()
        
    if wavenumber is not None:
        bax = brokenaxes(xlims=((1000, 1700), (2800, 3000)), wspace=0.15)
        X_pca = pca.components_
        # Iterate over each principal component
        for i in range(len(X_pca)):
            # Get the components
            X_filtered_pca = X_pca[i]
            # Plot the component with an offset
            bax.plot(wavenumber, X_filtered_pca.T + i*n, label=f'Component {i+1}')
    
        # Set labels and legend
        bax.set_xlabel('Wavenumber/cm$^{-1}$', labelpad=20, fontsize=12)
        bax.set_ylabel('Absorbance/ a.u.', labelpad=30, fontsize=12)
        bax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1, 1))
        plt.show()
    
def plot_pca_track(df, n_components, component_x_index=0, component_y_index=1, scaler=True, label_mapping=None, figsize=(10,10)):
    # Extract relevant data
    data = df.iloc[:, 2:]
    
    # Standardize data if scaler is set to True
    if scaler:
        data = StandardScaler().fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data)
    
    # Get unique groups
    unique_groups = df['group'].unique()
    
    # Define color map
    color_maps = [plt.cm.Greens, plt.cm.Reds,  plt.cm.Blues, plt.cm.Oranges, plt.cm.Purples, plt.cm.Paired]
    group_colors = {}
    group_color_maps = {group: color_maps[i % len(color_maps)] for i, group in enumerate(unique_groups)}
    
    for group in unique_groups:
        unique_patients = df[df['group'] == group]['patient'].unique()
        group_colors[group] = {patient: group_color_maps[group](i / len(unique_patients)) for i, patient in enumerate(unique_patients)}
    
    # Initialize plot
    plt.figure(figsize=figsize)
    marked_patients = set()
    
    # Plot scores for each group
    for group in unique_groups:
        unique_patients = df[df['group'] == group]['patient'].unique()
        for patient in unique_patients:
            patient_indices = np.where((df['group'] == group) & (df['patient'] == patient))[0]
            plt.scatter(scores[patient_indices, component_x_index], scores[patient_indices, component_y_index], color=group_colors[group][patient], label=f'{label_mapping[group]}: {patient}' if label_mapping else f'Group {group}: {patient}', alpha=0.5)
            # Add patient ID labels
            if patient not in marked_patients:
                plt.text(scores[patient_indices[0], component_x_index], scores[patient_indices[0], component_y_index], str(patient), fontsize=8)
                marked_patients.add(patient)
    
    # Add title and labels
    plt.title('PCA Scores Scatter of patients')
    plt.xlabel(f'Principal Component {component_x_index + 1}')
    plt.ylabel(f'Principal Component {component_y_index + 1}')
    
    # Add legend
    legend_samples = []
    for group in unique_groups:
        for patient, color in group_colors[group].items():
            legend_samples.append(plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color, label=f'{label_mapping[group]}: {patient}' if label_mapping else f'Group {group}: {patient}'))
    plt.legend(handles=legend_samples, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()




    

def plot_stacked_spectra(df, label_mapping, group_colors,n):
    # Extract spectra data and transpose for easier indexing
    x1 = pd.DataFrame(df.iloc[:, 2:], dtype=float).T
    bax = brokenaxes(xlims=((1000, 1700), (2800, 3000)), wspace=0.15)  # Create broken axes object
    
    # Initialize the base y coordinate for stacking
    base_yaxis = 0

    for cancer_type, label in label_mapping.items():
        rows_with_label = df['group'][df['group'] == cancer_type]
        valid_indices = rows_with_label.index.intersection(x1.columns)
        result_label = x1[valid_indices]

        dict_mean = {
            "mean": result_label.mean(axis=1),
            "std": result_label.std(axis=1)
        }
        dict_mean["mean+std"] = dict_mean["mean"] + dict_mean["std"]
        dict_mean["mean-std"] = dict_mean["mean"] - dict_mean["std"]
        dict_mean["Wavenumber"] = df.iloc[:, 2:].columns
        df_mean = pd.DataFrame.from_dict(dict_mean, dtype=float)

        # Plot mean spectra with the specified color
        color = group_colors[cancer_type]
        bax.plot(np.array(df_mean["Wavenumber"]).astype(float), 
                 np.array(df_mean["mean"]).astype(float) + base_yaxis, 
                 label=label, color=color(0.6))

        # Plot +- standard deviation lines with alpha for transparency
        bax.fill_between(np.array(df_mean["Wavenumber"]).astype(float), 
                         np.array(df_mean["mean-std"]).astype(float) + base_yaxis,
                         np.array(df_mean["mean+std"]).astype(float) + base_yaxis, 
                         color=color(0.6), alpha=0.3)

        # Update base y coordinate for next stack
        base_yaxis += max(df_mean["mean"].dropna()) * n  # Adjust 1.2 as needed to separate stacks

    bax.set_xlabel('Wavenumber/ cm$^{-1}$', labelpad=20, fontsize=12)
    bax.set_ylabel('Absorbance/ a.u.', labelpad=30, fontsize=12)
    bax.legend(loc='upper left', fontsize=8)

    plt.show()



def plot_patient_spectrum(dataframe, wavenumber=None, broken=True):
    unique_patients = dataframe['patient'].unique()
    
    for patient_id in unique_patients:
        patient_data = dataframe[dataframe['patient'] == patient_id].iloc[:, 2:]
        fs = 1
        freqs = np.fft.fftfreq(patient_data.shape[1], 1/fs)
        
        dict_mean = {}
        dict_mean["mean"] = patient_data.mean(axis=0)
        dict_mean["std"] = patient_data.std(axis=0)
        dict_mean["mean+std"] = dict_mean["mean"] + dict_mean["std"]
        dict_mean["mean-std"] = dict_mean["mean"] - dict_mean["std"]
        if wavenumber is not None:
            dict_mean["freqs"] = wavenumber
        else:
            dict_mean["freqs"] = freqs
        
        df_mean = pd.DataFrame.from_dict(dict_mean, dtype=float)
        
        if broken:
            bax = brokenaxes(xlims=((1000, 1700), (2800, 3000)), wspace=0.15)
            bax.plot(np.array(df_mean["freqs"]).astype(float), np.array(df_mean["mean"]).astype(float),)
            bax.fill_between(np.array(df_mean["freqs"]).astype(float), np.array(df_mean["mean-std"]).astype(float),
                               np.array(df_mean["mean+std"]).astype(float), alpha=0.3 )
            plt.title(f'Spectrum of {patient_id}')
            bax.set_xlabel('Wavenumber/ cm$^{-1}$', labelpad=20, fontsize=12)
            bax.set_ylabel('Absorbance/ a.u.', labelpad=40, fontsize=12)
            plt.show()
        else:
            plt.figure(figsize=(6, 3))
            plt.plot(np.array(df_mean["freqs"]).astype(float), np.array(df_mean["mean"]).astype(float))
            plt.fill_between(np.array(df_mean["freqs"]).astype(float), np.array(df_mean["mean-std"]).astype(float),
                             np.array(df_mean["mean+std"]).astype(float), alpha=0.3)
            plt.title(f'Frequency Spectrum of {patient_id}')
            plt.ylim(0, 0.2)
            if wavenumber is not None:
                plt.xlim(min(wavenumber), max(wavenumber))
            else:
                plt.xlim(0, 0.4)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.show()
            
def plot_stacked_spectra2(df, label_mapping, group_colors,n, title=None,type_col='group', figsize=(10, 6), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)
    # Extract spectra data and transpose for easier indexing
    
    x1 = pd.DataFrame(df.iloc[:, 3:], dtype=float).T
    bax = brokenaxes(xlims=((1000, 1700), (2800, 3000)), wspace=0.2)  # Create broken axes object
    
    # Initialize the base y coordinate for stacking
    base_yaxis = 0

    for cancer_type, label in label_mapping.items():
        rows_with_label = df[type_col][df[type_col] == cancer_type]
        valid_indices = rows_with_label.index.intersection(x1.columns)
        result_label = x1[valid_indices]

        dict_mean = {
            "mean": result_label.mean(axis=1),
            "std": result_label.std(axis=1)
        }
        dict_mean["mean+std"] = dict_mean["mean"] + dict_mean["std"]
        dict_mean["mean-std"] = dict_mean["mean"] - dict_mean["std"]
        dict_mean["Wavenumber"] = df.iloc[:, 3:].columns
        df_mean = pd.DataFrame.from_dict(dict_mean, dtype=float)

        # Plot mean spectra with the specified color
        color = group_colors[cancer_type]
        bax.plot(np.array(df_mean["Wavenumber"]).astype(float), 
                 np.array(df_mean["mean"]).astype(float) + base_yaxis, 
                 label=label, color=color(0.6))

        # Plot +- standard deviation lines with alpha for transparency
        bax.fill_between(np.array(df_mean["Wavenumber"]).astype(float), 
                         np.array(df_mean["mean-std"]).astype(float) + base_yaxis,
                         np.array(df_mean["mean+std"]).astype(float) + base_yaxis, 
                         color=color(0.6), alpha=0.3)

        # Update base y coordinate for next stack
        base_yaxis += max(df_mean["mean"].dropna()) * n  # Adjust 1.2 as needed to separate stacks
    
    bax.set_xlabel('Wavenumber/ cm$^{-1}$', labelpad=20, fontsize=10,fontweight='bold')
    bax.set_ylabel('Absorbance/ a.u.', labelpad=30, fontsize=10,fontweight='bold')
    for spine in bax.spines['bottom']: 
        spine.set_linewidth(1.5) 
    for spine in bax.spines['left']:  
        spine.set_linewidth(1.5)  

    bax.legend(loc='upper left', fontsize=6)
    bax.set_title(title,fontweight='bold', fontsize=12)
    plt.show()
    
    
def plot_individual_spectra(df, label_mapping, group_colors, type_col='group',figsize=(6,4), dpi=300):
    """
    Plots individual spectra based on the specified group column, each in a separate figure.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    label_mapping (dict): Dictionary mapping group labels to their names.
    group_colors (dict): Dictionary mapping group labels to their colors.
    type_col (str): Column name to use for grouping ('group' or 'cells').
    """
    
    # Extract spectra data and transpose for easier indexing
    x1 = pd.DataFrame(df.iloc[:, 3:], dtype=float).T
    for cancer_type, label in label_mapping.items():
        plt.figure(figsize=figsize, dpi=dpi)
        bax = brokenaxes(xlims=((1000, 1700), (2800, 3000)), wspace=0.2)  # Create broken axes object

        rows_with_label = df[type_col][df[type_col] == cancer_type]
        valid_indices = rows_with_label.index.intersection(x1.columns)
        result_label = x1[valid_indices]

        dict_mean = {
            "mean": result_label.mean(axis=1),
            "std": result_label.std(axis=1)
        }
        dict_mean["mean+std"] = dict_mean["mean"] + dict_mean["std"]
        dict_mean["mean-std"] = dict_mean["mean"] - dict_mean["std"]
        dict_mean["Wavenumber"] = df.iloc[:, 3:].columns
        df_mean = pd.DataFrame.from_dict(dict_mean, dtype=float)

        # Plot mean spectra with the specified color
        color = group_colors[cancer_type]
        bax.plot(np.array(df_mean["Wavenumber"]).astype(float), 
                 np.array(df_mean["mean"]).astype(float), 
                 label=label, color=color(0.7))

        # Plot +- standard deviation lines with alpha for transparency
        bax.fill_between(np.array(df_mean["Wavenumber"]).astype(float), 
                         np.array(df_mean["mean-std"]).astype(float),
                         np.array(df_mean["mean+std"]).astype(float), 
                         color=color(0.7), alpha=0.3)

        bax.set_xlabel('Wavenumber/ cm$^{-1}$', labelpad=20, fontsize=10,fontweight='bold')
        bax.set_ylabel('Absorbance/ a.u.', labelpad=35, fontsize=10, fontweight='bold')
        bax.tick_params(axis='x', labelsize=10)  
        bax.tick_params(axis='y', labelsize=10)  
        bax.set_ylim([-0.05, 0.25])  
        bax.set_yticks(np.arange(-0.05, 0.25, 0.05))  
        for spine in bax.spines['bottom']:  
            spine.set_linewidth(1.5)  
        for spine in bax.spines['left']:  
            spine.set_linewidth(1.5)  
        bax.set_title(f'Spectra for {label}', fontsize=12, fontweight='bold', pad=10)
        bax.legend(loc='upper left', fontsize=12)
        plt.show()
