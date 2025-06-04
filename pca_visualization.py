# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:40:12 2025

@author: admin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_visualization2(
    datasets,           # List of dicts, each with keys: 'data' (DataFrame), 'label' (str), 'marker' (str)
    groups_id,          # List of group identifiers (e.g., [0, 1])
    group_colors,       # Dict of group -> colormap (e.g., plt.cm.Reds)
    label_mapping,      # Dict mapping group IDs to display names (e.g., {0: "Control", 1: "Disease"})
    title,              # Plot title (string)
    xlabel,             # X-axis label
    ylabel,             # Y-axis label
    visible_groups      # Set or list of group IDs to be shown
):
    """
    Visualizes PCA results for multiple datasets using group-specific colors and per-patient scatter plots.
    """
    plt.figure(figsize=(6, 4), dpi=600)
    
    # Collect all patients across datasets for consistent color mapping
    all_data = pd.concat([dataset['data'] for dataset in datasets])
    custom_cmaps = {}

    # Generate a custom color for each patient within each visible group
    for group in groups_id:
        if group in visible_groups:
            unique_patients = all_data[all_data['group'] == group]['patient'].unique()
            custom_cmaps[group] = {
                patient: group_colors[group](i / len(unique_patients)) 
                for i, patient in enumerate(unique_patients)
            }

    handles = []
    labels = []

    # Plot each dataset
    for dataset in datasets:
        data = dataset['data']
        marker = dataset['marker']

        for group in groups_id:
            if group in visible_groups:
                group_data = data[data['group'] == group]
                
                # Plot each patient in the group
                for patient in group_data['patient'].unique():
                    patient_data = group_data[group_data['patient'] == patient]
                    plt.scatter(
                        patient_data['PC1'], patient_data['PC2'],
                        color=custom_cmaps[group][patient],
                        alpha=0.5,
                        marker=marker
                    )

                # Add legend entry (once per group)
                if label_mapping[group] not in labels:
                    handles.append(
                        plt.Line2D(
                            [0], [0], marker=marker, color='w',
                            markerfacecolor=group_colors[group](0.7),
                            markersize=10, alpha=0.5
                        )
                    )
                    labels.append(label_mapping[group])

    # Customize plot appearance
    ax = plt.gca()
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)

    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(xlabel, fontsize=10, fontweight='bold')
    plt.ylabel(ylabel, fontsize=10, fontweight='bold')

    # Optional axis limits and ticks (uncomment if needed)
    # plt.xlim(-0.1, 0.2)
    # plt.xticks(np.arange(-0.1, 0.3, 0.1))
    # plt.ylim(-0.1, 0.2)
    # plt.yticks(np.arange(-0.1, 0.3, 0.1))

    plt.legend(handles, labels, loc='upper left')
    plt.show()
