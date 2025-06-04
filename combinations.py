# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:27:00 2025

@author: admin
"""

def process_data_fusions(combined_df_filter, cell_combination):
    data = combined_df_filter[combined_df_filter['cells'].isin(cell_combination)].iloc[:,:-1]
    data['sample_id'] = data.groupby(['patient', 'cells']).cumcount() + 1
    data['patient'] = data['patient'] + '_' + data['sample_id'].astype(str)
    data.drop(columns=['sample_id'], inplace=True)

    feature_columns = data.columns[3:]
    pivoted_data = data.pivot(index=['patient'], columns='cells')
    pivoted_data.columns = [f"{col[0]}_{col[1]}" for col in pivoted_data.columns]
    pivoted_data.reset_index(inplace=True)

    pivoted_data['patient'] = pivoted_data['patient'].str.split('_').str[0]
    pivoted_data.rename(columns={pivoted_data.columns[1]: 'group'}, inplace=True)
    pivoted_data.drop(columns=[col for col in pivoted_data.columns if col.startswith('group_') and col != 'group'], inplace=True)

    return pivoted_data