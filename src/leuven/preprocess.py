""""""
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Literal


def summerize2allel(df):
    cols = list(set(df.columns) - set(['tx_id', 'Bead', 'Raw Value', 'BCM', 'BCR', 'AD-BCR', 'Assignment' ]))
    return df.assign(allele = 
        df[cols]
        .dropna(axis=1, how='all')
        .fillna(' ')
        .apply(lambda x: {val.strip() for val in x if '*' in val}, axis=1)
        .apply(lambda x: ', '.join(x) if x else np.nan)
    ).drop(cols, axis=1)

def merge_lsa(path_i, path_ii):
    paths = [path_i, path_ii]
    cols_i = ['Bead', 'Raw Value', 'BCM', 'BCR', 'AD-BCR', 'Assignment', 'A', 'B', 'C']
    cols_ii = ['Bead', 'Raw Value', 'BCM', 'BCR', 'AD-BCR', 'Assignment', 'DR/DR5x', 'DQA', 'DQB', 'DPA',  'DPB']
    hla_cols = [cols_i, cols_ii]
    dfs = [(pd.read_csv(path, skiprows=13)
            .rename(columns=lambda x: x.strip())[hla_cols[i]]
            .dropna(axis=0, how = 'all')
            )  for i, path in enumerate(paths) if path
    ]

    df = (
        pd.concat(dfs, axis=0)
        .reset_index(drop=True)
    ) 
    return df

def aggregate_lsa_per_patients(idvspath_csv):
    """ Aggregate the Luminex Single Antigen data per patient in single data frame """
    dfs = defaultdict()
    for id, paths in idvspath_csv.items():
        try:
            dfs[id] = (
                    merge_lsa(*paths)
                    .pipe(drop_samples_with_na, how='most')
                    .pipe(summerize2allel)
                    .pipe(flag_missing_values, 'Assignment')  
                    .pipe(add_id, id)
                    .pipe(strip_column, 'Assignment')
                    )
        except Exception as e:
            print(id, e)
    return pd.concat([df.reset_index(drop=True) for df in dfs.values()])

def aggregate_lsa_excel():
    excel_files = [
        ('LSA2 2838.xlsx', 2838), ('LSA2-2801.xlsx', 2801), 
        ('LSA2-2835.xlsx', 2835), ('LSA2-2896.xlsx', 2896),
        ('LSA2-2907.xlsx', 2907)
    ]
    excel_paths = [('~/UMCUtrecht/Leuven/Original/lsa_per_tx/' + file, id) for file, id in excel_files] 
    dfs = [
        pd.read_excel(path, skiprows=2) 
        .rename(columns={'Raw': 'Raw Value', 'AD- BCR':'AD-BCR', 'Assign': 'Assignment'})
        .dropna(axis=0, how = 'all')
        .pipe(add_id, id)
        for path, id in excel_paths]
    return (
            pd.concat(dfs)[['tx_id', 'Bead', 'Raw Value', 'BCM', 'BCR', 'AD-BCR', 'Assignment', 'DR/DR5x', 'DQA', 'DQB', 'DPA', 'DPB']]
            .pipe(summerize2allel)
            .pipe(flag_missing_values, 'Assignment')  
            .pipe(flag_missing_values, 'allele')  
            .pipe(strip_column, 'Assignment')
            .rename(columns={'Assignment': 'assignment'})
    )

def flag_missing_values(df, col):
    assert df[col].isna().sum() == 0, f'Column {col} has missing values'
    return df

def clean_index2int(df, col_index):
    import re
    df[col_index] = df[col_index].apply(lambda x: int(re.findall(r'\d+', str(x))[0]))
    return df

def drop_samples_with_na(df, how=Literal['most', 'all']):
    if how == 'all':
        return df.dropna(axis=0, how = 'all')
        
    if how == 'most':
        imp_cols = ['Bead', 'Raw Value', 'BCM',	'BCR', 'AD-BCR', 'Assignment']
        i = df[df[imp_cols].isnull().sum(axis=1) > 3].index
        if len(i):
            print(f'Indexes with mostly NaN #:{len(i)}, index:{i}')
        return df.drop(i)
        

def prepare_merged_desa(df, save_path=None):
    from src.constants import RELEVANT_DESA_BAD
    desa = df[['Tx_id', 'Epitope_Mismatch', 'DESA_Status', 'DESA_info']]
    desa = desa.assign(
        EpvsHLA_Donor = desa['DESA_info'].apply(lambda x: x.to_dict(key='epitope')),
        DESA = desa['DESA_info'].apply(lambda x: x.get_epitopes()),
        DESA_num = desa['DESA_info'].apply(len),
    ).drop('DESA_info', axis=1)
    desa = desa.assign(Relevant_DESA = desa['DESA'].apply(lambda x: 1 if (set(x) & RELEVANT_DESA_BAD) else 0))
    # Merge with Tx data
    tx = pd.read_excel('/Users/Danial/UMCUtrecht/Leuven/original/LEUVEN cohort_AS.xlsx')
    cols = ['TX ID number', 'surv_time_yr', 'donor_age', 'rec_age', 'donor_type', 'death_censor', 
            'repeat_tx', 'induction', 'CIT', 'donor_LD', 'donor_DCD', 'donor_DBD']
    desa = desa.merge(tx[cols].rename(columns={'TX ID number':'Tx_id'}), on='Tx_id')

    if save_path:
        if save_path.split('.')[-1] == 'csv':
            desa.to_csv(save_path, index=False)
        if save_path.split('.')[-1] == 'pickle':
            desa.to_pickle(save_path)
        else:
            raise ValueError(f'save_path recognized extentions are csv & pickle')
    return desa

def strip_column(df, *cols):
    for col in list(cols):
        df[col] = df[col].apply(str.strip)
    return df

def select_cols(df, *cols):
    return df[list(cols)]

def add_id(df, id):
    df['tx_id'] = id
    return df