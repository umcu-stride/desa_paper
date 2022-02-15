from collections import defaultdict
from typing import  Sequence, Union, List
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def _get_indexes(df, donor_type, desa_spec:Union[str, Sequence]=None):
    
    # donor_type = 'Deceased'
    desa_spec = set([desa_spec]) if isinstance(desa_spec, str) else desa_spec
    if 'TypeOfDonor_NOTR' in df.columns:
        ind_dead = df.TypeOfDonor_NOTR == donor_type 
    else:
        ind_dead = df.Donor_Type == donor_type

    ind_desa = df.DESA_Status == 'DESA'
    ind_T1 = ~ind_desa & ind_dead
    if desa_spec:
        ind_desa_spec = df.DESA.apply(lambda x: bool(desa_spec & x))
        ind_T2 = ind_desa & ~ind_desa_spec & ind_dead
        ind_T3 = ind_desa & ind_desa_spec & ind_dead
    else: 
        ind_ndesa = df['#DESA'].apply(lambda x:  x <= 2) 
        ind_T2 = ind_desa & ind_ndesa & ind_dead
        ind_T3 = ind_desa & ~ind_ndesa & ind_dead

    return ind_T1, ind_T2, ind_T3

def create_treatment_grups(data, desa_spec:Union[str, List[set]]=None):
    df = data.copy(deep=True)
    desa_spec = [set([desa_spec])] if isinstance(desa_spec, str) else desa_spec
    if desa_spec:
        if len(desa_spec) == 1:
            df = df.assign(
                No_DESA = df['DESA'].apply(lambda x: 0 if x else 1),
                Specific_DESA = df['DESA'].apply(lambda x: 1 if x & desa_spec[0] else 0),
                Other_DESA =  df['DESA'].apply(lambda x: 1 if (x and not x & desa_spec[0]) else 0),
            )
            return df.assign(
                Groups = df[['No_DESA', 'Specific_DESA', 'Other_DESA']].values.argmax(1) + 1
            )
        elif len(desa_spec) == 2:
            df = df.assign(
                No_DESA = df['DESA'].apply(lambda x: 0 if x else 1),
                Specific_DESA_1 = df['DESA'].apply(lambda x: 1 if x & desa_spec[0] else 0),
                Specific_DESA_2 = df['DESA'].apply(lambda x: 1 if x & desa_spec[1] else 0),
                Other_DESA =  df['DESA'].apply(
                    lambda x: 1 if (x and not x & desa_spec[0] and not x & desa_spec[1]) else 0
                ),
            )
            return df.assign(
                Groups = df[['No_DESA', 'Specific_DESA_1', 'Specific_DESA_2', 'Other_DESA']].values.argmax(1) + 1
            )
        else:
            raise ValueError('Specific set of DESA can not be larger than 2')
    df = df.assign(
        No_DESA = df['DESA'].apply(lambda x: 0 if x else 1),
        DESA_12 = df['#DESA'].apply(lambda x: 1 if x <= 2 else 0),
        DESA_3orMore = df['#DESA'].apply(lambda x: 1 if x >= 3 else 0),
    )
    return df.assign(
        Groups = df[['No_DESA', 'DESA_12', 'DESA_3orMore']].values.argmax(1) + 1
    )

def feature_scaler(df, scaler='standard'):
    num_vars = ['RecipientAge_NOTR', 'DonorAge_NOTR', 'CIPHour_DBD', 'CIPHour_DCD']
    for var in num_vars:
        try:
            if scaler == 'standard':
                mean = df[var].mean()
                std = df[var].std()
                df[var] = df[var].apply(lambda x: (x - mean) / std)
            if scaler == 'maxmin':
                max, min = df[var].max(), df[var].min()
                df[var] = df[var].apply(lambda x: (x - min) / (max - min))
        except KeyError:
            pass
    return df

def find_ipw(df, confounders, treatments, verbose=False) -> pd.Series:
    """ This method finds the inverse probability weights use in causal inference per treatment group """    
    # scale the features
    df = feature_scaler(df, scaler='maxmin')
    
    # Propensity model
    formulas = [treatment + ' ~ ' + ' + '.join(confounders) for treatment in treatments]
    models = [
        smf.glm(formula=formulas[i], data=df, family=sm.families.Binomial())
        .fit() for i in range(len(treatments))
    ]

    # Propensity scores 
    propensity_scores = np.array([model.fittedvalues for model in models])
    p = propensity_scores/sum(propensity_scores)

    if verbose:
        print(summary_col(models))
        # print('sum propensity score', sum(propensity_scores))

    # Calculate the weights
    df['w'] = sum([(df[treatment]==1) / p[i] for i, treatment in enumerate(treatments)])
    return df

def kaplan_meier_curves(df:pd.DataFrame, 
                        desa_spec:Union[str, Sequence]=None,
                        donor_type:str='Deceased',
                        labels:list=None,
                        adjust=False):

    assert 'Groups' in df.columns, 'Groups column is missing'
    if 'GraftSurvival10Y_R' in df.columns:
        failure, time = 'FailureCode10Y_R', 'GraftSurvival10Y_R'
    else:
        failure, time = 'Failure', 'Survival[Y]'

    ind_T1, ind_T2, ind_T3 = _get_indexes(df, donor_type=donor_type, desa_spec=desa_spec)

    T = [df.loc[ind_T1, time], df.loc[ind_T2, time], df.loc[ind_T3, time]]
    E = [df.loc[ind_T1, failure], df.loc[ind_T2, failure], df.loc[ind_T3, failure]]
    if adjust:
        assert 'w' in df.columns, 'Weight column is missing'
        W = [df.loc[ind_T1, 'w'], df.loc[ind_T2, 'w'], df.loc[ind_T3, 'w']]
    else:
        W = [None] * 3
    t = np.linspace(0, 10, 1000)
    kmfs = [
        KaplanMeierFitter(label=labels[i])
        .fit(T[i], event_observed=E[i], timeline=t, weights=W[i]) for i in range(len(T)) 
    ]
    p_value = multivariate_logrank_test(
        df['GraftSurvival10Y_R'], df['Groups'], df['FailureCode10Y_R']
        ).p_value
    return kmfs, p_value
    
def plot_kaplan_meier_curve(kmfs:list, 
                            p_value:float=None,
                            title:str=None,
                            ci_show:bool=False, 
                            grid:bool=False, 
                            add_counts:bool=True,
                            ax=None,
                            fontsize:int=15,
                            legend_size = 10,
                            xlabel:str='Years after Tranplantation',
                            ylabel:str='Graft Survival (%)',
                            verbose:bool=False):
    """ kmfs: is a list of KaplanMeierFitter instances already fitted """

    colors = ['black', 'blue', 'red']
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    for i, kmf in enumerate(kmfs):
        kmf.survival_function_ = kmf.survival_function_ * 100 
        kmf.plot(ci_show=ci_show, color=colors[i], ax=ax) 

    # Add at-risk, censored, and failure statistics
    if add_counts:
        add_at_risk_counts(*kmfs, ax=ax, fontsize=15)

    # add p-value 
    if p_value:
        p_value_string = 'p < 0.0001' if p_value < 0.0001 else f'p = {p_value}'
        ax.add_artist(AnchoredText(p_value_string, loc=4, frameon=False))

    # Other figure settings
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(prop={"size":legend_size})
    ax.grid(grid)
    ax.set_ylim(0.2)
    # plt.tight_layout()
    # plt.show()

    # Final values
    if verbose:
        kmfs_final = [kmf.survival_function_.values[-1][0] for kmf in kmfs]
        print(f'Top curve: {kmfs_final[0]:.2f}, Middle curve: {kmfs_final[1]: .2f}, Lower curve:{kmfs_final[2]: .2f}')
        print(f'10-Year Gap is: {kmfs_final[0] - kmfs_final[2] : .2f}')
        kmfs_100 = [kmf.survival_function_.values[100][0] for kmf in kmfs]
        print(f'1-Year Gap is: {kmfs_100[1] - kmfs_100[2] : .2f}')

def plot_scatter_diff(dataframe, desa_sets:list, confounders:list, donor_type:str='Deceased', adjust:bool=True, grid:bool=False):
    time_points = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 980]
    diffs = defaultdict(lambda : defaultdict(list))
    treatments = ['No_DESA', 'Other_DESA', 'Specific_DESA']
    for i, desa_set in enumerate(desa_sets):
        for desa in desa_set:
            # Compute weighted km curves )
            df_tret_group = create_treatment_grups(dataframe, desa)
            df_weight = find_ipw(df_tret_group, confounders, treatments, verbose=False)
            kmfs, p_value = kaplan_meier_curves(df_weight, desa, donor_type=donor_type, labels=treatments)
            # Subtract all DESA from Specific DESA
            diff = kmfs[1].subtract(kmfs[2])
            diffs[i]['x'].append(diff.index[time_points])
            diffs[i]['y'].append(diff.values[time_points])

    fig, ax = plt.subplots(figsize=(22, 10))
    colors = ['red', 'yellow', 'green']
    labels = [ 'Relevant DESA [Bad]', 'Irrelevant DESA', 'Relevant DESA [Good]']
    for i in range(len(desa_sets)):
        ax.scatter(diffs[i]['x'], diffs[i]['y'],  color=colors[i], label=labels[i], linewidth=2.5, alpha= 0.25)
    plt.axhline(0.1, color='black', linestyle='dashed')
    plt.axhline(-0.1, color='black', linestyle='dashed')
    ax.set_xlabel('Years after Tranplantation', fontsize=20)
    ax.set_ylabel('Diff Graft Survival (%)', fontsize=20)
    ax.legend(prop={"size":15})
    ax.grid(grid)
    plt.tight_layout()
