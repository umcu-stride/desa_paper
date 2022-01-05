from collections import defaultdict
from typing import  Sequence, Union
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def create_treatment_grups(df, desa_spec:set=None):
    if desa_spec:
        return df.assign(
            No_DESA = df['DESA'].apply(lambda x: 0 if x else 1),
            Specific_DESA = df['DESA'].apply(lambda x: 1 if x & desa_spec else 0),
            Other_DESA =  df['DESA'].apply(lambda x: 0 if x & desa_spec else 1),
        )
    return df.assign(
            No_DESA = df['DESA'].apply(lambda x: 0 if x else 1),
            DESA_12 = df['#DESA'].apply(lambda x: 1 if (x > 0 & x <= 2) else 0),
            DESA_3orMore = df['#DESA'].apply(lambda x: 1 if x >= 3 else 0),
    )

def normalization(df, variable):
    mean = df[variable].mean()
    std = df[variable].std()
    return df[variable].apply(lambda x: (x - mean) / std)

def find_ipw(df, confounders, treatments):
    """ This method finds the inverse probability weights use in causal inference per treatment group """
    formulas = [treatment + ' ~ ' + ' + '.join(confounders) for treatment in treatments]
    # Propensity model
    models = [
        smf.glm(formula=formulas[i], data=df, family=sm.families.Binomial())
        .fit() for i in range(len(treatments))
    ]
    # Propensity scores 
    propensity_scores = np.array([model.fittedvalues for model in models])
    print(propensity_scores)
    p = propensity_scores/sum(propensity_scores)

    print('sum propensity score', sum(propensity_scores))
    
    # Calculate the weights
    return sum([(df[treatment]==1) / p[i] for i, treatment in enumerate(treatments)])


def kaplan_meier_curves(df:pd.DataFrame, 
                        desa_spec:Union[str, Sequence]=None,
                        donor_type:str='Deceased',
                        weights=None):
    if isinstance(desa_spec, str):
        desa_spec = set([desa_spec])

    if 'TypeOfDonor_NOTR' in df.columns:
        ind_dead = df.TypeOfDonor_NOTR == donor_type 
    else:
        ind_dead = df.Donor_Type == donor_type

    ind_desa = df.DESA_Status == 'DESA'
    ind_desa_spec = df.DESA.apply(lambda x: bool(desa_spec & x))
    ind_T1 = ~ind_desa & ind_dead
    ind_T2 = ind_desa & ~ind_desa_spec & ind_dead
    ind_T3 = ind_desa & ind_desa_spec & ind_dead

    if 'GraftSurvival10Y_R' in df.columns:
        failure, time = 'FailureCode10Y_R', 'GraftSurvival10Y_R'
    else:
        failure, time = 'Failure', 'Survival[Y]'

    T = [df.loc[ind_T1, time], df.loc[ind_T2, time], df.loc[ind_T3, time]]
    E = [df.loc[ind_T1, failure], df.loc[ind_T2, failure], df.loc[ind_T3, failure]]

    t = np.linspace(0, 10, 1000)
    labels = ['No DESA', 'Other DESA', 'Specific DESA']
    return [
        KaplanMeierFitter(label=labels[i])
        .fit(T[i], event_observed=E[i], timeline=t, weights=weights) for i in range(len(T)) 
    ]
    
def plot_kaplan_meier_curve(kmfs:list, ci_show=False):
    """ kmfs: is a list of KaplanMeierFitter instances already fitted """

    colors = ['black', 'blue', 'red']
    fig, ax = plt.subplots(figsize=(8, 7))
    [kmf.plot(ci_show=ci_show, color=colors[i]) for i, kmf in enumerate(kmfs)]
    # Num=ind_dead.sum()
    # ax.set_title(f'Specific Epitopes, {donor_type} Donor (N={Num})', fontsize=20)
    ax.set_title(f'Deceased Donor', fontsize=20)
    ax.set_xlabel('Years after Tranplantation', fontsize=20)
    ax.set_ylabel('Graft Survival (%)', fontsize=20)
    ax.legend(prop={"size":15})
    ax.set_ylim(0.2)
    plt.tight_layout()
    add_at_risk_counts(*kmfs, ax=ax, fontsize=15)
    plt.show()
    # Final values
    kmfs_final = [kmf.survival_function_.values[-1][0]*100 for kmf in kmfs]
    print(f'Top KM curve: {kmfs_final[0]:.2f}, Middle KM curve: {kmfs_final[1]: .2f}')
    print(f'10-Year Gap is: {kmfs_final[0] - kmfs_final[2] : .2f}')
    kmfs_100 = [kmf.survival_function_.values[100][0]*100 for kmf in kmfs]
    print(f'1-Year Gap is: {kmfs_100[1] - kmfs_100[2] : .2f}')

def plot_scatter_diff(desa_df, desa_sets):
    time_points = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 980]
    diffs = defaultdict(lambda : defaultdict(list))
    for i, desa_set in enumerate(desa_sets):
        for desa in desa_set:
            kmfs = kaplan_meier_curves(desa_df, desa)
            # Subtract all DESA from Specific DESA
            diff = kmfs[1].subtract(kmfs[2])
            diffs[i]['x'].append(diff.index[time_points])
            diffs[i]['y'].append(diff.values[time_points])

    fig, ax = plt.subplots(figsize=(22, 10))
    colors = ['red', 'yellow', 'green']
    labels = [ 'Relevant DESA [Bad]', 'Relevant DESA [Good]', 'Irrelevant DESA']
    for i in range(len(desa_sets)):
        ax.scatter(diffs[i]['x'], diffs[i]['y'],  color=colors[i], label=labels[i], linewidth=2.5, alpha= 0.25)
    plt.axhline(0.1, color='black', linestyle='dashed')
    plt.axhline(-0.1, color='black', linestyle='dashed')
    ax.set_xlabel('Years after Tranplantation', fontsize=20)
    ax.set_ylabel('Diff Graft Survival (%)', fontsize=20)
    ax.legend(prop={"size":15})
    plt.tight_layout()