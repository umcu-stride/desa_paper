import logging
from functools import wraps
import datetime as dt
import pandas as pd
import numpy as np
from typing import List, Literal
from src.constants import (
    Epitope_DB,
    RELEVANT_DESA_BAD, 
    RELEVANT_DESA_GOOD,
    RELEVANT_DESA_GOOD_OLD,
)
from src.utility import get_hla_class, flatten2list, flatten2string, epitopeseq2str, sequence2string
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str((dt.datetime.now() - tic).total_seconds())
        logger.info(f"Step: {func.__name__} | Shape: {result.shape} | Computation Time: {time_taken}s")
        return result
    return wrapper

@logging
def data_loading(path):
    return pd.read_pickle(path)

@logging
def start_pipeline(dataf:pd.DataFrame, status:str, donor_type:str=None):
        """ 
        This function split the data set to 4 categories
            1. Deceased Donor with antibody epitopes (DESA)
            2. Deceased Donor without antibody epitopes 
            3. Living Donor with antibody epitopes
            4. Living Donor without antibody epitopes
        If not input is given the whole data frame is used without any split

        Parameters:
        -----------
        donor_type: str ['Living', 'Deceased'] 
            Type of the donor
        
        # antibody_epitope: bool
        #     Presence or absence of antibody epitope
        """
        df = dataf.copy()
        
        logging_indet = 6
        logger.info(
            f'- selceted cohort --> Donor Type: {donor_type}, Epitope Antibody Presence: {status}'
            .rjust(logging_indet)
        )

        if not donor_type:
            ind_dead = df.index
        else:
            ind_dead = df.Donor_Type == donor_type

        if status == 'All':
            ind_desa = True
        elif status == 'DESA':
            ind_desa = df.Status == 'DESA'
        elif status == 'No DESA':
            ind_desa = df.Status != 'DESA'

        ind = ind_dead & ind_desa
        
        return (
                df[ind].drop(['Status'], axis=1) if 'Status' in df.columns else df[ind].drop(['DESA_Status'], axis=1) 
                .reset_index(drop=True)
        )

def keeping_features(df:pd.DataFrame, *cols:str):
    """ Keeping the columns in the col """
    
    return df[list(cols)]


from typing import Sequence
def epitope_preparation(epitopes:Sequence):
    """ Make sentence from all epitopes in a DB and then joint them in a sentence """

    if isinstance(epitopes, dict):
        epitopes = set(epitopes.keys())
    _dict = {db : epitopeseq2str(epitopes & Epitope_DB[db]) for db in ['ABC', 'DR', 'DQ']}
    return sequence2string(list(_dict.values()))


def features_from_antibody_epitopes(df):
    def get_class(x):
        return { get_hla_class(hla.split('*')[0]) for hla in x.values() }
    df = df.assign(
        No_DESA = df['DESA'].apply(lambda x: 0 if len(x) else 1),
        Relevant_DESA_Bad = df['DESA'].apply(lambda x: int(bool(x & RELEVANT_DESA_BAD))),
        Relevant_DESA_Good = df['DESA'].apply(lambda x: int(bool(x & RELEVANT_DESA_GOOD))),
        Class_I = df.EpvsHLA_Donor.apply(
                    lambda x: int( ('I' in get_class(x)) & ('II' not in get_class(x)) ) 
                ),
        Class_II = df.EpvsHLA_Donor.apply(
                    lambda x: int( ('II' in get_class(x)) & ('I' not in get_class(x)) ) 
                ),
        Class_I_II  = df.EpvsHLA_Donor.apply(
                    lambda x: int( ('I' in get_class(x)) & ('II' in get_class(x)) ) 
                ),
        # Epitopes = df['DESA'].apply(epitope_preparation),
    )
    df.drop(['EpvsHLA_Donor'], axis=1, inplace=True)
    return df

def features_from_mismatch_epitopes(df):
    df = df.assign(
        ep_mismatch = df['Epitope_Mismatch'].apply(lambda x: 1 if len(x) else 0),
        Epitopes = df['Epitope_Mismatch'].apply(epitope_preparation)
    )
    return df.drop(['Epitope_Mismatch'], axis=1)

@logging
def eng_immunological_features(df:pd.DataFrame, antibody_epitope:bool) -> pd.DataFrame:
    """
    Engineering immunological features based on the presence of antibody epitopes. In the presence of 
    antibody epitopes relevance of the eptitopes is important. While in their absence presence or absence of
    epitope mismatch is important

    Parameters:
    ----------- 
    antibody_epitope: bool
        Presence or absence of antibody epitope
    """

    if antibody_epitope:
        return features_from_antibody_epitopes(df)
    return features_from_mismatch_epitopes(df)

@logging
def one_hot_encoder(df:pd.DataFrame, col, *cats):
    """
    Convert categorical variable into dummy variables and keeps the given categorical columns. 
    Under the hood uses pandas get_dummies method

    Parameters:
    -----------
    col: The specific column to be one hot encoded
    
    cats: the categories/columns that will be kept after one hot encoding.

    Example:
    --------
    >> s = pd.Series(list('abca'))
    >> pd.get_dummies(s)
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0
    """
    
    encoded_df = pd.get_dummies(df[col])    
    df.drop(col, axis=1, inplace=True)
    if not cats:
        return pd.concat([df, encoded_df], axis=1)
    cats = list(cats)    
    return pd.concat([df, encoded_df[cats]], axis=1)

@logging
def integer_encoder(df:pd.DataFrame, *cols):
    """
    Encode the object as an enumerated type, i.e. Integer Encoding. This function is particularly 
    used for features with 2 category. Under the hood uses pandas factorize method. For Features 
    with more categories consider one_hot_encoder.

    Parameters:
    -----------
    cols: Name of the column/feature in the dataset
    
    Example:
    --------
    >> pd.factorize(['b', 'b', 'a', 'c', 'b'])
    (
        array([0, 0, 1, 2, 0]...),
        array(['b', 'a', 'c'], dtype=object)
    )
    """

    if not cols:
        raise ValueError('Name of the dataset column is missing')

    for col in cols:
        if df[col].dtype == 'object':
            df[col], _ = pd.factorize(df[col])
    return df

@logging
def set_time_event_label(df:pd.DataFrame, E='Failure', T='Survival[Y]'):
    """ 
    Set the time and event label to T and E respectively 
    
    Parameters:
    -----------
    E: The event column in the dataset
    T: The time column in the dataset
    """

    df['E'] = df['Failure'].astype(bool)
    df['T'] = df['Survival[Y]']
    df.drop([E, T], axis=1, inplace=True)
    return df

def polynomial_power2(df:pd.DataFrame, *cols:str):
    for col in list(cols):
        df[col + '_power2'] = df[col] *  df[col]
    return df

@logging
def censoring_deaths(df:pd.DataFrame):
    """
    Transplants where patient dies with functioning graft, i.e. 2 is changed to no failure, i.e. 0
    Method set_time_event_label should be run beforehand
    """

    return df.assign(
        E=df['E'].map({0:0, 1:1, 2:0})
    )


@logging
def setting_prediction_horizon(df:pd.DataFrame, time:int, E='E', T='T'):
    """ 
    Setting the prediction horizon, i.e. keep events for T < year & make event to zero 
    for T > year. Method set_time_event_label should be run beforehand
    
    Parameters:
    -----------
    time: in Year
    E: event column
    T: survival time
    """

    ind = df[T].apply(lambda x: x <= time)
    return  df.assign(
        E = ind * df[E],
        T = df[T].apply(lambda x: x if x <= time else time)
    )


def scikit_lifeline_adapter(df:pd.DataFrame):
    """ Splits the data frame to input matrix and output vector suitable for lifeline sklearn adapter """

    X = df.drop('T', axis=1)
    y = df.pop('T')
    return (X, y)


def scikit_survival_adapter(df:pd.DataFrame):
    """ Splits the data frame to input matrix and output vector suitable for scikit survival """

    X = df.drop(['E', 'T'], axis=1)
    y = np.array(list(zip(df['E'], df['T'])), np.dtype('bool, float'))
    return (X, y)


def classification_adapter(df:pd.DataFrame, time:float=1, drop_censored=True):
    """ Transforms the survival analysis problem into a classification problem by creating a lable
    column L where
        - 1 denotes failed transplants within time horizon T
        - 0 denotes not-failed transplants within time horizon T  
        - 99 denotes censored transplants within time horizon T 

    Parameters:
    -----------
    df: dataframe
    time: Prediction horizon in Year
    drop_censored: Bool
        Drop the censored transplats denoted by 2 
    """

    df['L'] =  df[['E', 'T']].apply(lambda x: 
                    1 if (x[0] == 1) & (x[1] <= time) else 99 if (x[0] == 0) & (x[1] < time) else 0, 
                    axis=1,
                )
    if drop_censored:
        df = df[df.L != 99]
    X = df.drop(['E', 'T', 'L'], axis=1)
    y = df['L']
    return (X, y)

def feature_scaler(df, cols:list, scaler:Literal['standard', 'maxmin']='standard'):
    """ 
    Scales the given data frame features based on the selected scaler

    Parameters:
    -----------
    scaler: str
    Options are standard & maxmin
    """
    for var in cols:
        try:
            if scaler == 'standard':
                mean = df[var].mean()
                std = df[var].std()
                df[var] = df[var].apply(lambda x: (x - mean) / std)
            if scaler == 'maxmin':
                max, min = df[var].max(), df[var].min()
                df[var] = df[var].apply(lambda x: (x - min) / (max - min))
        except KeyError:
            print(f'Variable {var} does not exist in the data frame columns')
            pass
    return df

def dataframe_to_dataset(X, y):
    import tensorflow as tf
    
    X, y = X.copy(), y.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
    ds = ds.shuffle(buffer_size=len(X))
    return ds



def df_inv_val_split(df, val_size:float=0.25):
    msk = np.random.rand(len(df)) < val_size
    return (df[~msk], df[msk])