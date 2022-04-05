import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from joblib import dump

from lifelines import CoxPHFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from src.model_build.common.pipe_store import (
    data_loading,
    start_pipeline,
    keeping_features,
    eng_immunological_features,
    integer_encoder,
    one_hot_encoder,
    set_time_event_label,
    censoring_deaths,
    setting_prediction_horizon,
    scikit_lifeline_adapter,
    scikit_survival_adapter,
)

from src.model_build.common.model_utility import (
    numeric_features,
    CustomScaler,
    CustomPolyFeature
)


# path = '/Users/Danial/Repos/STRIDE/STRIDE-Analytics/data/20210325_pipeline_desa_extended.pickle'
path = '/Users/Danial/Repos/STRIDE/STRIDE-Analytics/data/20210614-mismatch_ep_db-extended.pickle'

donor_type ='Deceased'
antibody_epitope = True

X, y = (
    data_loading(path)
    .pipe(start_pipeline, donor_type,  antibody_epitope)
    .pipe(keeping_features,
        'Failure',
        'Survival[Y]',
        'DESA',
        'EpvsHLA_Donor',
        # 'Epitope_Mismatch',
        'CIPHour_DCD',
        'CIPHour_DBD',
        'Retransplant',
        'DialysisYears',
        'DonorAge_NOTR',
        'RecipientAge_NOTR',
        # 'TypeCadaveric_NOTR',
        'PrimaryDiseaseGroups'
    )
    .pipe(integer_encoder, 'Retransplant') #'TypeCadaveric_NOTR'
    # .pipe(integer_encoder, 'Retransplant', 'TypeCadaveric_NOTR' )
    .pipe(one_hot_encoder, 'PrimaryDiseaseGroups', 'DiabetesType_I')
    .pipe(eng_immunological_features, antibody_epitope=antibody_epitope)
    .pipe(set_time_event_label, E='Failure', T='Survival[Y]')
    .pipe(censoring_deaths)
    .pipe(setting_prediction_horizon, 1)
    .pipe(scikit_lifeline_adapter)
)

print(X)
X = X.drop('Epitopes', axis=1)
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

numeric_features = numeric_features.get(donor_type)

CoxRegression = sklearn_adapter(
    CoxPHFitter,
    event_col='E',
)

pipe = Pipeline(
    [
        ('scaler', CustomScaler(StandardScaler, *numeric_features)),
        # ('Poly Feature', CustomPolyFeature('DonorAge_NOTR', 'RecipientAge_NOTR')),
    ], verbose=True
)

X_train_ts = pipe.fit_transform(X_train)
print(X_train_ts)
cox_model = CoxRegression(n_baseline_knots=3)
cox_model.fit(X_train_ts, y_train)
cox_model.lifelines_model.print_summary()
print('Training Score:', round(cox_model.score(X_train_ts, y_train)*100, 2))
print('Test Score:',  round(cox_model.score(pipe.transform(X_test), y_test)*100, 2))
# joblib.dump(pipeline, 'pipeline.pkl')


# from sksurv.ensemble import RandomSurvivalForest
# X_train_ts = pipe.fit_transform(X_train)
# rsf = RandomSurvivalForest(n_estimators=100,
#                            min_samples_split=10,
#                            min_samples_leaf=15,
#                            max_features="sqrt",
#                            n_jobs=-1,
#                            random_state=random_state)
# rsf.fit(X_train_ts, y_train)

# print(rsf.score(pipe.transform(X_test), y_test))

