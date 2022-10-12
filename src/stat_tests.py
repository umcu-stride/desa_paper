from scipy.stats import mannwhitneyu, chi2_contingency

def mannwhitneyu_test(data, column,  **kwargs):
    desa_group = data[data.DESA_Status == 'DESA'][column]
    print(round(desa_group.mean(), 2), round(desa_group.std(), 2))
    no_desa_group = data[data.DESA_Status == 'No DESA'][column]
    print(round(no_desa_group.mean(), 2), round(no_desa_group.std(), 2))
    print(mannwhitneyu(desa_group, no_desa_group,  **kwargs))

def chi2_test(data, column):
    # https://pythonfordatascienceorg.wordpress.com/chi-square-python/
    desa_group = data[data.DESA_Status == 'DESA'][column]
    print(desa_group.value_counts())
    no_desa_group = data[data.DESA_Status == 'No DESA'][column]
    print(no_desa_group.value_counts())

    print(chi2_contingency(desa_group, no_desa_group))