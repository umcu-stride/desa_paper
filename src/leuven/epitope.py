""" This file contains all the methods consumed by Epitope Data Base """
import os
from collections import defaultdict
from typing import Union, List
import pandas as pd
from src.leuven.utility.common import flatten2set


class EpitopeRegistry:
    """ This is a class that entails the data base [Pandas DataFrame] of all epitopes and
        all the related methods tha can be applied to this data base  """

        # the path is consistent if dash_hla_3d/app.py is ran
    def __init__(self, path:str=None):
        if not path:
            pass
            path = '~/Repos/STRIDE/dash_hla_3d/data/EpitopevsHLA.pickle'
            # path = '~/Repos/STRIDE/20200804_EpitopevsHLA_distance.pickle'
        self.path = os.path.expanduser(path)
        self.df = pd.read_pickle(self.path) # pylint: disable=invalid-name
        # Get hlas with pdb files
        self._hlavsep = None
        self._hlavsep_df = None

    def __repr__(self):
        return f""" Epitope_DB(records={len(self.df)}, columns={self.df.columns}) """

    def __iter__(self):
        return iter(set(self.df.Epitope.values))

    def filter_mAb(self):
        ind = self.df.mAb == 'Yes'
        self.df = self.df[ind]
        return self

    def is_IgG(self):
        self.filter_isotype()
        if len(self.df != 0):
            return True
        else:
            return False

    def filter_isotype(self, isotype:str='IgG'):
        ind = self.df.isotype.apply(lambda x: isotype in x)
        self.df = self.df[ind]
        return self

    def get_epitopes(self, value: Union[str, List[str]]):
        """ get epitope info from the df
        value: can be str or a list of strings """

        if isinstance(value, str):
            ind = self.df.Epitope == value
        else:
            ind = self.df.Epitope.apply(lambda x: x in value)
        self.df = self.df[ind]
        return self

    def ellipro(self, value):
        """ filter EpitopeDB based on desired ellipro score """
        if isinstance(value, str):
            value = set([value])
        ind = self.df['ElliPro Score'].apply(lambda x: x in value)
        self.df = self.df[ind]
        return self

    def hlavsep(self,
                hla_allel:str='Luminex Alleles') -> pd.DataFrame:
        """ returns a DataFrame of HLA vs epitopes
        hla_allel [default is 'Luminex Alleles']: determines the allel type
        only_with_pdb [default is False]: If True, includes only luminex allels that pdb file
        is available:
        only_with_pdb: Include only Luminex Alleles that pdb file is available
        { 'HLA' : {'epitopes'}} """

        hlas = flatten2set(self.df[hla_allel].values)

        hlavsep_dict = defaultdict(list)
        for hla in hlas:
            ind = self.df[hla_allel].apply(lambda x: hla in x)
            epitopes = flatten2set(self.df[ind]['Epitope'].values)
            hlavsep_dict['HLA'].append(hla)
            hlavsep_dict['Epitope'].append(epitopes)
        self._hlavsep_df = pd.DataFrame(hlavsep_dict)
        return self._hlavsep_df

    def polymorphic_residues(self, epitope:str) -> set:
        """ Gets the aminoacide sequence of one epitope from
        the polymorphic residue column """
        return self.df[self.df.Epitope == epitope].PolymorphicResidues.values[0]

    def min_hlavsep(self, epitopes:set, ignore_hla:set=set()) -> dict:
        """ Returns the HLA vs epitope dictionary
            based on minimum number of HLA possible
            ignore_hla: ignores some hla
            format { 'HLA' : {'epitopes'} }
        """
        # Deep copy of epitopes set for later epitope removal
        _epitopes = epitopes.copy()
        hlavsep_df = self.hlavsep(only_with_pdb=True, ignore_hla=ignore_hla)
        hla_ep = defaultdict(set)
        for _ in range(10):
            intersect = hlavsep_df.Epitope.apply(
                lambda x: len(x.intersection(_epitopes))
            )
            # find the indexes with maximum value
            value_max = intersect.max()
            if value_max:
                ind_maxes = intersect == value_max
                hlavsep_max_df = hlavsep_df[ind_maxes]
                max_hla = hlavsep_max_df.HLA.values.tolist()[0]
                ind_max = hlavsep_max_df.HLA == max_hla
                set_of_ep = hlavsep_max_df[ind_max].Epitope.values[0].intersection(_epitopes)
                hla_ep[max_hla] = set_of_ep
                _epitopes.difference_update(set_of_ep)
        return dict(hla_ep)

    def epvshla2hlavsep(self, epvshla:dict) -> dict:
        """ Transform an ep vs hla dict 2 hla vs ep dict """
        hlavsep = defaultdict(set)
        for epitope, hla in epvshla.items():
            hlavsep[hla].add(epitope)
        return hlavsep

if __name__ == '__main__':

    epitope = EpitopeRegistry()
    print(epitope.df)
    print(epitope.hlavsep(hla_allel='All Alleles'))

