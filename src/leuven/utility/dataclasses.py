from dataclasses import dataclass, InitVar, field
from typing import Set, List
from src.leuven.utility.constants import MOL2SERO_CLASS_I, MOL2SERO_CLASS_II, HLA_SPLITS

@dataclass
class HLA:
    """ Class for all High Resolution HLA's """
    string: str = field(init=True, repr=False)
    gene: str = field(init=False, repr=False)
    allele: str = field(init=False, repr=False)
    protein: str = field(init=False, repr=False)
    _class: str = field(init=False, repr=False)
    locus: str = field(init=False, repr=False)
    high_res: str = field(init=False, repr=True)
    low_res: str = field(init=False, repr=True)

    def __post_init__(self):
        self.string = self.string.strip()
        self.gene, specificity = self.string.split('*')
        _spec = specificity.replace('G','').split(':')
        # filter higher resolutions with 2 or more ':'
        self.allele, self.protein = _spec[0], _spec[1]
        self.high_res = self.gene + '*' + self.allele + ':' + self.protein
        assert self.high_res == self.string.strip(), f'Error {self.high_res} != {self.string}' 
        self._class = 'I' if self.gene in ['A', 'B', 'C'] else 'II'
        self.locus = {'Cw': 'C', 'Bw': 'B'}.get(self.gene[0:2], self.gene[0:2])
        self.locus = self.gene[0:2]
        try:
            if self._class == 'I':
                self.low_res = MOL2SERO_CLASS_I[self.high_res]
                self.low_res = HLA_SPLITS.get(self.string) if self.low_res in HLA_SPLITS.keys() else self.low_res 
            if self._class == 'II':
                if self.gene in ['DQA1', 'DPA1', 'DPB1']:
                    self.low_res = None
                else: 
                    self.low_res = MOL2SERO_CLASS_II[self.high_res]
        except Exception as e:
            print(f'HLA {self.high_res} throws exception {e}')

@dataclass
class Epitope():
    ep: str = field(repr=True)
    hla: HLA = field(repr=True)

@dataclass
class DSA():
    """ Class contatining DSA data """
    hlas: List[HLA] = field(repr=False, default_factory=list)
    num: int = field(init=False)

    def __post_init__(self):
        self.num = self.__len__()

    def __iter__(self):
        return iter({hla.low_res for hla in self.hlas})

    def __len__(self):
        return len(self.hlas)

@dataclass
class DESA():
    """ Class contatining DESA data """
    epitopes: List[Epitope] = field(repr=False, default_factory=list)
    num: int = field(init=False)

    def __post_init__(self):
        self.num = self.__len__()

    def __len__(self):
        return len(self.epitopes)

    def __getitem__(self, key):
        return self.epitopes[key]

    def __iter__(self):
        for epitope in self.epitopes:
            yield (epitope.ep, epitope.hla.high_res)

    def to_dict(self, key:str='epitope'):
        """ key takes either epitope or hla as index """
        if key == 'epitope':
            return {epitope.ep:epitope.hla.high_res for epitope in self.epitopes}
        if key == 'hla':
            _dict = defaultdict(set)
            for epitope in self.epitopes:
                _dict[epitope.hla.high_res].add(epitope.ep)
            return dict(_dict)
    
    def get_epitopes(self):
        return {epitope.ep for epitope in self.epitopes}

@dataclass
class Participant():
    """ Either patient or donor """
    hla_str: InitVar
    hlas: List[HLA] = field(default_factory=list)

    def __post_init__(self, hla_str):
        if not hla_str:
            raise ValueError('HLA string is missing')
        self.hlas  = [HLA(hla) for hla in hla_str.split(', ') if '*' in hla]

    def get_hla(self, which=None):
        if which == 'high_res':
            return {hla.high_res for hla in self.hlas}
        if which == 'low_res':
            return {hla.low_res for hla in self.hlas if hla.low_res}

@dataclass
class Luminex():
    """ Class contatining Luminex data """
    _specificity: InitVar[set] = field(repr=False)
    assignment: str = field(repr=True)
    hlas: Set[HLA] = field(init=False, default_factory=list, repr=True)

    def __post_init__(self, _specificity):
        if isinstance(_specificity, str):
            self.hlas  = [HLA(hla) for hla in _specificity.split(',')]
        if isinstance(_specificity, set):
            self.hlas  = [HLA(hla) for hla in _specificity]


@dataclass
class Transplant():
    """ Class containing all the transplants """
    id: int = field(init=True, repr=True)
    donor: Participant = field(init=True, repr=False)
    recipient: Participant = field(init=True, repr=False)
    lsa: List[Luminex] = field(init=True, repr=False, default_factory=list)
    dsa: DSA = field(init=False, repr=False, default_factory=list)
    desa: DESA = field(init=False, repr=False, default_factory=list)
    