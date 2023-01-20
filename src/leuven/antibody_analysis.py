from collections import defaultdict
from src.leuven.epitope import EpitopeRegistry
from dataclasses import dataclass, InitVar, field
from typing import Set, List
from src.leuven.utility.common import flatten2set
from src.leuven.utility.dataclasses import (
    Transplant, 
    Participant,
    DESA,
    DSA, 
    Epitope, 
    HLA,
)

class AntibodyAnalysis:
    """ Pipeline to find DESA from transplants HLA typing & Luminex Single Antigen data """

    def __init__(self, epitope_db_path=None):
        self.epitope = EpitopeRegistry(path=epitope_db_path)
        self.hlavsep = self.epitope.hlavsep(hla_allel='All Alleles')
        # other variables
        self.luminex_beads = {}
        self.mismatch_ep = set()
        self.positive_ep = set()
        self.hla_abs = set()

    def _separate_luminex_hla_beads(self):
        beads = defaultdict(lambda: defaultdict(set))
        if len(self.tx.lsa):
            for bead in self.tx.lsa:
                if bead.assignment == 'Bead Failure':
                    continue
                elif bead.assignment == 'Positive':
                    beads['Positive']['HighResolution'].update({hla.high_res for hla in bead.hlas})
                    beads['Positive']['Serology'].update({hla.low_res for hla in bead.hlas if hla.low_res})
                else:
                    beads['non-Positive']['HighResolution'].update({hla.high_res for hla in bead.hlas})
                    beads['non-Positive']['Serology'].update({hla.low_res for hla in bead.hlas if hla.low_res})
            self.luminex_beads = beads


    def find_dsa(self, tx:Transplant):
        """ Intersection of donor HLA and patient HLA antibody """

        self.tx = tx
        donor_hlas = self.tx.donor.get_hla(which='low_res')
        if donor_hlas:
            self._separate_luminex_hla_beads()
            if self.luminex_beads:
                high_res_ab = self.luminex_beads['Positive']['HighResolution']
                # print(f'High resolution HLA Antibodies {high_res_ab}')
                self.hla_abs = self.luminex_beads['Positive']['Serology']
                print(f'Donor HLA {donor_hlas} and HLA Antibodies {self.hla_abs}')
                
                self.tx.dsa = [hla for hla in (donor_hlas & self.hla_abs)]
            else:
                print(f'Transplant with id: {self.tx.id} does not have LSA data')
        else:
            print('No Donor HLA Found')

    def _mismatched_epitopes(self) -> dict:
        """ find epitope mismatch between donor & patient. First, find
        donor epitope, then remove patient epitopes and keep hla molecule """

        recipient_hla, donor_hla = self.tx.recipient.get_hla(which='high_res'), self.tx.donor.get_hla(which='high_res')
        ind = self.hlavsep['HLA'].apply(lambda x: x in recipient_hla)
        recipient_ep = flatten2set(self.hlavsep[ind]['Epitope'].values)
        donor_ep_dic = {ep:hla for hla in donor_hla for ep in self.hlavsep[self.hlavsep['HLA'] == hla]['Epitope'].values[0]}
        return {ep:hla for ep, hla in donor_ep_dic.items() if ep not in recipient_ep}

    def _epitopes_only_pos_bead(self)-> set:
        """ Find epitopes that only belong to the positive bead """
        self._separate_luminex_hla_beads()
        ind_pos = self.hlavsep['HLA'].apply(lambda x: x in self.luminex_beads['Positive']['HighResolution'])
        ind_nonpos = self.hlavsep['HLA'].apply(lambda x: x in self.luminex_beads['non-Positive']['HighResolution'])
        ep_pos_bead = flatten2set(self.hlavsep[ind_pos]['Epitope'].values)
        ep_nonpos_bead = flatten2set(self.hlavsep[ind_nonpos]['Epitope'].values)
        return ep_pos_bead - ep_nonpos_bead

    def find_desa(self, tx:Transplant):
        
        self.tx = tx
        if len(self.tx.lsa):
            self.mismatch_ep = self._mismatched_epitopes()
            self.positive_ep = self._epitopes_only_pos_bead()
            self.tx.desa = DESA([Epitope(ep, HLA(hla)) for ep, hla in self.mismatch_ep.items()
                                                if ep in self.positive_ep])

    def export_results(self, format:str='dict') -> dict:
        result = defaultdict()
        result['Epitope_Mismatch'] = self.mismatch_ep
        result['Tx_id'] = self.tx.id
        result['Donor_HLA'] = self.tx.donor
        result['Recipient_HLA'] = self.tx.recipient
        result['LuminexBeads'] = {_:dict(self.luminex_beads[_]) for _ in iter(self.luminex_beads)}
        result['DESA_Status'] = 'No LSA' if len(self.tx.lsa) == 0 else \
                                'No HLAE-Abs' if len(self.positive_ep) == 0 else \
                                'No DESA' if len(self.tx.desa) == 0 else 'DESA'

        result['DSA_Status'] = 'No LSA' if len(self.tx.lsa) == 0  else \
                               'No HLA-Abs' if len(self.hla_abs) == 0 else \
                               'No DSA' if len(self.tx.dsa) == 0 else 'DSA'

        result['DESA_info'] = self.tx.desa if result['DESA_Status'] == 'DESA' else DESA(epitopes=[])
        result['DSA_info'] = self.tx.dsa if result['DSA_Status'] == 'DSA' else DSA(hlas=[])
        return dict(result)