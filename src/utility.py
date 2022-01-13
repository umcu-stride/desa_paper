""" common utility methods used in STRIDE """
import os
from collections import defaultdict
import pandas as pd
import regex as re
import numpy as np
from typing import Sequence


def flatten2list(object) -> list:
    """ This function flattens objects in a nested structure and returns a list"""
    gather = []
    for item in object:
        if isinstance(item, (list, set)):
            gather.extend(flatten2list(item))
        else:
            gather.append(item)
    return gather

def flatten2string(object):
    """ This function flattens objects in a nested structure and return a joint list"""
    
    return ' '.join(flatten2list(object))

def flatten2set(object) -> set:
    """ This function flattens objects in a nested structure and returns a set"""

    return set(flatten2list(object))

# def flatten2array(object) -> np.array:
#     """ This function flattens objects in a nested structure and returns an array"""
#     return np.asarray(flatten2list(object)).astype(np.float32)

def flatten_dict_values(dictionary:dict) -> set:
    """ This function flattens objects in a nested structure and returns a set"""

    return flatten2set(dictionary.values())


def get_hla_class(hla:str):
    """ get hla class from high resolution typing """
    if hla.split('*')[0] in ['A', 'B', 'C']:
        return 'I'
    return 'II'

def get_class(x):
    """ get hla class """
    _class = {get_hla_class(hla) for hla in set(x.values())}
    return ','.join(list(_class))

def sequence2string(epitopes:Sequence) -> str:
    """ Join a sequence of strings in one string """

    return ' '.join([ep for ep in epitopes if ep.strip() ])

def sort_epitopes(epitopes:Sequence) -> list:
    """ sorts a list or set of epitopes """

    temp = re.compile("([0-9]+)([a-zA-Z]+)")
    epitopes = [temp.match(ep).groups() for ep in epitopes]
    sorted_epitopes = sorted(epitopes, key=lambda x:int(x[0]))
    return [''.join(_tuple) for _tuple in sorted_epitopes]

def epitopeseq2str(epitopes:Sequence, sort=True) -> str:
    if sort:
        epitopes = sort_epitopes(epitopes)
    return sequence2string(epitopes)
