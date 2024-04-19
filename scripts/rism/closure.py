from enum import Enum

import numpy as np

class ClosureType(Enum):
    HNC = 1
    KH  = 2
    PY  = 3
    MSA = 4


def parseClosureType(rootDict):
    typestr = rootDict.get('closure', 'HNC')
    if typestr == 'HNC':
        return ClosureType.HNC
    elif typestr == 'KH':
        return ClosureType.KH
    elif typestr == 'PY':
        return ClosureType.PY
    elif typestr == 'MSA':
        return ClosureType.MSA
    else:
        raise ValueError()

class Closure():
    def __init__(self, closureType, rismType):
        self.__closureType = closureType
        self.__rismType = rismType
        self.__closureFunc = self.__selectFunc(closureType, rismType)

    def apply(self, **kwargs):
        return self.__closureFunc(**kwargs)

    def __selectFunc(self, closureType, rismType):
        # moduleの先頭でimportすると循環importになるのでこのタイミングでimportする
        from rism.rism import RISMType
        if rismType is RISMType.RISM1dN or rismType is RISMType.RISM3d:
            # 通常の1D-RISMと3D-RISMの場合
            if closureType is ClosureType.HNC:
                return HNCclosure
            elif closureType is ClosureType.KH:
                return KHclosure
            elif closureType is ClosureType.PY:
                return PYclosure
            elif closureType is ClosureType.MSA:
                return MSAclosure

        elif rismType is RISMType.RISM1dX:
            # XRISMの場合
            if closureType is ClosureType.HNC:
                return HNCclosure_XRISM
            elif closureType is ClosureType.KH:
                return KHclosure_XRISM
            elif closureType is ClosureType.PY:
                return PYclosure_XRISM
            elif closureType is ClosureType.MSA:
                return MSAclosure_XRISM

        raise ValueError()

"""
https://core.ac.uk/download/pdf/42591811.pdf
https://pubs.acs.org/doi/epdf/10.1021/acs.macromol.8b00011
"""
def HNCclosure(**kwargs):
    """
    Eta -> C
    """
    Us = kwargs['Us']
    Ul = kwargs['Ul']
    Eta = kwargs['Eta']
    U = Us + Ul
    return np.exp(-U+Eta) -Eta -1

def KHclosure(**kwargs):
    """
    Eta -> C
    """
    Us = kwargs['Us']
    Ul = kwargs['Ul']
    Eta = kwargs['Eta']
    U = Us + Ul
    d = -U + Eta
    return (np.exp(d)-1) * np.heaviside(-d,0.5) + d * np.heaviside(d,0.5) -Eta

def PYclosure(**kwargs):
    """
    Eta -> C
    """
    Us = kwargs['Us']
    Ul = kwargs['Ul']
    Eta = kwargs['Eta']
    U = Us + Ul
    return np.exp(-U) * (1+Eta) -Eta -1

def MSAclosure(**kwargs):
    """
    Eta -> C

    """
    Us = kwargs['Us']
    Ul = kwargs['Ul']
    U = Us + Ul
    return -U

def HNCclosure_XRISM(**kwargs):
    """
    Etas -> Cs
    """
    Us = kwargs['Us']
    Hl = kwargs['Hl']
    Etas = kwargs['Etas']
    Clc = kwargs['Clc']
    d = -Us + Hl + Etas - Clc
    return np.exp(d) -(Hl+Etas) -1

def KHclosure_XRISM(**kwargs):
    """
    Etas -> Cs
    """
    Us = kwargs['Us']
    Hl = kwargs['Hl']
    Etas = kwargs['Etas']
    Clc = kwargs['Clc']
    d = -Us + Hl + Etas - Clc
    return (np.exp(d)-1) * np.heaviside(-d,0.5) + d * np.heaviside(d,0.5) -(Hl+Etas)

def PYclosure_XRISM(**kwargs):
    """
    Etas -> Cs
    """
    Us = kwargs['Us']
    Ul = kwargs['Ul']
    Cl = kwargs['Cl']
    Hl = kwargs['Hl']
    Etas = kwargs['Etas']
    U = Us + Ul
    return np.exp(-U) * (1 + Etas + Hl - Cl) -(Hl+Etas) -1

def MSAclosure_XRISM(**kwargs):
    """
    Etas -> Cs
    """
    Us = kwargs['Us']
    return -Us


