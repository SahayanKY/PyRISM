from enum import Enum

import numpy as np

from rism.rism import RISMType


class ClosureType(Enum):
    HNC = 1
    KH  = 2
    PY  = 3


def parseClosureType(rootDict):
    typestr = rootDict.get('closure', 'HNC')
    if typestr is 'HNC':
        return ClosureType.HNC
    elif typestr is 'KH':
        return ClosureType.KH
    elif typestr is 'PY':
        return ClosureType.PY
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
        if rismType is RISMType.RISM1dN or rismType is RISMType.RISM3d:
            # 通常の1D-RISMと3D-RISMの場合
            if closureType is ClosureType.HNC:
                return HNCclosure
            elif closureType is ClosureType.KH:
                return KHclosure
            elif closureType is ClosureType.PY:
                return PYclosure

        elif rismType is RISMType.RISM1dX:
            # XRISMの場合
            if closureType is ClosureType.HNC:
                return HNCclosure_XRISM
            elif closureType is ClosureType.KH:
                return KHclosure_XRISM
            elif closureType is ClosureType.PY:
                return PYclosure_XRISM


def HNCclosure(**kwargs):
    """
    Eta -> C
    """
    U = kwargs['U']
    Eta = kwargs['Eta']
    return np.exp(-U+Eta) -Eta -1

def HNCclosure_XRISM(**kwargs):
    """
    Etas -> Cs
    """
    Us = kwargs['Us']
    Hl = kwargs['Hl']
    Etas = kwargs['Etas']
    return np.exp(-Us+Hl+Etas) -(Hl+Etas) -1