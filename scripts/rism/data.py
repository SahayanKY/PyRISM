from enum import Enum

import numpy as np
np.seterr(over='raise') # exp等でのオーバーフロー時に例外をスロー
import pandas as pd
from scipy.spatial import distance
from scipy.linalg import block_diag

from rism.grid import GridData
from rism.potential import *


class RISMInputData():
    def __init__(self, rismDict, temperature, closure):
        # 温度情報
        # temperature : 温度: K
        beta = 1 / (1.98720e-3 * temperature) # 逆温度: (kcal/mol)^-1

        # json読み込み
        # グリッド情報読み込み
        gridData = GridData(rismDict['discretize'])
        numgrid = gridData.numgrid # グリッド数
        r = gridData.r # shape: (numgrid,) # 動径座標: A
        k = gridData.k # shape: (numgrid,) # 動径座標: A^-1

        # 溶媒データ読み込み
        solventDict = rismDict['solvent']
        # 数密度: A^-3
        rhos = solventDict['rho'] # shape: (M,)
        # 溶媒種数
        M = len(rhos)
        # サイトデータ読み込み
        siteList = solventDict['site']
        # 総サイト数
        totalN = len(siteList)
        # サイトデータは縦に同じ種類のデータが配置されているのでまず転置させる
        siteList = list(zip(*siteList))
        # サイトの溶媒種への帰属
        belongList = siteList[0]          # shape: (totalN,)
        siteNameList = siteList[1]        # shape: (totalN,)
        sigma = np.array(siteList[2])     # shape: (totalN,)
        eps   = np.array(siteList[3])     # shape: (totalN,)
        z     = np.array(siteList[4])     # shape: (totalN,)
        xyz   = np.array(siteList[5:8]).T # shape: (totalN, 3)
        # belongListチェック
        # - 整数のみか
        # - 1始まりか
        # - 昇順か
        # - 要素数がtotalNに一致するか
        #   (siteListが中途半端になっているとzipの段階でtotalNに一致しなくなる)
        # - ユニークな値の数がMに一致するか
        if any([type(i) is not int for i in belongList]):
            raise ValueError()
        if belongList[0] != 1:
            raise ValueError()
        if not all([b1 <= b2 for b1,b2 in zip(belongList,belongList[1:])]):
            raise ValueError()
        if len(belongList) != totalN:
            raise ValueError()
        uniqList, countList = np.unique(belongList, return_counts=True)
        if len(uniqList) != M:
            raise ValueError()
        # 溶媒種毎のサイト数
        Ns = countList # shape: (M,)

        # 単位行列
        I = np.diag(np.ones(totalN)) # shape: (totalN, totalN)

        # 数密度行列: A^-3
        P = np.diag([rhos[i] for i in range(M) for j in range(Ns[i])]) # shape: (totalN,totalN)

        # 分子内サイト間距離行列: A
        L = distance.cdist(xyz, xyz) # shape: (totalN, totalN) # 本来は異なる分子種間の距離は計算する意味がないが面倒なのでこのままで
        # ブロック単位行列
        bI = block_diag(*[np.ones([n,n]) for n in Ns]) # shape: (totalN, totalN) # delta_st
        # 分子内相関行列(波数空間): [無次元]
        t_W = bI * np.sinc(k[:,np.newaxis,np.newaxis] * L / np.pi) # shape: (numgrid,totalN,totalN)

        # サイト間短距離ポテンシャル行列: [無次元]: shape: (numgrid, totalN, totalN)
        # Lorentz-Berthelot則
        Us = generateLennardJonesPotMatrix(sigma, sigma, eps, eps, r, beta)

        # サイト間長距離ポテンシャル行列: [無次元]: shape: (numgrid, totalN, totalN)
        Ul = generateCoulombPotMatrix(z, z, r, beta)
        t_Ul = generateFourierSpaceCoulombPotMatrix(z, z, k, beta)

        # -------------------------------------
        # インスタンス初期化
        self.gridData = gridData

        self.T = temperature
        self.beta = beta

        self.rhos = rhos
        self.belongList = belongList
        self.siteNameList = siteNameList
        self.z = z
        self.totalN = totalN

        self.I = I
        self.P = P
        self.t_W = t_W
        self.Us = Us
        self.Ul = Ul
        self.t_Ul = t_Ul
        self.corrFuncShape = t_W.shape

        self.closure = closure

        # 書き出し時に用いる相関のラベル(sigma-tau)の作成
        _snl = ['{}@{}'.format(s,b) for s,b in zip(siteNameList, belongList)]
        # ベクトル用
        self.siteLabelList = _snl
        # 正方(非対称)行列用
        self.sitesiteLabelList = ['{}-{}'.format(_snl[i],_snl[j]) for i in range(totalN) for j in range(totalN)]
        # 対称行列用
        self.sitesiteLabelListForSymmMatrix = ['{}-{}'.format(_snl[i],_snl[j]) for i in range(totalN) for j in range(i,totalN)]
        self.uniqIndexListForFlattenSymmMatrix = [i*totalN+j for i in range(totalN) for j in range(i,totalN)]

    def giveFuncsDict(self):
        # r, k, t_W, Us, Ul
        d = {
                'r': [self.gridData.r, DataAnnotation.Scaler],
                'k': [self.gridData.k, DataAnnotation.Scaler],
                't_W': [self.t_W, DataAnnotation.SymmMatrix],
                'Us': [self.Us, DataAnnotation.SymmMatrix],
                'Ul':[self.Ul, DataAnnotation.SymmMatrix]
            }
        return d


class DataAnnotation(Enum):
    """
    データ(ある量)がスカラー値なのか、ベクトル値なのか、対称行列なのかを区別するためのアノテーション
    convertRISMDataToDataFrame()にて用いる
    """
    Scaler = 1
    Vector = 2
    SquareMatrix = 3
    SymmMatrix = 4

def convertRISMDataToDataFrame(rismInpData, rismData):
    numgrid = rismInpData.numgrid
    uniqIndexListForFlattenSymmMatrix = rismInpData.uniqIndexListForFlattenSymmMatrix
    siteLabelList = rismInpData.siteLabelList
    sslabelList = rismInpData.sitesiteLabelList
    sslabelList_symm = rismInpData.sitesiteLabelListForSymmMatrix

    inpDataDict = rismInpData.giveFuncsDict()
    resultDataDict = rismData.giveFuncsDict()
    dataDict = {**inpDataDict, **resultDataDict}

    columns = []
    data = []
    for name, (value, annotation) in dataDict.items():
        value = value.reshape(numgrid, -1)
        if annotation is DataAnnotation.Scaler:
            # 結合の時のことを考慮して二次元配列に変えておく
            columns.append((name,'-'))
            data.append(value) # shape: (numgrid, 1)

        elif annotation is DataAnnotation.Vector:
            columns.extend([(name,s) for s in siteLabelList])
            data.append(value) # shape: (numgrid, N)

        elif annotation is DataAnnotation.SquareMatrix:
            columns.extend([(name,ss) for ss in sslabelList])
            data.append(value) # shape: (numgrid, N*N)

        elif annotation is DataAnnotation.SymmMatrix:
            # 対称行列は上半分だけ取り出して、平坦化する
            columns.extend([(name,ss) for ss in sslabelList_symm])
            data.append(value[:,uniqIndexListForFlattenSymmMatrix]) # shape: (numgrid, N(N+1)/2)
        else:
            raise ValueError()
    # 結合
    data = np.hstack(data) # shape: (numgrid, *)
    # MultiIndex
    columns = [[t[0] for t in columns], [t[1] for t in columns]]
    columns = pd.MultiIndex.from_arrays(columns, names=('name', 'site'))
    df = pd.DataFrame(data, columns=columns)
    return df


class RISMData():
    def __init__(self, **kwargs):
        self.inpData = kwargs['inpData']
        self.numLoop = kwargs['numLoop']
        self.maxError = kwargs['maxError']
        self.isConverged = kwargs['isConverged']
        self.chargeFactor = kwargs['chargeFactor']
        self.fUl = kwargs['fUl']
        self.t_C = kwargs['t_C']
        self.t_H = kwargs['t_H']
        self.t_X = kwargs['t_X']
        self.C = kwargs['C']
        self.H = kwargs['H']
        self.G = kwargs['G']
        self.Eta = kwargs['Eta']

    def giveFuncsDict(self):
        # fUl, t_C, t_H, t_X, C, H, G, Eta
        d = {
                'fUl': [self.fUl, DataAnnotation.SymmMatrix],
                't_C': [self.t_C, DataAnnotation.SymmMatrix],
                't_H': [self.t_H, DataAnnotation.SymmMatrix],
                't_X': [self.t_X, DataAnnotation.SquareMatrix], # Xは非対称行列なのでSquare指定
                'C': [self.C, DataAnnotation.SymmMatrix],
                'H': [self.H, DataAnnotation.SymmMatrix],
                'G': [self.G, DataAnnotation.SymmMatrix],
                'Eta': [self.Eta, DataAnnotation.SymmMatrix]
            }
        return d

class XRISMData(RISMData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.t_Cl = kwargs['t_Cl']
        self.t_Cs = kwargs['t_Cs']
        self.t_Hl = kwargs['t_Hl']
        self.t_Hs = kwargs['t_Hs']
        self.t_Xl = kwargs['t_Xl']

        self.Cl = kwargs['Cl']
        self.Cs = kwargs['Cs']
        self.Hl = kwargs['Hl']
        self.Hs = kwargs['Hs']
        self.Etas = kwargs['Etas']

    def giveFuncsDict(self):
        dic1 = super().giveFuncsDict()

        # t_Cl, t_Cs, t_Hl, t_Hs, t_Xl, Cl, Cs, Hl, Hs, Etas
        dic2 = {
                't_Cl': [self.t_Cl, DataAnnotation.SymmMatrix],
                't_Cs': [self.t_Cs, DataAnnotation.SymmMatrix],
                't_Hl': [self.t_Hl, DataAnnotation.SymmMatrix],
                't_Hs': [self.t_Hs, DataAnnotation.SymmMatrix],
                't_Xl': [self.t_Xl, DataAnnotation.SquareMatrix], # Xは非対称行列なのでSquare指定
                'Cl': [self.Cl, DataAnnotation.SymmMatrix],
                'Cs': [self.Cs, DataAnnotation.SymmMatrix],
                'Hl': [self.Hl, DataAnnotation.SymmMatrix],
                'Hs': [self.Hs, DataAnnotation.SymmMatrix],
                'Etas': [self.Etas, DataAnnotation.SymmMatrix]
            }
        d = {**dic1, **dic2}
        return d
