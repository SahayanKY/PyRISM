import json

import numpy as np
from scipy.spatial import distance
from scipy.linalg import block_diag

class RISM():
    def __init__(self, rismDict, temperature, closure):
        # json読み込み
        #jsonDict = json.load(open(jsonFile, 'r'))
        # .....

        # RISMの種類を判別
        isXRISM = rismDict['configure'].get('XRISM', True)
        # dataオブジェクトとoptimizerオブジェクトを生成
        if isXRISM:
            data = XRISMData(rismDict, temperature, closure)
            optimizer = XRISMOptimizer(rismDict, closure)
        else:
            data = RISMData(rismDict, temperature, closure)
            optimizer = RISMOptimizer(rismDict, closure)

        self.__data = data
        self.__optimizer = optimizer

        # optimizerにdataを紐づけ
        optimizer.setData(data)

    def solve(self):
        self.__optimizer.optimize()

    def write(self, csvFile):
        somedata = self.__data.give()
        # csvFileへ書き込み

class RISMInputData():
    def __init__(self, rismDict, temperature, closure):
        # 温度情報
        # temperature : 温度: K
        beta = 1 / (1.98720e-3 * temperature) # 逆温度: (kcal/mol)^-1

        # json読み込み
        # グリッド情報読み込み
        dr = rismDict['discretize']['dr'] # 刻み幅: A
        numgrid = rismDict['discretize']['n'] # グリッド数
        # DST(type4)に合わせてグリッド生成
        dk = np.pi / dr / numgrid # 刻み幅: A^-1
        r = (np.arange(numgrid) + 0.5) * dr # shape: (numgrid,) # 動径座標: A
        k = (np.arange(numgrid) + 0.5) * dk # shape: (numgrid,) # 動径座標: A^-1

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
        if all([b1 <= b2 for b1,b2 in zip(belongList,belongList[1:])]):
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
        Sigma = (sigma[:,np.newaxis] + sigma) / 2           # shape: (totalN,totalN) # LJ sigma_ij
        Eps = np.sqrt(eps[:,np.newaxis]*eps)                # shape: (totalN,totalN) # LJ eps_ij
        __sigmar6 = (Sigma / r[:,np.newaxis,np.newaxis])**6 # shape: (numgrid, totalN, totalN)
        Us = beta * 4 * Eps * (__sigmar6**2 - __sigmar6)

        # サイト間長距離ポテンシャル行列: [無次元]: shape: (numgrid, totalN, totalN)
        # 電荷行列: A: shape: (totalN,totalN)
        ZZ = z[:,np.newaxis] * z
        __ZZ = beta * 332.053 * ZZ
        Ul = __ZZ / r[:,np.newaxis,np.newaxis]
        t_Ul = __ZZ * 4*np.pi / (k[:,np.newaxis,np.newaxis]**2)

        # -------------------------------------
        # インスタンス初期化
        self.r = r
        self.dr = dr
        self.k = k
        self.dk = dk

        self.T = temperature
        self.beta = beta

        self.rhos = rhos
        self.belongList = belongList
        self.siteNameList = siteNameList
        self.Sigma = Sigma
        self.Eps = Eps
        self.z = z
        self.ZZ = ZZ

        self.I = I
        self.P = P
        self.t_W = t_W
        self.Us = Us
        self.Ul = Ul
        self.t_Ul = t_Ul

        self.closure = closure


class RISMData():
    def __init__(self):
        self.isConverged = False

        self.t_C = None # 直接相関
        self.t_H = None # 全相関
        self.t_X = None # 溶媒感受率
        self.C = None
        self.H = None
        self.X = None
        self.Eta = None # H - C


class XRISMData(RISMData):
    def __init__(self):
        super().__init__()

        self.t_Cl = None
        self.t_Cs = None
        self.t_Hl = None
        self.t_Hs = None
        self.t_Xl = None

        self.Cl = None
        self.Cs = None
        self.Hl = None
        self.Hs = None
        self.Etas = None


class RISMOptimizer():
    def __init__(self, configDict, closure, risminpdata):
        self.mixingParam = configDict.get('mixingParam', 0.5)

        self.chargeUp = configDict.get('chargeUp', 0.25)
        factorList = np.arange(0,1,self.chargeUp)
        if factorList[-1] != 1:
            factorList = np.append(factorList, 1)
        self.chargeFactorList = factorList

        self.converge = configDict.get('converge', 1e-8)
        self.maxiter = configDict.get('maxiter', 1000)
        self.closure = closure
        self.risminpdata = risminpdata
        self.rismdata = None

    def initialize(self):
        # RISMDataの初期化
        shape = self.risminpdata.t_W.shape
        self.rismdata.Eta = np.zeros(shape)

    def optimize(self):
        I = self.risminpdata.I
        P = self.risminpdata.P
        t_W = self.risminpdata.t_W
        Us = self.risminpdata.Us
        Ul = self.risminpdata.Ul

        Eta = self.rismdata.Eta

        for factor in self.chargeFactorList:
            _Ul = Ul * factor**2
            _U = Us + _Ul

            numLoop = 0
            while True:
                numLoop +=1
                C = np.exp(-_U+Eta) -Eta -1 # HNC closure
                # フーリエ変換
                t_C = None
                # RISM式
                t_H = t_W @ t_C @ t_W @ np.linalg.inv(I - P @ t_C @ t_W)
                # 逆フーリエ変換
                H = None

                # 更新
                newEta = H - C
                # 収束判定
                maxError = np.max(newEta - Eta)
                if maxError < self.converge:
                    print('converged: loop: {}'.format(numLoop))
                    break
                elif numLoop >= self.maxiter:
                    print('Maximum number of iterations exceeded: {}, Error: {}'.format(self.maxiter, maxError))
                    break
                Eta = self.mixingParam * newEta + (1-self.mixingParam) * Eta




        pass
        # returnでRISMDataを返す


class XRISMOptimizer(RISMOptimizer):
    def __init__(self, configDict, closure, risminpdata):
        super().__init__(configDict, closure, risminpdata)

    def initialize(self):
        pass

    def optimize(self):
        pass



class ThreeDimRISM():
    def __init__(self):
        pass
