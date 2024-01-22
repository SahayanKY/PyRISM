import json

import numpy as np
from scipy.spatial import distance
from scipy.linalg import block_diag

class RISM():
    def __init__(self, rismDict, temperature, closure):
        # json読み込み
        #jsonDict = json.load(open(jsonFile, 'r'))
        # .....

        # inpオブジェクト生成
        inpdata = RISMInputData(rismDict, temperature, closure)

        # RISMの種類を判別
        configDict = rismDict['configure']
        isXRISM = configDict.get('XRISM', True)
        # solverオブジェクト生成
        if isXRISM:
            solver = XRISMSolver(configDict, closure, inpdata)
        else:
            solver = RISMSolver(configDict, closure, inpdata)

        self.__rismDict = rismDict
        self.__inpdata = inpdata
        self.__isXRISM = isXRISM
        self.__solver = solver
        self.__resultDataList = None

    def solve(self):
        # 1D-RISM計算
        self.__solver.solve()
        self.__resultDataList = self.__solver.giveResult()

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
        self.corrFuncShape = t_W.shape

        self.closure = closure


class RISMData():
    def __init__(self, **kwargs):
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
        self.Eta = kwargs['Eta']


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

class RISMInitializer():
    def __init__(self, shape, method):
        self.shape = shape
        self.method = method

    def initializeEta0(self):
        # Etaの初期化
        if self.method == 'zeroization':
            Eta0 = np.zeros(self.shape)
            return Eta0
        else:
            return None

class RISMSolver():
    def __init__(self, configDict, closure, risminpdata):
        shape = risminpdata.corrFuncShape
        iniMethod = configDict.get('initialze', "zeroization")
        initializer = RISMInitializer(shape, iniMethod)
        self.initializer = initializer

        self.mixingParam = configDict.get('mixingParam', 0.5)

        self.chargeUp = configDict.get('chargeUp', 0.25)
        factorList = np.arange(0,1,self.chargeUp)
        if factorList[-1] != 1:
            factorList = np.append(factorList, 1)
        self.chargeFactorList = factorList

        self.converge = configDict.get('converge', 1e-8)
        self.maxIter = configDict.get('maxIter', 1000)

        self.saveInterval = configDict.get('saveInterval', None)
        self.isSaveIntermediate = self.saveInterval is not None

        self.closure = closure
        self.risminpdata = risminpdata

        self.rismdataList = []

    def solve(self):
        def __registerData(force=False):
            if self.isSaveIntermediate or force:
                t_X = t_W + P @ t_H
                data = RISMData(
                        numLoop=numTotalLoop,
                        maxError=maxError,
                        isConverged=isConverged,
                        chargeFactor=factor,
                        fUl=fUl,
                        t_C=t_C,
                        t_H=t_H,
                        t_X=t_X,
                        C=C,
                        H=H,
                        Eta=Eta
                    )
                self.rismdataList.append(data)
            else:
                return

        I = self.risminpdata.I
        P = self.risminpdata.P
        t_W = self.risminpdata.t_W
        Us = self.risminpdata.Us
        Ul = self.risminpdata.Ul

        Eta0 = self.initializer.initializeEta0()

        numTotalLoop = 0
        for factor in self.chargeFactorList:
            # 初期化
            fUl = Ul * factor**2
            fU = Us + fUl
            Eta = Eta0
            isConverged = False

            numLoop = 0
            while True:
                numLoop +=1
                numTotalLoop += 1
                C = np.exp(-fU+Eta) -Eta -1 # HNC closure
                # フーリエ変換
                t_C = None
                # RISM式
                t_H = t_W @ t_C @ t_W @ np.linalg.inv(I - P @ t_C @ t_W)
                # 逆フーリエ変換
                H = None

                newEta = H - C
                # 収束判定
                maxError = np.max(newEta - Eta)
                if numLoop % 100 == 0:
                    print('Factor: {}, Iter: {}, Error: {}'.format(factor, numLoop, maxError))
                if maxError < self.converge:
                    print('converged: Iter: {}'.format(numLoop))
                    isConverged = True
                    __registerData()
                    break
                elif numLoop >= self.maxIter:
                    print('Maximum number of iterations exceeded: {}, Error: {}'.format(self.maxIter, maxError))
                    isConverged = False
                    __registerData()
                    break
                if numTotalLoop % self.saveInterval == 0:
                    # data生成
                    __registerData()

                # 更新
                Eta = self.mixingParam * newEta + (1-self.mixingParam) * Eta

            # 初期値更新
            Eta0 = Eta

        # 最終結果は中間結果を残さない場合(not isSaveItermediate)にも残す(force=True)
        if not self.isSaveIntermediate:
            __registerData(force=True)

    def giveResult(self):
        # TODO
        return None

class XRISMSolver(RISMSolver):
    def __init__(self, configDict, closure, risminpdata):
        super().__init__(configDict, closure, risminpdata)

    def initialize(self):
        pass

    def solve(self):
        pass



class ThreeDimRISM():
    def __init__(self):
        pass
