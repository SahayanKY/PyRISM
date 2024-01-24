import json
from enum import Enum

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.linalg import block_diag
from scipy.fft import dst

class RISM():
    def __init__(self, rismDict, temperature, closure):
        # json読み込み
        #jsonDict = json.load(open(jsonFile, 'r'))
        # .....

        # inpオブジェクト生成
        inpData = RISMInputData(rismDict, temperature, closure)

        # RISMの種類を判別
        configDict = rismDict['configure']
        isXRISM = configDict.get('XRISM', True)
        # solverオブジェクト生成
        if isXRISM:
            solver = XRISMSolver(configDict, closure, inpData)
        else:
            solver = RISMSolver(configDict, closure, inpData)

        self.__rismDict = rismDict
        self.__inpData = inpData
        self.__isXRISM = isXRISM
        self.__solver = solver
        self.__resultDataList = None

    def solve(self):
        # 1D-RISM計算
        self.__solver.solve()
        self.__resultDataList = self.__solver.giveResult()

    def write(self, directory):
        if self.__resultDataList is None:
            raise RuntimeError()

        inpData = self.__inpData
        resultDataList = self.__resultDataList

        # inpData取り出し
        inpTableData, inpTableColumns = inpData.giveMergedFuncsData()

        finalLoop = resultDataList[-1].numLoop
        numDigit = len(str(finalLoop))
        for i, resultData in enumerate(resultDataList):
            # 出力先決定
            numLoop = resultData.numLoop
            isLastData = i == len(resultDataList) -1
            if isLastData:
                # 最終結果
                csvFile = 'rism_result.csv'
                float_format = '% .10E'
            else:
                # 途中経過
                csvFile = 'rism_result_loop{}.csv'.format(str(numLoop).zfill(numDigit))
                float_format = '% .4E'

            csvPath = directory + '/' + csvFile

            # resultData取り出し
            resultTableData, resultTableColumns = resultData.giveMergedFuncsData()

            # 結合
            tableData = np.hstack([inpTableData, resultTableData]) # shape: (numgrid, *)
            columns = [*inpTableColumns, *resultTableColumns]
            df = pd.DataFrame(tableData, columns=columns)

            # csv書き出し
            df.to_csv(csvPath, mode='w', index=False, float_format=float_format)

            # solve中に書き出していかないと
            # メモリ量の関係で途中経過のData全てを保持できない場合がある
            # -  4000点 *   3サイト * 13行列関数 * 10桁 -> 5644 KB
            # - 10000点 * 100サイト * 13行列関数 *  5桁 -> 7,838,888 KB = 7.8 GB (推定)

        # 最終結果のグラフ書き出し

        # 途中経過のアニメーション書き出し
        # https://qiita.com/yubais/items/c95ba9ff1b23dd33fde2

        # - 統計量書き出し


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
        self.gridData = GridData(r,dr,k,dk,ffttype=4)
        self.numgrid = numgrid

        self.T = temperature
        self.beta = beta

        self.rhos = rhos
        self.belongList = belongList
        self.siteNameList = siteNameList
        self.Sigma = Sigma
        self.Eps = Eps
        self.z = z
        self.ZZ = ZZ
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
        _snl = ['{}:{}'.format(b,s) for b,s in zip(belongList, siteNameList)]
        # ベクトル用
        self.siteLabelList = _snl
        # 正方(非対称)行列用
        self.sitesiteLabelList = ['{}-{}'.format(_snl[i],_snl[j]) for i in range(totalN) for j in range(totalN)]
        # 対称行列用
        self.sitesiteLabelListForSymmMatrix = ['{}-{}'.format(_snl[i],_snl[j]) for i in range(totalN) for j in range(i,totalN)]
        self.uniqIndexListForFlattenSymmMatrix = [i*totalN+j for i in range(totalN) for j in range(i,totalN)]

    def giveMergedFuncsData(self):
        # r, k, t_W, Us, Ul
        d = {
                'r': [self.r, DataAnnotation.Scaler],
                'k': [self.k, DataAnnotation.Vector],
                't_W': [self.t_W, DataAnnotation.SymmMatrix],
                'Us': [self.Us, DataAnnotation.SymmMatrix],
                'Ul':[self.Ul, DataAnnotation.SymmMatrix]
            }
        data, columns = convertToMergedFuncsData(self, d)
        return data, columns

class GridData():
    def __init__(self, r, dr, k, dk, ffttype=4):
        self.numgrid = len(r)
        self.r = r
        self.dr = dr
        self.k = k
        self.dk = dk
        self.ffttype = ffttype

    def fft3d_spsymm(self, f):
        """
        f: shape: (numgrid, *, *), or (numgrid, *)
        """
        r = self.r
        dr = self.dr
        k = self.k
        t_f = 4*np.pi/k[:,np.newaxis,np.newaxis] * dst(r[:,np.newaxis,np.newaxis] * f, type=self.ffttype, axis=0) / 2 * dr
        return t_f

    def ifft3d_spsymm(self, t_f):
        """
        t_f: shape: (numgrid, *, *), or (numgrid, *)
        """
        r = self.r
        k = self.k
        dk = self.dk
        f = 1/(2*np.pi**2 * r[:,np.newaxis,np.newaxis]) * dst(k[:,np.newaxis,np.newaxis] * t_f, type=self.ffttype, axis=0) / 2 * dk
        return f

class DataAnnotation(Enum):
    """
    データ(ある量)がスカラー値なのか、ベクトル値なのか、対称行列なのかを区別するためのアノテーション
    giveMergedFuncsData()にて用いる
    """
    Scaler = 1
    Vector = 2
    SquareMatrix = 3
    SymmMatrix = 4

def convertToMergedFuncsData(inpData, funcsDict):
    """
    d = {'r': [r, Scaler], 'k': [k, Scaler], 't_W':[t_W, SymmMatrix], 'Us':[Us, SymmMatrix], 'Ul':[Ul, SymmMatrix]}
    """
    numgrid = inpData.numgrid
    uniqIndexListForFlattenSymmMatrix = inpData.uniqIndexListForFlattenSymmMatrix
    siteLabelList = inpData.siteLabelList
    sslabelList = inpData.sitesiteLabelList
    sslabelList_symm = inpData.sitesiteLabelListForSymmMatrix

    columns = []
    data = []
    for name, (value, annotation) in funcsDict.items():
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
    data = np.hstack(data) # shape: (numgrid, N(N+1)/2 * 3)

    return data, columns


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

    def giveMergedFuncsData(self):
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
        data, columns = convertToMergedFuncsData(self.inpData, d)
        return data, columns

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

    def giveMergedFuncsData(self):
        data1, columns1 = super().giveMergedFuncsData()

        # t_Cl, t_Cs, t_Hl, t_Hs, t_Xl, Cl, Cs, Hl, Hs, Etas
        d = {
                't_Cl': [self.fUl, DataAnnotation.SymmMatrix],
                't_Cs': [self.t_C, DataAnnotation.SymmMatrix],
                't_Hl': [self.t_H, DataAnnotation.SymmMatrix],
                't_Hs': [self.t_X, DataAnnotation.SymmMatrix],
                't_Xl': [self.C, DataAnnotation.SquareMatrix], # Xは非対称行列なのでSquare指定
                'Cl': [self.H, DataAnnotation.SymmMatrix],
                'Cs': [self.G, DataAnnotation.SymmMatrix],
                'Hl': [self.Eta, DataAnnotation.SymmMatrix],
                'Hs': [self.H, DataAnnotation.SymmMatrix],
                'Etas': [self.G, DataAnnotation.SymmMatrix]
            }
        data2, columns2 = convertToMergedFuncsData(self.inpData, d)
        data = np.hstack([data1, data2])
        columns = [*columns1, *columns2]
        return data, columns


def RISM_HNC():
    """
    h = exp(-u + h - c) -1
        h -> c / c -> h
    h = exp(-u + eta) -1
        h -> eta / eta -> h
    eta + c = exp(-u + eta) -1
        eta -> c / c -> eta
    """
    pass

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

        self.rismdataList = None

    def solve(self):
        def __registerData(force=False):
            if self.isSaveIntermediate or force:
                t_X = t_W + P @ t_H
                data = RISMData(
                        inpData=self.risminpdata,
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
                        G=H+1,
                        Eta=Eta
                    )
                self.rismdataList.append(data)
            else:
                return

        self.rismdataList = []

        I = self.risminpdata.I
        P = self.risminpdata.P
        t_W = self.risminpdata.t_W
        Us = self.risminpdata.Us
        Ul = self.risminpdata.Ul
        grid = self.risminpdata.gridData

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
                t_C = grid.fft3d_spsymm(C)
                # RISM式
                t_H = t_W @ t_C @ t_W @ np.linalg.inv(I - P @ t_C @ t_W)
                # 逆フーリエ変換
                H = grid.ifft3d_spsymm(t_H)

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
        if self.rismdataList is None:
            raise RuntimeError()

        return self.rismdataList

class XRISMSolver(RISMSolver):
    def __init__(self, configDict, closure, risminpdata):
        super().__init__(configDict, closure, risminpdata)


    def solve(self):
        def __registerData(force=False):
            if self.isSaveIntermediate or force:
                t_C = t_Cl + t_Cs
                t_H = t_Hl + t_Hs
                C = Cl + Cs
                H = Hl + Hs
                Eta = H - C
                t_X = t_W + P @ t_H
                data = XRISMData(
                        inpData=self.risminpdata,
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
                        G=H+1,
                        Eta=Eta,
                        #
                        t_Cl = t_Cl,
                        t_Cs = t_Cs,
                        t_Hl = t_Hl,
                        t_Hs = t_Hs,
                        t_Xl = t_Xl,
                        Cl = Cl,
                        Cs = Cs,
                        Hl = Hl,
                        Hs = Hs,
                        Etas = Etas
                    )
                self.rismdataList.append(data)
            else:
                return

        self.rismdataList = []

        I = self.risminpdata.I
        P = self.risminpdata.P
        t_W = self.risminpdata.t_W
        Us = self.risminpdata.Us
        Ul = self.risminpdata.Ul
        t_Ul = self.risminpdata.t_Ul
        grid = self.risminpdata.gridData

        Etas0 = self.initializer.initializeEta0()

        numTotalLoop = 0
        for factor in self.chargeFactorList:
            # 初期化
            fUl = Ul * factor**2
            t_fUl = t_Ul * factor**2
            fU = Us + fUl
            Etas = Etas0
            isConverged = False

            # 長距離part
            Cl = fUl
            t_Cl = t_fUl
            t_Hl = t_W @ t_Cl @ t_W @ np.linalg.inv(I - P @ t_Cl @ t_W)
            Hl = grid.ifft3d_spsymm(t_Hl)
            t_Xl = t_W + P @ t_Hl
            t_XlT = t_Xl.transpose(0,2,1) # 転置

            # 短距離part
            numLoop = 0
            while True:
                numLoop +=1
                numTotalLoop += 1
                Cs = np.exp(-Us+Hl+Etas) -(Hl+Etas) -1 # HNC closure
                # フーリエ変換
                t_Cs = grid.fft3d_spsymm(Cs)
                # RISM式
                t_Hs = t_XlT @ t_Cs @ t_Xl @ np.linalg.inv(I - P @ t_Cs @ t_Xl)
                # 逆フーリエ変換
                Hs = grid.ifft3d_spsymm(t_Hs)

                newEtas = Hs - Cs
                # 収束判定
                maxError = np.max(newEtas - Etas)
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
                Etas = self.mixingParam * newEtas + (1-self.mixingParam) * Etas

            # 初期値更新
            Etas0 = Etas

        # 最終結果は中間結果を残さない場合(not isSaveItermediate)にも残す(force=True)
        if not self.isSaveIntermediate:
            __registerData(force=True)



class ThreeDimRISM():
    def __init__(self):
        pass
