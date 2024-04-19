import json
import re
import os
from enum import Enum

import numpy as np
np.seterr(over='raise') # exp等でのオーバーフロー時に例外をスロー
import pandas as pd
from scipy.spatial import distance
from scipy.linalg import block_diag

from rism.grid import GridData
from rism.potential import *
from rism.data import RISMInputData, DataAnnotation, convertRISMDataToDataFrame, RISMData, XRISMData
from rism.closure import Closure

class RISMType(Enum):
    RISM1dN = 1
    RISM1dX = 2
    RISM3d  = 3

class RISM():
    def __init__(self, rismDict, temperature, closureType):
        # json読み込み
        #jsonDict = json.load(open(jsonFile, 'r'))
        # .....
        configDict = rismDict['configure']
        saveDict = rismDict['save']
        # RISMの種類を判別
        rismTypeStr = configDict.get('RISMType', "XRISM")
        if rismTypeStr == 'RISM':
            rismType = RISMType.RISM1dN
            solvercls = RISMSolver
        elif rismTypeStr == 'XRISM':
            rismType = RISMType.RISM1dX
            solvercls = XRISMSolver
        else:
            raise ValueError()

        # closureオブジェクト生成
        closure = Closure(closureType, rismType)
        # inpオブジェクト生成
        inpData = RISMInputData(rismDict, temperature, closure)
        # writerオブジェクト生成
        writer = RISMWriter(saveDict)
        # solverオブジェクト生成
        solver = solvercls(configDict, closure, inpData, writer)

        self.__rismDict = rismDict
        self.__inpData = inpData
        self.__rismType = rismType
        self.__solver = solver
        self.__resultData = None

    def solve(self):
        # 1D-RISM計算
        self.__solver.solve()
        self.__resultData = self.__solver.giveResult()

    def giveResultData(self):
        if self.__resultData is None:
            raise RuntimeError()
        return self.__inpData, self.__resultData


class RISMWriter():
    """
    CSV書き出しと統計量書き出しを行う
    グラフ化については別にクラスを用意し、計算終了後に処理する形にする

    solve中に書き出していかないと
    メモリ量の関係で途中経過のData全てを保持できない場合がある
    (仮実装で生成したcsvの実測値)
    -  4000点 *   (3サイト * 13行列関数 =      78列) * 7+10桁 ->      5,644 KB
    (推定)
    - 10000点 * (100サイト * 21行列関数 = 106,050列) * 7+ 5桁 -> 13,541,769 KB = 12.9 GB
    - 10000点 * (100サイト * 11行列関数 =  55,550列) * 7+ 3桁 ->  5,911,090 KB =  5.6 GB (出力する行列関数の種類を削減、桁数削減)
        - +1.*****E+1, -> 7+5桁の計算
    なので、都度SolverクラスがWriterクラスを呼び出し、書き出す

    出力ファイルのサイズに関する仕様や削減策については後でまた考え直す
    - サイトペアの特徴が似てないペア100組ぐらいまでに絞ればサイズを2桁落とすことができるのでその方向を検討してみる
    """

    def __init__(self, saveDict):
        self.__onlyFinal = saveDict.get('onlyFinal', False)
        self.__interval =  saveDict.get('interval', 100)
        self.__directory = saveDict.get('directory', './')

        size = saveDict.get('maxFileSize', '1GB') # csvファイル1つ辺りのサイズ
        # 単位変換: Byte単位に変換
        m = re.match(r'([0-9]+) *([kKMGT]?)B', size)
        if m is None:
            raise ValueError()
        exponentDict = {'': 0, 'k': 1, 'K': 1, 'M': 2, 'G': 3, 'T': 4}
        self.__maxFileSize = int(m.groups()[0]) * 1024 ** exponentDict[m.groups()[1]]

        self.__notWrittenYet = True
        self.__digit_rough = 3 # 小数部桁数
        self.__digit_detail = 8
        self.__float_format_rough = '% .{}E'.format(self.__digit_rough)
        self.__float_format_detail = '% .{}E'.format(self.__digit_detail)
        self.__size_csv_rough = None
        self.__size_csv_detail = None
        self.__saveFileList = []

    def judgeInterval(self, numLoop):
        return numLoop % self.__interval == 0

    def saveOnlyFinal(self):
        return self.__onlyFinal

    def __appendSaveFileList(self, filePath):
        self.__saveFileList.append(filePath)

    def __checkFileSize(self, filePath):
        """
        実際出力した結果、超えていた場合は例外をスロー
        """
        # totalで何Byteか計算
        size = os.path.getsize(filePath)
        if size > self.__maxFileSize:
            raise RuntimeError()

    def __checkEstimatedFileSize(self, df):
        numGrid = len(df.index)
        numColumn = len(df.columns)
        digit_rough = 3 # 小数部桁数
        digit_detail = 8

        self.__size_csv_rough = int(5644 * 1024 * (numGrid/4000) * (numColumn/88) * ((7+digit_rough)/(7+10)))
        self.__size_csv_detail= int(5644 * 1024 * (numGrid/4000) * (numColumn/88) * ((7+digit_detail)/(7+10)))
        if self.__size_csv_detail > self.__maxFileSize or self.__size_csv_rough > self.__maxFileSize:
            raise RuntimeError()

    def writeIntermediate(self, rismData):
        """
        途中経過の書き出し
        - csv書き出し
        - グラフのアニメーション作成
        """
        if self.__onlyFinal:
            # 中間結果を保存しない場合
            return
        numLoop = rismData.numLoop
        if not self.judgeInterval(numLoop):
            # intervalが合ってない場合
            return

        rismInpData = rismData.inpData
        # RISMDataをDataFrameに変換
        df = convertRISMDataToDataFrame(rismInpData, rismData)

        # ファイルサイズの推定パラメータの設定
        if self.__notWrittenYet:
            self.__checkEstimatedFileSize(df)

        # 出力先設定
        csvFile = 'rism_result_loop{}.csv'.format(numLoop)
        csvPath = self.__directory + '/' + csvFile
        # 出力桁数
        float_format = self.__float_format_rough

        # csv書き出し
        df.to_csv(csvPath, mode='w', index=False, float_format=float_format)
        self.__appendSaveFileList(csvPath)
        self.__checkFileSize(csvPath)
        if self.__notWrittenYet:
            self.__notWrittenYet = False

        # アニメーション用データ作成
        # https://qiita.com/yubais/items/c95ba9ff1b23dd33fde2
        # -> 別クラスに実装する

    def writeFinal(self, rismData):
        """
        最終結果の書き出し
        - csv書き出し
        - グラフ作成
        - 統計量計算
        """
        rismInpData = rismData.inpData
        # RISMDataをDataFrameに変換
        df = convertRISMDataToDataFrame(rismInpData, rismData)

        # 出力先設定
        csvFile = 'rism_result.csv'
        csvPath = self.__directory + '/' + csvFile
        # 出力桁数
        float_format = self.__float_format_detail

        # csv書き出し
        df.to_csv(csvPath, mode='w', index=False, float_format=float_format)
        self.__appendSaveFileList(csvPath)

        # 途中経過のアニメーション書き出し

        # 最終結果のグラフ書き出し

        # 統計量書き出し




class RISMInitializer():
    def __init__(self, rismInpData, method):
        self.rismInpData = rismInpData
        self.shape = rismInpData.corrFuncShape
        self.method = method

    def initializeEta0(self):
        # Etaの初期化
        if self.method == 'zeroization':
            Eta0 = np.zeros(self.shape)
            return Eta0
        else:
            return None

class RISMSolver():
    def __init__(self, configDict, closure, rismInpData, rismWriter):
        iniMethod = configDict.get('initialize', "zeroization")
        initializer = RISMInitializer(rismInpData, iniMethod)
        self.initializer = initializer

        self.mixingParam = configDict.get('mixingParam', 0.5)

        self.chargeUp = configDict.get('chargeUp', 0.25)
        factorList = np.arange(0,1,self.chargeUp)
        if factorList[-1] != 1:
            factorList = np.append(factorList, 1)
        self.chargeFactorList = factorList

        self.converge = configDict.get('converge', 1e-8)
        self.maxIter = configDict.get('maxIter', 1000)

        self.closure = closure
        self.rismInpData = rismInpData
        self.rismWriter = rismWriter
        self.rismData = None

    def solve(self):
        def __registerData(final=False):
            if not final and self.rismWriter.saveOnlyFinal():
                return

            t_X = t_W + P @ t_H
            data = RISMData(
                    inpData=self.rismInpData,
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
            self.rismData = data

            if not final:
                self.rismWriter.writeIntermediate(data)

            else: # final
                self.rismWriter.writeFinal(data)

        I = self.rismInpData.I
        P = self.rismInpData.P
        t_W = self.rismInpData.t_W
        Us = self.rismInpData.Us
        Ul = self.rismInpData.Ul
        grid = self.rismInpData.gridData
        closure = self.closure

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
                C = closure.apply(Us=Us, Ul=fUl, Eta=Eta)
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
                    print('TotalIter: {}, Factor: {}, Iter: {}, Error: {:.6e}, RangeEta: ({:.2e}, {:.2e})'.format(numTotalLoop, factor, numLoop, maxError, np.min(newEta),np.max(newEta)))
                if maxError < self.converge:
                    print('converged: TotalIter: {}, Factor: {}, Iter: {}, Error: {:.6e}, RangeEta: ({:.2e}, {:.2e})'.format(numTotalLoop, factor, numLoop, maxError, np.min(newEta),np.max(newEta)))
                    isConverged = True
                    __registerData(final=False)
                    break
                elif numLoop >= self.maxIter:
                    print('Maximum number of iterations exceeded: {}, Error: {}'.format(self.maxIter, maxError))
                    isConverged = False
                    __registerData(final=False)
                    break
                if self.rismWriter.judgeInterval(numTotalLoop):
                    # data生成
                    __registerData(final=False)

                # 更新
                Eta = self.mixingParam * newEta + (1-self.mixingParam) * Eta

            # 初期値更新
            Eta0 = Eta

        # 最終結果のdata生成及び書き出し
        __registerData(final=True)


    def giveResult(self):
        if self.rismData is None:
            raise RuntimeError()

        return self.rismData

class XRISMSolver(RISMSolver):
    def __init__(self, configDict, closure, rismInpData, rismWriter):
        super().__init__(configDict, closure, rismInpData, rismWriter)


    def solve(self):
        def __registerData(final=False):
            if not final and self.rismWriter.saveOnlyFinal():
                return
            t_C = t_Cl + t_Cs
            t_H = t_Hl + t_Hs
            C = Cl + Cs
            H = Hl + Hs
            Eta = H - C
            t_X = t_W + P @ t_H
            data = XRISMData(
                    inpData=self.rismInpData,
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
            self.rismData = data

            if not final:
                self.rismWriter.writeIntermediate(data)

            else: # final
                self.rismWriter.writeFinal(data)

        I = self.rismInpData.I
        P = self.rismInpData.P
        t_W = self.rismInpData.t_W
        Us = self.rismInpData.Us
        Ul1 = self.rismInpData.Ul
        Cl1 = - self.rismInpData.Fb
        t_Cl1 = - self.rismInpData.t_Fb
        Clc1 = self.rismInpData.Fbc # Ul+Cl
        grid = self.rismInpData.gridData
        closure = self.closure

        Etas0 = self.initializer.initializeEta0()
        Cs = self.initializer.initializeEta0() # TODO 別のメソッドを定義する
        Hs = self.initializer.initializeEta0() # TODO 別のメソッドを定義する

        numTotalLoop = 0
        for factor in self.chargeFactorList:
            # 初期化
            Ul = Ul1 * factor**2
            Cl = Cl1 * factor**2
            t_Cl = t_Cl1 * factor**2
            Clc = Clc1 * factor**2
            U = Us + Ul
            Etas = Etas0
            isConverged = False

            # 長距離part
            t_Hl = t_W @ t_Cl @ t_W @ np.linalg.inv(I - P @ t_Cl @ t_W)
            Hl = grid.ifft3d_spsymm(t_Hl)
            t_Xl = t_W + P @ t_Hl
            t_XlT = t_Xl.transpose(0,2,1) # 転置

            # 短距離part
            numLoop = 0
            while True:
                numLoop +=1
                numTotalLoop += 1
                Cs = closure.apply(Us=Us, Ul=Ul, Cs=Cs, Cl=Cl, Clc=Clc, Hs=Hs, Hl=Hl, Etas=Etas)
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
                    print('TotalIter: {}, Factor: {}, Iter: {}, Error: {:.6e}, RangeEtas: ({:.2e}, {:.2e})'.format(numTotalLoop, factor, numLoop, maxError, np.min(newEtas),np.max(newEtas)))
                if maxError < self.converge:
                    print('converged: TotalIter: {}, Factor: {}, Iter: {}, Error: {:.6e}, RangeEtas: ({:.2e}, {:.2e})'.format(numTotalLoop, factor, numLoop, maxError, np.min(newEtas),np.max(newEtas)))
                    isConverged = True
                    __registerData(final=False)
                    break
                elif numLoop >= self.maxIter:
                    print('Maximum number of iterations exceeded: {}, Error: {}'.format(self.maxIter, maxError))
                    isConverged = False
                    __registerData(final=False)
                    break
                if self.rismWriter.judgeInterval(numTotalLoop):
                    # data生成
                    __registerData(final=False)

                # 更新
                Etas = self.mixingParam * newEtas + (1-self.mixingParam) * Etas

            # 初期値更新
            Etas0 = Etas

        # 最終結果のdata生成及び書き出し
        __registerData(final=True)




class ThreeDimRISM():
    def __init__(self):
        pass
