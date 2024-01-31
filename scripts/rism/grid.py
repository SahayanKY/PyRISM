import numpy as np
from scipy.fft import dst

class GridData():
    def __init__(self, gridDict):
        numgrid = gridDict['n'] # グリッド数
        dr = gridDict['dr'] # 刻み幅: A
        ffttype = gridDict.get('ffttype', 4) # 離散フーリエ変換タイプ

        dk = np.pi / dr / numgrid
        r = (np.arange(numgrid) + 0.5) * dr
        k = (np.arange(numgrid) + 0.5) * dk

        self.numgrid = numgrid
        self.__ffttype = ffttype
        self.r = r
        self.k = k
        self.dr = dr
        self.dk = dk


    def fft3d_spsymm(self, f):
        """
        f: shape: (numgrid, *, *), or (numgrid, *)
        """
        r = self.r
        dr = self.dr
        k = self.k
        t_f = 4*np.pi/k[:,np.newaxis,np.newaxis] * dst(r[:,np.newaxis,np.newaxis] * f, type=self.__ffttype, axis=0) / 2 * dr
        return t_f

    def ifft3d_spsymm(self, t_f):
        """
        t_f: shape: (numgrid, *, *), or (numgrid, *)
        """
        r = self.r
        k = self.k
        dk = self.dk
        f = 1/(2*np.pi**2 * r[:,np.newaxis,np.newaxis]) * dst(k[:,np.newaxis,np.newaxis] * t_f, type=self.__ffttype, axis=0) / 2 * dk
        return f
