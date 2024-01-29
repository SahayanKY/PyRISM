import numpy as np
from scipy.fft import dst

class GridData():
    def __init__(self, dr=None, numgrid=None, ffttype=None):
        if dr is None or numgrid is None:
            raise ValueError()
        if ffttype is None:
            ffttype = 4

        self.__numgrid = numgrid
        self.__ffttype = ffttype
        self.__dr = dr

        # ffttypeに応じてグリッド生成
        # 暫定的にffttype4の場合を記述
        self.__r = (np.arange(numgrid) + 0.5) * dr
        self.__k = (np.arange(numgrid) + 0.5) * dk
        self.__dk = np.pi / dr / numgrid

    def give_r(self):
        return self.__r
    def give_k(self):
        return self.__k
    def give_dr(self):
        return self.__dr
    def give_dk(self):
        return self.__dk

    def fft3d_spsymm(self, f):
        """
        f: shape: (numgrid, *, *), or (numgrid, *)
        """
        r = self.__r
        dr = self.__dr
        k = self.__k
        t_f = 4*np.pi/k[:,np.newaxis,np.newaxis] * dst(r[:,np.newaxis,np.newaxis] * f, type=self.__ffttype, axis=0) / 2 * dr
        return t_f

    def ifft3d_spsymm(self, t_f):
        """
        t_f: shape: (numgrid, *, *), or (numgrid, *)
        """
        r = self.__r
        k = self.__k
        dk = self.__dk
        f = 1/(2*np.pi**2 * r[:,np.newaxis,np.newaxis]) * dst(k[:,np.newaxis,np.newaxis] * t_f, type=self.__ffttype, axis=0) / 2 * dk
        return f
