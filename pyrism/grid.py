import numpy as np
from scipy.fft import dst

class GridData():
    def __init__(self, r, dr, k, dk, ffttype=4):
        self.numgrid = len(r)
        self.__r = r
        self.__dr = dr
        self.__k = k
        self.__dk = dk
        self.__ffttype = ffttype

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
