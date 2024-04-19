import numpy as np
from scipy.special import erf, erfc


def generateLennardJonesPotMatrix(sigma1, sigma2, eps1, eps2, r, beta):
    """
    sigma1: np.ndarray: shape: (numsiteA,): A
    sigma2: np.ndarray: shape: (numsiteB,): A
    eps1  : np.ndarray: shape: (numsiteA,): kcal/mol
    eps2  : np.ndarray: shape: (numsiteB,): kcal/mol
    r     : np.ndarray: shape: (numgrid,) : A
    beta  : float                         : (kcal/mol)^-1

    return: np.ndarray: shape: (numgrid, numsiteA, numsiteB): 無次元
    """
    # Lorentz-Berthelot則
    Sigma = (sigma1[:,np.newaxis] + sigma2) / 2           # shape: (numsiteA,numsiteB) # LJ sigma_ij
    Eps = np.sqrt(eps1[:,np.newaxis]*eps2)                # shape: (numsiteA,numsiteB) # LJ eps_ij
    __sigmar6 = (Sigma / r[:,np.newaxis,np.newaxis])**6 # shape: (numgrid, numsiteA, numsiteB)
    Us = beta * 4 * Eps * (__sigmar6**2 - __sigmar6)

    return Us


def generateCoulombPotMatrix(z1, z2, r, beta):
    """
    z1  : np.ndarray: shape: (numsiteA,): e
    z2  : np.ndarray: shape: (numsiteB,): e
    r   : np.ndarray: shape: (numgrid,) : A
    beta: float                         : (kcal/mol)^-1
    """
    return generateFbondMatrix(z1, z2, r, beta, np.inf)

def generateFbondMatrix(z1, z2, r, beta, alpha):
    """

    """
    ZZ = z1[:,np.newaxis] * z2
    Fb = beta * 332.053 * ZZ / r[:,np.newaxis,np.newaxis] * erf(alpha * r[:,np.newaxis,np.newaxis])
    return Fb

def generateComplementaryFbondMatrix(z1, z2, r, beta, alpha):
    """
    Ul - Fb
    """
    ZZ = z1[:,np.newaxis] * z2
    Fbc = beta * 332.053 * ZZ / r[:,np.newaxis,np.newaxis] * erfc(alpha * r[:,np.newaxis,np.newaxis])
    return Fbc

def generateFourierSpaceFbondMatrix(z1, z2, k, beta, alpha):
    """

    """
    ZZ = z1[:,np.newaxis] * z2
    t_Fb = beta * 332.053 * ZZ * 4*np.pi / (k[:,np.newaxis,np.newaxis]**2) * np.exp(-(k[:,np.newaxis,np.newaxis]/2/alpha)**2)
    return t_Fb